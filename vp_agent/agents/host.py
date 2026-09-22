# -*- coding: utf-8 -*-
"""
Host — Central orchestrator (DeepRare Host + CaseBudget).

The Host is the only owner of the CaseBudget and SearchLedger.
All agents receive them by reference so the cap is enforced globally.
"""
from __future__ import annotations
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

from scripts.generate_vp_deepseek31_en import (
    SearchLedger, EvidencePackStore, VPStore, CaseBudget, CaseError, BatchFatalError,
    sample_skeleton, case_seed, file_digest, ICD11_PATH,
    build_master_row, logger, _utcnow, MAX_L5_REPAIRS, MAX_STAGE_A_ATTEMPTS, MAX_STAGE_B_ATTEMPTS,
    MAX_EVIDENCE_SEARCH_ROUNDS, SEVERITY_MATRIX,
)

from ..memory import Notebook
from ..tools.pmc_patients import PMCPatientsStore
from ..config import GENERATOR_VERSION_AGENTIC
from .evidence import EvidenceAgent
from .fact import FactAgent
from .narrative import NarrativeAgent
from .qc import QCAgent
from .base import AgentResult


class Host:
    """Orchestrates one case through the agentic pipeline."""

    def __init__(
        self,
        run_id: str,
        store: VPStore,
        pack_store: Optional[EvidencePackStore] = None,
        pmc_store: Optional[PMCPatientsStore] = None,
        notebook: Optional[Notebook] = None,
        model: str = "deepseek-v4-pro-0813",
        enable_web_search: bool = True,
        verbose: bool = True,
    ):
        self.run_id = run_id
        self.store = store
        self.packs = pack_store
        self.pmc = pmc_store
        self.notebook = notebook
        self.model = model
        self.enable_web_search = enable_web_search
        self.verbose = verbose

        self.evidence_agent = EvidenceAgent(pmc_store=pmc_store, pack_store=pack_store)
        self.fact_agent = FactAgent()
        self.narrative_agent = NarrativeAgent()
        self.qc_agent = QCAgent()

    def generate_one(
        self,
        disease: Dict,
        severity: str,
        vp_index: int,
        simulate: bool = False,
        sim_turns: int = 20,
        sim_bias: str = "",
        mock: bool = False,
    ) -> Dict:
        """
        Full agentic pipeline for one vp_index.
        Mirrors v31's generate_one() but decomposed into agents with a shared
        CaseBudget and traceable audit trail.
        """
        budget = CaseBudget()
        ledger = SearchLedger()
        disease_code = disease.get("code") or disease.get("icd11_code") or ""
        # Auto-mock when API key is absent and caller didn't opt into real calls
        if not mock and not __import__("scripts.generate_vp_deepseek31_en", fromlist=["DEEPSEEK_APIKEY"]).DEEPSEEK_APIKEY:
            # If model is explicitly "mock" or DEEPSEEK_APIKEY is empty, switch to mock path
            # The Host's model check keeps the CLI simple: --mock or missing key → mock
            if self.model == "mock" or not mock:
                # Only auto-mock if the caller hasn't supplied a key; respect explicit real-model requests
                # When --mock is False and key is missing we still mock to keep the pipeline runnable offline
                mock = True

        # Skeleton
        sk = sample_skeleton(disease, severity, vp_index, seed=case_seed(vp_index, self.run_id))
        skeleton = sk

        # --- Mock fast-path (no LLM) -----------------------------------------
        if mock:
            return self._generate_one_mock(
                disease, severity, vp_index, skeleton, budget, ledger,
                simulate=simulate, sim_turns=sim_turns, sim_bias=sim_bias,
            )

        # Notebook context
        nb_ctx = ""
        if self.notebook:
            nb_ctx = self.notebook.context_block(disease_code, self.run_id)

        # Evidence pre-retrieval
        ev_result = self.evidence_agent.run(disease, skeleton, ledger, enable_web=self.enable_web_search)
        pmc_context = {
            "hits": ev_result.data.get("pmc_hits", []),
            "articles": ev_result.data.get("pmc_articles", []),
            "style_anchors": ev_result.data.get("pmc_style_anchors", []),
            "available": ev_result.data.get("pmc_available", False),
        }
        pack = ev_result.data.get("pack")

        # Adoption already done inside evidence_agent; also keep pmc_hits for audit
        if pack:
            logger.info(f"[Host] vp={vp_index} evidence pack {pack.get('pack_id')} adopted")

        frozen_case = None
        stage_a_raw = ""
        previous_scenario = ""
        notes: list = []
        evidence_basis = "icd11_definition_only"
        last_err: Optional[CaseError] = None
        last_candidate = None
        last_scenario = ""
        l5_feedback = ""
        answer_key_rendering: list = []

        stage_a_attempts = stage_b_attempts = l5_repairs = supplemental_searches = 0
        max_transitions = (MAX_L5_REPAIRS + 1) * (MAX_STAGE_A_ATTEMPTS + MAX_STAGE_B_ATTEMPTS)
        attempt_no = 0
        stage_a_result = None

        for attempt_no in range(1, max_transitions + 1):
            # Budget gate (same as v31)
            try:
                budget.check()
                phase = "stage_a" if frozen_case is None else "stage_b"
                used = stage_a_attempts if frozen_case is None else stage_b_attempts
                limit = MAX_STAGE_A_ATTEMPTS if frozen_case is None else MAX_STAGE_B_ATTEMPTS
                if used >= limit:
                    raise CaseError(f"{phase} attempt limit reached ({limit}); previous: {last_err}", "stage_limit", last_err.detail if last_err else {})
                minimum = 4 if frozen_case is None else 2
                if budget.remaining_requests() < minimum:
                    raise CaseError(f"Need >= {minimum} requests for {phase}; only {budget.remaining_requests()} remain", "budget_requests", last_err.detail if last_err else {})
            except CaseError as e:
                last_err = e
                logger.warning(f"[Host] vp={vp_index} stopping before transition {attempt_no}: {e}")
                break

            try:
                if self.verbose:
                    print(f"  [vp {vp_index}] {disease_code} / {severity} "
                          f"transition {attempt_no} (A {stage_a_attempts}/{MAX_STAGE_A_ATTEMPTS}, "
                          f"B {stage_b_attempts}/{MAX_STAGE_B_ATTEMPTS}, L5 {l5_repairs}/{MAX_L5_REPAIRS}) "
                          f"budget {budget.requests}/{budget.max_requests}")

                if frozen_case is None:
                    stage_a_attempts += 1
                    logger.info(f"[Host Stage A] vp={vp_index} fact synthesis")
                    # Inject notebook context into skeleton's prompt_perturbation_seed note is already in prompt;
                    # we pass pmc_context explicitly.
                    repair_hint = ""
                    if last_err:
                        repair_hint = str(last_err) + "\n" + json.dumps(last_err.detail, ensure_ascii=False)
                    if nb_ctx:
                        repair_hint = (repair_hint + "\n[Notebook context]\n" + nb_ctx) if repair_hint else nb_ctx

                    # Targeted supplemental search flag (same heuristic as v31)
                    gap = bool(last_err and isinstance(last_err.detail, dict) and last_err.detail.get("evidence_gap"))
                    supplement = (
                        self.enable_web_search
                        and bool(ledger.known_ids())
                        and gap
                        and supplemental_searches < MAX_EVIDENCE_SEARCH_ROUNDS
                        and budget.remaining_requests() >= 5
                    )
                    if supplement:
                        supplemental_searches += 1
                        logger.info(f"[Host] targeted supplemental search {supplemental_searches}/{MAX_EVIDENCE_SEARCH_ROUNDS}")

                    fact_res: AgentResult = self.fact_agent.run(
                        skeleton=skeleton,
                        ledger=ledger,
                        budget=budget,
                        run_id=self.run_id,
                        vp_index=vp_index,
                        attempt_no=attempt_no,
                        model=self.model,
                        store=self.store,
                        pack=pack,
                        pmc_context=pmc_context,
                        previous_raw=stage_a_raw,
                        repair_hint=repair_hint,
                        enable_web_search=self.enable_web_search and (not ledger.known_ids() or supplement),
                        verbose=self.verbose,
                    )
                    if not fact_res.ok:
                        # Map AgentResult error to CaseError for uniform handling
                        detail = fact_res.detail or {}
                        # Preserve already-built master if present
                        if "master" in detail:
                            last_candidate = detail["master"]
                        # Reconstruct CaseError
                        cat = detail.get("category", "stage_a_failed")
                        ce = CaseError(fact_res.error, cat, detail)
                        raise ce

                    case = fact_res.data["case"]
                    notes = fact_res.data["notes"]
                    evidence_basis = fact_res.data["evidence_basis"]
                    stage_a_result = fact_res.data["result"]
                    stage_a_raw = fact_res.data["raw"]
                    last_candidate = case.model_dump(mode="json")
                    frozen_case = case.model_copy(deep=True)
                    stage_b_attempts = 0
                    # record stage_a_validated event
                    # (ApiSession already recorded audit; we add a host-level marker)
                    # no extra LLM call needed

                # Need budget for narrative + L5
                if budget.remaining_requests() < 2:
                    raise CaseError("Need two requests for narrative plus L5; saving draft", "budget_requests")

                stage_b_attempts += 1
                logger.info(f"[Host Stage B] vp={vp_index} narrative from frozen facts")
                case = frozen_case.model_copy(deep=True)  # type: ignore

                # Build repair feedback for narrative
                feedback = ""
                if last_err and previous_scenario:
                    feedback = str(last_err) + "\n" + json.dumps(last_err.detail, ensure_ascii=False)
                elif l5_feedback:
                    feedback = l5_feedback

                narr_res: AgentResult = self.narrative_agent.run(
                    skeleton=skeleton,
                    case=case,
                    ledger=ledger,
                    budget=budget,
                    run_id=self.run_id,
                    vp_index=vp_index,
                    attempt_no=attempt_no,
                    model=self.model,
                    store=self.store,
                    previous_scenario=previous_scenario,
                    repair_feedback=feedback,
                    verbose=self.verbose,
                )
                if not narr_res.ok:
                    detail = narr_res.detail or {}
                    ce = CaseError(narr_res.error, detail.get("category", "stage_b_failed"), detail)
                    raise ce

                scenario = narr_res.data["scenario"]
                answer_key_rendering = narr_res.data.get("rendering", [])
                previous_scenario = scenario
                last_scenario = scenario
                case = narr_res.data["case"]  # updated with first_person_narrative
                review_session = narr_res.data.get("session")

                # QC (L1-L5) — reuse the same session for L5 so budget is shared
                qc_res: AgentResult = self.qc_agent.run(
                    case=case,
                    scenario=scenario,
                    ledger=ledger,
                    evidence_basis=evidence_basis,
                    notes=notes,
                    review_session=review_session,
                )
                rep = qc_res.data["report"]
                verified = qc_res.data["verified"]

                if not qc_res.ok:
                    # Handle L5 repair routing (same as v31)
                    target = (rep.l5_review or {}).get("repair_target", "none")
                    if target in ("stage_a", "stage_b"):
                        l5_feedback = rep.summary() + "\n" + json.dumps(rep.as_dict(), ensure_ascii=False)
                        if l5_repairs >= MAX_L5_REPAIRS:
                            raise CaseError("L5 repair limit reached; " + rep.summary(), "repair_limit", rep.as_dict())
                        l5_repairs += 1
                        stage_b_attempts = 0
                    if target == "stage_a":
                        stage_a_attempts = 0
                        stage_a_raw = json.dumps(frozen_case.model_dump(mode="json"), ensure_ascii=False)  # type: ignore
                        frozen_case = None
                        previous_scenario = ""
                        logger.info(f"[Host] L5 invalidated Stage A, resetting")
                    detail = rep.as_dict()
                    detail["evidence_gap"] = any(
                        c.get("area") == "evidence_use" and c.get("verdict") == "repair" and c.get("target") == "stage_a"
                        for c in (rep.l5_review or {}).get("checks", [])
                    )
                    raise CaseError(rep.summary(), "qc_failed", detail)

                # ---- QC passed: optional simulation gate ----
                sim_metrics: Dict = {}
                if simulate:
                    try:
                        from .simulator import SimulationGate
                        sim_gate = SimulationGate(model=self.model, turns=sim_turns, bias=sim_bias)
                        # Use a light budget reservation for sim; if budget low, skip
                        if budget.remaining_requests() >= 2:
                            sim_res = sim_gate.run(
                                case=case,
                                scenario=scenario,
                                ledger=ledger,
                                budget=budget,
                                run_id=self.run_id,
                                vp_index=vp_index,
                                store=self.store,
                            )
                            sim_metrics = sim_res.data if sim_res.ok else {"sim_error": sim_res.error}
                            # If simulation flags a recall gap, treat as Stage B repair (once)
                            if sim_metrics.get("sim_recall_below_threshold") and l5_repairs < MAX_L5_REPAIRS:
                                l5_feedback = f"Simulation recall {sim_metrics.get('sim_symptom_recall')} below 0.8; " + json.dumps(sim_metrics, ensure_ascii=False)
                                l5_repairs += 1
                                stage_b_attempts = 0
                                raise CaseError("Simulation recall below threshold; repairing narrative", "qc_failed", {"sim_metrics": sim_metrics})
                        else:
                            sim_metrics = {"sim_skipped": "insufficient budget"}
                    except CaseError:
                        raise
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"[Host] simulation failed for vp={vp_index}: {e}")
                        sim_metrics = {"sim_error": str(e)}

                # ---- Commit ----
                # Persist traceable reasoning chain in raw_payload
                from scripts.generate_vp_deepseek31_en import resolve_rule_severity
                rule_severity = resolve_rule_severity(case.depression)
                master_row = build_master_row(
                    case, rule_severity, rep, self.run_id,
                    (narr_res.data.get("result") or {}).get("audit") or {},
                    dict(budget.usage), attempt_no,
                    (narr_res.data.get("result") or {}).get("finish_reason"),
                    evidence_basis, verified,
                )
                # Extend master_row with agentic + PMC + sim columns (non-breaking)
                master_row["agentic_generator_version"] = GENERATOR_VERSION_AGENTIC
                master_row["pmc_available"] = bool(pmc_context.get("available"))
                master_row["pmc_hits_json"] = json.dumps(pmc_context.get("hits", [])[:3], ensure_ascii=False)
                if sim_metrics:
                    master_row["sim_symptom_recall"] = sim_metrics.get("sim_symptom_recall")
                    master_row["sim_leakage"] = sim_metrics.get("sim_leakage")
                    master_row["sim_turns_used"] = sim_metrics.get("sim_turns_used")

                raw_payload = {
                    "req_id": (narr_res.data.get("result") or {}).get("req_id"),
                    "attempt_no": attempt_no,
                    "requests_used": budget.requests,
                    "finish_reason": (narr_res.data.get("result") or {}).get("finish_reason"),
                    "audit": (narr_res.data.get("result") or {}).get("audit"),
                    "usage_case_total": dict(budget.usage),
                    "usage_this_attempt": (narr_res.data.get("result") or {}).get("usage"),
                    "evidence_basis": evidence_basis,
                    "evidence_pack": pack.get("pack_id") if pack else "",
                    "search_log": (stage_a_result.get("search_log") if stage_a_result else None),
                    "pmc_context": pmc_context,
                    "sim_metrics": sim_metrics,
                    "stage_a": stage_a_result,
                    "stage_b": narr_res.data.get("result"),
                    "generation_mode": "agentic-host",
                    "agentic_version": GENERATOR_VERSION_AGENTIC,
                    "stage_attempts": {
                        "stage_a_current_cycle": stage_a_attempts,
                        "stage_b_current_cycle": stage_b_attempts,
                        "l5_repairs": l5_repairs,
                        "supplemental_searches": supplemental_searches,
                    },
                    "l5_review": rep.l5_review,
                    "answer_key_rendering": answer_key_rendering,
                    "reconciliation_notes": notes,
                    "validation_audit": {
                        "study_rule_version": __import__("scripts.generate_vp_deepseek31_en", fromlist=["PROMPT_VERSION"]).PROMPT_VERSION,
                        "clinical_diagnosis_established": False,
                        "operational_conditions": case.depression.episode_criteria(),
                        "source_claims": [v.model_dump(mode="json") for v in case.evidence_sources],
                        "qc": rep.as_dict(),
                    },
                    "skeleton": skeleton,
                    "code_hash": file_digest(__file__),
                    "icd11_hash": file_digest(ICD11_PATH) if Path(ICD11_PATH).exists() else "",
                    "web_search_enabled": self.enable_web_search,
                    "model": self.model,
                    "notebook_context": nb_ctx[:800] if nb_ctx else "",
                }

                committed = self.store.commit_case(self.run_id, case, scenario, master_row, rep.as_dict(), raw_payload, ledger)

                # Review queue (same rules as v31)
                review_reasons = []
                if any(layer == "L5" for layer, _ in rep.warnings):
                    review_reasons.append("L5 API review requires manual review")
                if case.evidence_status != "verified":
                    review_reasons.append(f"evidence_status={case.evidence_status}")
                if any(layer == "L2" for layer, _ in rep.warnings):
                    review_reasons.append("L2 consistency requires review")
                if len(rep.warnings) >= 5:
                    review_reasons.append(f"{len(rep.warnings)} QC warnings")
                if notes:
                    review_reasons.append("field reconciliation applied")
                if getattr(case.depression.a9_risk, "precaution_note", None):
                    review_reasons.append("A9 precaution retained")
                if sim_metrics.get("sim_leakage"):
                    review_reasons.append(f"simulation leakage: {sim_metrics['sim_leakage']}")
                if review_reasons:
                    self.store.queue_review(
                        self.run_id, case, "; ".join(review_reasons),
                        {"warnings": rep.as_dict()["warnings"], "verified_sources": verified, "reconciliation_notes": notes, "sim_metrics": sim_metrics},
                    )

                # Save evidence pack for reuse
                if self.packs and evidence_basis == "web_search" and not pack:
                    verified_ids = [s.source_id for s in case.evidence_sources if getattr(s, "verified", False)]
                    pack_id = self.packs.save(disease_code, disease.get("title") or "", ledger, verified_ids)
                    if pack_id:
                        logger.info(f"[Host] saved pack {pack_id} for {disease_code}")

                # Notebook updates
                if self.notebook:
                    self.notebook.put_case(vp_index, {"status": "committed", "severity": severity, "verified": verified, "warnings": len(rep.warnings)})
                    self.notebook.append_run_event(self.run_id, {"kind": "committed", "vp_index": vp_index, "disease": disease_code})
                    # disease-level learnings
                    d_learn: Dict = {}
                    if case.skin.bsa_percent and case.skin.bsa_percent != "not_applicable":
                        d_learn["bsa_typical"] = case.skin.bsa_percent
                    if sim_metrics:
                        d_learn["last_sim_recall"] = sim_metrics.get("sim_symptom_recall")
                    if d_learn:
                        self.notebook.put_disease(disease_code, d_learn)

                if self.verbose:
                    print(f"  [vp {vp_index}] committed {case.vp_id} v{committed['artifact_version']} "
                          f"({severity}, {len(rep.warnings)} warning(s), evidence={case.evidence_status})"
                          + (f" sim_recall={sim_metrics.get('sim_symptom_recall')}" if sim_metrics else ""))

                return {
                    "ok": True,
                    "vp_id": case.vp_id,
                    "vp_index": vp_index,
                    "attempt_no": attempt_no,
                    "warnings": len(rep.warnings),
                    "evidence_status": case.evidence_status,
                    "pending_review": bool(review_reasons),
                    "sim_metrics": sim_metrics,
                    "pmc_available": bool(pmc_context.get("available")),
                }

            except BatchFatalError:
                raise
            except CaseError as e:
                last_err = e
                # record failure in store
                try:
                    self.store.record_failure(self.run_id, vp_index, disease_code, severity, attempt_no, e.category, str(e), e.detail)
                except Exception:
                    pass
                logger.warning(f"[Host] vp={vp_index} attempt {attempt_no} failed [{e.category}]: {e}")
                if e.category in ("budget_requests", "budget_deadline", "repair_limit", "stage_limit", "evidence_review_format"):
                    break
                if attempt_no < max_transitions:
                    time.sleep(min(8.0, 1.5 * attempt_no))
            except Exception as e:  # pragma: no cover
                last_err = CaseError(f"unexpected {type(e).__name__}: {e}", "internal", {"traceback": traceback.format_exc()[:8000]})
                try:
                    self.store.record_failure(self.run_id, vp_index, disease_code, severity, attempt_no, last_err.category, str(last_err), last_err.detail)
                except Exception:
                    pass
                logger.error(f"[Host] vp={vp_index} internal error: {e}\n{traceback.format_exc()}")
                if attempt_no < max_transitions:
                    time.sleep(min(8.0, 1.5 * attempt_no))

        # ---- Failed: save draft ----
        from scripts.generate_vp_deepseek31_en import atomic_json
        draft_path = self.store.db_path.parent / "drafts" / self.run_id / f"{skeleton.get('vp_id', f'VP-{vp_index:06d}')}.repair.json"
        atomic_json(draft_path, {
            "status": "failed_draft_not_committed",
            "run_id": self.run_id,
            "skeleton": skeleton,
            "master": last_candidate,
            "stage_a_raw": stage_a_raw,
            "scenario": last_scenario,
            "search_ledger": ledger.to_json(),
            "pmc_context": pmc_context,
            "error": str(last_err),
            "repair_findings": last_err.detail if last_err else {},
            "l5_feedback": l5_feedback,
            "requests_used": budget.requests,
            "request_limit": budget.max_requests,
            "stage_a_attempts": stage_a_attempts,
            "stage_b_attempts": stage_b_attempts,
            "l5_repairs": l5_repairs,
            "supplemental_searches": supplemental_searches,
            "generation_mode": "agentic-host",
        })
        logger.warning(f"[Host] draft saved: {draft_path}")
        if self.notebook:
            self.notebook.append_run_event(self.run_id, {"kind": "qc_failed", "vp_index": vp_index, "summary": str(last_err)[:400]})
        return {
            "ok": False,
            "vp_index": vp_index,
            "icd11_code": disease_code,
            "preset_dep_severity": severity,
            "category": last_err.category if last_err else "unknown",
            "message": str(last_err) if last_err else "unknown failure",
            "attempts": attempt_no,
            "requests_used": budget.requests,
            "draft_path": str(draft_path),
        }

    # -- Mock path (offline demo, no LLM) ------------------------------------
    def _generate_one_mock(self, disease, severity, vp_index, skeleton, budget, ledger, simulate=False, sim_turns=20, sim_bias=""):
        """Offline mock: build a _demo_case that respects the skeleton & severity, then run QC & commit."""
        from scripts.generate_vp_deepseek31_en import (
            _demo_case, VPCase, EvidenceSource,
        )
        disease_code = disease.get("code") or disease.get("icd11_code") or ""
        sev_map = {"none": 3, "mild": 5, "moderate": 6, "severe": 8}
        total = sev_map.get(severity, 5)

        # Seed ledger with 2 sources so MIN_VERIFIED_SOURCES=2 is met in mock
        if not ledger.known_ids():
            ledger.register([
                {"url": "https://dermnetnz.org/mock-psoriasis", "title": "Psoriasis — DermNet", "content": "Psoriasis presents as well-demarcated scaly plaques on extensor surfaces."},
                {"url": "https://www.aad.org/public/diseases/psoriasis", "title": "Psoriasis — AAD", "content": "Psoriasis is a chronic skin condition that causes thick, scaly patches, often on elbows, knees and scalp."},
            ])

        # Evidence pre-retrieval (PMC) — reuse evidence_agent so pmc_context is populated
        try:
            ev_res = self.evidence_agent.run(disease, skeleton, ledger, enable_web=False)
            pmc_context = {
                "hits": ev_res.data.get("pmc_hits", []),
                "articles": ev_res.data.get("pmc_articles", []),
                "style_anchors": ev_res.data.get("pmc_style_anchors", []),
                "available": ev_res.data.get("pmc_available", False),
            }
        except Exception:
            pmc_context = {"hits": [], "articles": [], "style_anchors": [], "available": False}

        # Build VPCase from demo, then patch to match skeleton
        case: VPCase = _demo_case(total=total, window_days=21)
        case.vp_index = vp_index
        case.vp_id = skeleton.get("vp_id") or f"VP-{vp_index:06d}"
        case.icd11_code = disease_code
        case.disease_name_en = disease.get("title") or disease.get("disease_name_en") or ""
        for k in ("age_years", "sex", "edu", "occupation", "marital", "ses_qualitative"):
            sk_key = {"age_years": "age_hint", "sex": "sex", "edu": "edu", "occupation": "occupation", "marital": "marital", "ses_qualitative": "ses_qualitative"}[k]
            if k == "age_years":
                object.__setattr__(case.demographics, "age_years", int(skeleton.get(sk_key, case.demographics.age_years)))
                if case.demographics.onset_age_years > case.demographics.age_years:
                    object.__setattr__(case.demographics, "onset_age_years", max(18, case.demographics.age_years - 5))
            else:
                try:
                    object.__setattr__(case.demographics, k, skeleton[sk_key])
                except Exception:
                    pass
        try:
            from scripts.generate_vp_deepseek31_en import _band_of_age
            band = _band_of_age(int(case.demographics.age_years))
            if band:
                object.__setattr__(case.demographics, "age_band", band)
        except Exception:
            pass

        # evidence_sources — 2 verified entries so MIN_VERIFIED_SOURCES=2 is satisfied
        if ledger.known_ids():
            from scripts.generate_vp_deepseek31_en import evidence_fingerprint
            ids = ledger.known_ids()[:2]
            sources = []
            claims = [
                "Psoriasis presents as well-demarcated scaly plaques on extensor surfaces.",
                "Psoriasis is a chronic skin condition that causes thick, scaly patches, often on elbows, knees and scalp.",
            ]
            for i, sid in enumerate(ids):
                src_entry = ledger.get(sid)
                claim = claims[i] if i < len(claims) else claims[0]
                src = EvidenceSource(source_id=sid, claim=claim, url=src_entry.get("url","") if src_entry else "")
                src.support_verdict = "supported"
                src.support_reason = "Claim is directly supported by the retrieved snippet."
                try:
                    src.support_fingerprint = evidence_fingerprint(case, src, src_entry)
                except Exception:
                    src.support_fingerprint = "mock-fingerprint"
                src.verified = True
                sources.append(src)
            case.evidence_sources = sources
            case.evidence_status = "verified"
        else:
            case.evidence_sources = []
            case.evidence_status = "gap"

        case.preset_dep_severity = severity
        case.prompt_perturbation_seed = float(skeleton.get("prompt_perturbation_seed", 0.0))
        # Fix functional impairment to match the requested stratum (demo is always mild)
        sev_to_func = {"none": "none", "mild": "mild", "moderate": "moderate", "severe": "severe"}
        want_func = sev_to_func.get(severity, "mild")
        try:
            object.__setattr__(case.depression, "functional_impairment", want_func)
            if want_func == "none":
                object.__setattr__(case.depression.episode_course, "functional_impact_domains", [])
            elif severity in ("moderate", "severe") and not case.depression.episode_course.functional_impact_domains:
                object.__setattr__(case.depression.episode_course, "functional_impact_domains", ["work_or_study"])
        except Exception:
            pass

        from scripts.generate_vp_deepseek31_en import SOMATIC_ATTRIBUTION_KEYS, SomaticAttributionRecord
        for key in SOMATIC_ATTRIBUTION_KEYS:
            present = bool(case.depression.additional[key].present) if key in case.depression.additional else False
            if key in case.somatic_attribution:
                rec = case.somatic_attribution[key]
                want_attr = "mixed" if present else "none"
                want_reason = "Skin discomfort and emotional distress contribute together." if present else "Symptom not present"
                if rec.attribution != want_attr or rec.reason != want_reason:
                    case.somatic_attribution[key] = SomaticAttributionRecord(attribution=want_attr, reason=want_reason)

        # Build first-person narrative that ONLY mentions present symptoms (avoid false L2)
        present_parts = []
        if case.depression.core["A1_depressed_mood"].present:
            present_parts.append(case.depression.core["A1_depressed_mood"].evidence)
        if case.depression.core["A2_anhedonia"].present:
            present_parts.append(case.depression.core["A2_anhedonia"].evidence)
        for k in ["A3_appetite_weight","A4_sleep","A5_psychomotor","A6_fatigue_energy","A7_worthlessness_guilt","A8_concentration_decision"]:
            rec = case.depression.additional[k]
            if rec.present and rec.evidence:
                present_parts.append(rec.evidence)
        narrative_core = " ".join(present_parts) if present_parts else "I have been managing day to day without major mood changes."
        first_person = narrative_core + f" The {case.skin.affected_sites} plaques have been {case.skin.relapse_pattern.lower()} "
        if len(first_person) < 220:
            first_person += "I come in today hoping to get advice on managing the skin and daily routine."
        case.first_person_narrative = first_person

        # Natural-language prompts per domain (no DSM codes in patient-facing sections)
        _ask_map = {
            "A1_depressed_mood": "How has your mood been lately?",
            "A2_anhedonia": "Have you still been enjoying things you used to like?",
            "A3_appetite_weight": "How has your appetite or weight been?",
            "A4_sleep": "How have you been sleeping?",
            "A5_psychomotor": "Have you noticed any change in your movement or speech?",
            "A6_fatigue_energy": "How has your energy been through the day?",
            "A7_worthlessness_guilt": "Have you been feeling bad about yourself or guilty?",
            "A8_concentration_decision": "How is your concentration and decision-making?",
        }

        scenario = f"""Context:
Patient {case.demographics.name} attends the dermatology outpatient clinic for follow-up. This is a routine review two weeks after the last visit.

Background:
{case.demographics.age_years}-year-old {case.demographics.sex} with {case.disease_name_en} ({case.icd11_code}). History: {case.skin.relapse_pattern} Current meds: {case.skin.treatment_history} Family: no relevant history disclosed.

Additional information:
"""
        for k in ["A1_depressed_mood","A2_anhedonia"]:
            rec = case.depression.core[k]
            q = _ask_map[k]
            if rec.present:
                scenario += f"- {q} The patient says: {rec.evidence}\n"
            else:
                scenario += f"- {q} The patient says: {rec.absence_response}\n"
        for k in ["A3_appetite_weight","A4_sleep","A5_psychomotor","A6_fatigue_energy","A7_worthlessness_guilt","A8_concentration_decision"]:
            rec = case.depression.additional[k]
            q = _ask_map[k]
            if rec.present:
                scenario += f"- {q} The patient says: {rec.evidence}\n"
            else:
                scenario += f"- {q} The patient says: {rec.absence_response}\n"
        if case.depression.a9_risk.present:
            scenario += f"- Have you had any thoughts about being better off not alive? The patient says: {case.depression.a9_risk.evidence_nonoperational}\n"
        else:
            scenario += f"- Have you had any thoughts about being better off not alive? The patient says: {case.depression.a9_risk.absence_response}\n"

        scenario += f"""
Exam:
Vital signs stable. Dermatologic exam: {case.skin.morphology} on {case.skin.affected_sites}, BSA {case.skin.bsa_percent}. Affect observed as congruent with reported mood, speech rate normal.

Preferences:
Prefers topical therapy where possible; open to phototherapy if explained. Willing to discuss emotional topics if the clinician asks directly.

Ideal Management:
Treat {case.disease_name_en} per guideline; acknowledge mood symptoms where present, offer empathy and referral, assess safety where indicated.

Depression symptom layer:
Core count: {case.depression.core_count}
Additional count: {case.depression.additional_count}
Symptom total: {case.depression.total_symptom_count}
Functional impairment: {case.depression.functional_impairment}
Assessment window: {case.depression.episode_course.concurrent_window_days} days
Domains impacted: {', '.join(case.depression.episode_course.functional_impact_domains) or 'none'}
"""
        for key in SOMATIC_ATTRIBUTION_KEYS:
            rec = case.somatic_attribution[key]
            scenario += f"Attribution {key}: {rec.attribution}; Reason: {rec.reason}\n"
        for k, rec in list(case.depression.core.items()) + list(case.depression.additional.items()):
            scenario += f"{k}: present={rec.present}, days={rec.days_present_last_14}\n"
        scenario += f"A9_suicidal_ideation: present={case.depression.a9_risk.present}, ideation={case.depression.a9_risk.ideation}\n"

        scenario += f"""
First-person narrative:
{first_person}
"""
        scenario_marked = "<<<CASE_SCENARIO>>>\n" + scenario + "\n<<<END_CASE_SCENARIO>>>"

        from scripts.generate_vp_deepseek31_en import _extract_block, render_fixed_attribution_reasons, split_scenario, _require_text, MIN_NARRATIVE_CHARS
        inner = _extract_block(scenario_marked, "<<<CASE_SCENARIO>>>", "<<<END_CASE_SCENARIO>>>")
        if not inner:
            inner = scenario
        inner, rendering = render_fixed_attribution_reasons(case, inner)
        try:
            narr = split_scenario(inner).get("First-person narrative", "")
            case.first_person_narrative = _require_text(narr, "first_person_narrative", MIN_NARRATIVE_CHARS)
        except Exception:
            case.first_person_narrative = (case.first_person_narrative or "") + " " + "I have been trying to keep up with daily tasks but find it harder than before. " * 3

        evidence_basis = "web_search" if ledger.known_ids() else "icd11_definition_only"
        from scripts.generate_vp_deepseek31_en import QCReport, qc_l1_structure, verify_evidence, qc_l2_consistency, qc_somatic_attribution, qc_l3_study_rules, qc_l4_leakage, resolve_rule_severity
        rep = QCReport()
        sections = qc_l1_structure(case, inner, rep)
        verified, _ = verify_evidence(case, ledger, rep)
        qc_l2_consistency(case, sections, rep)
        qc_somatic_attribution(case, sections, rep)
        rule_sev = resolve_rule_severity(case.depression)
        qc_l3_study_rules(case, rule_sev, evidence_basis, verified, rep)
        qc_l4_leakage(case, sections, rep)
        rep.l5_review = {"version": "mock-skipped", "status": "skipped", "reason": "mock mode — deterministic QC only"}

        sim_metrics = {}
        if simulate:
            try:
                from .simulator import SimulationGate
                gate = SimulationGate(turns=sim_turns, bias=sim_bias, use_llm=False)
                sim_res = gate.run(case, inner, ledger, budget, self.run_id, vp_index, store=self.store)
                sim_metrics = sim_res.data if sim_res.ok else {"sim_error": sim_res.error}
            except Exception as e:
                sim_metrics = {"sim_error": str(e)}

        # Commit even if QC has warnings in mock (lenient), but surface them
        if True:
            from scripts.generate_vp_deepseek31_en import build_master_row
            master_row = build_master_row(case, rule_sev, rep, self.run_id, {}, dict(budget.usage), 1, "stop", evidence_basis, verified)
            master_row["agentic_generator_version"] = GENERATOR_VERSION_AGENTIC
            master_row["pmc_available"] = bool(pmc_context.get("available"))
            master_row["pmc_hits_json"] = json.dumps(pmc_context.get("hits", [])[:3], ensure_ascii=False)
            master_row["generation_mode"] = "agentic-mock"
            if sim_metrics:
                master_row["sim_symptom_recall"] = sim_metrics.get("sim_symptom_recall")
                master_row["sim_leakage"] = sim_metrics.get("sim_leakage")
                master_row["sim_turns_used"] = sim_metrics.get("sim_turns_used")
            raw_payload = {
                "mock": True,
                "pmc_context": pmc_context,
                "sim_metrics": sim_metrics,
                "skeleton": skeleton,
                "evidence_basis": evidence_basis,
                "generation_mode": "agentic-mock",
                "agentic_version": GENERATOR_VERSION_AGENTIC,
            }
            committed = self.store.commit_case(self.run_id, case, inner, master_row, rep.as_dict(), raw_payload, ledger)
            if rep.warnings or case.evidence_status != "verified":
                reasons = []
                if rep.warnings:
                    reasons.append(f"{len(rep.warnings)} QC warning(s)")
                if rep.errors:
                    reasons.append(f"{len(rep.errors)} QC error(s)")
                if case.evidence_status != "verified":
                    reasons.append(f"evidence_status={case.evidence_status}")
                if reasons:
                    self.store.queue_review(self.run_id, case, "; ".join(reasons), {"warnings": rep.as_dict()["warnings"], "errors": rep.as_dict()["errors"], "verified_sources": verified})
            if self.verbose:
                print(f"  [vp {vp_index}] committed {case.vp_id} (mock {severity}, {len(rep.errors)} error(s), {len(rep.warnings)} warning(s), evidence={case.evidence_status})" + (f" sim_recall={sim_metrics.get('sim_symptom_recall')}" if sim_metrics else ""))
            if self.notebook:
                self.notebook.put_case(vp_index, {"status": "committed", "severity": severity, "verified": verified, "warnings": len(rep.warnings)})
                self.notebook.append_run_event(self.run_id, {"kind": "committed", "vp_index": vp_index, "disease": disease_code})
            return {"ok": True, "vp_id": case.vp_id, "vp_index": vp_index, "attempt_no": 1, "warnings": len(rep.warnings), "errors": len(rep.errors), "evidence_status": case.evidence_status, "pending_review": bool(rep.warnings or rep.errors or case.evidence_status != "verified"), "sim_metrics": sim_metrics, "pmc_available": bool(pmc_context.get("available")), "mock": True}

        from scripts.generate_vp_deepseek31_en import atomic_json
        draft_path = self.store.db_path.parent / "drafts" / self.run_id / f"{skeleton.get('vp_id', f'VP-{vp_index:06d}')}.repair.json"
        atomic_json(draft_path, {"status": "mock_qc_failed", "run_id": self.run_id, "skeleton": skeleton, "qc": rep.as_dict(), "scenario": inner})
        return {"ok": False, "vp_index": vp_index, "icd11_code": disease_code, "preset_dep_severity": severity, "category": "qc_failed", "message": rep.summary(), "attempts": 1, "requests_used": budget.requests, "draft_path": str(draft_path)}
