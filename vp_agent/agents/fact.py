# -*- coding: utf-8 -*-
"""
FactAgent — Stage A (frozen facts).

Wraps v31's build_stage_a_prompt + ApiSession + canonicalize + stage_a_qc
so the agentic host can treat it as a single tool call.
"""
from __future__ import annotations
import json
from typing import Dict, Optional

from scripts.generate_vp_deepseek31_en import (
    SearchLedger, EvidencePackStore, ApiSession, CaseBudget,
    build_stage_a_prompt, canonicalize, stage_a_qc,
    _load_json_tolerant, _STAGE_A_PLACEHOLDER, CaseError, logger,
)
from .base import Agent, AgentResult


class FactAgent(Agent):
    name = "fact"

    def run(
        self,
        skeleton: Dict,
        ledger: SearchLedger,
        budget: CaseBudget,
        run_id: str,
        vp_index: int,
        attempt_no: int,
        model: str,
        store=None,
        pack: Optional[Dict] = None,
        pmc_context: Optional[Dict] = None,
        previous_raw: str = "",
        repair_hint: str = "",
        enable_web_search: bool = True,
        verbose: bool = True,
    ) -> AgentResult:
        # Build prompt — inject PMC style anchors as an extra block
        prompt = build_stage_a_prompt(skeleton, pack)
        if pmc_context and pmc_context.get("style_anchors"):
            anchors = "\n".join(f"- {a}" for a in pmc_context["style_anchors"][:3])
            prompt += (
                "\n[PMC-Patients style anchors — real patient phrasing, use as voice reference only, "
                "do not copy verbatim:]\n" + anchors + "\n"
            )
        if pmc_context and pmc_context.get("hits"):
            # also expose top hit titles as background
            hits_block = "\n".join(
                f"  pmc_uid={h['patient_uid']} | {h.get('title','')[:120]} | {(h.get('patient') or '')[:180]}"
                for h in pmc_context["hits"][:3]
            )
            prompt += "\n[PMC similar patients (background, not citable as source_id):]\n" + hits_block + "\n"
        if ledger.known_ids():
            prompt += "\nPreviously retrieved sources (data; reuse these):\n" + json.dumps(ledger.to_json(), ensure_ascii=False)
        if previous_raw:
            prompt += "\nPrevious draft (data):\n" + previous_raw
        if repair_hint:
            prompt += "\nRepair these findings:\n" + repair_hint

        # Decide whether to allow search in this turn
        gap = "evidence_gap" in repair_hint or "support" in repair_hint.lower()
        allow_search = enable_web_search and (not ledger.known_ids() or gap)

        session = ApiSession(run_id, vp_index, attempt_no, model, budget, ledger, store=store, verbose=verbose)
        try:
            result = session.run(
                __import__("scripts.generate_vp_deepseek31_en", fromlist=["STAGE_A_SYSTEM"]).STAGE_A_SYSTEM,
                prompt,
                enable_web_search=allow_search,
                final_instruction="Return only the complete Stage A master JSON now. No scenario or narrative.",
                reserve_requests=3,
                search_limit=2 if gap else None,
            )
        except Exception as e:
            return AgentResult(ok=False, error=str(e), detail={"category": getattr(e, "category", "api_error")})

        raw = result.get("content", "")
        if result.get("finish_reason") != "stop":
            return AgentResult(ok=False, error="Stage A response incomplete", detail={"category": "stage_a_parse", "raw": raw[:2000]})

        try:
            parsed = _load_json_tolerant(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Stage A must return one JSON object")
        except Exception as e:
            return AgentResult(ok=False, error=str(e), detail={"category": "stage_a_parse", "raw": raw[:3000]})

        parsed["first_person_narrative"] = _STAGE_A_PLACEHOLDER
        try:
            case, notes = canonicalize(parsed, skeleton)
        except Exception as e:
            return AgentResult(ok=False, error=f"canonicalize: {e}", detail={"category": "stage_a_canonicalize", "raw": raw[:3000]})

        if pack:
            case.evidence_pack_id = pack.get("pack_id", "")

        evidence_basis = "web_search" if ledger.known_ids() else "icd11_definition_only"
        try:
            stage_a_qc(case, ledger, session, evidence_basis)
        except CaseError as e:
            return AgentResult(ok=False, error=str(e), detail={"category": e.category, "detail": e.detail, "master": case.model_dump(mode="json")})
        except Exception as e:  # pragma: no cover
            return AgentResult(ok=False, error=str(e), detail={"category": "stage_a_qc"})

        return AgentResult(ok=True, data={
            "case": case,
            "notes": notes,
            "evidence_basis": evidence_basis,
            "session": session,
            "result": result,
            "raw": raw,
        })
