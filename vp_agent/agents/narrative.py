# -*- coding: utf-8 -*-
"""
NarrativeAgent — Stage B (blinded narrative from frozen facts).

No retrieval is allowed; every claim must already be in the frozen VPCase.
"""
from __future__ import annotations
import json
from typing import Dict

from scripts.generate_vp_deepseek31_en import (
    SearchLedger, ApiSession, CaseBudget,
    build_stage_b_prompt, split_scenario, render_fixed_attribution_reasons,
    _extract_block, _require_text, MIN_NARRATIVE_CHARS, CaseError, logger,
)
from .base import Agent, AgentResult


class NarrativeAgent(Agent):
    name = "narrative"

    def run(
        self,
        skeleton: Dict,
        case,  # VPCase
        ledger: SearchLedger,
        budget: CaseBudget,
        run_id: str,
        vp_index: int,
        attempt_no: int,
        model: str,
        store=None,
        previous_scenario: str = "",
        repair_feedback: str = "",
        verbose: bool = True,
    ) -> AgentResult:
        session = ApiSession(run_id, vp_index, attempt_no, model, budget, ledger, store=store, verbose=verbose)
        prompt = build_stage_b_prompt(skeleton, case, previous_scenario, repair_feedback)
        try:
            result = session.run(
                __import__("scripts.generate_vp_deepseek31_en", fromlist=["STAGE_B_SYSTEM"]).STAGE_B_SYSTEM,
                prompt,
                enable_web_search=False,
                reserve_requests=1,
            )
        except Exception as e:
            return AgentResult(ok=False, error=str(e), detail={"category": getattr(e, "category", "api_error")})

        if result.get("finish_reason") != "stop":
            return AgentResult(ok=False, error="Stage B response incomplete", detail={"category": "stage_b_parse"})
        scenario = _extract_block(result.get("content", ""), "<<<CASE_SCENARIO>>>", "<<<END_CASE_SCENARIO>>>")
        if not scenario:
            return AgentResult(ok=False, error="Stage B scenario markers missing", detail={"category": "stage_b_parse", "raw": result.get("content","")[:3000]})

        scenario, rendering = render_fixed_attribution_reasons(case, scenario)
        narrative = split_scenario(scenario).get("First-person narrative", "")
        try:
            case.first_person_narrative = _require_text(narrative, "first_person_narrative", MIN_NARRATIVE_CHARS)
        except ValueError as e:
            return AgentResult(ok=False, error=str(e), detail={"category": "stage_b_parse", "scenario": scenario[:3000]})

        return AgentResult(ok=True, data={
            "case": case,
            "scenario": scenario,
            "rendering": rendering,
            "session": session,
            "result": result,
        })
