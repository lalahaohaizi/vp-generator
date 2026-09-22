# -*- coding: utf-8 -*-
"""
QCAgent — wraps v31's run_qc (L1-L5).

The Host treats QC as a gate: ok → commit or simulate; fail → reflection.
"""
from __future__ import annotations
from typing import List

from scripts.generate_vp_deepseek31_en import SearchLedger, run_qc, resolve_rule_severity, logger
from .base import Agent, AgentResult


class QCAgent(Agent):
    name = "qc"

    def run(self, case, scenario: str, ledger: SearchLedger, evidence_basis: str, notes: List[str], review_session=None, review_cached=None) -> AgentResult:
        rule_severity = resolve_rule_severity(case.depression)
        rep, verified = run_qc(case, scenario, rule_severity, evidence_basis, ledger, notes,
                               review_session=review_session, review_cached=review_cached)
        for layer, msg in rep.warnings:
            logger.warning(f"[QC] {layer}: {msg}")
        data = {
            "report": rep,
            "verified": verified,
            "rule_severity": rule_severity,
            "ok": bool(rep.ok),
            "repair_target": (rep.l5_review or {}).get("repair_target", "none"),
            "l5_review": rep.l5_review,
        }
        return AgentResult(ok=bool(rep.ok), data=data, error=rep.summary() if not rep.ok else "", detail=rep.as_dict())
