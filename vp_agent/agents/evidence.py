# -*- coding: utf-8 -*-
"""
EvidenceAgent — owns all retrieval.

Order:
  1. PMC-PPR: similar real patients (style + phenotype anchors)
  2. PMC-PAR: relevant articles (citation-graph expansion)
  3. Web search (SearXNG) for anything PMC does not cover
  4. Merge + dedup into a single SearchLedger
  5. Optionally save an EvidencePack for disease-level reuse
"""
from __future__ import annotations
from typing import Dict, List, Optional

from scripts.generate_vp_deepseek31_en import SearchLedger, EvidencePackStore, logger
from ..tools.web_search import WebSearchTool
from ..tools.pmc_patients import PMCPatientsStore
from ..config import PMC_TOPK
from .base import Agent, AgentResult


class EvidenceAgent(Agent):
    name = "evidence"

    def __init__(
        self,
        pmc_store: Optional[PMCPatientsStore] = None,
        web_tool: Optional[WebSearchTool] = None,
        pack_store: Optional[EvidencePackStore] = None,
    ):
        self.pmc = pmc_store
        self.web = web_tool or WebSearchTool()
        self.packs = pack_store

    def run(
        self,
        disease: Dict,
        skeleton: Dict,
        ledger: SearchLedger,
        enable_web: bool = True,
        topk: int = PMC_TOPK,
    ) -> AgentResult:
        disease_name = disease.get("title") or disease.get("disease_name_en") or ""
        disease_code = disease.get("code") or disease.get("icd11_code") or ""
        pmc_hits: List[Dict] = []
        pmc_articles: List[Dict] = []
        pmc_style: List[str] = []

        # 1) Try to load a disease-level evidence pack first
        pack = None
        if self.packs:
            pack = self.packs.load(disease_code)
            if pack:
                ledger.adopt_pack(pack)
                logger.info(f"[EvidenceAgent] reused pack {pack.get('pack_id')} ({len(pack.get('sources') or [])} sources)")

        # 2) PMC retrieval (degraded gracefully if no index)
        if self.pmc and self.pmc.available:
            try:
                # PPR — similar patients
                pmc_hits = self.pmc.search_similar_patients(
                    query=disease_name,
                    k=topk,
                    age_band=skeleton.get("age_band"),
                    gender=skeleton.get("sex"),
                    disease_hint=disease_name,
                )
                pmc_style = self.pmc.get_style_anchors(disease_name, k=min(3, topk))
                pmc_articles = self.pmc.search_relevant_articles(disease_name, k=topk)
                logger.info(f"[EvidenceAgent] PMC hits={len(pmc_hits)} articles={len(pmc_articles)} style={len(pmc_style)}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"[EvidenceAgent] PMC search failed: {e}")
                pmc_hits, pmc_articles, pmc_style = [], [], []
        else:
            if self.pmc and self.pmc.degraded_reason:
                logger.warning(f"[EvidenceAgent] PMC unavailable ({self.pmc.degraded_reason}); web-only mode")

        # 3) Web search — only if no pack and web enabled
        # The actual LLM-driven web_search tool calls happen inside FactAgent's ApiSession;
        # here we do a *pre-retrieval* to seed the ledger so the LLM can cite without searching.
        # Keep it cheap: 1 query.
        if enable_web and not ledger.known_ids():
            q = f"{disease_name} clinical features treatment guideline"
            raw = self.web(q, max_results=3)
            if raw:
                ledger.register(raw)
                logger.info(f"[EvidenceAgent] pre-seeded {len(raw)} web sources for {disease_code}")

        return AgentResult(
            ok=True,
            data={
                "ledger": ledger,
                "pack": pack,
                "pmc_hits": pmc_hits,
                "pmc_articles": pmc_articles,
                "pmc_style_anchors": pmc_style,
                "pmc_available": bool(self.pmc and self.pmc.available),
            },
        )
