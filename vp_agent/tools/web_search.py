# -*- coding: utf-8 -*-
"""
WebSearchTool — thin wrapper over v31 SearXNG layer.

Keeps the exact filtering / reranking / cache behaviour so the agentic
path and the standalone v31 path share one SearchLedger contract.
"""
from __future__ import annotations
from typing import List

from scripts.generate_vp_deepseek31_en import (
    _searxng_web_search,
    SEARCH_RESULTS_N,
    logger,
)


class WebSearchTool:
    name = "web_search"
    description = "Web-search the clinical literature; each hit carries a program-assigned source_id."

    def __call__(self, query: str, max_results: int = SEARCH_RESULTS_N) -> List[dict]:
        if not query or not query.strip():
            return []
        try:
            return _searxng_web_search(query, max_results=max_results)
        except Exception as e:  # pragma: no cover
            logger.warning(f"[WebSearchTool] query={query!r} failed: {e}")
            return []

    @staticmethod
    def tool_spec() -> dict:
        from scripts.generate_vp_deepseek31_en import WEB_SEARCH_TOOL
        return WEB_SEARCH_TOOL
