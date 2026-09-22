# -*- coding: utf-8 -*-
"""
vp_agent — Agentic wrapper over generate_vp_deepseek31_en.py (v31).

Three-tier hybrid (DeepRare × AgentClinic) + PMC-Patients retrieval.
See docs/agentic_design.md for the full spec.

Usage:
    python -m vp_agent --help
    python -m vp_agent --n 5 --simulate
"""
from __future__ import annotations

__version__ = "0.1.0-agentic"
GENERATOR_VERSION_AGENTIC = "v32-agentic-hybrid-20260922"

# Re-export key constants from v31 for convenience
try:
    import scripts.generate_vp_deepseek31_en as _v31  # noqa: F401
except Exception:  # pragma: no cover
    _v31 = None  # type: ignore
