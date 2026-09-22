# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentResult:
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)


class Agent:
    name: str = "base"

    def run(self, **kwargs) -> AgentResult:  # pragma: no cover
        raise NotImplementedError
