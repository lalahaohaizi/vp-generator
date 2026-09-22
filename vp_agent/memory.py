# -*- coding: utf-8 -*-
"""
Memory Bank / Notebook — AgentClinic-inspired persistent memory.

Three tiers: run / disease / case.
Backed by SQLite so it survives process restarts; also mirrors to
vp_output/audit/{run_id}/notebook.json for human inspection.

Llama-3 showed +92% relative gain with the notebook tool in AgentClinic;
here the gain is expected in evidence-query refinement and narrative de-homogenisation.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import NOTEBOOK_PATH, logger, _utcnow


_DDL = """
CREATE TABLE IF NOT EXISTS notebook (
    tier TEXT NOT NULL,          -- run | disease | case
    key  TEXT NOT NULL,
    value TEXT NOT NULL,         -- JSON
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tier, key)
);
"""


class Notebook:
    """Thread-safe, cross-case memory."""

    def __init__(self, path: str = NOTEBOOK_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()

    # -- low-level -----------------------------------------------------------
    def _put(self, tier: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO notebook(tier,key,value,updated_at) VALUES(?,?,?,?)",
                (tier, key, json.dumps(value, ensure_ascii=False), _utcnow()),
            )
            self._conn.commit()

    def _get(self, tier: str, key: str, default=None):
        with self._lock:
            cur = self._conn.execute("SELECT value FROM notebook WHERE tier=? AND key=?", (tier, key))
            row = cur.fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

    def _scan(self, tier: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT key,value,updated_at FROM notebook WHERE tier=?", (tier,))
            out = []
            for k, v, ts in cur.fetchall():
                try:
                    val = json.loads(v)
                except Exception:
                    val = v
                out.append({"key": k, "value": val, "updated_at": ts})
            return out

    # -- high-level helpers --------------------------------------------------
    # Run tier
    def put_run(self, run_id: str, data: Dict[str, Any]) -> None:
        self._put("run", run_id, data)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._get("run", run_id)

    def append_run_event(self, run_id: str, event: Dict[str, Any]) -> None:
        cur = self.get_run(run_id) or {"events": []}
        cur.setdefault("events", []).append({**event, "at": _utcnow()})
        # keep last 100 events
        cur["events"] = cur["events"][-100:]
        self.put_run(run_id, cur)

    # Disease tier
    def put_disease(self, icd11_code: str, data: Dict[str, Any]) -> None:
        prev = self._get("disease", icd11_code, {}) or {}
        prev.update(data)
        prev["updated_at"] = _utcnow()
        self._put("disease", icd11_code, prev)

    def get_disease(self, icd11_code: str) -> Dict[str, Any]:
        return self._get("disease", icd11_code, {}) or {}

    # Case tier
    def put_case(self, vp_index: int, data: Dict[str, Any]) -> None:
        self._put("case", str(vp_index), data)

    def get_case(self, vp_index: int) -> Optional[Dict[str, Any]]:
        return self._get("case", str(vp_index))

    # Convenience: build a compact context block for prompting
    def context_block(self, icd11_code: str, run_id: str, max_chars: int = 1200) -> str:
        """Return a short textual context for inclusion in prompts."""
        parts: List[str] = []
        d = self.get_disease(icd11_code)
        if d:
            # only include stable, non-identifying learnings
            if d.get("bsa_typical"):
                parts.append(f"Disease {icd11_code} typical BSA: {d['bsa_typical']}")
            if d.get("common_treatments"):
                parts.append(f"Common treatments: {', '.join(d['common_treatments'][:4])}")
            if d.get("retrieval_hints"):
                parts.append(f"Effective retrieval hints: {', '.join(d['retrieval_hints'][:3])}")
        r = self.get_run(run_id)
        if r and r.get("events"):
            # last 2 failure patterns
            fails = [e for e in r["events"] if e.get("kind") == "qc_failed"][-2:]
            for f in fails:
                parts.append(f"Recent QC pattern: {f.get('summary','')[:160]}")
        text = "\n".join(f"- {p}" for p in parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
