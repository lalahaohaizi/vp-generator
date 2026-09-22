# -*- coding: utf-8 -*-
"""
PMCPatientsStore — local retrieval over PMC-Patients (167k summaries).

Design goals (see docs/agentic_design.md §4.2):
  - Offline-first: if the index is absent, search() degrades to [] + warning.
  - Two indices: full (167k) and derm_subset (~8-12k) — caller picks via --pmc-mode.
  - Three operations: PPR (similar patients), PAR (relevant articles), distribution calibration.
  - SQLite FTS5 when available, else Python fallback (substring + ranking).

Data source (choose one, both have identical keys):
  figshare  PMC-Patients.json.tar.gz  (195 MB gz, 167034 rows)
  huggingface zhengyun21/PMC-Patients (1.38 GB)

The builder (build_pmc_index) is also exposed as scripts/build_pmc_index.py.
"""
from __future__ import annotations
import json
import re
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import logger, _utcnow, PMC_TOPK


# Keyword heuristic for derm_subset (used both at build and at query time)
DERM_KEYWORDS = re.compile(
    r"dermatolog|psoriasis|eczema|atopic|acne|vitiligo|melanoma|dermatitis|urticaria|alopecia|lichen|pemphig|blister|cutaneous|skin",
    re.I,
)

# Minimal stop-word list for naive ranking when FTS5 is unavailable
_STOP = frozenset({"a", "an", "the", "is", "are", "was", "were", "with", "for", "and", "or", "of", "in", "on", "to", "by", "as", "at"})


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z]{3,}", text) if t.lower() not in _STOP]


def _score(query_tokens: List[str], doc_text: str) -> float:
    """Naive TF overlap score — used only when FTS5 is missing."""
    if not query_tokens or not doc_text:
        return 0.0
    doc_tokens = set(_tokenize(doc_text))
    if not doc_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if t in doc_text.lower())
    # Jaccard-ish
    overlap = len(set(query_tokens) & doc_tokens)
    return hits * 0.5 + overlap * 1.0


class PMCPatientsStore:
    """
    Retrieval store over PMC-Patients.

    Parameters
    ----------
    index_path: path to pmc_patients.sqlite (built by build_pmc_index).
                If missing, the store operates in degraded mode (all searches return []).
    json_path:  optional direct path to PMC-Patients.json for fallback scan
                (used when sqlite is absent but json is present).
    """

    def __init__(self, index_path: Optional[str] = None, json_path: Optional[str] = None):
        self.index_path = Path(index_path) if index_path else None
        self.json_path = Path(json_path) if json_path else None
        self._conn: Optional[sqlite3.Connection] = None
        self._has_fts = False
        self._json_cache: Optional[List[dict]] = None
        self._degraded_reason = ""

        if self.index_path and self.index_path.exists():
            try:
                self._conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                # probe FTS
                cur = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {r[0] for r in cur.fetchall()}
                self._has_fts = "patients_fts" in tables or "patients" in tables
                if not self._has_fts:
                    self._degraded_reason = "sqlite exists but no patients table"
            except Exception as e:  # pragma: no cover
                logger.warning(f"[PMC] cannot open index {self.index_path}: {e}")
                self._conn = None
                self._degraded_reason = str(e)
        else:
            if self.index_path:
                self._degraded_reason = f"index not found: {self.index_path}"
            # try json fallback
            if self.json_path and self.json_path.exists():
                self._degraded_reason += " (json fallback available)"

    # -- lifecycle -----------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._conn is not None or (self.json_path and self.json_path.exists())

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # -- internal: json scan fallback ---------------------------------------
    def _load_json(self, limit: Optional[int] = None) -> List[dict]:
        if self._json_cache is not None:
            return self._json_cache[:limit] if limit else self._json_cache
        if not self.json_path or not self.json_path.exists():
            return []
        try:
            import gzip
            # handle .tar.gz containing the json — caller should have extracted, but be helpful
            if str(self.json_path).endswith(".tar.gz") or str(self.json_path).endswith(".tgz"):
                import tarfile
                with tarfile.open(str(self.json_path), "r:gz") as tf:
                    member = next((m for m in tf.getmembers() if m.name.endswith(".json")), None)
                    if member is None:
                        return []
                    f = tf.extractfile(member)
                    assert f is not None
                    data = json.load(f)  # type: ignore
            elif str(self.json_path).endswith(".gz"):
                with gzip.open(str(self.json_path), "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(str(self.json_path), "r", encoding="utf-8") as f:
                    data = json.load(f)
            # data is a list
            if isinstance(data, dict) and "patients" in data:
                data = data["patients"]
            self._json_cache = data if isinstance(data, list) else []
            if limit:
                return self._json_cache[:limit]
            return self._json_cache
        except Exception as e:  # pragma: no cover
            logger.warning(f"[PMC] json load failed {self.json_path}: {e}")
            return []

    # -- public API ----------------------------------------------------------
    def search_similar_patients(
        self,
        query: str,
        k: int = PMC_TOPK,
        age_band: Optional[str] = None,
        gender: Optional[str] = None,
        disease_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        ReCDS-PPR: retrieve patients whose summary is BM25-similar to query.
        Returns list of {patient_uid, patient, title, age, gender, score, pmid}.
        """
        if not query or not query.strip():
            return []
        k = max(1, min(k, 20))

        # 1) try sqlite FTS5
        if self._conn:
            try:
                return self._fts_search(query, k, age_band, gender, disease_hint)
            except Exception as e:  # pragma: no cover
                logger.warning(f"[PMC] FTS search failed, falling back to scan: {e}")

        # 2) json scan fallback
        rows = self._load_json()
        if not rows:
            if self._degraded_reason:
                logger.warning(f"[PMC] search degraded ({self._degraded_reason}); returning []")
            return []
        qtok = _tokenize(query)
        scored: List[Tuple[float, dict]] = []
        for r in rows:
            txt = r.get("patient") or r.get("patient_note") or ""
            # optional filters
            if disease_hint and DERM_KEYWORDS.search(disease_hint):
                # derm_subset hint — prefer derm-relevant rows but don't exclude others
                bonus = 2.0 if DERM_KEYWORDS.search(txt) or DERM_KEYWORDS.search(r.get("title") or "") else 0.0
            else:
                bonus = 0.0
            # gender filter
            if gender and r.get("gender"):
                g = r["gender"]
                # map M/F to Male/Female vocab
                want = "M" if "male" in gender.lower() else "F" if "female" in gender.lower() else None
                if want and g != want:
                    continue
            s = _score(qtok, txt) + bonus
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, r in scored[:k]:
            out.append({
                "patient_uid": r.get("patient_uid") or r.get("patient_id") or "",
                "pmid": r.get("PMID") or r.get("pmid") or "",
                "title": r.get("title") or "",
                "patient": (r.get("patient") or "")[:2000],
                "age": r.get("age"),
                "gender": r.get("gender"),
                "score": round(float(score), 3),
            })
        return out

    def _fts_search(self, query: str, k: int, age_band, gender, disease_hint) -> List[Dict[str, Any]]:
        assert self._conn is not None
        # Use FTS5 MATCH if available, else LIKE fallback
        # Build gender filter
        where = []
        params: List[Any] = []
        # We store gender as M/F in the table
        if gender:
            want = "M" if "male" in gender.lower() else "F" if "female" in gender.lower() else None
            if want:
                where.append("gender = ?")
                params.append(want)

        # FTS query — escape quotes
        fts_q = query.replace('"', " ").strip()
        # If FTS table exists, use it
        cur = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients_fts'")
        has_fts = cur.fetchone() is not None
        if has_fts:
            sql = "SELECT patient_uid, pmid, title, patient, age_json, gender, rank FROM patients_fts WHERE patients_fts MATCH ?"
            fts_params: List[Any] = [fts_q]
            if where:
                # FTS5 doesn't support extra WHERE on content columns directly in all builds;
                # filter in Python instead — keep simple.
                pass
            # order by rank (lower is better in FTS5 bm25), limit k*3 then filter
            sql += " ORDER BY rank LIMIT ?"
            fts_params.append(k * 3)
            rows = self._conn.execute(sql, fts_params).fetchall()
            # apply gender filter in python
            filtered = []
            for r in rows:
                if gender and r["gender"] and want and r["gender"] != want:
                    continue
                filtered.append(r)
                if len(filtered) >= k:
                    break
            out = []
            for r in filtered[:k]:
                # rank is negative bm25 in some builds; normalise
                try:
                    score = float(r["rank"])
                    # FTS5 rank is negative; flip
                    score = -score if score < 0 else 1.0 / (1.0 + score)
                except Exception:
                    score = 1.0
                out.append({
                    "patient_uid": r["patient_uid"],
                    "pmid": r["pmid"],
                    "title": r["title"] or "",
                    "patient": (r["patient"] or "")[:2000],
                    "age": json.loads(r["age_json"]) if r["age_json"] else None,
                    "gender": r["gender"],
                    "score": round(float(score), 4),
                })
            return out
        else:
            # no FTS, LIKE scan
            like = f"%{query.split()[0]}%"
            sql = "SELECT patient_uid, pmid, title, patient, age_json, gender FROM patients WHERE patient LIKE ?"
            params_like = [like]
            if where:
                sql += " AND " + " AND ".join(where)
                params_like.extend(params)
            sql += " LIMIT ?"
            params_like.append(k * 5)
            rows = self._conn.execute(sql, params_like).fetchall()
            qtok = _tokenize(query)
            scored = []
            for r in rows:
                s = _score(qtok, r["patient"] or "")
                scored.append((s, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for s, r in scored[:k]:
                out.append({
                    "patient_uid": r["patient_uid"],
                    "pmid": r["pmid"],
                    "title": r["title"] or "",
                    "patient": (r["patient"] or "")[:2000],
                    "age": json.loads(r["age_json"]) if r["age_json"] else None,
                    "gender": r["gender"],
                    "score": round(float(s), 3),
                })
            return out

    def search_relevant_articles(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        ReCDS-PAR (lightweight): expand similar patients' relevant_articles.
        Without a full PubMed dump, we return the PMIDs + titles from the
        similar-patients' citation graph.
        """
        hits = self.search_similar_patients(query, k=k)
        seen: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            # If we have the raw json row, pull relevant_articles
            pmid = h.get("pmid")
            if not pmid:
                continue
            # Try to find the original row to read relevant_articles
            raw = None
            if self._conn:
                try:
                    cur = self._conn.execute("SELECT relevant_json FROM patients WHERE pmid=? LIMIT 1", (pmid,))
                    row = cur.fetchone()
                    if row and row[0]:
                        raw = json.loads(row[0])
                except Exception:
                    pass
            if raw is None:
                # json scan
                for r in self._load_json():
                    if str(r.get("PMID")) == str(pmid):
                        raw = r.get("relevant_articles") or r.get("relevantArticles") or {}
                        break
                if isinstance(raw, dict):
                    pass
                else:
                    raw = {}
            if isinstance(raw, dict):
                for art_pmid, rel in list(raw.items())[:5]:
                    if art_pmid not in seen:
                        seen[art_pmid] = {"pmid": art_pmid, "relevance": rel, "source_patient": h["patient_uid"]}
        # sort by relevance desc
        arts = sorted(seen.values(), key=lambda x: x["relevance"], reverse=True)
        return arts[:k]

    def age_gender_distribution(self, disease_hint: Optional[str] = None) -> Dict[str, Any]:
        """Return empirical age/gender distribution for calibration."""
        rows = self._load_json(limit=5000) if not self._conn else None
        if self._conn:
            try:
                cur = self._conn.execute("SELECT age_json, gender FROM patients LIMIT 5000")
                rows_sql = cur.fetchall()
                ages: List[float] = []
                genders: List[str] = []
                for r in rows_sql:
                    aj = r["age_json"]
                    if aj:
                        try:
                            arr = json.loads(aj)
                            # arr like [[34.0, "year"]]
                            for val, unit in arr:
                                if unit == "year":
                                    ages.append(float(val))
                                elif unit == "month":
                                    ages.append(float(val) / 12)
                        except Exception:
                            pass
                    if r["gender"]:
                        genders.append(r["gender"])
                return _dist_summary(ages, genders)
            except Exception as e:  # pragma: no cover
                logger.warning(f"[PMC] distribution sql failed: {e}")
        if rows is not None:
            ages = []
            genders = []
            for r in rows:
                for val, unit in (r.get("age") or []):
                    if unit == "year":
                        ages.append(float(val))
                    elif unit == "month":
                        ages.append(float(val) / 12)
                if r.get("gender"):
                    genders.append(r["gender"])
            return _dist_summary(ages, genders)
        return {"n": 0, "note": "no index available"}

    def get_style_anchors(self, disease_hint: str, k: int = 3) -> List[str]:
        """Return k real patient first-sentences as narrative style anchors."""
        hits = self.search_similar_patients(disease_hint, k=k)
        anchors = []
        for h in hits:
            txt = (h.get("patient") or "").strip()
            if txt:
                # first 1-2 sentences
                sents = re.split(r"(?<=[.!?])\s+", txt)
                anchor = " ".join(sents[:2])[:320]
                anchors.append(anchor)
        return anchors


def _dist_summary(ages: List[float], genders: List[str]) -> Dict[str, Any]:
    import statistics
    n = len(ages)
    out: Dict[str, Any] = {"n": n}
    if n:
        ages_sorted = sorted(ages)
        out["age_median"] = round(statistics.median(ages_sorted), 1)
        out["age_mean"] = round(statistics.mean(ages_sorted), 1)
        out["age_p25"] = round(ages_sorted[n // 4], 1) if n >= 4 else ages_sorted[0]
        out["age_p75"] = round(ages_sorted[3 * n // 4], 1) if n >= 4 else ages_sorted[-1]
        out["age_min"] = round(min(ages_sorted), 1)
        out["age_max"] = round(max(ages_sorted), 1)
    if genders:
        from collections import Counter
        c = Counter(genders)
        total = len(genders)
        out["gender"] = {k: round(v / total, 3) for k, v in c.items()}
    return out


# ---------------------------------------------------------------------------
# Index builder (also used by scripts/build_pmc_index.py)
# ---------------------------------------------------------------------------

def build_pmc_index(
    source: str | Path,
    out: str | Path,
    mode: str = "full",
    limit: Optional[int] = None,
) -> Path:
    """
    Build pmc_patients.sqlite from PMC-Patients.json.

    mode:
      full        — index every row
      derm_subset — only rows where title or patient mentions derm keywords
    """
    source = Path(source)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load
    if not source.exists():
        raise FileNotFoundError(f"PMC source not found: {source}")
    logger.info(f"[PMC] loading {source} (mode={mode})…")

    # Support tar.gz / gz / json
    import gzip, tarfile
    data: List[dict]
    if str(source).endswith(".tar.gz") or str(source).endswith(".tgz"):
        with tarfile.open(str(source), "r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith(".json")), None)
            if member is None:
                raise ValueError("tar.gz contains no .json")
            f = tf.extractfile(member)
            assert f is not None
            data = json.load(f)  # type: ignore
    elif str(source).endswith(".gz"):
        with gzip.open(str(source), "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(str(source), "r", encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, dict) and "patients" in data:
        data = data["patients"]
    assert isinstance(data, list), "PMC-Patients.json must be a list"
    orig_n = len(data)
    if limit:
        data = data[:limit]
    if mode == "derm_subset":
        filtered = []
        for r in data:
            txt = (r.get("title") or "") + " " + (r.get("patient") or "")
            if DERM_KEYWORDS.search(txt):
                filtered.append(r)
        data = filtered
        logger.info(f"[PMC] derm_subset: {len(data)}/{orig_n} rows kept")

    # Build SQLite
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(str(out))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE patients(
            patient_uid TEXT PRIMARY KEY,
            patient_id  TEXT,
            pmid        TEXT,
            title       TEXT,
            patient     TEXT,
            age_json    TEXT,
            gender      TEXT,
            relevant_json TEXT,
            similar_json  TEXT
        );
    """)
    # FTS5 virtual table for BM25
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE patients_fts USING fts5(
                patient_uid, pmid, title, patient, age_json, gender,
                content='patients', content_rowid='rowid', tokenize='porter'
            );
        """)
        has_fts = True
    except Exception as e:  # pragma: no cover
        logger.warning(f"[PMC] FTS5 not available ({e}); LIKE fallback will be used")
        has_fts = False

    # triggers to keep FTS in sync
    if has_fts:
        conn.execute("""
            CREATE TRIGGER patients_ai AFTER INSERT ON patients BEGIN
                INSERT INTO patients_fts(rowid, patient_uid, pmid, title, patient, age_json, gender)
                VALUES (new.rowid, new.patient_uid, new.pmid, new.title, new.patient, new.age_json, new.gender);
            END;
        """)

    rows = []
    for r in data:
        pid = str(r.get("patient_id") or r.get("patient_uid") or "")
        puid = str(r.get("patient_uid") or pid)
        pmid = str(r.get("PMID") or r.get("pmid") or "")
        title = r.get("title") or ""
        patient = r.get("patient") or r.get("patient_note") or ""
        # length / language / demographic filters (mirror paper's pipeline, but lenient)
        if len(patient.split()) < 10:
            continue
        age = r.get("age")
        gender = r.get("gender")
        # keep only rows with at least one of age/gender? paper requires both, but be lenient for derm
        # we keep all for full mode; for derm we also keep all that passed keyword
        rows.append((
            puid, pid, pmid, title, patient,
            json.dumps(age, ensure_ascii=False) if age else "",
            gender or "",
            json.dumps(r.get("relevant_articles") or {}, ensure_ascii=False),
            json.dumps(r.get("similar_patients") or {}, ensure_ascii=False),
        ))

    conn.executemany(
        "INSERT INTO patients(patient_uid, patient_id, pmid, title, patient, age_json, gender, relevant_json, similar_json) VALUES(?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    # meta
    meta = {
        "built_at": _utcnow(),
        "source": str(source),
        "mode": mode,
        "orig_rows": orig_n,
        "indexed_rows": len(rows),
        "has_fts": has_fts,
        "pmc_patients_version": "2023-11-06",
    }
    conn.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    for k, v in meta.items():
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, json.dumps(v, ensure_ascii=False)))
    conn.commit()
    conn.close()

    # also write json meta alongside
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[PMC] index built: {out} ({len(rows)} rows, fts={has_fts})")
    return out
