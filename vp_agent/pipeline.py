# -*- coding: utf-8 -*-
"""
Pipeline — batch orchestration for the agentic host.

Mirrors v31's run_batch / _run_batch_locked but routes each case through Host.
All store / export / locking semantics are preserved.
"""
from __future__ import annotations
import os
import time
import json
import traceback
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List

from scripts.generate_vp_deepseek31_en import (
    load_icd11_diseases, load_or_create_index_map, file_digest, atomic_json,
    VPStore, EvidencePackStore, logger, _attach_file_logger,
    SLEEP_BETWEEN_VP, MAX_BATCH_TOKENS, BatchFatalError,
    export_master_table, export_pending_review, export_run_summary,
    prompt_digest, OUTPUT_DIR, DB_PATH, ICD11_PATH, DEFAULT_MODEL,
)

from .config import PMC_INDEX_PATH
from .memory import Notebook
from .tools.pmc_patients import PMCPatientsStore
from .agents.host import Host, GENERATOR_VERSION_AGENTIC
from .agents.simulator import SimulationGate  # noqa: F401


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_batch_agentic(
    limit: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    enable_web_search: bool = True,
    only_severity: Optional[str] = None,
    only_code: Optional[str] = None,
    resume: bool = True,
    verbose: bool = True,
    pmc_index: Optional[str] = None,
    pmc_calibrate: bool = False,
    simulate: bool = False,
    sim_turns: int = 20,
    sim_bias: str = "",
    mock: bool = False,
) -> int:
    """Agentic batch — Host per case, shared index map & store."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(OUTPUT_DIR / "generation.log")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    diseases = load_icd11_diseases(ICD11_PATH)
    plan = load_or_create_index_map(diseases, OUTPUT_DIR / "vp_index_map.csv")
    by_code = {d["code"]: d for d in diseases}

    store = VPStore(DB_PATH, OUTPUT_DIR / "cases")
    packs = EvidencePackStore(OUTPUT_DIR / "evidence_packs")

    # PMC store (optional)
    pmc_path = pmc_index or PMC_INDEX_PATH
    pmc_store: Optional[PMCPatientsStore] = None
    if pmc_path and Path(pmc_path).exists():
        pmc_store = PMCPatientsStore(index_path=pmc_path)
        logger.info(f"[PMC] index loaded: {pmc_path}")
    elif pmc_path:
        logger.warning(f"[PMC] index not found at {pmc_path}; retrieval will be degraded (web-only)")
        pmc_store = PMCPatientsStore(index_path=pmc_path)  # degraded mode

    notebook: Optional[Notebook] = None
    try:
        from .memory import Notebook as _NB
        from .config import NOTEBOOK_PATH, NOTEBOOK_ENABLED
        if NOTEBOOK_ENABLED:
            notebook = _NB(NOTEBOOK_PATH)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[Notebook] disabled: {e}")

    try:
        if not resume and store.committed_indices():
            raise BatchFatalError("--no-resume cannot overwrite committed cases; use a separate output directory")
        done = store.validated_indices(plan, by_code, model, enable_web_search) if resume else set()
    except BaseException:
        store.close()
        if notebook:
            notebook.close()
        if pmc_store:
            pmc_store.close()
        raise

    todo = []
    for row in plan:
        if only_severity and row["preset_dep_severity"] != only_severity:
            continue
        if only_code and row["icd11_code"] != only_code:
            continue
        if row["vp_index"] in done:
            continue
        if row["icd11_code"] not in by_code:
            logger.warning(f"[plan] vp_index {row['vp_index']} references {row['icd11_code']}, absent from frame; skipped")
            continue
        todo.append(row)
    if limit is not None:
        todo = todo[:limit]

    store.start_run(run_id, model, enable_web_search, len(todo))
    # manifest — extend v31 manifest with agentic fields
    try:
        import importlib.metadata
        deps = {name: importlib.metadata.version(name) for name in ("pydantic", "openai")}
    except Exception:
        deps = {}
    # mock auto-detect: missing key or explicit --mock
    if not mock:
        try:
            from scripts.generate_vp_deepseek31_en import DEEPSEEK_APIKEY
            if not DEEPSEEK_APIKEY or model == "mock":
                mock = True
                logger.info("[run] mock mode auto-enabled (no DEEPSEEK_API_KEY or model=mock)")
        except Exception:
            pass
    manifest = {
        "run_id": run_id,
        "state": "running",
        "plan": todo,
        "code_hash": file_digest(__file__),
        "prompt_hash": prompt_digest(),
        "icd11_hash": file_digest(ICD11_PATH) if Path(ICD11_PATH).exists() else "",
        "model": model,
        "dependencies": deps,
        "agentic_version": GENERATOR_VERSION_AGENTIC,
        "agentic": True,
        "mock": bool(mock),
        "pmc_index": str(pmc_path) if pmc_path else None,
        "pmc_calibrate": pmc_calibrate,
        "simulate": simulate,
        "sim_turns": sim_turns,
        "sim_bias": sim_bias,
    }
    manifest_path = OUTPUT_DIR / "runs" / run_id / "manifest.json"
    atomic_json(manifest_path, manifest)
    logger.info(f"[run] {run_id} model={model} agentic={GENERATOR_VERSION_AGENTIC} web_search={enable_web_search} pmc={bool(pmc_store and pmc_store.available)} simulate={simulate}")
    logger.info(f"[run] {len(plan)} planned cells, {len(done)} already committed, {len(todo)} to generate this run")
    if not enable_web_search:
        logger.warning("[run] web search disabled: evidence_basis will be icd11_definition_only")

    host = Host(
        run_id=run_id,
        store=store,
        pack_store=packs,
        pmc_store=pmc_store,
        notebook=notebook,
        model=model,
        enable_web_search=enable_web_search,
        verbose=verbose,
    )

    committed, pending, failed = 0, 0, []
    exit_code = 0
    try:
        for i, row in enumerate(todo, 1):
            if MAX_BATCH_TOKENS and store.run_token_total(run_id) >= MAX_BATCH_TOKENS:
                logger.error(f"[run] batch token ceiling {MAX_BATCH_TOKENS} reached; stopping with {len(todo)-i+1} unstarted")
                exit_code = 1
                break
            if verbose:
                print(f"[{i}/{len(todo)}] vp_index={row['vp_index']} {row['icd11_code']} / {row['preset_dep_severity']}")
            res = host.generate_one(
                by_code[row["icd11_code"]], row["preset_dep_severity"], row["vp_index"],
                simulate=simulate, sim_turns=sim_turns, sim_bias=sim_bias, mock=mock,
            )
            if res.get("ok"):
                committed += 1
                pending += int(bool(res.get("pending_review")))
            else:
                failed.append(res)
                _append_jsonl(OUTPUT_DIR / "failed_cases.jsonl", {**res, "run_id": run_id, "recorded_at": __import__("scripts.generate_vp_deepseek31_en", fromlist=["_utcnow"])._utcnow()})
            if i < len(todo) and SLEEP_BETWEEN_VP > 0:
                time.sleep(SLEEP_BETWEEN_VP)
    except KeyboardInterrupt:
        logger.warning("[run] interrupted; committed cases are intact and --resume will pick up the remainder")
        exit_code = 130
    except BatchFatalError as e:
        logger.error(f"[run] fatal: {e}")
        exit_code = 2
    finally:
        store.finish_run(run_id)
        try:
            committed_cnt = store.conn.execute("SELECT COUNT(*) FROM cases WHERE run_id=?", (run_id,)).fetchone()[0]
        except Exception:
            committed_cnt = committed
        manifest.update(
            state="interrupted" if exit_code == 130 else "failed" if exit_code else "partial" if failed or committed < len(todo) else "completed",
            committed=committed_cnt,
            failed=len(failed),
            usage=store.run_token_breakdown(run_id),
            finished_at=__import__("scripts.generate_vp_deepseek31_en", fromlist=["_utcnow"])._utcnow(),
        )
        atomic_json(manifest_path, manifest)
        try:
            export_master_table(store, OUTPUT_DIR)
            export_pending_review(store, OUTPUT_DIR)
            export_run_summary(store, run_id, OUTPUT_DIR, len(todo), committed, failed, model)
        except Exception as e:
            logger.error(f"[export] failed: {e}\n{traceback.format_exc()}")
        store.close()
        if notebook:
            notebook.close()
        if pmc_store:
            pmc_store.close()

    if exit_code == 0 and (failed or pending):
        exit_code = 1
    print(f"\n[run {run_id}] committed {committed}/{len(todo)}, queued for review {pending}, failed {len(failed)}")
    if failed:
        print(f"  failures: {OUTPUT_DIR / 'failed_cases.jsonl'} and the failures table")
    if pending:
        print(f"  review queue: {OUTPUT_DIR / 'pending_review.csv'}")
    return exit_code


def run_batch(*args, **kwargs):
    """Lock-aware wrapper (same lock file as v31 so they don't collide)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / ".generator.lock", "a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BatchFatalError("Another generator is using this output directory") from exc
        try:
            return run_batch_agentic(*args, **kwargs)
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
