#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a local PMC-Patients SQLite index for vp_agent.

Supports both the Figshare tarball and the Hugging Face JSON.

Usage:
    python scripts/build_pmc_index.py --source /data/PMC-Patients.json --out vp_output/pmc_patients/pmc_patients.sqlite
    python scripts/build_pmc_index.py --source /data/PMC-Patients.json.tar.gz --out vp_output/pmc_patients/pmc_patients.sqlite --mode derm_subset
    python scripts/build_pmc_index.py --help
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `vp_agent` is importable when run as `python scripts/build_pmc_index.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vp_agent.tools.pmc_patients import build_pmc_index  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build PMC-Patients SQLite index (FTS5 + BM25)")
    p.add_argument("--source", required=True, help="path to PMC-Patients.json / .json.gz / .tar.gz (figshare 195 MB or HF)")
    p.add_argument("--out", required=True, help="output sqlite path, e.g. vp_output/pmc_patients/pmc_patients.sqlite")
    p.add_argument("--mode", choices=["full", "derm_subset"], default="full", help="full=167k, derm_subset=~8-12k skin-related only")
    p.add_argument("--limit", type=int, default=None, help="for testing: only index first N rows")
    args = p.parse_args(argv)

    src = Path(args.source)
    out = Path(args.out)
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        print("Download PMC-Patients first:", file=sys.stderr)
        print("  figshare: https://figshare.com/articles/dataset/PMC-Patients_Dataset/24504115  (PMC-Patients.json.tar.gz, 195 MB)", file=sys.stderr)
        print("  HF:       https://huggingface.co/datasets/zhengyun21/PMC-Patients  (1.38 GB)", file=sys.stderr)
        print("  HF CLI:   huggingface-cli download zhengyun21/PMC-Patients --repo-type dataset --local-dir ./pmc_data", file=sys.stderr)
        return 2

    print(f"Building PMC index: source={src} out={out} mode={args.mode} limit={args.limit}")
    result = build_pmc_index(str(src), str(out), mode=args.mode, limit=args.limit)
    print(f"Done: {result}")
    print(f"Meta: {result.with_suffix('.meta.json')}")
    print(f"Use it: python -m vp_agent --n 5 --pmc-index {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
