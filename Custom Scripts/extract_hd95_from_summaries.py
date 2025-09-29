#!/usr/bin/env python
"""
Extract HD95 (mm) from nnU-Net v2 cross-validation summary.json files and
aggregate per-trainer values.

Usage:
  python extract_hd95_from_summaries.py \
    --root nnUNet_results/Dataset201_ATLAS2 \
    --out hd95_from_summaries.csv

Notes:
  - Tries to be robust to different summary.json structures.
  - Prefers foreground class id '1' where available; otherwise falls back to any
    available hd95 metric in the fold dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def is_number(x: Any) -> bool:
    try:
        _ = float(x)
        return True
    except Exception:
        return False


HD_KEYS = (
    "hd95",
    "hd95_rounded",
    "Hausdorff95",
    "hd_95",
    "hd95_mm",
)


def pull_hd_from_dict(d: Dict[str, Any]) -> float | None:
    for k in HD_KEYS:
        if k in d and is_number(d[k]):
            return float(d[k])
    return None


def collect_fold_hd95(res: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    # Prefer class id '1'
    for key in ("1", 1, "foreground", "fg"):
        if key in res and isinstance(res[key], dict):
            v = pull_hd_from_dict(res[key])
            if v is not None:
                vals.append(v)
                return vals
    # Else try directly in this dict
    v = pull_hd_from_dict(res)
    if v is not None:
        vals.append(v)
    return vals


def extract_hd95_from_summary(summary_path: Path) -> Tuple[str, float | None, int]:
    with summary_path.open("r", encoding="utf-8") as f:
        js = json.load(f)

    # Trainer name from path .../<trainer>/crossval_results/summary.json
    trainer = summary_path.parts[-3]

    # Try direct aggregate
    agg = pull_hd_from_dict(js) if isinstance(js, dict) else None
    if agg is not None:
        return trainer, agg, 0

    vals: List[float] = []
    if isinstance(js, dict) and "results" in js and isinstance(js["results"], dict):
        for _fold, fold_res in js["results"].items():
            if isinstance(fold_res, dict):
                vals.extend(collect_fold_hd95(fold_res))

    if vals:
        return trainer, float(np.mean(vals)), len(vals)

    # Fallback: recursive search for hd keys anywhere (last resort)
    def walk(obj: Any, acc: List[float]):
        if isinstance(obj, dict):
            v = pull_hd_from_dict(obj)
            if v is not None:
                acc.append(v)
            for vv in obj.values():
                walk(vv, acc)
        elif isinstance(obj, (list, tuple)):
            for vv in obj:
                walk(vv, acc)

    walk(js, vals)
    if vals:
        return trainer, float(np.mean(vals)), len(vals)
    return trainer, None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("nnUNet_results"))
    ap.add_argument("--out", type=Path, default=Path("hd95_from_summaries.csv"))
    args = ap.parse_args()

    root: Path = args.root
    out_csv: Path = args.out

    summary_files = sorted(root.glob("**/crossval_results/summary.json"))
    rows: List[Tuple[str, float | None, int, Path]] = []
    for p in summary_files:
        trainer, val, n = extract_hd95_from_summary(p)
        rows.append((trainer, val, n, p))

    # Print and write CSV
    print("trainer,hd95_mm,n_folds,summary_path")
    for t, v, n, p in rows:
        print(f"{t},{'' if v is None else v},{n},{p}")

    try:
        import pandas as pd

        df = pd.DataFrame(
            [{"trainer": t, "hd95_mm": v, "n_folds": n, "summary_path": str(p)} for t, v, n, p in rows]
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        finite = df["hd95_mm"].dropna().astype(float)
        if len(finite):
            print(
                f"Overall: mean={finite.mean():.2f} mm, std={finite.std():.2f}, "
                f"median={finite.median():.2f} mm across {len(finite)} trainers"
            )
        print(f"Saved: {out_csv}")
    except Exception:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("trainer,hd95_mm,n_folds,summary_path\n")
            for t, v, n, p in rows:
                f.write(f"{t},{'' if v is None else v},{n},{p}\n")
        print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()


