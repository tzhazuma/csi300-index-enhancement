#!/usr/bin/env python
"""Prove that the training feature path and the inference feature path agree.

This is the Lead's independent check of the property that matters most: `predict()` must
compute, for its signal date, exactly the features the model was trained on. The two paths
are structurally different:

  training   build_features(full 2016-2022 panel)              -> row for date t
  inference  build_features(normalise_input(judge dict))       -> row for date t
             where the judge dict holds at most 120 days

Any disagreement is a silent, invisible score loss on the hidden board, so it is checked
rather than assumed.

Usage
-----
    ./venv/bin/python tools/test_train_inference_parity.py [--dates 20180621,20200630,...]
                                                            [--lookback 120]
Exit code is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtcomp.data import BARRA, BARRA_COLS, FACTOR33, load_panel  # noqa: E402

MARKET_SENT = ["日期", "代码", "名称", "open", "high", "low", "close",
               "volume", "amount", "is_suspend"]
FACTOR_SENT = ["日期", "代码", "名称", *FACTOR33]
BARRA_SENT = ["日期", "代码", "名称", *BARRA_COLS]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def judge_window(panel: pd.DataFrame, date: int, lookback: int) -> dict[str, pd.DataFrame]:
    """Exactly what the platform hands `predict()` for this signal date."""
    calendar = np.sort(panel["日期"].unique())
    upto = calendar[calendar <= date]
    window = upto[-lookback:] if len(upto) > lookback else upto
    sub = panel.loc[panel["日期"].between(int(window[0]), date)]
    barra_rows = sub.loc[sub["in_barra"]]
    return {
        "MarketData": sub.loc[:, MARKET_SENT].copy(),
        "Factor33_winsor": sub.loc[:, FACTOR_SENT].copy(),
        "Barra_factor": barra_rows.loc[:, BARRA_SENT].rename(
            columns={f"barra_{n}": n for n in BARRA}).copy(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-src", type=Path, default=ROOT / "dtcomp" / "features.py")
    ap.add_argument("--schema-src", type=Path, default=ROOT / "pkg" / "schema.py")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "Data")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--dates", default="",
                    help="comma-separated YYYYMMDD signal dates (default: a spread)")
    ap.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args()

    features_mod = load_module("parity_features", args.features_src)
    schema_mod = load_module("parity_schema", args.schema_src)
    build_features = features_mod.build_features
    columns = list(features_mod.FEATURE_COLUMNS)
    print(f"[parity] {len(columns)} features from {args.features_src.name}")

    panel = load_panel(args.data_dir)
    calendar = np.sort(panel["日期"].unique())
    if args.dates:
        dates = [int(d) for d in args.dates.split(",")]
    else:
        idx = [len(calendar) - 1, len(calendar) - 120, len(calendar) - 500,
               len(calendar) - 900, len(calendar) - 1300]
        dates = sorted({int(calendar[i]) for i in idx if i >= 120})

    print("[parity] building the training-path features on the full panel ...")
    full = build_features(panel)
    full_keys = full.loc[:, ["日期", "代码"]]
    full_keys["代码"] = full_keys["代码"].astype(str)

    failures = 0
    for date in dates:
        train_rows = full.loc[full["日期"].eq(date)].reset_index(drop=True)
        train_keys = full_keys.loc[full["日期"].eq(date)].reset_index(drop=True)

        data = judge_window(panel, date, args.lookback)
        panel_in = schema_mod.normalise_input(data)
        n_days = panel_in["日期"].nunique()
        infer_all = build_features(panel_in)
        infer_rows = infer_all.loc[infer_all["日期"].eq(date)].reset_index(drop=True)

        # Align on (日期, 代码) so a row-order difference is not mistaken for a value bug.
        merged = train_keys.merge(
            pd.DataFrame({"代码": infer_rows["代码"].astype(str)}).assign(_i=np.arange(len(infer_rows))),
            on="代码", how="outer", indicator=True)
        if (merged["_merge"] != "both").any():
            print(f"[FAIL] {date}: row sets differ "
                  f"({int((merged['_merge'] == 'left_only').sum())} train-only, "
                  f"{int((merged['_merge'] == 'right_only').sum())} inference-only)")
            failures += 1
            continue
        order = merged["_i"].to_numpy()
        infer_rows = infer_rows.iloc[order].reset_index(drop=True)

        a = train_rows.loc[:, columns].to_numpy(dtype="float64")
        b = infer_rows.loc[:, columns].to_numpy(dtype="float64")

        nan_a, nan_b = np.isnan(a), np.isnan(b)
        nan_mismatch = int((nan_a != nan_b).sum())
        both = ~nan_a & ~nan_b
        with np.errstate(invalid="ignore"):
            diff = np.abs(a - b)
        max_diff = float(np.nanmax(np.where(both, diff, np.nan))) if both.any() else 0.0
        scale = float(np.nanmax(np.abs(a[both]))) if both.any() else 1.0
        bad = int((both & (diff > args.rtol * max(scale, 1.0))).sum())

        status = "ok" if (nan_mismatch == 0 and bad == 0) else "FAIL"
        if status == "FAIL":
            failures += 1
            worst = np.argsort(-np.where(both, diff, -np.inf).ravel())[:6]
            detail = [(columns[j % len(columns)], float(diff.ravel()[j]))
                      for j in worst if np.isfinite(diff.ravel()[j])]
            print(f"[{status}] {date}: {n_days}-day window, "
                  f"NaN-pattern mismatches={nan_mismatch}, |diff|>tol={bad}, "
                  f"max|diff|={max_diff:.4g}, examples={detail}")
        else:
            print(f"[{status}] {date}: {n_days}-day window, {len(train_rows)} rows, "
                  f"max|diff|={max_diff:.3g}, NaN mismatches=0")

    print()
    if failures:
        print(f"[parity] FAIL — {failures}/{len(dates)} dates disagree between the training "
              f"and inference feature paths")
        print("[parity] This means predict() sees different numbers than the model was "
              "trained on. Fix before submitting.")
        return 1
    print(f"[parity] PASS — training and inference paths agree on all {len(dates)} dates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
