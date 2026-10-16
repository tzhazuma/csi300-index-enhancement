#!/usr/bin/env python
"""Replay the platform's calling convention against a submission package.

This is our safety net.  The organizers run:

    model = InferenceModel(model_dir="./submission/model/")   # once
    for each signal date t:
        data = {"MarketData":..., "Factor33_winsor":..., "Barra_factor":...}
        out  = model.predict("YYYY-MM-DD", data)              # per day
        check(out)

`simulate_judge.py` reproduces that loop on our public 2016-2022 data, applies every hard
rule from Submission_Rules.md section 5, and scores the result with RankIC.  Anything the
real judge would reject must be rejected here first, because a failed submission burns one
of our three daily slots.

Usage
-----
    ./venv/bin/python tools/simulate_judge.py --submission submission \
        --start 2022-01-04 --end 2022-12-30 [--lookback 120] [--stride 1]

Exit code is non-zero if any hard rule is violated.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtcomp.data import (  # noqa: E402
    BARRA, BARRA_COLS, FACTOR33, load_panel, build_labels, tradable_mask,
)
from dtcomp.cv import ic_summary, rankic_by_year  # noqa: E402

# The three keys the platform guarantees, with the columns it says it sends.
PLATFORM_KEYS = ("MarketData", "Factor33_winsor", "Barra_factor")
MARKET_SENT = ["日期", "代码", "名称", "open", "high", "low", "close",
               "volume", "amount", "is_suspend"]
FACTOR_SENT = ["日期", "代码", "名称", *FACTOR33]
BARRA_SENT = ["日期", "代码", "名称", *BARRA]


class RuleError(AssertionError):
    """A hard rule from the submission guide was violated."""


def load_inference(path: Path):
    """Import `inference.py` from a submission directory as a standalone module."""
    target = path / "inference.py"
    if not target.exists():
        raise FileNotFoundError(f"{target} not found")
    # The submission may use package-relative imports (`from .helper import ...`), so give
    # it a synthetic package so both plain and relative imports resolve.
    import types
    pkg_name = "submission_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(path)]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.inference", target)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def slice_asof(panel: pd.DataFrame, date: int, lookback: int) -> dict[str, pd.DataFrame]:
    """Build the `data` dict exactly as the platform describes it.

    "数据范围为截至当前交易日、最多 lookback 个交易日的研究历史" and
    "平台保证三张输入表中的最大日期恰好等于 as_of_date".
    """
    calendar = np.sort(panel["日期"].unique())
    upto = calendar[calendar <= date]
    window = upto[-lookback:] if len(upto) > lookback else upto
    lo = int(window[0])
    in_window = panel["日期"].between(lo, date)
    sub = panel.loc[in_window]
    # MarketData / Factor33_winsor are the yearly HS300 *union* (~327-354 names/day), but
    # Barra_factor holds ONLY the true members -- exactly 300 rows/day (verified across all
    # 1703 public days). Emitting the union here would overstate the universe and make the
    # simulator reject a perfectly correct submission.
    barra_rows = sub.loc[sub["in_barra"]]
    return {
        "MarketData": sub.loc[:, MARKET_SENT].copy(),
        "Factor33_winsor": sub.loc[:, FACTOR_SENT].copy(),
        # The platform sends Barra under its ORIGINAL column names (市值, 贝塔, ...); our
        # panel carries the `barra_` prefix internally, so strip it before handing the
        # frame to the model -- otherwise a model reading `data["Barra_factor"]` directly
        # would see different column names here than it will on the real judge.
        "Barra_factor": barra_rows.loc[:, ["日期", "代码", "名称", *BARRA_COLS]].rename(
            columns={f"barra_{name}": name for name in BARRA}).copy(),
    }


def check_output(out, as_of_date: str, universe: set[str]) -> pd.DataFrame:
    """Apply every hard rule from Submission_Rules.md section 5."""
    if not isinstance(out, pd.DataFrame):
        raise RuleError(f"predict returned {type(out).__name__}, expected pd.DataFrame")
    if list(out.columns) != ["date", "code", "score"]:
        raise RuleError(f"columns must be exactly ['date','code','score'], got {list(out.columns)}")
    if len(out) == 0:
        raise RuleError("predict returned an empty frame")

    dates = out["date"].unique()
    if len(dates) != 1 or dates[0] != as_of_date:
        raise RuleError(f"date column must be exactly {as_of_date!r}, got {list(dates)[:3]}")
    if not isinstance(dates[0], str):
        raise RuleError(f"date must be a str, got {type(dates[0]).__name__}")

    codes = out["code"]
    if codes.duplicated().any():
        dupes = codes[codes.duplicated()].unique()[:5]
        raise RuleError(f"duplicate codes: {list(dupes)}")
    got = set(codes.astype(str))
    missing, extra = universe - got, got - universe
    if missing:
        raise RuleError(f"{len(missing)} universe codes missing, e.g. {sorted(missing)[:5]}")
    if extra:
        raise RuleError(f"{len(extra)} codes outside the universe, e.g. {sorted(extra)[:5]}")

    score = pd.to_numeric(out["score"], errors="coerce")
    if score.isna().any():
        raise RuleError(f"{(score.isna()).sum()} non-numeric/NaN scores")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise RuleError("score contains inf/-inf")
    if score.nunique() < 2:
        raise RuleError("score is constant across the cross-section")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=ROOT / "submission")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "Data")
    ap.add_argument("--start", default="2022-01-04")
    ap.add_argument("--end", default="2022-12-30")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth signal date")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of signal dates")
    args = ap.parse_args()

    panel = load_panel(args.data_dir)
    labels = build_labels(panel)
    labels = labels.loc[tradable_mask(labels)]

    calendar = np.sort(panel["日期"].unique())
    start, end = int(args.start.replace("-", "")), int(args.end.replace("-", ""))
    dates = [int(d) for d in calendar if start <= d <= end][:: args.stride]
    if args.limit:
        dates = dates[: args.limit]
    if not dates:
        print("no signal dates in range", file=sys.stderr)
        return 2

    print(f"[judge] importing InferenceModel from {args.submission}")
    module = load_inference(args.submission)
    if not hasattr(module, "InferenceModel"):
        raise RuleError("inference.py does not define InferenceModel")

    t0 = time.time()
    model = module.InferenceModel(str(args.submission / "model"))
    print(f"[judge] __init__ ok in {time.time() - t0:.1f}s")

    # Sanity: the platform passes these exact types.
    if not isinstance(args.submission, Path):
        raise RuleError("internal: bad model_dir")

    frames, failures = [], []
    t0 = time.time()
    for i, date in enumerate(dates, 1):
        as_of = f"{date // 10000:04d}-{date // 100 % 100:02d}-{date % 100:02d}"
        data = slice_asof(panel, date, args.lookback)
        barra_today = data["Barra_factor"]
        universe = set(barra_today.loc[barra_today["日期"] == date, "代码"].astype(str))
        if not universe:
            raise RuleError(f"simulator produced an empty universe for {as_of}")
        try:
            out = model.predict(as_of, data)
            out = check_output(out, as_of, universe)
        except Exception as exc:  # noqa: BLE001 - we want every failure reported
            failures.append((as_of, f"{type(exc).__name__}: {exc}"))
            if len(failures) <= 3:
                traceback.print_exc()
            continue
        frames.append(out.assign(日期=date))
        if i % 25 == 0 or i == len(dates):
            print(f"[judge] {i}/{len(dates)} days, {time.time() - t0:.1f}s elapsed")

    if failures:
        print(f"\n[judge] FAIL: {len(failures)}/{len(dates)} signal dates violated a hard rule")
        for as_of, msg in failures[:10]:
            print(f"  {as_of}: {msg}")
        return 1

    scored = pd.concat(frames, ignore_index=True)
    # predict() emits `date, code, score`; our labels key on `日期, 代码`.
    scored["代码"] = scored["code"].astype(str)
    labels = labels.copy()
    labels["代码"] = labels["代码"].astype(str)
    merged = scored.merge(labels.loc[:, ["日期", "代码", "future_return"]],
                          on=["日期", "代码"], how="inner", validate="one_to_one")
    if merged.empty:
        print("[judge] FAIL: no overlap between predictions and labels", file=sys.stderr)
        return 1

    summary = ic_summary(merged, "score", "future_return")
    print("\n[judge] PASS — all hard rules satisfied on "
          f"{len(dates)} signal dates ({time.time() - t0:.1f}s)")
    print(f"[judge] coverage: {len(merged)}/{len(scored)} predictions matched a label")
    for key, value in summary.items():
        print(f"  {key:16s} {value:.6f}" if isinstance(value, float) else f"  {key:16s} {value}")
    print("\n[judge] per year:")
    print(rankic_by_year(merged, "score", "future_return").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
