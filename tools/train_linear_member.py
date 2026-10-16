"""Fit a LINEAR member and export it in the same 1-layer format the MLP member uses.

Qlib's published CSI300 benchmarks put Linear at Rank IC 0.0475 against LightGBM's 0.0485 -- i.e.
a linear model on the same features is nearly as good a ranker as the trees, while being a
completely different function class.  That makes it a strong candidate for the cross-family
diversity the board's IR rewards, and it costs no inference code: a single weight matrix plus bias
is exactly what `_mlp_scores` already evaluates (one layer, so no relu is applied).

Inputs are `cs_rank`-normalised, matching the MLP member's path.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtcomp.run_experiment import build_dataset   # noqa: E402
from dtcomp.models import DATE, TARGET            # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "submission" / "model" / "linear_member.npz"))
    ap.add_argument("--alpha", type=float, default=100.0)
    args = ap.parse_args(argv)

    ds = build_dataset(Path(args.data_dir))
    frame = ds.frame.loc[ds.frame["is_tradable"]].copy()
    frame = frame.loc[np.isfinite(frame[TARGET].to_numpy(dtype="float64"))]
    cols = list(ds.feature_columns)
    dates = frame[DATE].to_numpy()
    print(f"[linear] {len(frame):,} rows, {len(cols)} features")

    block = frame.loc[:, cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    X = (block.groupby(dates).rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype="float64")
    y = (pd.to_numeric(frame[TARGET], errors="coerce")
         .groupby(dates).rank(pct=True) - 0.5).to_numpy(dtype="float64")

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=args.alpha, random_state=42)
    model.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coefs_0=np.asarray(model.coef_, dtype="float64").reshape(-1, 1),
        intercepts_0=np.asarray([model.intercept_], dtype="float64"),
        n_layers=np.asarray([1], dtype="int64"),
    )
    print(f"[linear] wrote {out} ({out.stat().st_size/1024:.0f} KiB)")

    h = X @ model.coef_.reshape(-1, 1) + model.intercept_
    print(f"[linear] forward-pass max |diff| vs sklearn: {np.abs(h.ravel()-model.predict(X)).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
