"""Fit the MLP member on ALL public data and export its weights as .npz (no pickle).

The net needs `cs_rank`-normalised inputs (F80: raw costs it ~10x), so the transform is
reproduced exactly here and again at inference time:

    block = features.astype(float).replace([inf,-inf], nan)
    block = block.groupby(day).rank(pct=True) - 0.5
    block = block.fillna(0.0)

Weights are written as `coefs_<i>` / `intercepts_<i>` arrays; the forward pass is relu on hidden
layers and identity on the output, which was verified bit-identical to `sklearn.predict`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dtcomp.run_experiment import build_dataset   # noqa: E402
from dtcomp.models import DATE                     # noqa: E402
from dtcomp.models import TARGET                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "submission" / "model" / "mlp_member.npz"))
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    ds = build_dataset(Path(args.data_dir))
    frame = ds.frame.loc[ds.frame["is_tradable"]].copy()
    frame = frame.loc[np.isfinite(frame[TARGET].to_numpy(dtype="float64"))]
    cols = list(ds.feature_columns)
    dates = frame[DATE].to_numpy()
    print(f"[mlp] {len(frame):,} rows, {len(cols)} features, {frame[DATE].nunique()} days")

    block = frame.loc[:, cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    X = (block.groupby(dates).rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype="float64")
    y = (pd.to_numeric(frame[TARGET], errors="coerce")
         .groupby(dates).rank(pct=True) - 0.5).to_numpy(dtype="float64")

    from sklearn.neural_network import MLPRegressor
    hidden = tuple(int(v) for v in args.hidden.split(","))
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=args.epochs, batch_size=4096,
                       learning_rate_init=1e-3, alpha=args.alpha, early_stopping=False,
                       shuffle=True, random_state=args.seed)
    net.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for i, (w, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        payload[f"coefs_{i}"] = np.asarray(w, dtype="float64")
        payload[f"intercepts_{i}"] = np.asarray(b, dtype="float64")
    payload["n_layers"] = np.asarray([len(net.coefs_)], dtype="int64")
    np.savez_compressed(out, **payload)
    print(f"[mlp] wrote {out}  ({out.stat().st_size/1024:.0f} KiB, {len(net.coefs_)} layers)")

    # self-check: numpy forward pass must reproduce sklearn exactly
    h = X
    for i in range(len(net.coefs_)):
        h = h @ net.coefs_[i] + net.intercepts_[i]
        if i < len(net.coefs_) - 1:
            h = np.maximum(h, 0.0)
    ref = net.predict(X)
    print(f"[mlp] forward-pass max |diff| vs sklearn: {np.abs(h.ravel()-ref).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
