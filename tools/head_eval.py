"""Fast local evaluator on the COMPOSITE objective the board actually scores.

IR is driven by the HEAD (top_k=50 equal-weighted), so full-cross-section RankIC is only
70% of the story.  This reports both terms plus a cheap proxy for the IR term, so candidate
models can be screened in seconds before spending ~25s on the official engine.

    ./venv/bin/python tools/head_eval.py <pred_glob> [<pred_glob> ...]
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPK = 50

def load(pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) \
             .drop_duplicates(subset=["日期", "代码"])

def evaluate(frame: pd.DataFrame) -> dict:
    fr = "future_return" if "future_return" in frame.columns else "future_return_g"
    d = frame.dropna(subset=[fr]).copy()
    d["r"] = d.groupby("日期")["score"].rank(ascending=False, method="first")
    full = d.groupby("日期").apply(
        lambda x: x["score"].corr(x[fr], method="spearman"), include_groups=False).dropna()
    top = d[d["r"] <= TOPK]
    t50 = top.groupby("日期")[fr].mean()
    univ = d.groupby("日期")[fr].mean()
    head = (t50 - univ).dropna()
    t50ic = d[d["r"] <= 100].groupby("日期").apply(
        lambda x: x["score"].corr(x[fr], method="spearman"), include_groups=False).dropna()
    return {"rankic": full.mean(), "head_spread": head.mean(),
            "head_t": head.mean() / head.std() * np.sqrt(len(head)),
            "top50ic": t50ic.mean(), "days": len(full)}

def main(argv: list[str]) -> int:
    print(f"{'candidate':34s} {'RankIC':>9s} {'head/d':>9s} {'head t':>8s} {'top50 IC':>9s} {'days':>6s}")
    rows = []
    for pat in argv:
        try:
            m = evaluate(load(pat))
        except Exception as exc:
            print(f"{pat[:34]:34s}  FAILED {type(exc).__name__}: {exc}")
            continue
        rows.append((pat, m))
        print(f"{Path(pat).stem[:34]:34s} {m['rankic']:+.5f} {m['head_spread']*100:+8.3f}% "
              f"{m['head_t']:+8.2f} {m['top50ic']:+.5f} {m['days']:6d}")
    if rows:
        base = rows[0][1]
        print()
        for pat, m in rows[1:]:
            print(f"  {Path(pat).stem[:30]:30s} dRankIC {m['rankic']-base['rankic']:+.5f}   "
                  f"dhead {((m['head_spread']-base['head_spread'])*100):+.3f}%/d")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
