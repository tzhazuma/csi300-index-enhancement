# CSI 300 Index Enhancement — 上海科技大学第一届量化金融挑战赛

Solution for the **DataTech × 上海TopK AI量化实验室** quantitative finance challenge
(`csi300-2026`): cross-sectional alpha prediction on CSI 300 constituents, evaluated as an
index-enhancement portfolio.

**Final result — public leaderboard: rank #30 of 153, score 0.7992** (RankIC 0.0695, IR 4.3083).

---

## The task

Predict a cross-sectional score for every CSI 300 constituent each day. The label is the
back-adjusted open-to-open return `open(t+2)/open(t+1) - 1`, and the organizers build the portfolio
themselves with a Barra-style optimizer (long-only, CSI 300 only, position/industry-deviation/style
constraints, turnover limits, fees, price-limit constraints) — **identical rules for every team**, so
all differences come from the alpha ranking.

```
Final score = 0.70 × s_rankic + 0.30 × s_ir
IR          = Annualized Excess Return / Tracking Error
```

Both terms are min-max normalized across the whole field.

## The model

A two-family rank-blend. Nothing exotic — the interesting part is *which* families and *what weight*.

| member | configuration | weight |
|---|---|---|
| 3 × LightGBM | 63 leaves, depth 8, lr 0.03, min_data 200, ff 0.70, bagging 0.80, l2 10.0, **400 fixed rounds (no early stopping)**, seeds 42/43/44 | 0.20 total |
| 1 × CatBoost | 400 iterations, depth 6, lr 0.04, l2 5.0 | **0.80** |

* **136 features** (momentum, reversal, volatility, volume, turnover, liquidity, and Barra-style
  exposures), all with a look-back of at most **120 trading days**.
* **Target**: within-day percentile rank of the label.
* **Blending**: each member's output is converted to a cross-sectional percentile rank and averaged
  (rank space is scale-free, so members with different output scales combine safely).
* **Post-processing**: `rank`.

## The finding worth reporting

**RankIC did not move; the entire gain came from IR.**

| CatBoost weight | RankIC | **IR** | score |
|---|---|---|---|
| 0.00 (LightGBM only) | 0.0711 | **0.8139** | 0.6130 |
| 0.35 | 0.0719 | 2.9945 | 0.6945 |
| 0.50 | 0.0713 | 3.4477 | 0.7051 |
| 0.65 | 0.0705 | 3.7939 | 0.7104 |
| **0.80** | 0.0695 | **4.3083** | **0.7199** |
| 1.00 (CatBoost only) | 0.0679 | 3.4143 | 0.6761 |

Every submission in this project scored RankIC in a **0.0602–0.0719** band regardless of recipe, while
IR spanned **0.81 to 4.67**. The weight axis is almost pure IR, and it has an **interior peak at
0.80** — CatBoost alone (1.00) falls back to 3.41.

The same is true of depth: 6 → IR 4.31, **4 → IR 4.6740**, 3 → IR 3.55.

### What did *not* work, with measurements

| direction | result |
|---|---|
| MLP as a third family | **IR 3.45 → 1.09** (score −0.113) |
| CatBoost seed-averaging (3 seeds, same total weight) | IR 4.31 → 3.43 |
| L2 regularization (l2 = 20) | IR 4.67 → 4.43 |
| Head-focused binary targets ("is this a top-30% name?") | RankIC 0.0838 → **0.0304** (top-decile: −0.0062) |
| XGBoost, ridge, low-capacity variants, time decay | neutral to negative |

**Cross-family blending helps only when the added family is competent.** CatBoost (RankIC 0.0483 in
Qlib's published CSI300 benchmarks) raised IR 5.3×; the MLP (0.0425 — the weakest tabular model in
that same table) destroyed it. Diversity is not free.

### The methodological lesson

> **Our local backtest engine could not see the leaderboard's IR — its sign was wrong.**
>
> It ranked CatBoost *below* LightGBM-only (IR 0.418 vs 0.446) when the leaderboard ranked it 5.3×
> *above* (4.31 vs 0.81).

Acting on that local signal cost **0.111 of score** until it was reversed. RankIC *can* be screened
locally (a long-holdout fold reproduces the leaderboard's degradation), but **IR can only be measured
by submitting**. The full write-up, including five separate "silent no-op" bugs caught along the way,
is in [`FINDINGS.md`](FINDINGS.md).

## Repository layout

```
submission/            the exact artifact that was submitted
  inference.py         self-contained inference (the only file the judge runs)
  train.py             reproduces the artifact from public data
  requirements.txt
  model/               3 LightGBM boosters, 1 CatBoost model, metadata.json
SUMMARY.md             one-page project summary
FINDINGS.md            130 measured findings, append-only, including every self-correction
dtcomp/                the research harness (feature library, CV, model configs)
tools/                 evaluation, packaging, and parity-check utilities
```

## Reproducing

```bash
pip install -r submission/requirements.txt
python -m submission.train --data-dir /path/to/Data --submission-dir submission \
    --lightgbm-seeds 42,43,44 --catboost-seeds 42 --catboost-weight 0.80
```

`train.py` reads only the organizers' public 2016–2022 data and rebuilds the model directory.

## Compliance

The submission was built to the competition's submission rules and passes a code review check:

* **No banned constructs** — `eval`, `exec`, `compile`, `__import__`, `getattr`, `subprocess`,
  `os.system`, `sys.popen`, `sys.exit`, pickle/marshal/base64. Verified with both word-boundary and
  raw-substring scans (the latter because the builder flags `eval`/`exec` even inside comments).
* **No serialized objects** — LightGBM `.txt` and CatBoost `.cbm` only. Model weights cannot carry a
  payload.
* **Self-contained** — `inference.py` needs only numpy / pandas / lightgbm / catboost, and every
  dependency it imports is declared in `requirements.txt`.
* **No data redistribution** — no organizer data or competition documents are included here.
* **Model classes are explicitly permitted** by the problem statement (LightGBM, CatBoost, ensembles).
