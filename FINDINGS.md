# Findings — CSI300 Index Enhancement Challenge

Owner: Lead. Append-only log of verified results. Everything here was measured, not assumed.

> **This log is append-only, so it contains SEVERAL "Archive of record" statements with different
> hashes.** Each was correct when written and is a historical record of how the package evolved.
> The **current** artifact is:
>
> * `out/submission.zip`, sha256 `06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`
> * config `lgbm_sub1_l63d8`: 3x LightGBM, lr 0.03 / **num_leaves 63** / **max_depth 8** /
>   min_data 200 / ff 0.70 / bagging 0.80 / l2 10.0 / 400 fixed rounds, seeds 42,43,44
>   rank-blended at 1/3, 136 features, no CatBoost
>
> Earlier hashes in this file (`314398fb`, `cd6385e0`, `0b9ea409`, `c1d332f5`) are superseded. The
> authoritative statement of what ships is `SUBMISSION_PLAN.md`; this file is the evidence trail.

---

## F1. The shipped baseline's 2022 number is in-sample — do not trust it

**Measured** with `tools/simulate_judge.py` (replays the platform's exact calling convention:
`InferenceModel(model_dir)` once, then `predict(as_of_date, data)` per signal day with a
rolling 120-trading-day window, enforcing every hard rule in `Submission_Rules.md` §5).

| model | evaluation period | mean RankIC | days |
|---|---|---|---|
| organizers' shipped `baseline_submission/model/` | 2022 | **+0.0870** | 240 |

That is **not** an out-of-sample number. The baseline's own `train.py` refits on
`final = samples with exit_date <= 2022-12-31` (`final_train_rows: 499799` in its
`model/metadata.json`). The shipped artifact has therefore already seen 2022.

Its metadata quotes `0.0566` for the early-stopped *selection* model, which is the
honest-ish figure the organizers intended to report.

## F2. Honest walk-forward numbers for the baseline recipe

Same feature set (`baseline_submission.inference.FEATURE_COLUMNS`, 49 columns), same
LightGBM hyper-parameters, 163 rounds, within-day percentile-rank target. Expanding window,
test period strictly excluded from training.

| train | test | mean RankIC | days |
|---|---|---|---|
| ≤ 2019-12-31 | 2020 | **+0.0521** | 243 |
| ≤ 2020-12-31 | 2021 | **+0.0378** | 243 |
| ≤ 2021-12-31 | 2022 | **+0.0496** | 240 |
| ≤ 2022-12-31 | 2022 | +0.0936 | 240 |

**Conclusions**

1. In-sample evaluation roughly **doubles** the apparent RankIC (0.0936 vs 0.0496). Any number
   produced without excluding the test period is worthless.
2. The baseline recipe's honest skill is **≈ 0.038–0.052**, mean ≈ 0.047.
3. Year-to-year spread is large: 2021 (0.038) is materially harder than 2020 (0.052). A model
   selected on one year will not transfer. **Report per-year RankIC for every candidate.**
4. Our final submission model is trained through 2022-12-31, so it can never be honestly
   evaluated on 2016-2022. Walk-forward folds are the only evidence we will ever have.

## F3. Leaderboard calibration (public board, 2026-10-09)

131 rows. RankIC min 0.0018 / p25 0.0504 / median 0.0652 / p75 0.0707 / **max 0.0832**.
IR min −1.243 / median 1.664 / max 6.433.
`score = 0.7*s_rankic + 0.3*s_ir`, `s_*` min-max normalised over the whole field.

Since the baseline recipe honestly scores ≈0.047 and the field median is 0.065, the median
competitor is already beating the baseline recipe out-of-sample. The leaders at 0.083 have a
genuinely better pipeline, not just a lucky seed. Headroom is real, and a robust 0.07+ is a
credible target.

## F4. Contract details confirmed against the real data

- Platform `data` dict keys: `MarketData`, `Factor33_winsor`, `Barra_factor`.
- The ten Barra style factors arrive under their **original** names (`市值`, `贝塔`, …), not
  our internal `barra_` prefix.
- Table dates are `int` YYYYMMDD; `as_of_date` is a `str` "YYYY-MM-DD"; the returned `date`
  column must be that same `str`.
- Universe = the codes present in that day's `Barra_factor` (exactly 300/day, verified across
  all 1703 public days).
- `is_suspend` is identical in `MarketData` and `Factor33_winsor` (0 mismatches over a test
  month), so either may be used.
- The judge scores **all 300** universe members including suspended ones, so local evaluation
  must use the full universe rather than a tradability-filtered subset.

## F5. Environment facts worth remembering

- Dev venv: `.venv/bin/python` (Python 3.11.15), with a `venv -> .venv` symlink for
  convenience. This matches the judge's Python 3.11.
- `files.pythonhosted.org` is **unreachable** from this machine; a plain `pip install` hangs.
  Use `UV_DEFAULT_INDEX="https://mirrors.cloud.tencent.com/pypi/simple/"` (≈34 MB/s).
- The DSH built-in browser needed its Electron binary: the `dsh-builtin-browser` plugin was
  installed in profile `desktop` but `node_modules/electron/dist` was empty. Fixed by running
  the plugin's documented `install-electron` (`node install.js` in
  `~/.dsh/profiles/desktop/node_modules/electron`). No DSH restart was required.

## F6. Platform status

Registered and **joined** as `Azuma` on https://datatech.geekpie.club (slug `csi300-2026`).
Daily submission limit **3**; none used yet. Submission is a 3-step presigned upload
(`POST /uploads` → `PUT` bytes → `POST /submissions`), single reserved upload slot.
Full spec at `/api/openapi.json`.

## F7. The submission pipeline is built and passes the judge contract

`tools/build_submission.py` assembles `submission/` from single sources of truth
(`pkg/inference.py`, `pkg/train.py`, `pkg/schema.py`, and the feature module copied in as
`features.py`), then runs a static self-containment/compliance check (relative imports
resolve locally, no import outside the package or the declared deps, no
`eval`/`exec`/`compile`/`__import__`/`subprocess`/`os.system`).

`tools/simulate_judge.py` replays the judge: `InferenceModel(model_dir)` once, then
`predict(as_of_date, data)` per signal day with a rolling 120-day window, then enforces every
hard rule from `Submission_Rules.md` §5 (exact three columns, `date` string equal to
`as_of_date`, universe covered exactly once, no NaN/Inf, suspended names included).

End-to-end run with the safety-net feature set: **PASS on 242/242 signal dates**, 27 s.
Honest walk-forward for that packaged model: 2020 **+0.0512**, 2021 **+0.0416**,
2022 **+0.0558** (mean **+0.0495**). Consistent with F2 — this is our floor, not our target.

### Two contract details the simulator forced us to get right

1. **`Barra_factor` has exactly 300 rows per day — the true members only.** `MarketData` and
   `Factor33_winsor` are the yearly HS300 *union* (327–354 names/day). Emitting the union in
   the Barra table overstates the universe and makes a correct submission look like it is
   missing ~42 names.
2. **Universe membership must be "the Barra table has a row for this (date, code)", not
   "some Barra factor is non-NaN".** Individual Barra factors carry ~12–13% missing values
   even for genuine members, so a `notna()` membership test silently drops ~42 names a day
   and fails the coverage rule. `pkg/schema.py` now marks membership explicitly.

## F8. In-sample vs out-of-sample, one more time

The packaged model scores **+0.1129** on 2022 through the simulator and **+0.0558** on the
same year via walk-forward. Same code, same data, ~2x difference — purely because the
shipped artifact is refit through 2022-12-31. Never quote a simulator number as a result.

## F9. model-lab's first nested walk-forward sweep (baseline feature set)

Nested protocol (selection decisions confined to training folds), full 300-name universe,
2018–2022, 1213 test days:

| config | mean RankIC | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|---|
| `lgbm_rank_raw` | **+0.0516** | +0.075 | +0.056 | +0.049 | +0.038 | +0.039 |
| `lgbm_rank_csz` | +0.0457 | +0.048 | +0.047 | +0.046 | +0.042 | +0.045 |
| `ridge_csrank` | +0.0452 | +0.063 | +0.046 | +0.043 | +0.042 | +0.031 |
| `lgbm_rank_csrank` | +0.0450 | +0.055 | +0.042 | +0.043 | +0.046 | +0.040 |
| `xgb_reg_csrank` | +0.0444 | +0.055 | +0.041 | +0.043 | +0.046 | +0.038 |
| `lgbm_zscore_csrank` | +0.0302 | +0.059 | +0.027 | +0.031 | +0.020 | +0.013 |

Notes: a **raw-return target with a raw (untransformed) score output** leads, and the
z-scored target is clearly worst. Skill decays over time (2018 +0.075 → 2021/2022 ≈ +0.039),
which is the regime risk that matters for the private board. Every config here is still on
the organizers' 49 provided columns — the feature library is the outstanding lever.

## F10. Two of our submissions were rejected; the cause was a compliance token scan

We spent real slots to validate the judge path rather than assume it. Results:

| # | package | outcome |
|---|---|---|
| 1 | 49 fallback features, relative imports, pinned deps | **failed** — "evaluation failed with exit code 1" |
| 2 | same, after making `inference.py` self-contained and relaxing the pins | **failed** — same error |
| 3 | **the organizers' own `baseline_submission.zip`** (control) | **finished** — rankic 0.0623, ir 2.5694, score 0.6694, **rank #61/133** |

The control is what made this tractable: the platform, our account, our auth and the judge
queue are all fine, so the fault had to be inside our package.

Root cause: `inference.py` contained the literal §9.3 vocabulary — `eval`, `exec`, `compile`,
`__import__`, `getattr`, `subprocess`, `os.system`, `sys.exit`, `pickle` — mostly inside a
docstring, plus a real but unreachable `pickle.load` branch, plus a genuine `getattr`-based
dynamic dispatch in the feature code. The organizers' accepted `inference.py` contains **none**
of those tokens. A static compliance scan cannot distinguish a banned word in a comment from a
real call, so it fails the submission.

Fixes applied: tokens purged from every shipped file; the `pickle` branch deleted; the
`getattr` dispatch replaced with explicit `if/elif`; and `tools/build_submission.py` now
hard-fails the build on any banned token, including a strict raw-substring check for `eval`
and `exec` anywhere in a `.py` file.

Two further observations that shape the gate:
- The baseline ships the raw substring `eval` inside `model/lightgbm_model.txt` and
  `model/model.txt` **and still passed** — so the scanner is not a raw substring scan over the
  whole archive. It is either word-boundary based or scoped to Python sources. We cannot tell
  which from outside, so we satisfy the stricter hypothesis.
- **Failed evaluations do not consume quota.** After 4 submissions the platform reported
  `2/3 used today` — only the successful one counted. Retrying is cheap, but we do not rely on
  it.

## F11. The feature library is worth +0.016 honest RankIC

Same model recipe (LightGBM + CatBoost rank blend, within-day percentile-rank target,
identical hyper-parameters), expanding-window walk-forward with the test year excluded:

| feature set | 2020 | 2021 | 2022 | mean |
|---|---|---|---|---|
| 49 provided columns (safety-net fallback) | +0.0512 | +0.0416 | +0.0558 | +0.0495 |
| **136 engineered features** | **+0.0705** | **+0.0585** | **+0.0683** | **+0.0658** |

feature-eng's independent paired ablation agrees: provided-only 0.0345 → frozen 136 0.0576,
**+0.0231 paired, positive in 12/12 (year, seed) comparisons**.

Honest null recorded: the 12 `*_xr` cross-sectional columns are worth +0.0012 (t = 1.52) on
their own — not significant, and flagged as the first block to drop if inference time binds.

## F12. Annual folds understate the shipped model (model-lab)

A 12-month test block lets the model go stale *inside* the block. Same frozen model, trained
through 2020-12-31, evaluated on 2021H1 vs 2021H2: **+0.0713 vs +0.0285**. A model retrained
through 2021-06-30 scores +0.0467 on that same 2021H2 block, so 6 months of extra history is
worth ~+0.018 to the same model family.

Whole-fold-set comparison, identical config: annual **+0.0516** vs semiannual **+0.0721**.
The most submission-like fold (train ≤ 2022-06-30, test 2022H2) gives **+0.0703** — and our
submission is applied immediately after its 2022-12-31 training cut, exactly like that fold.
So treat annual-fold numbers (including F11's +0.0658) as a **lower bound**.

Also from that sweep, two honest negatives: raw-return targets degenerate (constant daily
scores), and learning-to-rank objectives underperform here (LightGBM LambdaRank +0.034,
XGBoost rank:pairwise +0.016, vs +0.044 for XGBoost regression). Directly optimising the
order is not paying off on this data.

## F13. The DSH built-in browser needed its Electron binary

`dsh-builtin-browser` was installed in profile `desktop` but `node_modules/electron/dist` was
empty, so every `browser_*` tool reported the provider unavailable. Fixed by running the
plugin's documented installer (`node install.js` in
`~/.dsh/profiles/desktop/node_modules/electron`); no DSH restart was needed. The browser then
worked for the whole Casdoor login and all platform recon.



## F14. FIRST ACCEPTED SUBMISSION — rank #29 of 133

Cause of the earlier failures was the compliance token scan (F10). After purging the tokens,
the same 136-feature package was accepted:

| package | RankIC | IR | s_rankic | s_ir | score | public rank |
|---|---|---|---|---|---|---|
| **ours** (136 features, LGBM+CatBoost rank blend, percentile-rank target, `rank` postprocess) | **0.0712** | **2.8105** | 0.8524 | 0.5281 | **0.7551** | **#29 / 133** |
| organizers' baseline (control) | 0.0623 | 2.5694 | 0.7435 | 0.4967 | 0.6694 | #61 / 133 |

Field of 133: RankIC min 0.0018 / median 0.0657 / max 0.0832. Our 0.0712 sits at the 75.9th
percentile. Predicted well by F11's honest walk-forward (mean +0.0658), and above it in the
direction F12 predicted for a freshly-trained model.

**Where the remaining score is:** s_rankic 0.8524 but **s_ir only 0.5281**. IR carries 0.3 of
the score with far more headroom (field max 6.4329 vs our 2.8105). Per backtest-recon's engine
analysis, RankIC is the best proxy for realised IR (Spearman +0.79/+0.86) and score
post-processing is a dead end -- any strictly increasing transform is bit-identical, while
neutralisation costs up to -0.81 IR and tail ties cost up to -0.32 IR. So the lever remains
RankIC, and the 70% and 30% components are aligned.

## F15. Runtime is a live risk

Per `predict()` call on the real judge schema: `build_features` **1.514 s** of 1.523 s total
(97%); `normalise_input` 0.053 s. The cost is 180 `_Roller.roll` calls expanding to 61,740
per-group pandas `rolling` invocations.

The organizers' baseline package finished its entire hidden-period evaluation in **47 s**; ours
took roughly **45 minutes**. We do not know the judge's timeout, and a timeout on the private
board would forfeit the competition. feature-eng is doing a bit-exact numpy rewrite of the
rolling layer with a >= 5x target, gated on `np.array_equal(old, new, equal_nan=True)` over the
full panel.

## F16. `train.py` reproduces the shipped model byte-for-byte

The organizers re-run `train.py` during code review and void the result if it does not
reproduce `model/` (Submission_Rules.md §2). Verified rather than assumed:

    ./venv/bin/python -m submission.train --data-dir data/Data --submission-dir /tmp/repro

    IDENTICAL  lightgbm_seed42.txt
    IDENTICAL  lightgbm_seed202.txt
    IDENTICAL  lightgbm_seed2024.txt
    metadata identical (ignoring timestamps): True

All three boosters hash identically to the shipped artifacts, so training is deterministic
given the seed set (`deterministic: True`, fixed `num_threads`, no early stopping in the
frozen fit).

## F17. Second submission — seed-bagged LightGBM (submitted 2026-10-09 02:15 CST)

Recipe = model-lab's winning nested semiannual config `lgbm_rank_raw__seedbag3`, frozen into
the submission: 136 features, within-day percentile-rank target, LightGBM regression,
`{learning_rate 0.04, num_leaves 15, max_depth 6, min_data_in_leaf 100, feature_fraction 0.85,
bagging_fraction 0.8, lambda_l2 5.0}`, 500 rounds, **seeds [42, 202, 2024] blended by
cross-sectional rank**, CatBoost dropped (the validated winner is LightGBM-only),
postprocess `rank`.

Measured locally on nested semiannual folds: mean RankIC **+0.0800**, RankICIR **+0.556**,
1338 test days, 0 degenerate days. Versus +0.0720 for the same config single-seed, so seed
bagging is worth ~+0.008 mean RankIC and a clear RankICIR gain — which is the component that
should feed IR, our weakest leaderboard column (s_ir 0.5281 vs s_rankic 0.8524).

Cost: 79 s to fit 3 boosters, 2.4 MB of artifacts. Inference cost is unchanged in practice —
`build_features` is 1.514 s of a 1.523 s `predict()` call, so three boosters instead of two
adds well under 1%.

Quota is now 3/3 for 2026-10-09; the next slot opens 2026-10-10 00:00 +08:00.

## F18. The 2017-2022 folds did NOT predict the hidden period

Two submissions, same 136 features, same target, same postprocess; only the model recipe differs:

| # | recipe | RankIC | IR | score | public rank |
|---|---|---|---|---|---|
| 1 | LGBM(lr .03, leaves 31, depth 7, min_data 200, ff .70, l2 10, 400 r) + CatBoost(400 it), 50/50 | **0.0712** | **2.8105** | **0.7551** | #29 |
| 2 | `lgbm_rank_raw__seedbag3`: 3x LGBM(lr .04, leaves 15, depth 6, min_data 100, ff .85, l2 5, 500 r), no CatBoost | 0.0602 | 1.0785 | 0.5931 | — |

Recipe 2 was **selected by nested semiannual walk-forward on 2017H2-2022H2** and was the best
config in that harness (mean RankIC +0.0800, RankICIR +0.556). It lost ~0.011 RankIC and
~1.73 IR on the real target. Recipe 1 was chosen with no fold selection at all -- it is
essentially the organizers' baseline architecture with our features, more rounds, and deeper
trees.

**Conclusion: fold-based hyper-parameter selection over 2017-2022 is selecting for a different
market.** The hidden period follows 2022-12-31 and appears in none of our folds.

Honest caveat: with ~250 hidden days and daily RankIC sd ~0.144, the standard error on a mean
RankIC is ~0.009, so the 0.011 RankIC gap is only ~1.2 SE and is not decisive on its own. The
IR gap is larger relative to its own scale, but IR is the noisier statistic. We are treating
this as evidence, not proof, and we are not spending more slots on fold-tuned configs.

## F19. CatBoost is deterministic; only its container metadata differs

`catboost_seed42.cbm` hashes differently between two identical training runs, which looked like
a reproducibility failure (Submission_Rules.md §2 voids the result if `train.py` does not
reproduce `model/`). It is not:

    max|prediction difference| : 0.0000e+00
    spearman(pred_a, pred_b)   : 1.000000000000
    tree count                 : 400 vs 400
    file size                  : 464272 vs 464272

The trained model is bit-identical in behaviour; only bytes inside the `.cbm` container differ.
All three LightGBM boosters hash **byte-identically**, and `metadata.json` matches apart from
timestamps. So `train.py` does reproduce the submission.

## F20. Third submission prepared: proven architecture + seed bagging

Reverted to submission #1's architecture (the only recipe with direct evidence on the target
period) and layered seed bagging on top, keeping CatBoost:

  * 3x LightGBM, lr 0.03 / num_leaves 31 / max_depth 7 / min_data_in_leaf 200 /
    feature_fraction 0.70 / bagging_fraction 0.80 / lambda_l2 10.0 / 400 rounds, seeds
    [42, 43, 44], rank-blended at 0.1667 each (0.50 total)
  * 1x CatBoost, lr 0.04 / depth 6 / l2_leaf_reg 5.0 / 400 iterations, weight 0.50

Bagging is pure variance reduction on an architecture that demonstrably works, and it preserves
the LightGBM/CatBoost blend whose removal coincided with the IR collapse. Built, trained
(109 s), simulator-validated, and reproducibility-checked. Ready for the 2026-10-10 00:00 slot.

Quota: 3/3 used on 2026-10-09; next slot 2026-10-10 00:00 +08:00. 5 days x 3 slots remain
before the 2026-10-14 23:59 CST deadline.

## F21. Runtime risk closed — verified independently, not taken on trust

feature-eng rewrote the rolling layer to call pandas' private Cython kernels
(`pandas._libs.window.aggregations.roll_*`) once per column with per-block window bounds,
removing pandas' per-column Python loop. Because that is a private API and a wrong value here
would silently corrupt every prediction, I verified it myself rather than accepting the report:

| check | result |
|---|---|
| fast path vs pre-optimisation path, full 590,238-row panel | **bit-identical** (all 136 columns, NaN patterns and row order identical) |
| fast path vs pre-optimisation path, judge windows 20221230 / 20211008 / 20190422 | **bit-identical** |
| fast path vs forced slow path (kernels disabled), 3 frames incl. full panel | **bit-identical** |
| kernels enabled by the import-time probe | `mean`, `std`, `min`, `max` — `skew` correctly rejected |
| module import with the private API unavailable | imports cleanly, falls back |

Measured `predict()` cost: **1523 ms -> 197 ms** (7.7x). A 250-day hidden period drops from
~6.3 min to **~0.8 min**, against a baseline package that took 47 s for the whole run. The
judge's own pandas is not guaranteed to be ours (`requirements.txt` says `pandas>=2.2`), and the
probe plus fallback means a different pandas can only cost speed, never values.

Full-year simulator after the rebuild: **PASS on 242/242 signal dates in 51 s** (was 267 s for
146 days). Reproducibility re-checked on the rebuilt package: all three LightGBM boosters
byte-identical, CatBoost identical tree count.

## F22. Submission #3 is built, validated and reproducible — waiting on the quota

`submission/` currently holds: 136 fast features, 3x LightGBM (lr 0.03 / leaves 31 / depth 7 /
min_data 200 / ff 0.70 / bagging 0.80 / l2 10.0 / 400 rounds, seeds [42,43,44], 0.1667 each)
plus CatBoost (lr 0.04 / depth 6 / l2 5.0 / 400 iterations, weight 0.50) -- submission #1's
architecture, the only one with direct evidence on the target period, plus seed bagging as pure
variance reduction.

Verified: `tools/build_submission.py` compliance gate PASS, `tools/simulate_judge.py` PASS on
242/242 days, `train.py` reproduces `model/` byte-for-byte.

Quota 3/3 used on 2026-10-09; next slot 2026-10-10 00:00 +08:00.

## F23. The submission is robust to short look-back windows

The judge provides "at most 120 trading days" and "sufficient history per the competition
config", but the warm-up days at the start of the hidden period may carry far less. A missing
history that makes `predict()` raise would fail the whole submission, so this is checked rather
than assumed. Same package, 22 signal dates, only the window length varied:

| look-back given | result | in-sample mean RankIC |
|---|---|---|
| 120 days | PASS | 0.1683 |
| 60 days | PASS | 0.1577 |
| 20 days | PASS | 0.1506 |
| 5 days | PASS | 0.1281 |
| 2 days | PASS | 0.1137 |
| **1 day** | **PASS** | 0.0754 |

Every hard rule holds at every window length and the score degrades smoothly instead of
raising. Features simply go NaN and are filled with the training medians, which is exactly the
intended fallback. (The RankIC column is in-sample and meaningless as a score -- it is here only
to show the graceful degradation.)

## F24. What the public board can and cannot tell us

Each distinct configuration yields exactly **one** board number: the judge re-runs the same
model on the same hidden data, so re-submitting a package reproduces its score and there are no
repeats to average. With ~250 hidden days and daily RankIC sd ~0.144, a single observation
carries ~0.009 standard error. Distinguishing two configs therefore needs an effect of ~0.02+
to be credible, and we will never resolve small differences.

Consequence for the remaining 15 slots: do not chase marginal gains. Spend them on changes with
a strong independent prior (more bagging, genuinely new feature information), keep the proven
configuration as the fallback, and make sure the LAST submission -- the one the private board
actually scores -- is the best-evidenced one.

## F25. RESOLVED: the fold harness IS valid — we were in the wrong hyper-parameter region

F18 concluded that 2017-2022 folds "did not predict the hidden period". That conclusion was
**wrong, or at least far too strong**, and model-lab's calibration experiment shows why: their
62-config sweep simply never contained the region that works. None of the 62 had
`num_leaves=31` with `max_depth=7`.

Running submission #1's architecture through the SAME nested harness:

| config | semiannual | annual |
|---|---|---|
| `lgbm_sub1__sb3` (sub #1 arch, 3 seeds) | **+0.0823** | +0.0750 |
| `arch_lgbm_sub1_sb3+cat_sub1_fx` | **+0.0808** | +0.0743 |
| `lgbm_rank_raw__fxseedbag5` | +0.0804 | +0.0736 |
| `lgbm_rank_raw__seedbag3` (what we shipped as #2) | +0.0800 | — |
| `cat_sub1__fx` (CatBoost alone, sub #1 params) | +0.0748 | +0.0701 |
| `lgbm_sub1__single` (sub #1 arch, 1 seed) | +0.0738 | — |

So the harness **does** rank submission #1's architecture at the top, agreeing with the board.
The earlier "folds are useless" reading was an artifact of an unexplored search space, not a
broken protocol. The real lesson is narrower and more actionable:

**The winning region is DEEPER and slower than anything model-lab tried** — 31 leaves / depth 7 /
min_data 200 / feature_fraction 0.70 / lambda_l2 10 at learning rate 0.03 with 400 rounds, versus
their 15 leaves / depth 6 / min_data 100 / ff 0.85 / l2 5 at 0.04. CatBoost at those same
settings is also strong on its own (+0.0748 semiannual), which independently supports keeping it
in the blend.

Also note seed bagging helps *within* this region too: `lgbm_sub1__single` +0.0738 ->
`lgbm_sub1__sb3` +0.0823 semiannual, +0.0085.

## F26. Our prepared submission is the harness's #2 config

The package already built, validated and waiting for the 2026-10-10 00:00 slot is exactly
`arch_lgbm_sub1_sb3 + cat_sub1_fx`:

  * 3x LightGBM, sub #1 params (lr 0.03 / leaves 31 / depth 7 / min_data 200 / ff 0.70 /
    bagging 0.80 / l2 10.0 / 400 rounds), seeds [42,43,44], 0.1667 each
  * 1x CatBoost, lr 0.04 / depth 6 / l2_leaf_reg 5.0 / 400 iterations, weight 0.50

It is simultaneously (a) the architecture with direct evidence on the target period, (b) the
harness's #2 configuration at +0.0808 semiannual / +0.0743 annual, and (c) submission #1 plus
pure variance reduction. The harness's #1 (`lgbm_sub1__sb3`, no CatBoost) is only +0.0015 ahead
on semiannual RankIC, and we have board evidence that dropping CatBoost coincided with IR
falling 2.81 -> 1.08, so keeping CatBoost hedges the 30% IR component at negligible RankIC cost.

## F27. Judge-faithful validation in a clean environment

The strongest test we can run without spending a slot: a **fresh Python 3.11 venv with only
`submission/requirements.txt` installed**, the submission loaded in the **hardest import mode**
(`spec_from_file_location`, no package context, nothing on `sys.path`), fed the real three-table
data over a 120-day window, and checked against every rule in Submission_Rules.md §5.

    PASS: 30/30 signal dates, clean venv, no package context (~230-520 ms/day)
    stderr: 0 lines

`requirements.txt` resolved to exactly the versions we developed against
(numpy 2.4.6, pandas 2.3.3, lightgbm 4.7.0, catboost 1.2.10, pyarrow 25.0.1), so the judge's
`pip install -r requirements.txt` step is known-good rather than assumed.

One polish applied: `normalise_input` used `fillna(False).astype(bool)` on the membership marker,
which made pandas emit a downcasting FutureWarning once per `predict()` call (~30 lines of
stderr per evaluation). Replaced with `.eq(True)`, which yields a clean bool array. Judge stderr
is now empty.

## F28. Residual reversal was a bad hypothesis — an honest null, and a standing warning

I proposed beta-hedged (residual) reversal as the most promising untried direction: the library
is all raw reversal, and removing the market-beta component is the textbook next step. It does
not work, and the measurement says my reasoning was backwards.

Paired walk-forward ablation, 4 test years x 3 seeds, against the frozen 136:

| candidate set (n) | paired RankIC | years up | t |
|---|---|---|---|
| extended, no `beta_x_rev` (151) | +0.0028 | 9/12 | +2.92 |
| extended, all 16 new (152) | +0.0027 | 9/12 | +2.65 |
| frozen + core 3 (139) | +0.0019 | 11/12 | +3.04 |
| frozen + `mkt_ret` only (139) | +0.0016 | 8/12 | +1.61 |
| frozen + `idio_vol` only (138) | +0.0015 | 10/12 | +2.74 |
| **frozen + residual reversal only (144)** | **+0.0003** | 7/12 | +0.38 |
| **frozen + `beta_x_rev` only (137)** | **-0.0000** | 6/12 | -0.04 |

**Beta-hedging makes raw reversal WEAKER, not cleaner**: `resid_ret_5` -0.0257 vs `ret_5`
-0.0325, `resid_ret_60` -0.0031 vs `ret_60` -0.0070. The market component was *helping* the
univariate signal. Residual reversal as a block adds +0.0003 (t=0.38) and is negative in 2021
and 2022.

**The most instructive result in the project**: `beta_x_rev` is the strongest reliable univariate
feature in the entire 136-feature library (-0.0333, sign-stable in all seven years) and adds
*exactly nothing* (-0.0000, t=-0.04, 6/12). It is `beta * ret_5` — an interaction of two columns
the frozen set already contains, and the trees had already learned it. **Univariate RankIC is not
incremental value.** Rank features by paired ablation, never by standalone IC.

The one survivor is `idio_vol_20/60` (std of `ret_1 - beta*mkt_ret_1`): +0.0015, t=2.74, 10/12
positive, and 40% better univariately than total volatility (-0.0255 vs -0.0183). Also worth
recording: `mkt_ret_*` are constant within a day, so their univariate IC is *undefined* rather
than zero — they can only act as regime conditioning, and at t=1.61 with negative 2019-2020 they
are not shippable.

### Decision: keep the frozen 136, do not ship the 138

Not because `idio_vol` is uninteresting, but because the effect is unmeasurable:

* one public-board observation carries ~0.009 standard error, and the judge is deterministic so
  repeats are impossible;
* the measured expected value of the whole direction is **+0.0015**, one sixth of one
  observation's noise;
* the frozen 136 holds the only direct evidence on the target period in this project (0.0712 on
  the hidden board), and adding two columns retrains every tree — it is a different model, not a
  pure addition.

`EXTENDED_FEATURE_COLUMNS` (152) and the extended code path remain in place and tested (15
tests green), so the option is open if we ever have slots we cannot use better.

feature-eng also flagged the right caveat: ten candidate sets compared on the same folds that
selected everything else, with t of 2.6-3.0 and the largest deltas the least stable, is exactly
the selection pressure that produced submission #2's failure.

## F29. RankIC predicts realized IR ACROSS SIGNALS (+0.963) — one objective, not two

backtest-recon ran the **official** `skd-backtest` engine over model-lab's cached out-of-sample
fold predictions (`reports/preds/*__semiannual__outer.parquet`), i.e. a genuine walk-forward
replay through the organizers' own optimizer and cost model:

| config | realized IR | mean RankIC |
|---|---|---|
| `arch_sub1` (bagged LGBM + CatBoost) | **+0.3075** | 0.0675 |
| `arch_lgbm_sub1_sb3+cat_sub1_fx` (what we have built) | **+0.3075** | 0.0675 |
| `lgbm_sub1__sb3` (bagged LGBM, no CatBoost) | +0.3061 | **0.0702** |
| `lgbm_rank_raw__seedbag3` (submission #2's recipe) | +0.0801 | 0.0674 |
| `lgbm_sub1__single` (sub #1 arch, ONE seed) | +0.0306 | 0.0618 |
| `arch_lgbm_sub1_single+cat_sub1_fx` | -0.1539 | 0.0628 |
| `cat_sub1__fx` (CatBoost alone) | -0.2063 | 0.0610 |

**Cross-signal Spearman(RankIC, realized IR) = +0.963, p < 0.001** over 11 configurations. The
earlier +0.79/+0.86 was measured across *transformations of one signal*; this is across
*different signals*, which is the question that actually mattered. Conclusion: RankIC and IR are
near-redundant here, the 0.70/0.30 split is effectively one objective, and there is no trade-off
to exploit. **Maximise RankIC and IR follows.**

**Seed bagging is worth ~10x on realized IR**: `lgbm_sub1__single` +0.0306 -> `lgbm_sub1__sb3`
+0.3061, with RankIC 0.0618 -> 0.0702. This is by far the largest single effect we have measured
on the 30% component, and it is pure variance reduction.

**CatBoost is not earning its place.** `cat_sub1__fx` alone has *negative* realized IR (-0.2063)
despite respectable RankIC (0.0610) -- its scores rank acceptably but build a poor portfolio.
Adding it at weight 0.5 leaves IR unchanged (0.3075 vs 0.3061, inside noise) while *lowering*
RankIC in both independent measurements (engine 0.0675 vs 0.0702; harness semiannual 0.0808 vs
0.0823). By F29's rule, the lower-RankIC blend is the worse choice.

Note the engine's ordering validates against the only two board observations we have: it ranks
sub #1's family IR +0.3075 vs sub #2's recipe +0.0801, and the board gave 2.8105 vs 1.0785 --
same direction, same rough magnitude. The engine's *RankIC* ordering, by contrast, does not
reproduce the board gap (0.0675 vs 0.0674 locally, 0.0712 vs 0.0602 on the board), which is
consistent with that gap being largely one-observation noise.

## F30. DECISION: drop CatBoost — ship the seed-bagged LightGBM alone

F29 says maximise RankIC, because RankIC predicts realized IR across signals at +0.963. Three
independent measurements all put the bagged LightGBM above the LightGBM+CatBoost blend:

| measurement | LGBM-only (`lgbm_sub1__sb3`) | blend (`arch_lgbm_sub1_sb3+cat_sub1_fx`) |
|---|---|---|
| harness mean RankIC, semiannual (1338 d) | **+0.0823** | +0.0808 |
| harness mean RankIC, annual (1213 d) | **+0.0750** | +0.0743 |
| official engine mean RankIC | **0.0702** | 0.0675 |
| official engine realized IR | 0.3061 | 0.3075 (inside noise) |
| harness RankICIR, semiannual | **+0.570** | +0.547 |
| in-sample simulator RankIC / RankICIR | **0.1995 / 1.511** | 0.1689 / 1.240 |

And per year it is not one lucky period — LGBM-only wins **5 of 6** semiannual years and **4 of 5**
annual years:

    2017  LGBM +0.103  blend +0.101
    2018  LGBM +0.107  blend +0.108   <- the only year the blend wins
    2019  LGBM +0.075  blend +0.074
    2020  LGBM +0.081  blend +0.076
    2021  LGBM +0.063  blend +0.062   <- the hardest year for every config
    2022  LGBM +0.074  blend +0.073

The decisive mechanism is that **CatBoost's scores are pathologically shaped for this optimizer**:
`cat_sub1__fx` alone has respectable RankIC (0.0610) but a *negative* realized IR (-0.2063). Since
the optimizer converts scores to percentile ranks and only ORDER matters, that means CatBoost's
ordering has bad tail structure in exactly the region the optimizer trades. At weight 0.5 it is
diluted to neutral on IR while still costing RankIC.

The one counter-argument is our only direct board observation (submission #1, a 1-seed
LightGBM+CatBoost blend, 0.0712 / IR 2.8105 / rank #29). I am overriding it, for three reasons:
it is a **single** observation carrying ~0.009 standard error; it used a **1-seed** model, not the
bagged one; and the engine rates that exact 1-seed configuration at IR -0.1539, so the engine and
the board disagree about it specifically -- it is the least reliable data point we have.

**Shipped package** (`submission/`, archive of record):
3x LightGBM, lr 0.03 / num_leaves 31 / max_depth 7 / min_data_in_leaf 200 / feature_fraction 0.70
/ bagging_fraction 0.80 / lambda_l2 10.0 / 400 rounds, seeds [42, 43, 44], rank-blended at 1/3
each, 136 frozen features, within-day percentile-rank target, `rank` postprocess. 3.7 MB of
artifacts. The blend package is preserved at `/tmp/blend_submission` as a fallback.

Verified on the swapped-in package: `build_submission.py` gate PASS and **build is deterministic**
(all four generated files hash identically on rebuild), `simulate_judge.py` PASS on 242/242
signal dates, clean-venv judge smoke test PASS 30/30 with **0 stderr lines**, and `train.py`
reproduces all three boosters byte-for-byte.

**Archive of record**: `out/submission.zip`, sha256
`314398fbc4406ebdd78b8dd052ae4ed6038de02ba5e0c9b70aa6c6bb454abdcd`, 1.41 MiB, 8 files + 2
directories, container modes matching the organizers' own accepted package (dirs `0o40777`,
files `0o100666`). Confirmed by listing that it carries the three LightGBM boosters and **no**
CatBoost artifact.

## F31. Two operational traps found and neutralised

1. **Bare `python3` on this machine prints nothing and exits 0.**
   `which python3` -> `/Users/azuma/toolchains/pyocd-venv/bin/python3`, and
   `python3 -c "print('x')"` produces no output with exit status 0. A script that writes files
   still writes them, so a run can look successful while emitting nothing, and anything reading
   results from stdout would silently record empties. All commands now invoke
   `./venv/bin/python` explicitly. feature-eng found this; I verified it.
2. **A stale archive is easy to ship by accident.** After swapping the package I checked
   `out/submission.zip` and found the *previous* package's hash -- `zip_submission.py --list`
   only lists, it does not rebuild. The archive must be regenerated after any package change, and
   the hash compared against the expected one. Now recorded above.

## F32. `lgbm_sub1__sb3` is Pareto-optimal across all 35 engine-measured configs

Sorted by engine mean RankIC, with the official engine's realized IR beside it:

| engine RankIC | realized IR | config |
|---|---|---|
| **0.0702** | **+0.3061** | **`lgbm_sub1__sb3` (shipped)** |
| 0.0691 | +0.1891 | `lgbm_sub1__fixed400` |
| 0.0677 | +0.0061 | `lgbm_rank_raw__fxseedbag5` |
| 0.0675 | +0.3075 | `arch_lgbm_sub1_sb3+cat_sub1_fx` |
| 0.0674 | +0.0801 | `lgbm_rank_raw__seedbag3` |
| 0.0628 | -0.1539 | `arch_lgbm_sub1_single+cat_sub1_fx` |
| 0.0610 | -0.2063 | `cat_sub1__fx` |

Nothing dominates it: it has the highest engine RankIC **and** the second-highest IR (0.3061 vs
0.3075, a difference far inside noise). Over all 35 cached configs, Spearman(RankIC, IR) =
**+0.879 (p < 0.0001)** and Spearman(RankICIR, IR) = +0.836, so RankIC is the objective and the
shipped config leads on it.

## F33. Early stopping HURTS — and our frozen model already avoids it

The sub1 variants separate two effects that had never been measured independently:

| config | seeds | early stopping | mean RankIC | engine IR |
|---|---|---|---|---|
| `lgbm_sub1__single` | 1 | yes (35) | +0.0738 | +0.0306 |
| `lgbm_sub1__fixed400` | 1 | **no** | **+0.0809** | **+0.1891** |
| `lgbm_sub1__sb3` | 3 | yes (35) | +0.0823 | +0.3061 |

* **Turning early stopping off at one seed is worth +0.0071 mean RankIC and +0.16 realized IR.**
* **Seed bagging is worth +0.0085 mean RankIC and +0.28 realized IR.**

Both effects are large and both point the same way. Our frozen `train.py` fits to a **fixed 400
rounds with no validation set**, so it already sits on the good side of the first effect, and it
bags 3 seeds for the second. The exact combination (3 seeds + fixed 400) is the one cell the
sweep never measured, which is why I asked model-lab to run it before we spend a slot.

Round count is otherwise flat: `R700` +0.0813, `fixed400` +0.0809, `R1000` +0.0807 semiannual, so
400 is not a knife-edge.

## F34. Rough board projection for the shipped package

We have exactly one (board, engine) pair to calibrate against: submission #1's architecture scored
board RankIC 0.0712 while the engine rates it 0.0628, a ratio of ~1.13. Applying that same ratio
to the shipped config's engine RankIC of 0.0702 gives a projected board RankIC of **~0.080**.

This is an extrapolation from a single point and should be treated as a sanity check, not a
forecast -- the ratio itself carries the full ~0.009 standard error of one board observation. It
is consistent with the direction everything else points, and with `lgbm_sub1__sb3` being the
best RankIC config we have found. If it holds, it would place us near the top of the RankIC
column (field max 0.0832).

## F35. FINAL BLEND DECISION — LightGBM-only, and the arithmetic that settles it

backtest-recon recommended keeping the CatBoost blend, on the grounds that the two finalists are a
statistical dead heat on IR (paired daily active-return test t=-0.06, **p=0.9502**) and that in the
top RankIC band the RankIC->IR relationship collapses (RankIC >= 0.070: Spearman +0.537;
>= 0.075: +0.771, p=0.072, n.s.), so a RankIC-driven switch is "a coin flip dressed up as a
decision". I verified the collapse independently on n=37 (global Spearman +0.891, p=1.5e-13;
residual sigma 0.187 IR, and a 0.0027 engine-RankIC gap moves *fitted* IR by only 0.56 sigma).
That part is correct and I adopted it.

**But that analysis treats RankIC only as a predictor of IR. The score uses RankIC directly:**

    score = 0.70 * s_rankic + 0.30 * s_ir

so the right question is what each gap is worth in score units, not whether RankIC predicts IR.
With the field's RankIC range (0.0018..0.0832, span 0.0814) and IR range (span ~7.7):

| gap | value | -> score |
|---|---|---|
| RankIC 0.0027 (LGBM-only ahead) | 0.0027 / 0.0814 = 0.0332 s_rankic | **+0.023** |
| IR 0.0014 (blend ahead, p=0.95) | 0.0014 / 7.7 = 0.00018 s_ir | +0.00005 |

**The RankIC gap is worth ~450x the IR gap per unit.** If the RankIC gap is noise then the IR gap
is noise too, and the asymmetry says take the positive RankIC point estimate rather than a
non-scored risk improvement.

On the secondary criteria: the blend's advantages are max drawdown (30.9% vs 35.6%) and tracking
error (7.34% vs 7.42%), and **neither is in the score** -- the board reports `rankic` and `ir` only.
On the one stability measure that does inform IR, the blend is *worse*: RankICIR +0.547 vs +0.570,
so the model-diversification argument does not show up as lower IC variance.

Also adopted from that report: CatBoost **alone** is the weakest of the three (IR -0.2063) and the
blend beats it significantly (p=0.026), so the correct reading of the board is "the combination is
fine", not "CatBoost is good" -- my earlier confounded interpretation is dead. And the
positive-day-ratio correlation (+0.83 in the top band) was correctly *declined* as a selection
objective: with n=6 it is driven by one config and would not survive out of sample.

**Shipped: `lgbm_sub1__sb3`.** Leads on engine RankIC (0.0702 vs 0.0675), harness RankIC (+0.0823
vs +0.0808), RankICIR (+0.570 vs +0.547), and wins 5 of 6 semiannual years. The blend remains at
`/tmp/blend_submission` as a fallback.

## F36. The blend decision is now independently verified — and the diversification value is ~zero

backtest-recon reproduced my score-unit arithmetic and **withdrew their recommendation**. The
decisive additions they made, which nobody had computed:

**The correct test is PAIRED, not two means.** Both configs are scored on the same days:

| label | RankIC gap (sb3 - blend) | t | p | sb3 wins |
|---|---|---|---|---|
| harness (future_return, 1338 d) | +0.001425 | +1.857 | 0.0634 | 50.4% of days |
| engine (5d open-to-open, 267 d) | +0.002664 | +1.597 | 0.1103 | 52.1% |
| *sanity:* sb3 vs CatBoost-only | +0.007473 | +4.983 | <0.0001 | 55.4% |
| *sanity:* blend vs CatBoost-only | +0.006049 | +7.807 | <0.0001 | 59.1% |

The gap is **borderline-suggestive, not established** — but the sanity rows prove the test does
detect real differences when they exist. Propagating both gaps' standard errors:

    E[d_score]  = +0.01220      SD[d_score] = 0.01176      P(sb3 outscores blend) = 0.85

**The diversification value I asked them to quantify is zero or negative.** This is the answer that
settled it:

| | `lgbm_sub1__sb3` | blend | verdict |
|---|---|---|---|
| mean RankIC | +0.082270 | +0.080845 | sb3 |
| RankIC std | 0.144340 | 0.147852 | **sb3 — the blend is NOISIER** |
| RankICIR | 0.5700 | 0.5468 | sb3 |
| RankIC 5th percentile | -0.15510 | -0.16473 | **sb3 — blend tail is worse** |
| worst single day | -0.34676 | -0.38900 | sb3 |
| variance ratio | — | 0.9531 | F-test p=0.3795, no variance reduction |

A two-family blend does **not** reduce RankIC variance here, and its RankIC tail is worse on both
the 5th percentile and the worst day. The diversification is real in drawdown and tracking error —
neither of which is in the score — and does not appear in RankIC at all. Persistence: sb3 wins 7/11
folds (binomial p=0.27) and 5/6 years (p=0.11); the one year the blend wins is **2018, the easiest
year for both configs** (mean RankIC 0.107), so the blend's only edge is a 0.9bp difference in the
easy regime, not a cushion in a hard one.

One asymmetry recorded for completeness, which does **not** change the decision: the IR term is the
noisier of the two (paired active-return t=-0.06 gives the IR gap a standard error of ~0.25 IR,
i.e. d_score = 0.0000 +/- 0.0096). So the IR contribution is centred on ~0 but is wide — the blend
is the hedge against a bad IR draw, and hedging costs ~0.012 score in expectation. A risk-neutral
score-maximiser should not pay it.

**Final: ship `lgbm_sub1__sb3`.** Confirmed by two independent analyses.

## F37. Shipped recipe finalised: `lgbm_sub1__R700sb3` (700 rounds, not 400)

model-lab closed the exact gap I identified — our recipe is "3 seeds + fixed rounds + no early
stopping", and that cell had never been measured. Clean apples-to-apples on our own recipe:

| config | semiannual mean | semiannual RankICIR | annual mean | annual RankICIR |
|---|---|---|---|---|
| `sb3_fixed400` (3 seeds, 400 r, no ES) | +0.0823 | +0.568 | +0.0750 | +0.520 |
| **`R700sb3` (3 seeds, 700 r, no ES)** | **+0.0827** | **+0.583** | **+0.0751** | **+0.531** |
| `sb3` (3 seeds, 400 r, early stopping) | +0.0823 | +0.570 | +0.0750 | +0.522 |
| `fixed400` (1 seed, 400 r, no ES) | +0.0809 | +0.566 | +0.0737 | +0.517 |

700 rounds dominates 400 on **every axis of both fold sets**. The mean-RankIC differences are
noise-sized (+0.0004 / +0.0001) but the RankICIR differences are consistent (+0.015 / +0.011), and
RankICIR correlates +0.836 with the realized IR that carries 30% of the score. Mean RankIC peaks
near 700 — `R1000` is lower at +0.0807 — so 700 sits at the top of the curve rather than past it.

Also confirmed: **early stopping is actively harmful here.** At one seed it costs +0.0071 mean
RankIC and +0.16 realized IR versus a fixed round count (`single` +0.0738/IR +0.0306 vs `fixed400`
+0.0809/IR +0.1891). Our `train.py` fits to a fixed round count with no validation set, so it was
already on the right side of this; the round count is now aligned too.

**Shipped artifact** (`submission/`, defaults in `train.py` reproduce it exactly):
3x LightGBM, lr 0.03 / num_leaves 31 / max_depth 7 / min_data_in_leaf 200 / feature_fraction 0.70
/ bagging_fraction 0.80 / lambda_l2 10.0 / **700 rounds**, seeds [42, 43, 44], rank-blended at 1/3
each, 136 frozen features, within-day percentile-rank target, `rank` postprocess, **no CatBoost**.

**Archive of record**: `out/submission.zip`, sha256
`bc5c2471962dd1149c6f15fc71de1eb6758b33e0594b477516256f275c218f6a`, 2.43 MiB, 8 files + 2 dirs,
container modes matching the organizers' accepted package.

Verified on this exact build: compliance gate PASS; **build deterministic** (all four generated
files hash identically on rebuild); **`train.py` reproduces all three boosters byte-for-byte**;
clean-venv judge smoke test PASS 30/30 with **0 stderr lines**; `simulate_judge` PASS on 242/242
signal dates. In-sample simulator RankIC/RankICIR rose to 0.2497 / 1.957 (from 0.1995 / 1.511 at
400 rounds) — in-sample, so not evidence, but consistent with the round count now being aligned.

## F38. Final compliance audit of the shipped package

A disqualification would be catastrophic and the organizers audit top submissions, so the shipped
files were parsed rather than eyeballed:

| check | `inference.py` | `train.py` |
|---|---|---|
| `open()` call sites | none | none |
| modules imported | json, pathlib, numpy, pandas, lightgbm, catboost, warnings, pandas._libs.window.aggregations | + argparse, sys |
| risky attribute access (network/process/env/unlink) | none | none (`write_text` only, for `metadata.json`) |
| absolute path literals | none | none |
| hard-coded YYYYMMDD | none | walk-forward fold boundaries only, in the `--walk-forward` reporting branch |

`inference.py` reads exactly one thing: `Path(model_dir) / "metadata.json"`. No network, no
subprocess, no environment access, no file writes, no cross-call state, no refitting. The
`train.py` date literals are the documented walk-forward fold cut-offs used only by the optional
`--walk-forward` report; the frozen training path is driven by `--train-end` (default 2022-12-31,
the end of the public data). Both files pass the builder's banned-token and self-containment gates.

## F39. model-lab found an override bug that voids the coordinate sweep — but not our decision

**The bug**: a shallow `dict.update` in their `--override` path replaced the whole `params` mapping
instead of merging into it. So `--override params.min_data_in_leaf=800` produced
`params = {"min_data_in_leaf": 800}` and every other hyper-parameter silently fell back to
LightGBM's defaults (learning_rate 0.1 not 0.03, max_depth -1 not 7, feature_fraction 1.0 not 0.70,
bagging 1.0 not 0.80, lambda_l2 0 not 10). That is why nearly every sweep row came out
0.005-0.018 worse: they were not "sub1 with one axis moved", they were "LightGBM defaults with one
axis set". **The coordinate sweep (18 rows) and the conclusion "the sub1 corner is a local optimum"
are void, and the search is not closed.**

**Verified independently** — I checked the saved `config_kwargs.params` of every config the
400-vs-700 decision rests on:

| config | params keys | verdict |
|---|---|---|
| `fixed400`, `R700`, `R1000` | 8 | valid |
| `sb3_fixed400`, `R700sb3`, `sb3` | 8 | valid |
| `MD800`, `L63` (coordinate sweep) | 1 | **bug-affected, void** |

So the round-count comparison is unaffected: `R700` and `R1000` used only **top-level** overrides
(`early_stopping`, `num_boost_round`), never `params.*`, and all carry the full 8-key dict. That
means **the decision to ship 400 rounds stands on valid evidence**:

* 1 seed: `R700` +0.0813 vs `fixed400` +0.0809 = **+0.0004** (t=0.99 semiannual, t=0.29 annual)
* 3 seeds: `R700sb3` +0.0827 vs `sb3_fixed400` +0.0823 = **+0.0004**
* `R1000` is already **negative** (-0.0002), so the curve turns over just past 700
* and 700 fits the training data **25% harder in sample** (simulator RankIC 0.1995 -> 0.2497) for
  no measurable out-of-sample gain — the overfitting signature

Also unaffected: the entire shipped sub1 architecture, which was never overridden at the `params.*`
level. model-lab is re-running the sweep with a deep merge, and the architecture-ensemble members
(`R700sb3` / `MD800` / `R1000`, both fold sets).

**Archive of record re-frozen after the revert to 400 rounds**: `out/submission.zip`, sha256
`cd6385e0eb9b3a96961c3519174eca5f2706262c0320faada6344e79f0a12bcb`, 1.41 MiB, 8 files + 2 dirs.
Re-verified on this exact build: compliance gate PASS, build deterministic, `train.py` reproduces
all three boosters byte-for-byte, clean-venv judge smoke PASS 30/30 with 0 stderr, `simulate_judge`
PASS on 242/242 signal dates.

## F40. The architecture ensemble does not clear the bar — shipped package confirmed

model-lab ran the architecture-level ensembles I asked for. Two of the three are unusable or worse:

| blend | members | semiannual | annual | verdict |
|---|---|---|---|---|
| `jitter_fixed400+R700+MD800` | fixed400, R700, **MD800** | +0.0827 (ICIR +0.593) | not run | **untrustworthy — `MD800` is still bug-affected (n_params=1)** |
| `jitter_fixed400+R700` | fixed400, R700 (both valid) | +0.0816 (ICIR +0.577) | +0.0741 (ICIR +0.526) | **worse than shipped on mean, both fold sets** |
| **shipped `sb3_fixed400`** | 3 seeds of sub1 | **+0.0823** (ICIR +0.568) | **+0.0750** (ICIR +0.520) | — |

The headline number that looked like an ensemble win (+0.0827 with the best ICIR in the sweep,
+0.593) came from a blend whose `MD800` member carries `params = {"min_data_in_leaf": 800}` only —
i.e. LightGBM defaults for everything else. The clean two-member blend, which uses only validated
members, loses to the shipped config on mean RankIC in **both** fold sets. So the ensemble does not
meet the pre-registered bar ("beat R700sb3 on mean in both fold sets, or match mean while clearly
beating RankICIR") and **we ship what we have**.

Also from the corrected sweep: the depth axis is now cleanly tested and does not help — `D8`
+0.0809 and `D9` +0.0806 against the base `fixed400` +0.0809. The corner is reasonable on that
axis even though the full coordinate sweep still needs re-running.

**model-lab's own final recommendation (`reports/RECOMMENDED.json`) is
`lgbm_sub1__sb3_fixed400`** — the exact configuration already frozen in `submission/`, with
`num_boost_round=400`, `early_stopping=null`, `seeds=[42,43,44]`, the 8-key params dict, 136
features, and the explicit instruction "emit the rank-average of the three boosters as-is. Do NOT
winsorise, round, bucket or neutralise." That matches the shipped artifact on every field.

Their `noise_floor` statement is worth recording because it is the rule we have been applying:
"unpaired SE on a fold mean ~0.004; paired SE between correlated models ~0.0005-0.0012. Only act
on >= ~0.005 that also holds on the second fold set." Every candidate considered since has failed
that bar, which is why the package has not moved.

## F41. Corrected sweep: the corner IS a local optimum — and the bug had inverted the conclusion

With the deep-merge fix applied, the coordinate sweep looks completely different. Same base
(`lgbm_sub1__fixed400`, 1 seed, valid, +0.0809), semiannual:

| config | mean | delta | params keys | status |
|---|---|---|---|---|
| `R700sb3` (3 seeds, 700 r) | +0.0827 | +0.0018 | 8 | valid |
| **`sb3_fixed400` (SHIPPED)** | **+0.0823** | **+0.0014** | 8 | valid |
| `R700` (1 seed, 700 r) | +0.0813 | +0.0004 | 8 | valid |
| `D8`, `fixed400` (base) | +0.0809 | 0.0000 | 8 | valid |
| `R1000` | +0.0807 | -0.0002 | 8 | valid |
| **`L127`** | **+0.0806** | **-0.0003** | **8** | **valid (was -0.0176 pre-fix)** |
| `D9` | +0.0806 | -0.0003 | 8 | valid |
| **`L63`** | **+0.0805** | **-0.0004** | **8** | **valid (was -0.0146 pre-fix)** |
| `single` (1 seed, early-stop) | +0.0738 | -0.0071 | 8 | valid |
| `MD800`, `MD400`, `L2_30`, `FF50`, `FF85`, `MD100`, `LR02R700`, `Dfull`, `BF60`, `L2_5` | -0.003 to -0.011 | | **1** | still bug-affected, pending re-run |

**The bug had inverted the conclusion.** Pre-fix, `leaves=63` looked catastrophic at -0.0146 and
`leaves=127` at -0.0176, which is what produced the (now retracted) claim that the corner is a local
optimum with everything worse around it. Post-fix they are -0.0004 and -0.0003 — **statistically
indistinguishable from the base**, not worse. The corrected picture is therefore not "everything
around here is worse" but "nothing around here is better", which is a weaker and more honest claim.

What survives: **no single-axis move beats the base beyond the noise floor** (best is `R700` at
+0.0004, t=0.99), and the two real effects are the ones we already shipped — turning early stopping
off (+0.0071, t=6.2) and seed bagging (+0.0014, t=3.3). `sb3_fixed400` at +0.0823 is the best
*validated* configuration; `R700sb3` is +0.0004 ahead, which is one tenth of the noise floor and
buys +25% in-sample fit for no out-of-sample gain.

Ten configs are still flagged `n_params=1` and awaiting re-run. If any of them turns out to beat the
base by >=0.005 on both fold sets once corrected, that would be the first genuine reason to move the
package. Nothing so far has.

## F42. Corrected sweep complete — nothing clears the bar

model-lab finished the deep-merge fix and re-ran every axis; **0 configs remain bug-affected**
(26 valid). Corrected semiannual ranking, top 10:

| config | mean RankIC | RankICIR |
|---|---|---|
| `R700sb3` (3 seeds, 700 r) | +0.0827 | +0.583 |
| **`sb3_fixed400` — SHIPPED** | **+0.0823** | **+0.568** |
| `L63D8` | +0.0820 | — |
| `LR02R700` | +0.0817 | — |
| `MD800sb3` | +0.0816 | — |
| … | | |

**Nothing beats the shipped configuration by >=0.005**, which was the pre-registered bar (the
unpaired standard error on a fold mean is ~0.004). The best challenger, `R700sb3`, is +0.0004
ahead — one tenth of the noise floor — and we already rejected it on the grounds that it buys +25%
in-sample fit for no measurable out-of-sample gain while costing 1.75x inference.

So the search is genuinely closed now, and it closed on the package we already have. The three
effects that matter are all in the shipped artifact: **no early stopping** (+0.0071, t=6.2),
**seed bagging** (+0.0014, t=3.3), and the **136 frozen features** (+0.0231 paired ablation,
12/12 positive). Everything else we tested — round count, leaves, depth, min_data, lambda_l2,
feature_fraction, bagging fraction, learning rate, CatBoost, architecture ensembles, residual
features — is either negative or inside the noise.

## F43. OPERATIONAL RISK: the private board scores our LAST submission, and our last one is the bad one

Submission_Rules and the competition rules are explicit: "比赛结束后我们会对所有选手的**最后一次
提交**统一在私榜数据集上进行重新评测". The private board scores the **last** submission, not the
best one.

Our submission history right now:

| # | id | status | board RankIC |
|---|---|---|---|
| 1 | `95cc01f9` | finished | **0.0712** |
| 2 | `36fed7c3` | finished | 0.0602 |
| 3 | `ffeeb8b3` / `8cbff65c` | failed (compliance tokens, pre-fix) | — |
| control | `6077b16d` | finished (organizers' baseline) | 0.0623 |

**Our last *successful* submission is #2 — the fold-tuned `seedbag3` that scored 0.0602, our worst
result.** If the competition ended right now, the private board would score that package, not the
0.0712 one and not the much better one sitting frozen on disk.

This makes the next slot the highest-priority action of the whole session, above any further
modelling work: **we must land the frozen package before the deadline.** Quota resets
2026-10-10 00:00 +08:00 and we have 15 slots across 2026-10-10..14, so there is ample margin, but
the current state is genuinely exposed and nothing else matters until it is fixed.

Corollary for the endgame: after the good package lands, the *last* submission is the one that
counts, so any later experiment that replaces it must be better-evidenced than what it displaces.
Do not use a late slot on a speculative variant, because that variant becomes the scored artifact.

## F44. `train.py` hardened against the import failure mode that cost us two submissions

The judge only ever loads `inference.py`, and that file is self-contained by design. But the
organizers also re-run `train.py` during code review, and a naive reviewer harness can load a file
by path with nothing on `sys.path` — which is exactly the failure mode that made our first two
submissions die with `ModuleNotFoundError`. Verified before the fix:

| invocation | `train.py` before | after |
|---|---|---|
| `python -m submission.train` (package) | OK | OK |
| `cd submission && python train.py` (script) | OK | OK |
| `spec_from_file_location` with no package context | **FAILED** | **OK** |

Fix: a plain `sys.path.insert` of the file's own directory at the top of `train.py` (not
reflection, and scoped to the training entry point — the scored `inference.py` neither does this
nor needs it). All three modes now import cleanly.

**Archive of record re-frozen**: sha256
`0b9ea409802f8fed9a2609aac5021915bc0733845ab0721f64ba7df1c8603291`, 1.41 MiB. Re-verified on this
exact build: compliance gate PASS, build deterministic (all four generated files hash identically),
`train.py` reproduces all three boosters byte-for-byte, clean-venv judge smoke PASS 30/30 with 0
stderr, `simulate_judge` PASS.

## F45. SWITCHED THE SHIPPED MODEL to 63 leaves / depth 8 — the paired test, applied consistently

The final bagged variants from model-lab's corrected sweep produced one config that beats the
shipped one on **every** measurement:

| | `sb3_fixed400` (31 leaves / depth 7) | **`L63D8sb3` (63 leaves / depth 8)** |
|---|---|---|
| semiannual mean RankIC | +0.0823 | **+0.0838** |
| semiannual RankICIR | +0.568 | **+0.591** |
| annual mean RankIC | +0.0750 | **+0.0770** |
| annual RankICIR | +0.520 | **+0.544** |
| paired daily RankIC difference | — | **t=+2.82, p=0.0049 (semiannual)** |
| paired daily RankIC difference | — | **t=+3.47, p=0.0005 (annual)** |
| per-year record | — | **wins 9 of 11 test years** |

**Why I overrode my own pre-registered +0.005 bar, and why that is not rationalisation.** The bar
was calibrated on the *unpaired* standard error (~0.004), which treats the two models as
independent samples. They are not: they share seeds, features and training data, and are scored on
the same days and stocks. The correct comparison is **paired**, where the standard error is
~0.00055 — about 7x smaller. By that standard the improvement is 2.8σ and 3.5σ, significant on both
fold sets.

The consistency check that matters: **this is the same test framework that settled the CatBoost
question.** There it returned t=+1.86 (p=0.063) and t=+1.60 (p=0.110) — no significant difference —
and I did **not** act on it. Here it returns p=0.005 and p=0.0005, and I do. Applying one standard
to both decisions is the point; the bar was the mis-specified part, not the principle behind it.

It is also the only change that survived: `L63D8sb3` beats every architecture ensemble too
(`ens_R700sb3+MD800sb3+L63D8sb3` +0.0834/+0.0763, `+R1000sb3` +0.0831/+0.0757, `+L2_30sb3`
+0.0828/+0.0758 — all below `L63D8sb3` alone). The change from the previous package is exactly two
hyper-parameters; seeds, rounds, early stopping, features, target and postprocess are unchanged.

**Archive of record re-frozen**: sha256
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`, 2.79 MiB (larger because
deeper trees; still 0.3% of the 1024 MB upload limit). Re-verified on this exact build: compliance
gate PASS, build deterministic, `train.py` reproduces all three boosters byte-for-byte, clean-venv
judge smoke PASS 30/30 with 0 stderr, `simulate_judge` PASS.

## F46. Time-decay sample weighting is a NEGATIVE result — the last open direction is closed

model-lab implemented exponential time-decay sample weighting (`weight = 0.5 ** (age_trading_days /
half_life)`) and swept it on the shipped recipe, both fold sets:

| half-life (trading days) | semiannual | annual |
|---|---|---|
| 125 | +0.0772 (ICIR +0.522) | +0.0700 (ICIR +0.480) |
| 250 | +0.0805 (ICIR +0.544) | +0.0735 (ICIR +0.500) |
| 500 | +0.0820 (ICIR +0.556) | +0.0756 (ICIR +0.516) |
| 1000 | +0.0819 (ICIR +0.559) | +0.0753 (ICIR +0.516) |
| no decay (`sb3_fixed400`, 31/7) | +0.0823 (ICIR +0.568) | +0.0750 (ICIR +0.520) |
| **no decay, SHIPPED (`L63D8sb3`, 63/8)** | **+0.0838 (ICIR +0.591)** | **+0.0770 (ICIR +0.544)** |

**Recency weighting does not help.** The dose-response is monotone in the *wrong* direction: the
more aggressively we weight recent data, the worse it gets (`td125` is -0.0066 semiannual against
the no-decay baseline), and as the half-life lengthens the result converges back to — but never
past — the unweighted model. No half-life reaches the shipped configuration on either fold set.

**My pre-stated prediction failed as a recency test.** I said in advance that a genuine recency
effect should show up in 2021-2022 specifically, and that an even spread across years would be
evidence of noise rather than recency. The per-year split does technically put a (tiny) positive in
the recent years (+0.0010 mean) and a (tiny) negative in the early years (-0.0009) — but the
**largest single effect is 2018, an early year, at +0.0037**, larger than either 2021 (-0.0003) or
2022 (+0.0023). A real recency effect cannot be largest in the oldest-but-one year. So the pattern
is noise, and the prediction is not supported.

This closes the last economically-motivated direction. The interpretation is genuinely informative:
this model does **not** benefit from favouring recent history — the full 2016-2022 sample is more
valuable than a recency-weighted subset, which is consistent with the feature library being built
from stationary-ish cross-sectional constructs (reversal, volatility, beta) rather than
regime-dependent levels.

Also worth recording: `td180`/`td365`/`td730` (history *truncation*) were already known to hurt
badly (+0.0677/+0.0707/+0.0720 semiannual on the other param set). Both ways of discounting old
data — dropping it and down-weighting it — are now measured and both are negative.

**Nothing changes in the shipped package.** It remains `L63D8sb3`, archive sha256
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`.

## F47. Full reviewer simulation — the code-review half of the ranking is now evidence-backed

The final ranking is "私榜排名和代码审查结果共同决定", so the review half deserves the same
evidence standard as the model half. I ran the organizers' likely workflow end-to-end against the
**frozen archive**, not the working tree:

| step | what a reviewer does | result |
|---|---|---|
| 1 | unzip `out/submission.zip` | 8 entries unpack, modes preserved (`0o40777` dirs / `0o100666` files) |
| 2 | create a fresh venv from the package's **own** `requirements.txt` | install OK; lightgbm 4.7.0, pandas 2.3.3 |
| 3 | `python train.py --data-dir <public data> --submission-dir <tmp>` from the unpacked copy, in that fresh venv | **all four artifacts byte-for-byte identical** — `lightgbm_seed42/43/44.txt` **and** `metadata.json`, in 66 s |
| 4 | load `inference.py` under the judge contract (no package context, no cwd on `sys.path`) | **PASS 30/30 signal dates, 0 stderr lines**, 242 ms/day |
| 5 | compliance scan of the unpacked sources | **0 banned-token hits** in both `inference.py` and `train.py` |

This is the strongest reproducibility evidence in the project: a different virtualenv, a different
directory, an unpacked copy, and the script invoked as a plain file rather than as a module — and
the models still come out bit-identical. It also re-confirms the two properties that were *not*
guaranteed earlier in the session: `train.py` now works when loaded with nothing on `sys.path`
(F44), and `inference.py` works with no package context (the fix for the two failed submissions).

Nothing in the package changed as a result. Archive remains sha256
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`.

## F48. The submission is now armed on a timer, not on chance

Everything is verified and frozen, so the only remaining failure mode is *operational*: the
critical submission simply not happening. Rounds do not reliably land near 00:00, so waiting for
one is not a control.

Armed: a scheduled reminder `schedule-27908252-0c4b-4711-9a4b-8b625d836c9a` at
**2026-10-10 00:00:30 +08:00** (30 s after the quota reset), carrying the full procedure —
hash check first, then `submit.py --zip out/submission.zip --yes --poll-timeout 5400`, then confirm
the new submission is our LAST one. It also carries the recovery step (`dtapi.py release` first if
a submit dies halfway) and the exact expected hash, so it cannot submit a stale archive by accident.

The 30-second offset is deliberate: the reset is at exactly 00:00:00 and a request that races it
could still read `daily_limit`.

## F49. Corrections: my truncation claim was wrong, and `L63D8sb3` is Pareto-optimal in model-lab's own data

**Correction to F46 (my error).** I wrote that `td180`/`td365`/`td730` were *history truncation* and
that time-decay was therefore an untested intervention. That is wrong. model-lab confirms those
runs were **already proper decay weighting** via `sample_weight`, just using a *calendar-day*
half-life; on this data there are ~1.45 calendar days per trading day, so `td365` is roughly a
250-trading-day half-life. They were on the older `lgbm_rank_raw` config, which is why the axis was
re-run on `lgbm_sub1`. So the conclusion in F46 stands (decay is null) but the reason I gave for
running it was incorrect — both unit conventions are now supported and both are null.

**`L63D8sb3` is Pareto-optimal, and it is model-lab's own leaderboard that says so.** I checked
their `reports/leaderboard_local.md` rather than only their prose:

| config | semiannual | annual |
|---|---|---|
| **`lgbm_sub1__L63D8sb3` (SHIPPED)** | **+0.0838** | **+0.0770** |
| `ens_R700sb3+MD800sb3+L63D8sb3` | +0.0834 | +0.0763 |
| `ens_R700sb3+MD800sb3+R1000sb3` | +0.0831 | +0.0757 |
| `ens_R700sb3+MD800sb3+L2_30sb3` | +0.0828 | +0.0758 |
| `lgbm_sub1__R700sb3` | +0.0827 | +0.0751 |
| `lgbm_sub1__sb3_fixed400` (their stated recommendation) | +0.0823 | +0.0750 |

**Configs beating `L63D8sb3` on both fold sets: none.** It is #1 on both, and it also beats every
architecture ensemble. Their `RECOMMENDED.json` still names `sb3_fixed400` — it was written before
the `L63D8sb3` run and is **stale relative to their own leaderboard**, so the switch I made in F45
is consistent with their data rather than in tension with it.

**Architecture ensembles confirmed null, with a mechanism.** All three beat the baseline on both
fold sets but by only +0.0005 to +0.0013, against the pre-registered +0.005, and RankICIR is flat.
The reason is structural: the 18 sweep neighbours span just +0.0797 to +0.0820, so the members are
near-identical models — rank-averaging them is more seed bagging, not diversity. The diversity
hypothesis needed the members to differ, and they do not.

Also noted: model-lab added `params_keys` / `effective_params` / `effective_sample_weighting` /
`effective_early_stopping` / `effective_num_boost_round` to every report, which is exactly the
defence against the silent-override class of bug that produced two false conclusions today.

## F50. The package is interpreter-portable — verified on Python 3.11, 3.12 and 3.13

The judge's Python version is not published, so assuming it matches ours was an untested risk.
Each version was tested with a **fresh venv built from the package's own `requirements.txt`**, then
put through the judge contract with no package context:

| interpreter | fresh install | judge contract | stderr | ms/day |
|---|---|---|---|---|
| 3.11.15 (our venv) | OK | PASS 30/30 | 0 | 226-339 |
| 3.12.7 | OK | PASS 30/30 | 0 | 216 |
| 3.13.14 | OK | PASS 30/30 | 0 | 258 |

And `train.py` under **Python 3.12** reproduces all three boosters **byte-identically** — a
different interpreter *and* different compiled library builds, same bits out. That is a stronger
statement than reproducibility in our own environment.

The reason it ports cleanly: every `X | Y` union in the shipped files is an *annotation*, and both
`inference.py` and `train.py` carry `from __future__ import annotations`, so the syntax is inert at
runtime on 3.9+. Verified by test rather than by reading, since a single runtime union would have
been a hard failure on the judge.

**Archive unchanged**: `06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`.

## F51. Where we actually stand: RankIC is our strength, IR is our weakness

Read the live public board directly (134 rows) rather than reasoning about our own row:

| rank | name | rankic | ir | s_rankic | s_ir | score |
|---|---|---|---|---|---|---|
| 1 | sh | 0.0832 | 5.5684 | 1.000 | 0.887 | 0.9662 |
| 2 | Eason | 0.0778 | 6.4329 | 0.934 | 1.000 | 0.9540 |
| 3 | Summer | 0.0826 | 4.7800 | 0.993 | 0.785 | 0.9303 |
| 10 | Water-Bucket | 0.0831 | 2.1816 | 0.999 | 0.446 | 0.8332 |
| **29** | **us (submission #1)** | **0.0712** | **2.8105** | **0.853** | **0.528** | **0.7551** |

**Our RankIC sits at the 85th percentile of the field; our IR sits at the 53rd.** The top ten
almost all have IR between 4.2 and 6.4 while ours is 2.81, and `Water-Bucket` at #10 shows the
trade-off from the other side — near-maximal RankIC (0.0831) with a weak IR (2.18) still lands #10,
because RankIC carries 70% of the weight.

Marginal value on the board, per unit: **+0.01 RankIC is worth +0.086 score; +1.0 IR is worth
+0.039** — so RankIC is worth ~2.2x IR per unit, and we are strong exactly where it counts most.

**Projection for the shipped package.** Chaining two measured ratios — harness->engine 0.853
(from `sb3_fixed400` 0.0823 -> 0.0702) and engine->board 1.134 (from submission #1, the single
board observation we have):

    L63D8sb3  harness +0.0838 -> engine ~0.0715 -> board ~0.0811

against a field max of 0.0832. Holding our previous IR of 2.81, that is a score of ~**0.840**,
which would sit around **rank #9**. If IR also improves to 4.0 the score goes to ~0.887 (~#5).

**Caveats, stated plainly:** this chains two extrapolations, one of which rests on a *single* board
observation carrying ~0.009 standard error, and the private period is a different regime entirely.
Treat ~#9 as the central estimate and the honest range as wide. What it does establish is the
*shape* of our position: we are near the top of the field on the metric that carries 70% of the
score, and the IR column is where the remaining headroom is.

**Is there a lever left for IR?** The feature library was audited for gaps (136 columns: returns,
momentum, five volatility families, liquidity/Amihud/turnover, overnight/intraday microstructure,
RSI/CCI/KDJ/MACD/BIAS, eight fundamental ratios, size, cross-sectional interactions, and all 11
Barra factors) and nothing obvious is missing. backtest-recon established that only the *ordering*
of the score reaches the optimizer, so transforms cannot move IR. The modelling levers are
exhausted; the IR gap is signal quality, not a fixable bug.

## F52. Primary-source confirmation that only ORDERING matters — and that the private board uses a DIFFERENT optimizer

I read the organizers' own `BACKTEST_PLATFORM_SPEC_v2.md` (cloned official repo) rather than relying
on our reconstruction. Section "Score 到组合权重":

> 不同队伍的 `score` 数值尺度可能完全不同，因此不能直接将原始 score 当作预期收益。
> 统一先做截面变换：
> **Raw Score → Cross-sectional Rank → Percentile / Normal Score → Unified Alpha Signal**
> 这样私榜组合主要取决于股票相对排序，而不是 score 人为放大或缩小。

**This settles the transform question with primary-source evidence.** The platform converts our raw
score to a cross-sectional rank/percentile *before* it reaches the optimizer, precisely so that
"人为放大或缩小" (artificial scaling) cannot matter. Two consequences:

1. **backtest-recon's transform-invariance result was correct**, and our `postprocess: rank` is
   exactly what the platform does anyway. There is no score-shaping lever — not for RankIC and not
   for IR. Our earlier conclusion that the IR gap is signal quality rather than a fixable bug now
   rests on the organizers' spec, not on our inference.
2. **The private board uses a different optimizer from the public one.** The same section says
   "第一版可先使用 Top-K 或简单 benchmark tilt，**正式私榜再替换为 Barra 风险约束优化器**" — the
   first version may use Top-K or a simple benchmark tilt, while the official private board
   replaces it with a **Barra risk-constrained optimizer**. The public board's `ir` column is
   therefore partly an artifact of a simpler optimizer, and our 53rd-percentile IR need not carry
   over.

**Strategic consequence: we are optimising the right thing.** The private Barra-constrained
optimizer consumes a *unified alpha signal* derived purely from our ordering, so ordering quality
(RankIC) is the only thing that transfers, and it carries 70% of the score directly. Our projected
0.0811 RankIC is near the field max of 0.0832. The public `ir` column should be read as a weak,
optimizer-specific signal, not as the thing to optimise — which is exactly the position the whole
session's evidence has been pointing at, now confirmed at the source.

## F53. Correction to F52: our engine runs DID use the private board's optimizer method

In F52 I noted the spec's line "正式私榜再替换为 Barra 风险约束优化器" and implied our IR
measurements might be an artifact of a simpler public optimizer. **That implication was wrong, and
I checked before letting it stand.** The engine config backtest-recon used is:

    C7_platform_guess:  method = "barra"
                        single_name_weight_limit    = 0.05
                        industry_exposure_limit     = 0.02
                        barra_style_exposure_limit  = 0.10

and **all 37 cached engine runs used it** (verified by counting the `optimizer` field across
`research/out/preds_metrics/*.json`). The engine's `OptimizerConfig.method` *defaults* to `"top_k"`,
which is what made the question worth asking — but the platform-guess config explicitly selects
`barra`, matching the method the spec says the private board uses.

So the corrected picture:

* **Our IR measurements are under the correct optimizer family**, and backtest-recon picked it
  deliberately. The 35-config RankIC↔IR correlation (+0.879), the CatBoost verdict, and the
  Pareto ranking all stand as measured.
* What is *guessed* is the **constraint parameters** (single-name 0.05, industry 0.02, Barra style
  0.10) and the cost model — not the method. Those are the residual unknowns, and they affect the
  absolute IR level rather than which config orders better.
* The public board's `ir` column may well come from a simpler optimizer, so our 53rd-percentile
  board IR is not necessarily what a Barra-constrained run would produce — but our own
  Barra-constrained engine already rates the shipped family at the top of our config space
  (+0.3061 for `sb3_fixed400`, against a field of -0.36 to +0.31 across our configs).

Net effect on the decision: none. The package stays. But the log would otherwise have carried a
wrong reason for a right conclusion, which is exactly the failure mode this project has already hit
twice today.

## F54. The switch to `L63D8sb3` trades IR for RankIC — and still wins on the combined score

I ran both configs through the official engine under **five different constraint settings**
(`research/constraint_robustness.py`), because the private board's constraint *parameters* are the
one thing we guessed. This is the first time the shipped config has been tested against the
runner-up under anything other than `C7_platform_guess`.

| optimizer setting | RankIC delta (L63D8sb3 - sb3_fixed400) | IR delta |
|---|---|---|
| `C1_tight` | +0.00078 | **-0.2554** |
| `C2_typical` | +0.00078 | -0.0708 |
| `C3_loose` | +0.00078 | +0.0235 |
| `C5_topk50` | +0.00078 | -0.1244 |
| `C7_platform_guess` | +0.00078 | -0.0683 |

Two things fall out. **RankIC is identical across all five settings** (as it must be — RankIC is a
property of the score ordering, not of the portfolio built from it), and `L63D8sb3` is ahead by
+0.00078 in every one: a clean sanity check that the engine is measuring what we think.
**IR is not**, and `sb3_fixed400` is ahead in 4 of 5 settings, by an average of -0.099.

**Evaluating IR in isolation is the wrong lens**, because the score weights them 70/30. Using the
board's own normalisation spans (RankIC 0.0814, IR 7.6759):

    score value per unit:  RankIC 8.60,  IR 0.0391      -> +0.01 RankIC is worth 2.20 IR

so the measured deltas give

    RankIC term  +0.7 * (0.00078 / 0.0814) = +0.00671
    IR term      +0.3 * (-0.0991 / 7.6759) = -0.00387
    NET                                    = +0.00283

**`L63D8sb3` still wins, and it wins in 4 of the 5 constraint settings individually** — only
`C1_tight` flips, because its IR penalty (-0.2554) exceeds the -0.172 that would be needed to
overturn the RankIC gain. So the decision does **not** depend on our constraint guess.

Honest caveats: the IR deltas are noisy across settings (range 0.28) and we have already seen the
same config measure 0.3061 and 0.3402 on separate runs, so a single IR delta of ~0.07 is only ~2x
the run-to-run variation. The net margin is +0.0028, not the +0.015 a RankIC-only view suggested.
This is a genuinely close call and I am recording it as one rather than as a clear win.

## F55. The 6-model hedge blend does NOT beat the shipped config — and the reason is the label

F54 exposed a real trade-off: `L63D8sb3` (63/8) leads on RankIC but gives back IR versus
`sb3_fixed400` (31/7), netting only +0.0028. The obvious fix is to blend them — each is already a
3-seed rank average, so the blend is a 6-model, 2-architecture ensemble. If it kept `L63D8sb3`'s
RankIC and `sb3_fixed400`'s IR it would dominate both, and blending is the standard hedge when you
cannot tell which of two close configs generalises better.

I built it (`research/blend_finalists.py`, `research/blend_engine.py`) and measured both halves.

**RankIC, on the OFFICIAL label** (`future_return` = open(t+2)/open(t+1)-1, the one the board
scores):

| fold set | blend | `L63D8sb3` | delta | paired test |
|---|---|---|---|---|
| semiannual | +0.083571 | +0.083800 | -0.000229 | t=-0.81, p=0.42 |
| annual | +0.076473 | +0.077019 | -0.000546 | t=-1.81, p=0.071 |
| vs `sb3_fixed400` | | | **+0.001304 / +0.001498** | t=+4.83 / +5.06, p<0.0001 |

**IR, across the five constraint settings** (averages): blend 0.3527, `L63D8sb3` 0.3246,
`sb3_fixed400` 0.4238. The blend sits *between* the two — it averages, it does not improve.

**Putting both halves on the score:**

    RankIC term  0.7 * (-0.000387 / 0.0814) = -0.00333
    IR term      0.3 * (+0.0281  / 7.6759) = +0.00110
    NET                                     = -0.00223   -> L63D8sb3 wins

**The label is what decides it.** The engine's *own* RankIC has the blend marginally ahead
(+0.00008), but the engine scores with a 5-day open-to-open label, whereas the leaderboard scores
the official `open(t+2)/open(t+1)-1`. On the label that actually counts, `L63D8sb3` leads in both
fold sets, so the 70%-weight component goes to the shipped config and the blend's IR edge (worth
0.3 * 0.028/7.68) cannot close a 3x larger RankIC gap.

So the hedge is rejected, and for a concrete reason rather than a bar: blending two highly
correlated members (same features, seeds, target and rounds; only leaves/depth differ) buys a
mid-point, and the mid-point is not on the score-optimal frontier. This also confirms model-lab's
architecture-ensemble null from a different direction — there, the members were near-identical;
here they differ exactly where the trade-off lives, and it *still* does not help.

**Package unchanged**: `L63D8sb3`, sha256
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`.

## F56. The engine is deterministic, and that makes the IR differences real-but-fragile

Re-ran `L63D8sb3` under `C7_platform_guess` from scratch and compared to the cached result:

    cached: rankic=0.07050205154091164  ir=0.271843879526695
    rerun : rankic=0.07050205154091164  ir=0.271843879526695

**Bit-identical.** The official engine is fully deterministic given the same score series, so the IR
differences reported in F54/F55 are *real measurements*, not run-to-run noise. That matters: it
means we cannot dismiss the -0.068 IR gap between `L63D8sb3` and `sb3_fixed400` as jitter.

**But it also means something more uncomfortable.** model-lab stated that `sb3` and `sb3_fixed400`
are "the same cell" — the same configuration measured twice. **They are not.** Their cached
out-of-sample predictions differ:

| fold set | rows | max abs score diff | Spearman |
|---|---|---|---|
| semiannual | 396,237 | 1.73e-01 | 0.99859 |
| annual | 360,763 | 1.48e-01 | 0.99850 |

So they are two distinct score series that happen to share a mean RankIC (+0.0823), and backtest-recon
measured IR +0.3061 on one while I measured +0.3402 on the other. **A 0.034 IR swing from score
series correlated at 0.9986.**

Putting the two findings together gives the correct reading of F54/F55: the engine is deterministic,
so the IR gaps are genuine properties of those exact score vectors — but IR is *highly sensitive* to
small perturbations in the score ordering, whereas RankIC is not. Among configs this similar, IR is
a fragile statistic and RankIC is the stable one. That is precisely why the decision rule weights
RankIC at 70% and why the F54 conclusion (net +0.0028 for `L63D8sb3`) should be read as "RankIC
decides, IR is a small noisy counterweight" rather than as a precise trade-off.

It also means any IR-based claim about two near-identical configs carries more uncertainty than its
determinism suggests, and we should not chase IR directly. No change to the shipped package.

## F57. A clean measurement of how much IR can move while RankIC stands still

Following F56, I located where the `sb3` and `sb3_fixed400` score series differ. The answer is
**everywhere, uniformly** — there is no fold-boundary or date-localised artifact:

| fold | mean abs score diff | max |
|---|---|---|
| 2017H2 | 0.00998 | 0.1176 |
| 2018H1 | 0.01127 | 0.1421 |
| … (all 11 folds) … | 0.0094 - 0.0113 | 0.13 - 0.17 |
| 2022H2 | 0.00960 | 0.1301 |

Per-date mean absolute difference ranges only from 0.0061 to 0.0174 across all 1338 signal dates,
with median 0.0098. So these are two genuinely different models drawn from the same recipe, not one
model measured twice with a boundary quirk — model-lab's "same cell" description is simply wrong,
and the harness is not introducing a fold artifact.

**That makes this the cleanest available measurement of IR's fragility**, and it is starker than the
correlation argument: two score series

* with the **same mean RankIC to four decimals (+0.0823)**, and
* correlated at Spearman **0.9986**,

produce engine IRs of **+0.3061 and +0.3402 — a 0.034 gap**, larger than the entire net effect that
decided the last two rounds (F54's +0.0028, F55's +0.0022).

Two conclusions, both of which reinforce the shipped package:

1. **IR is not redundant with RankIC at this resolution.** Two models RankIC cannot tell apart can
   differ by 0.034 IR. So the 30% component does carry independent information.
2. **But that information is not actionable.** A 0.034 IR swing from an 0.0098 mean score
   perturbation means IR differences between configs this similar are effectively unpredictable
   from the config itself — we cannot select for them, and a measured IR edge of 0.03 between two
   near-identical models should not be trusted to replicate on a new period.

This is the quantitative justification for the decision rule we have been applying all along:
**optimise RankIC (stable, 70% of the score, measurable), treat IR as a small counterweight you
cannot steer.** It also retroactively justifies not switching to the hedge blend in F55 on the
strength of its +0.028 IR edge, which is now known to be the same size as the noise between two
models RankIC cannot distinguish.

## F58. No industry data exists — and the obvious proxy is already known to fail

Audited the panel for a sector/industry dimension, since "Industry Exposure Limit" appears in the
platform spec's constraint list and industry-relative features are a standard alpha family. The
panel has **54 columns and no industry field of any kind**:

    日期 代码 名称 open high low close volume amount is_suspend raw_open turn change_ratio
    daily_return momentum_5 reversal_5 volatility_5 sma_20 ema_20 macd_* rsi_12 kdj_* bias_20
    cci_14 atr_14 total_market_cap float_market_cap pe_ttm pb ps_ttm roe_avg_ttm roa_avg_ttm
    gross_profit_rate_ttm net_profit_rate_ttm debt_to_asset_lf current_ratio_lf
    beta_000300SH_22 list_days is_st is_new  barra_市值 贝塔 动量 残差波动 非线性市值
    账面市值比 流动性 盈利 成长 杠杆  in_barra

The 11 Barra columns are all **style** factors; the industry block of a standard Barra model is not
provided. So the platform's industry constraint must draw on data held on their side, and
industry-relative features are not constructible from what we are given.

**The obvious workaround is a data-driven industry proxy** — cluster stocks by recent return
co-movement into K groups, then build cluster-relative features (stock return minus cluster mean).
I am not proposing it, because F28 already measured the closest available version of exactly that
idea and it failed: residual (beta-hedged) reversal added **+0.0003 (t=0.38)**, was negative in
2021 and 2022, and beta-hedging made raw reversal *weaker* univariately. Clustering is a
higher-variance way of estimating the same systematic component, so it inherits that null with more
estimation error on top. Proposing it would be re-running a known-null experiment with a noisier
estimator.

This is worth recording as a **data-availability limit rather than a modelling gap**: it is one of
the few directions we can rule out without spending compute, and it closes the last feature family
that our 136-column library does not already cover.

## F59. Learning-to-rank objectives were properly tested and fail decisively

Auditing for an untested lever, I found the objective family is the one place our coverage is
lopsided: **99 report rows use `lgbm_reg` and exactly one uses `lgbm_rank`.** That looked like a
gap worth closing, because RankIC *is* a rank correlation and a LambdaRank objective optimises
ranking directly — the most natural hypothesis available.

It was already tested, and it fails badly. The run is **valid**, not a casualty of the two silent
bugs model-lab found:

    lgbm_lambdarank   semiannual   degenerate_days=0   reliable=1   days=1338
                      mean RankIC 0.0219   RankICIR 0.103
    xgb_rank_csrank   annual       mean RankIC 0.0160   RankICIR 0.073
    (our shipped config, same fold set)         0.0823 / 0.568

`degenerate_days=0` and a full 1338 days means the harness measured it cleanly — this is not the
constant-score failure mode that silently voided `lgbm_raw_csrank` and `lgbm_winsor_csrank`
(those report `days=368`, which is how a degenerate run looks). And `params_keys=9` confirms its
parameter dict survived the shallow-merge bug.

**So regression-on-within-day-ranks beats learning-to-rank by roughly 4x on this task**, which is
counterintuitive and worth stating plainly: the objective that most directly matches the evaluation
metric is the worst performer. The likely mechanism is that LambdaRank's relevance labels are
within-day integer grades truncated at `lambdarank_truncation_level=30`, so it optimises ordering
among the top 30 names per day and discards the cross-sectional information that dominates a
268-day-average Spearman over the full 300-name universe. RankIC scores the whole cross-section;
LambdaRank was pointed at a corner of it.

This closes the last modelling lever. The objective family is now covered: regression on raw /
rank / rank_z / winsor / zscore targets, `l2` and `rank:pairwise` and `lambdarank` objectives,
LightGBM / XGBoost / CatBoost / ridge, with and without the 136-feature library. Nothing beats
regression on within-day percentile ranks with the shipped parameters.

## F60. model-lab's artifacts now agree with the shipped package — verified field by field

After I flagged the stale recommendation, model-lab regenerated it. I verified the agreement
programmatically rather than accepting the message:

| field | `RECOMMENDED.json` | shipped `metadata.json` | match |
|---|---|---|---|
| config | `lgbm_sub1_l63d8` | — | — |
| `num_boost_round` | 400 | 400 | yes |
| `early_stopping` | None | None (fixed rounds) | yes |
| seeds | [42, 43, 44] | [42, 43, 44] | yes |
| learning_rate | 0.03 | 0.03 | yes |
| **num_leaves** | **63** | **63** | yes |
| **max_depth** | **8** | **8** | yes |
| min_data_in_leaf | 200 | 200 | yes |
| feature_fraction | 0.70 | 0.70 | yes |
| bagging_fraction | 0.80 | 0.80 | yes |
| lambda_l2 | 10.0 | 10.0 | yes |
| features | `list(FEATURE_COLUMNS)  # 136 columns` | 136 columns | yes |

They also promoted it to a **first-class `CONFIGS` entry** so reproduction no longer depends on
getting a CLI override string right, and the new entry reproduces the override-built run
**bit-identically**: `lgbm_sub1_l63d8__recommended` gives semiannual **+0.083800** and annual
**+0.077019**, exactly matching `L63D8sb3`.

This matters beyond tidiness: **the organizers read our report during code review.** Had
`RECOMMENDED.json` kept naming the 31/7 config while `submission/model/` shipped the 63/8 one, the
audit would have found a documented recipe that does not reproduce the submitted artifact — a
credibility problem at exactly the moment we can least afford one, since "私榜排名和代码审查结果共同
决定" the final ranking. All three artifacts (recommendation, leaderboard, report) and the shipped
package now agree.

**Two corrections from model-lab, both adopted:**

1. **Their ensemble deltas were measured against `sb3_fixed400`, not against the recommendation.**
   Against `L63D8sb3` all three architecture ensembles are **negative on both fold sets**
   (-0.0004 to -0.0013). So they are not "#2 behind it" — they are strictly below it, and the
   ensemble null is *stronger* than their earlier message implied. This independently confirms F55's
   rejection of the hedge blend from a different direction.
2. **The 63-leaf family is a coherent plateau, not a lucky point.** They ran the missing
   seed-bagged challengers: `L63D8MD400sb3` +0.0835/+0.0762, `L63MD400sb3` +0.0832/+0.0761,
   `LR02R700sb3` +0.0824/+0.0760, `MD400sb3` +0.0819/+0.0751, with the two `min_data 400` siblings
   statistically tied with the recommendation (p=0.48/0.10 and 0.22/0.08). A neighbourhood of
   configs all landing at +0.083 is better evidence for the *region* than a single point estimate,
   and it is a stronger reason to trust `L63D8` than the paired test alone.

They also independently reached the same conclusion about the mis-calibrated bar, and noted the same
error in their own §3 reasoning — a 1-seed sweep carries noise of the same order as the entire
neighbourhood spread, so it is a screening pass rather than a decision procedure.

## F61. Added a contract docstring to `predict()` — and caught a self-inflicted hazard

The organizers read `inference.py` during code review, and that is half the final ranking. The
implementation was already well-commented inline, but the contract method itself had no docstring.
Added one documenting the signature, the input window, the exact output columns, the
suspended-name rule, and the fact that only the *ordering* of `score` reaches the optimizer (with
the spec reference). It is a string literal, so behaviour cannot change — confirmed by rebuilding
and finding all three boosters byte-identical to the pre-docstring build.

**The docstring had to be written around a trap.** The builder's own gate is stricter than the
rules: alongside word-boundary token matching it runs `RAW_SUBSTRINGS_PY_ONLY = ("eval", "exec")`
as a plain substring scan, so `evaluation`, `evaluate` and `execute` are fatal *even in a comment*
— which is exactly the class of problem that lost us two submissions early in the session. The
new text avoids every such word; verified with an explicit raw-substring scan of the rebuilt file
(0 hits for both needles).

**The hazard I created and then caught:** the archive hash changed
(`c1d332f5...` -> `06ea9637...`), but the **armed submission timer still carried the old hash** and
was instructed to *stop and investigate* on a mismatch. Left alone, the timer would have fired at
00:00:30, correctly refused to submit, and burned the slot window on a false alarm. I deleted
`schedule-27908252-...` and re-armed `schedule-682ca9c1-...` with the new hash and an explicit note
that the old value is superseded.

The general lesson, which this project keeps re-learning: **any change to the artifact must be
propagated to every place that asserts a fact about it.** The hash was asserted in three places
(plan, findings, timer) and only the timer was load-bearing.

**Archive of record re-frozen**: sha256
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`, 2.79 MiB. Re-verified:
build deterministic, models byte-identical, judge contract PASS 30/30 with 0 stderr on **four**
independent virtualenvs (3.11 x2, 3.12, 3.13), `simulate_judge` PASS, 0 banned-token hits.

## F62. Consistency audit of the documentation itself — found and fixed a real defect

Applying F61's lesson ("any change to the artifact must be propagated to every place that asserts
a fact about it") systematically, rather than only to the hash:

**Defect found.** `SUBMISSION_PLAN.md` still described the **superseded** configuration. My earlier
string-level edit had left *both* parameter sets in one sentence:

    3x LightGBM (lr 0.03 / num_leaves 31 / max_depth 7 / min_data 200 / ff 0.70 / bagging 0.80 /
    l2 10.0 / 400 rounds, **num_leaves 63 / max_depth 8**, seeds 42, 43, 44, ...

and the evidence bullets beneath it quoted the old config's numbers (+0.0823 / +0.0750, engine
0.0702 / IR 0.3061). A reviewer opening the plan would have read a self-contradictory description
of the submitted artifact. The whole section is now rewritten around `lgbm_sub1_l63d8`, with the
parameters in a table, the official-label evidence, and an explicit list of what was rejected and
why.

**Systematic hash sweep.** Scanned every `.md`/`.py`/`.json` for all four superseded hashes:

| hash | files still referencing |
|---|---|
| `314398fb` (first LGBM-only, 400 r) | FINDINGS only — historical |
| `cd6385e0` (after the `train.py` path fix) | FINDINGS only — historical |
| `0b9ea409` (after the 700->400 revert) | FINDINGS only — historical |
| `c1d332f5` (63/8, pre-docstring) | **none** |
| `06ea9637` (current) | SUBMISSION_PLAN + FINDINGS |

The survivors are all inside FINDINGS, which is **append-only by design** — an evidence log *should*
record what was true at the time, and rewriting history there would be worse than the confusion it
avoids. The fix is therefore a pointer rather than a rewrite: FINDINGS now opens with a header
block naming the current artifact and explicitly marking the earlier hashes as superseded, and
naming `SUBMISSION_PLAN.md` as the authoritative statement of what ships.

**Why this is worth the round.** The final ranking is "私榜排名和代码审查结果共同决定", so the
documents are read by a human deciding our placement. A plan that contradicts itself about the
submitted parameters, or a log whose six "Archive of record" lines disagree, invites exactly the
credibility doubt we cannot afford — and both were introduced by *my own* edits, not by teammates.

## F63. Package-level integrity: documentation, data provenance, and archive hygiene

Closing the consistency work at the level of the artifact itself, rather than the documents about it.

**Internal documentation agrees with content.** `submission/model/metadata.json` records exactly
what the shipped boosters were fitted with:

    lightgbm_params : lr 0.03, num_leaves 63, max_depth 8, min_data_in_leaf 200,
                      feature_fraction 0.70, bagging_fraction 0.80, lambda_l2 10.0
    lightgbm_seeds  : [42, 43, 44]        catboost_seeds: []        catboost_weight: 0.0
    postprocess     : {"kind": "rank"}    members: 3x lightgbm @ 0.3333

and `train.py --help` documents the LightGBM-only default *with its rationale*, so a reviewer who
runs `--help` sees the shipped configuration and why it was chosen.

**Data provenance is verifiable, not just asserted.** `metadata.json` declares
`public_data: "2016-01-04..2022-12-30"`, `train_end: 2022-12-31`, `train_rows: 499799` over
`train_days: 1701`. Checked against the panel: its range is **20160104 -> 20221230** and it
contains **no row after 20221231**. So the declaration matches the data actually used, and there is
no hidden-period artifact anywhere in the package — which matters because the rules forbid
re-fitting on hidden labels, and the organizers verify exactly this during review.

**Archive hygiene.** `submission/__pycache__/` exists on disk (3 `.pyc` files, regenerated by every
local run) but the archive contains **0** pycache/pyc entries — the zipper's exclusion rules cover
`__pycache__`, `.git`, `.venv`, `.DS_Store`, `*.pyc/.pyo/.log/.zip/.tmp` and `._*` prefixes. The
shipped archive is exactly 10 entries: 2 directories and 8 files, nothing stray. Shipping stale
bytecode would have been harmless in practice (Python regenerates it) but it is the kind of thing
that looks careless in an audit, and it is now confirmed absent rather than assumed.

## F64. The private-pandas fast path is verified to be value-neutral — the last portability risk

`inference.py` imports `pandas._libs.window.aggregations` to get ~10x faster rolling kernels. That
is a **private** API, so it is the one dependency that could behave differently in the judge's
environment, where we do not control the pandas build. The code claims:

> "Anything that fails the probe -- or any pandas where the import fails -- silently keeps the
> public slow path, so this can never change a value, only the runtime."

I tested that claim rather than trusting it. Ran the judge contract on three signal dates under two
conditions — normally, and with `builtins.__import__` patched to raise `ImportError` for any
`pandas._libs.window` import:

| condition | kernels active | scores |
|---|---|---|
| normal | `['max', 'mean', 'min', 'std']` | 900 values |
| private API blocked | **`[]`** | 900 values |
| comparison | | **max abs diff = 0.000e+00, bit-identical** |

So the fallback engages cleanly and produces *exactly* the same scores. The guard is well built:
the import is wrapped, and even when it succeeds each kernel is probed against the public per-group
result on the awkward cases (block shorter than the window, block longer, leading NaN, all-NaN
block, two window/min_periods pairs) before being used.

This closes the last environment assumption. Combined with F50 (Python 3.11/3.12/3.13 all pass) and
the clean-venv installs from the package's own `requirements.txt`, every dependency-level risk we
can test has now been tested: the package degrades to identical output rather than failing.

## F65. Input-contract robustness verified by test, including the `MarketDataRawOpen` case

The judge's `data` dict is the one interface we do not control, so I exercised it against malformed
and unexpected inputs rather than assuming the happy path:

| payload | result |
|---|---|
| baseline | **300 rows** scored — exactly the legal universe |
| missing `Barra_factor` | `KeyError: "data is missing required keys: ['Barra_factor']"` |
| missing `MarketData` | `KeyError: "data is missing required keys: ['MarketData']"` |
| **extra key `MarketDataRawOpen`** | **OK, 300 rows** |
| extra junk key | OK, 300 rows |
| `Barra_factor` not a DataFrame | `TypeError: data['Barra_factor'] must be a pd.DataFrame` |
| `data` not a dict | `TypeError: data must be a dict` |

The behaviour is exactly right: **fail loudly on malformed input, tolerate extras.** Silently
producing scores from a partial dict would be far worse than raising, because a missing risk-factor
table would quietly degrade the feature set rather than announcing itself.

The `MarketDataRawOpen` row is the one that matters in practice. The spec lists that table as
existing but **not** being sent to `inference.py`, and an early finding in this log established that
using it would be a leak (it is the unadjusted open, and the label is built on back-adjusted
prices). Two things are now confirmed: the judge may include it without breaking us, and we never
read it — `PLATFORM_KEYS` is exactly the three tables we are entitled to, and `raw_open` appears
nowhere in the shipped feature list.

Combined with F64, every interface the platform can present has now been tested: the three input
tables (present, absent, wrong type, extra), three Python versions, and two pandas builds (private
API available and blocked).

## F66. Full-year end-to-end run against the UNPACKED ARCHIVE (not the working tree)

Every previous `simulate_judge` run pointed at the working `submission/` directory. That is not
quite what ships. This one unpacked `out/submission.zip` into a clean directory and ran the judge
simulation against **that** copy for a full year:

    [judge] PASS — all hard rules satisfied on 240 signal dates
    coverage        71,940 / 72,600 predictions matched a label   (99.1%)
    mean_rankic     0.281179
    std_rankic      0.126500
    rankicir        2.222760
    positive_ratio  0.983333
    per-year 2022   240 days

240/240 signal dates, every hard rule satisfied, from the exact bytes we will upload.

**In-sample, so not evidence of generalisation** — 2022 was in the training window, and the
0.2812 figure is the model fitting data it has seen. It is recorded here only as an end-to-end
liveness check on the archive. The number that matters is the out-of-sample walk-forward one
(semiannual +0.0838 / annual +0.0770 on the official label), and the honest walk-forward runs on
the 136-feature library were 2020 +0.0705 / 2021 +0.0585 / 2022 +0.0683.

The higher in-sample figure than the earlier 31/7 package (0.1995) is expected — deeper trees fit
training data harder — and is exactly why the round-count decision in F37/F39 was made on
*out-of-sample* evidence rather than on this.

## F67. The risk-adjusted target is decisively WORSE — the last untested target family is closed

I found a real gap in the target coverage. All five existing kinds — `raw`, `rank`, `rank_z`,
`winsor`, `zscore` — are **monotone transforms of the same label**. They change how the loss
weights the tails; none of them changes *which stock the model is told to prefer*. A target that
**reorders** the cross-section had never been tested.

The natural candidate is a risk-adjusted label: within-day percentile rank of `return / vol_20`.
`vol_20` is a frozen member of `FEATURE_COLUMNS`, so it is present in every dataset frame.

**I measured the reordering before spending compute on it**, since a near-monotone target would
make the experiment vacuous:

    within-day Spearman( rank(raw return), rank(return / vol_20) )
        mean 0.9480   median 0.9660   min 0.1902   max 0.9925   (n = 396,190 rows, 1338 days)

so ~5.2% of the cross-section is genuinely reordered. Not vacuous. And the stakes were real:
monotone transforms alone span a huge range in this harness (`rank` +0.0721 vs `winsor` +0.0326 vs
`raw` +0.0251 on the same config), so target shaping clearly matters a lot.

**Result: it is worse, decisively, in every period.**

| | shipped (`lgbm_sub1_l63d8`) | risk-adjusted target | delta |
|---|---|---|---|
| semiannual mean RankIC | **+0.0838** | +0.0744 | **-0.0094** |
| annual mean RankIC | **+0.0770** | +0.0696 | **-0.0074** |
| semiannual RankICIR | +0.591 | +0.515 | -0.076 |
| annual RankICIR | +0.544 | +0.497 | -0.047 |
| years won | — | **0 of 6** | — |

Per-year deltas: 2017 -0.0074, 2018 -0.0096, 2019 -0.0111, 2020 -0.0079, 2021 -0.0090,
2022 -0.0101. Losing by 0.007-0.011 in **every single year**, on both fold sets, against a
pre-registered +0.005 bar — this is one of the least ambiguous results in the whole log.

**The interpretation is clean and worth stating:** the metric rewards ordering by expected **raw**
return, so training on a risk-adjusted target moves the model *away* from the thing it is scored
on. A "cleaner" label is not a better label when the loss and the metric disagree about what
"better" means. Combined with F59 (learning-to-rank, the objective that most directly matches the
metric, was the *worst* performer at 0.0219), the pattern is that **the shipped recipe wins by
aligning the training target with the evaluation metric as closely as possible** — and the two
attempts to be clever about that alignment both failed.

Nothing changes in the shipped package. Implementation left in place as `lgbm_sub1_voladj` and
`voladj_rank` so the negative is reproducible rather than merely asserted.

## F68. The F67 harness edit is provably non-invasive

Adding `voladj_rank` to `build_target` and two entries to `CONFIGS` meant editing
`dtcomp/models.py` — a **shared** file that every report and both teammate artifacts depend on. A
change there could silently alter results recorded earlier, so I verified it rather than reasoning
about it: re-ran the shipped config through the harness after the edit and compared against the
pre-edit report at full float precision.

| | after the edit | before the edit |
|---|---|---|
| semiannual mean RankIC | `0.08379950544016695` | `0.08379950544016695` |
| semiannual RankICIR | `0.5909973429605818` | `0.5909973429605818` |
| predictions (396,237 rows) | — | **max abs diff `0.000e+00`, bit-identical** |

So every number recorded before F67 remains valid, and the new target kind is purely additive. The
check run's reports were deleted so the leaderboard is not polluted by a duplicate row.

This matters more than it looks. The two most damaging failures in this session were both
**silent** — a shallow `dict.update` that replaced a params dict (F39) and a degenerate-score path
that dropped 845 of 1213 days (model-lab). Both produced plausible numbers rather than errors, and
both were only caught by comparing against a known-good reference. Editing a shared harness late in
the session is exactly the situation that creates a third one, so the comparison is the control,
not a formality.

## F69. Upload-path pre-flight, and a misleading error hint that could cost us the slot

I exercised the full reserve -> PUT -> release handshake with `dtapi.py upload-test`, which takes
the upload slot **without** consuming a submission slot. It was correctly refused:

    POST .../uploads -> 403
    body: {"detail":"daily_limit"}
    hint: session expired or not logged in -- re-run ./venv/bin/python platform/import_cookies.py
    preflight: {"joined": true, "can_submit": false, "reason": "daily_limit"}

**Two things confirmed:**

1. **No stuck slot.** `uploading: False` afterwards, so the failed reserve did not leave the global
   upload reservation held. That was the specific risk worth checking — a stuck slot sets
   `can_submit: false` and would have silently blocked the 00:00 submission, turning a
   belt-and-braces check into the cause of the failure. The slot is clean and the path is clear.
2. The reserve is gated on the daily limit, so the pre-flight cannot be run until quota resets.
   That is the correct behaviour, not a defect.

**The hazard worth recording: the tool's `hint` is wrong here.** It says "session expired or not
logged in -- re-run import_cookies.py", but the actual cause is in the body: `daily_limit`. A human
(or an agent) following the hint at 00:00 would re-import cookies, see the same 403, and have no
idea the real problem was quota — or worse, conclude the session was broken and start debugging
authentication while the slot window ticked.

**Rule for the submission moment: read the `body`, not the `hint`.** The body carries the
machine-readable reason (`daily_limit`, and the same channel would carry any other 403 cause). The
timer's procedure already tells the reader to check `can_submit`/`reason` before acting, which
covers this — the note is here so that anyone debugging a failure at 00:00 does not chase
authentication on the strength of a generic hint string.

## F70. Fixed the misleading 403 hint at its source, not just in the docs

F69 documented that `dtapi.py`'s 403 hint always claims "session expired or not logged in" even
when the platform's own body says `daily_limit`. Documenting a hazard that fires at the single most
important moment of the session is not enough — the fix belongs in the code.

`platform/dtapi.py` now parses the response body and leads with its own reason:

| | hint text on a real `daily_limit` 403 |
|---|---|
| before | `session expired or not logged in -- re-run ./venv/bin/python platform/import_cookies.py` |
| after | `the platform refused on QUOTA (detail='daily_limit'), not auth -- check dtapi.py subs for next allowed, and do not re-import cookies` |

It also handles the general case: any other 403 with a `detail` string surfaces that reason and says
to read the body rather than assume an expired session, and a 403 with no parseable body still falls
back to the original auth hint. Verified against the live endpoint (the same request that produced
the wrong hint a moment earlier now produces the right one), and `subs` / `lb` still work, so the
read-only paths are unaffected.

This does **not** touch the submission package — `platform/dtapi.py` is our own tooling, so the
archive hash is unchanged at
`06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723`.

The general principle, which this session has now applied three times (F44's import path, F61's
propagation sweep, and here): **when a failure mode is identified in a tool that will be used at a
decisive moment, fix the tool rather than writing a note telling the future reader to be careful.**
Notes get skipped under time pressure; code does not.

## F71. Closed the only non-quota failure path in the submission

With ~18 hours between now and the slot, one residual risk was unaddressed: **the session cookie
has to still be valid at 00:00.** Every submission path needs auth, and an expired session is the
one failure mode that is neither a quota problem nor recoverable by the documented `release` step.

Checked the current state: `dtapi.py me` returns `verify_status: verified` for
`tangzhh2026@shanghaitech.edu.cn` (verified 2026-10-08T15:48Z), so the session is live now. Whether
it survives to 00:00 is not knowable, so the *recovery* had to be written down rather than assumed:

    log in at https://datatech.geekpie.club in the DSH built-in browser (user Azuma)
    -> ./venv/bin/python platform/import_cookies.py
    -> retry the submit

`import_cookies.py` reads the DSH browser host's own Chromium cookie store
(`~/Library/Application Support/dsh-builtin-browser-host/Cookies`) into a 0600 JSON file that
`dtapi.py` consumes — our own browser profile, our own account, on our own machine. The cookie
values never reach stdout or a transcript.

The armed timer was re-armed as `schedule-75bab5b8-...` (replacing `schedule-682ca9c1-...`) with a
triage block that distinguishes the three failure classes explicitly:

| symptom | cause | action |
|---|---|---|
| 403, body `{"detail":"daily_limit"}` | quota | check `next allowed`; **do not** re-import cookies |
| 403/401, no quota in body | session expired | browser login -> `import_cookies.py` -> retry |
| reserve succeeded then died | stuck upload slot | `dtapi.py release` **first** |

It also carries the post-success instruction to record the board numbers and then **stop
submitting**, so the timer cannot be read as licence to keep uploading.

## F72. The LightGBM STRUCTURAL options were never swept — now they are, and none help

Auditing what the hyper-parameter sweep had actually covered, I found it was **numeric-only**.
Across 107 report files, the only `params` keys ever set were `learning_rate`, `num_leaves`,
`max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`,
`lambda_l2` (plus the XGBoost/CatBoost/Ridge equivalents). These appeared in **zero** reports:

    extra_trees   linear_tree   path_smooth   min_sum_hessian_in_leaf   max_bin

Those are *structural* options, not scale knobs — they change the tree-building algorithm rather
than its capacity, which is a genuinely different axis from "more leaves / slower learning rate".
Tested on the shipped recipe (3 seeds, fixed 400 rounds, 63 leaves / depth 8), both fold sets:

| variant | semiannual | delta | annual | delta |
|---|---|---|---|---|
| **SHIPPED** | **+0.083800** (ICIR +0.591) | — | **+0.077019** (ICIR +0.544) | — |
| `extra_trees=true` | +0.083590 (0.572) | -0.000210 | +0.076259 (0.523) | -0.000760 |
| `path_smooth=10` | +0.083571 (0.586) | -0.000228 | +0.076679 (0.537) | -0.000339 |
| `max_bin=63` | +0.083573 (0.588) | -0.000227 | +0.075987 (0.540) | -0.001032 |
| `min_sum_hessian_in_leaf=1e-2` | +0.083800 (0.591) | **+0.000000** | +0.077019 (0.544) | **+0.000000** |

**None helps.** The three that change behaviour are all slightly worse on both fold sets, and none
comes close to the +0.005 bar in either direction.

The fourth is a useful control rather than a result: `min_sum_hessian_in_leaf=1e-2` reproduces the
shipped config to **six decimal places on both fold sets** — because with `min_data_in_leaf=200`
the hessian constraint never binds, so LightGBM's default (1e-3) and 1e-2 build the same trees. It
is a free confirmation that the harness is deterministic and that the override path reaches the
model, which is exactly the property whose absence caused the F39 shallow-merge bug.

`linear_tree` was not run: it requires dense numeric input and this feature set is deliberately
NaN-rich (missing fundamentals, short listing histories), so a fair test would need its own
imputation design rather than an override. Recorded as untested rather than assumed neutral.

## F73. Leaderboard rebuilt and re-checked — the shipped config is still tied #1 on both fold sets

After adding the structural variants, `reports/leaderboard_local.md` was regenerated from all 148
report files (it is rebuilt from disk, not appended, so the new rows are ranked with everything
else rather than bolted on):

| semiannual top 3 | annual top 3 |
|---|---|
| +0.0838 `lgbm_sub1__L63D8sb3` | +0.0770 `lgbm_sub1__L63D8sb3` |
| +0.0838 `lgbm_sub1_l63d8__MSH` | +0.0770 `lgbm_sub1_l63d8__MSH` |
| +0.0838 **`lgbm_sub1_l63d8__recommended`** | +0.0770 **`lgbm_sub1_l63d8__recommended`** |

The top three are the **same configuration reached three ways** — the override-built run, the
`min_sum_hessian_in_leaf=1e-2` control (which is bit-identical, F72), and the first-class `CONFIGS`
entry. They tie to four decimals on both fold sets, which is itself the cleanest possible
demonstration that the three reproduction paths agree.

So after testing every structural option the sweep had missed, **nothing displaced the shipped
config on either fold set** — the two leaderboard ranks shown as "3/82" and "3/66" are an artefact
of alphabetical tie-breaking among three identical measurements, not a position behind two better
models.

## F74. `preprocess=raw` is decisively validated for the shipped config — the cross-sectional modes are 3x worse

Another instance of the F67/F72 pattern. All three preprocessing modes (`raw`, `cs_rank`, `cs_z`)
are either identity or **cross-sectional** (per-day) transforms — a time-series normalisation has
never existed here. More importantly, our shipped config uses `preprocess="raw"`, and that choice
had never been re-verified **for this config**: model-lab's early work established `cs_rank` as
better on a *different* configuration family, and the mode was inherited rather than retested after
the architecture changed.

Tested on the shipped recipe (3 seeds, fixed 400 rounds, 63 leaves / depth 8), both fold sets:

| preprocess | semiannual | delta | annual | delta |
|---|---|---|---|---|
| **`raw` — SHIPPED** | **+0.083800** | — | **+0.077019** | — |
| `cs_rank` | +0.067953 | **-0.015847** | +0.062297 | **-0.014722** |
| `cs_z` | +0.070034 | **-0.013766** | +0.064294 | **-0.012725** |

**Both cross-sectional modes lose by 0.013-0.016 on both fold sets — roughly 3x the pre-registered
+0.005 bar, in the wrong direction.** This is not a marginal result; it is the largest single-axis
effect measured in this project outside the seed-bagging and early-stopping findings.

The interpretation matters for anyone reading the leaderboard: **the preprocessing choice interacts
with the configuration.** `cs_rank` being better on the `lgbm_rank_raw` family (where it was
established) does not carry over to the deep-63-leaf sub1 family, where the raw feature values let
the trees find thresholds the cross-sectional ranks destroy. That is exactly the kind of inherited
assumption worth re-testing whenever the architecture moves, and it is why this was run rather than
assumed.

Nothing changes in the shipped package — but the choice is now **verified** rather than inherited.

## F75. `--train-filter` is a no-op, and the shipped `train.py` filter matches the harness exactly

Chasing the same "inherited assumption" pattern, I checked the training-row filter, since it is a
harness-level default rather than something the config pins down. Two results:

**1. The shipped `train.py` and the harness use the SAME definition.** `train.py` fits only on rows
where both legs were unsuspended and both opens are finite and positive; `dtcomp.data.tradable_mask`
is the identical predicate (entry/exit suspension flags, finite positive opens, finite label). So
the harness numbers describe the model we actually ship — worth confirming rather than assuming,
because a mismatch here would invalidate every fold number in this log at once.

**2. `--train-filter all` changes nothing, because it cannot.**

| filter | semiannual | delta | annual | delta | train_rows (first fold) |
|---|---|---|---|---|---|
| `tradable` (default) | +0.083800 | — | +0.077019 | — | 102,996 |
| `all` | +0.083800 | **+0.000000** | +0.077019 | **+0.000000** | **102,996** |

Identical to six decimals *and* identical row counts. The reason is structural: `build_dataset`
already drops every row whose label is non-finite —

    frame = frame.loc[np.isfinite(frame[TARGET].to_numpy(dtype="float64"))].copy()

— so by the time the filter is applied there are no non-tradable rows left to include. The flag
cannot do what its name says.

**This is not a bug** (you cannot fit on a row that has no label), but it is a **misleading
control**: anyone reaching for `--train-filter all` to test "should we train on suspended names
too?" will get a bit-identical result and could reasonably conclude "it makes no difference", when
the truth is "the experiment did not run". That is the same failure shape as the shallow-merge bug
(F39) and the degenerate-score path — a plausible number standing in for a measurement that never
happened. Recorded so the flag is not mistaken for a tested axis.

## F76. The frozen model trains on 8% more data than any fold — the fold numbers are conservative

Quantifying a structural point that had been implicit in every projection: the walk-forward folds
are the closest proxy we have for the hidden period, but **none of them trains on as much data as
the model we actually ship**.

| | training rows | window |
|---|---|---|
| first fold (2017H2) | 102,996 | <= 2017-06-30 |
| … | … | … |
| last fold (2022H2) | 462,332 | <= 2022-06-30 |
| **FROZEN shipped model** | **499,799** | **<= 2022-12-31 (all public data)** |

The frozen model sees **37,467 more rows (+8%)** than the longest fold, and its window extends a
further two quarters — including the 2022H2 period that the last fold is *scored* on. Two
consequences worth recording:

1. **The fold numbers are a mild underestimate, not an overestimate.** More training history
   generally helps this model family (the fold-by-fold train-row column rises monotonically and
   mean RankIC does not degrade with it), so a fold figure of +0.0838 is more likely to understate
   the shipped model than to flatter it. That is the safe direction for a projection to err in.
2. **The hidden period immediately follows the training window**, with no gap and no overlap. The
   frozen model is trained through 2022-12-31 and scored on data after it, which is exactly the
   walk-forward shape the folds simulate — the last fold's train<=2022-06-30 / test=2022H2 split is
   the closest analogue in the whole harness, and it is one of the strongest folds.

This does **not** upgrade the projection in F51 (~0.081 board RankIC). It bounds the direction of
the error rather than shrinking it: the extrapolation is still chained from a single board
observation, and a different market regime remains the dominant uncertainty.

## F77. Where the shipped model's signal actually comes from — it is short-horizon microstructure

Read the gain-based feature importances straight off the three shipped boosters (mean share across
seeds). Top 20:

| feature | gain share | | feature | gain share |
|---|---|---|---|---|
| **`close_pos_day`** | **4.29%** | | `ret_3` | 1.29% |
| `gap_prev_low` | 1.79% | | `hl_range_20_xr` | 1.25% |
| `close_over_ma60_xr` | 1.59% | | `range_pos_120` | 1.23% |
| `beta_000300SH_22` | 1.58% | | `illiq_ratio_20_120` | 1.21% |
| `ret_2` | 1.58% | | `ret_119_minus_20` | 1.21% |
| `hl_range_1` | 1.52% | | `barra_贝塔` | 1.20% |
| `intraday_mean_5` | 1.45% | | `low_prox_60` | 1.17% |
| `overnight_mean_5` | 1.43% | | `gap_prev_high` | 1.17% |
| `ret_1` | 1.38% | | `ret_119` | 1.10% |
| `intraday_1` | 1.37% | | `turn_z_20_xr` | 1.09% |

Aggregated by the largest window number appearing in each feature name:

| window | gain share |
|---|---|
| < 60 days | **72.0%** |
| 60-99 days | 14.5% |
| >= 100 days (e.g. 120) | 13.4% |

**Three things worth recording:**

1. **The signal is short-horizon.** Nearly three quarters of the model's gain comes from windows
   under 60 days, and the single most important feature is `close_pos_day` — where the close sits
   within the day's high-low range — at 4.29%, more than double the next feature. The top ten is
   almost entirely 1-5 day microstructure: `gap_prev_low`, `ret_1/2/3`, `hl_range_1`, `intraday_1`,
   `intraday_mean_5`, `overnight_mean_5`. This is consistent with everything else in the log: the
   library's edge is reversal and microstructure, and the residual-reversal idea (F28) failed
   precisely because the raw short-horizon version already captures it.
2. **Long-window features carry little weight** (13.4% for >= 100 days). That is a *reason to be
   sceptical* of any proposal to shorten the look-back further — there is not much long-window mass
   left to remove, and F76 showed the frozen model's extra history is an advantage rather than a
   liability.
3. **This is NOT an argument for dropping low-importance features.** F28 established the hard way
   that univariate importance is not incremental value: `beta_x_rev` was the strongest univariate
   feature in the library and added exactly zero. Gain shares say what the model *uses*, not what it
   *needs*, and the tree can route around any single column. No feature selection was performed on
   the strength of this table, and none should be.

## F78. Closing integrity check: the session's edits are all outside the artifact

Several shared files were edited after the package was frozen — `dtcomp/models.py` (F67's target
kind, F72's configs), `platform/dtapi.py` (F70's hint fix), plus four new `research/` scripts. Any
one of them could in principle have reached the artifact, so the package was re-checked against
itself rather than assumed intact:

| check | result |
|---|---|
| archive sha256 | `06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723` — **unchanged** |
| archive vs `submission/`, all 8 entries re-hashed from inside the zip | **0 mismatches** |

**Every edit this session is outside the submission.** The frozen package is byte-identical to the
build that passed the full validation suite (build determinism, `train.py` byte-for-byte
reproduction in a fresh venv from an unpacked copy, judge contract 30/30 with 0 stderr on four
virtualenvs across Python 3.11/3.12/3.13, 0 banned-token hits, full-year `simulate_judge` PASS on
the unpacked archive).

This is the check that makes all the exploratory work safe to have done: the harness, the platform
tooling and the research scripts can be modified freely, because the artifact is verified separately
and its hash is asserted in three places (this file, `SUBMISSION_PLAN.md`, and the armed timer).

## F79. The one obviously-missing microstructure feature is not constructible from this data

F77 showed the model's signal is dominated by short-horizon microstructure, with `close_pos_day`
(where the close sits in the day's range) the single most important feature at 4.29%. The natural
next question is what else belongs to that family. The obvious candidate is **VWAP deviation** —
`close / VWAP - 1` — which is standard, cheap, and absent from all 136 features.

`VWAP` would have to be proxied by `amount / volume`, and **it is not a price**. Measured over the
whole panel (577,827 rows with positive volume):

| `close * volume / amount` bucket | share of rows |
|---|---|
| [0.9, 1.1) — i.e. `amount/volume` really is the close | **41.2%** |
| [1.1, 2) | 28.9% |
| [2, 10) | 29.5% |
| [10, 200) | 0.4% |

and the per-stock ratio is unstable as well: median within-stock standard deviation 0.047, but
**75th percentile 0.225 and a maximum of 5.73**. A unit convention would be constant per stock; this
is not.

So `amount/volume` agrees with the close for fewer than half the rows and drifts systematically over
time (per-year median ratio 1.03 in 2016 rising to 1.42 in 2022). **A VWAP-deviation feature cannot
be built from it.** This is consistent with what feature-eng did: `volume` appears in
`_MARKET_REQUIRED` and in a docstring, and in **no feature computation anywhere** — they reached the
same conclusion and simply did not spend a finding on it.

**Why this matters:** it closes the "add more microstructure features" direction with a
data-availability argument rather than an experiment, exactly as F58 closed the industry direction.
The dominant feature family is already as complete as the data allows, and the one obvious gap is
not fillable. It also means no retraining is warranted on this axis — which matters, because
retraining would discard the only artifact with board evidence behind it.

## F80. Neural network: the last untested model FAMILY, and it loses by 7x the bar

Every model tested in this project was a tree ensemble or a linear model. A multilayer perceptron
has a genuinely different inductive bias — a smooth global function rather than axis-aligned splits
— so it was the last untested direction. Added as `mlp_reg` / `mlp_sub1` (additive; sklearn 1.9.1
is available, torch is not), fitted through the same `FeatureProcessor` and the same causal folds as
everything else. **Feasible at this scale**: 60 epochs on 500k rows takes ~0.6 min, not the hours a
CPU MLP would suggest.

**First run, and why it was wrong.** With `preprocess="raw"` — copied from the shipped recipe — the
MLP scored **+0.0048**, essentially nothing. That result was **unfair, not informative**: MLPs are
acutely scale-sensitive and raw features are unnormalised. Re-run properly:

| model | mean RankIC | RankICIR |
|---|---|---|
| **1-seed LightGBM (31/7), same folds** | **+0.0809** | +0.566 |
| MLP, `preprocess=raw` (unnormalised) | +0.0048 | +0.060 |
| MLP, `preprocess=cs_z` | +0.0391 | +0.383 |
| MLP, `preprocess=cs_rank` | **+0.0456** | +0.461 |

**Even correctly normalised, the MLP reaches less than 60% of the gradient-boosted model's
RankIC** — a deficit of 0.035, or **7x the pre-registered +0.005 bar, in the wrong direction.**
That is decisive, and it closes the last model family.

**Two things worth keeping:**

1. **Normalisation bought the MLP 9.5x** (+0.0048 -> +0.0456) while the *same* choice costs the
   LightGBM 0.013-0.016 (F74). The preprocessing preference is **model-family-dependent** — trees
   want raw thresholds, neural nets need standardised inputs — so an inherited preprocessing choice
   is not transferable across families in either direction. Running the MLP only on `raw` would have
   produced a confident and completely misleading negative.
2. **The result is consistent with the project's other findings.** The two attempts to be cleverer
   than "regress within-day ranks with a GBM" — learning-to-rank (F59) and a risk-adjusted target
   (F67) — also failed decisively. A neural net is the third, and the pattern is stable.

**Harness edit verified non-invasive** (the F68 control): after adding the `mlp` backend,
`lgbm_sub1_l63d8` still reproduces `mean_rankic = 0.08379950544016695`, `rankicir =
0.5909973429605818` and **bit-identical predictions** across 396,237 rows. Every earlier number
remains valid.

## F81. The board shows our LAST submission — confirmed empirically, and we are currently #94

Until now the "last submission is what counts" rule was known only from the rules text. The public
board proves it directly. Our row today:

    rank #94   user "Azuma"
      rankic        = 0.06021     <- submission #2, our WORST result
      ir            = 1.0785
      score         = 0.5931
      submission_id = 36fed7c3    <- our LAST submission
      submissions   = 5

**This is not the same row I read earlier in the session.** When submission #1 was the most recent,
our row showed `rankic 0.0712 / score 0.7551 / rank #29`. Once submission #2 replaced it, the board
moved us to **#94 / score 0.5931** — the same numbers the public board originally reported for
submission #2 in isolation. The board is a live view of the *last* submission per user, not the
best one, and it silently drops our standing when a worse artifact is uploaded.

**Two corrections to the record:**

1. **"We are rank #29" was stale.** It described the period when submission #1 was last. Our
   current standing is **#94 of 134**, and that is the number the private board would carry forward
   if the competition ended now. Everything in `SUBMISSION_PLAN.md` that quotes #29 needs reading
   as historical.
2. **The exposure is larger than the plan stated.** I had been describing the risk as "the private
   board would score a 0.0602 package instead of our better one". It is worse than that: we are
   *currently ranked 94th*, in the bottom third, and the frozen package is the fix.

**What does not change:** the plan, the timer, the artifact, and the conclusion that landing the
package at 00:00 is the single highest-value action available. What changes is the framing — this
is not protecting a good position, it is **recovering from a self-inflicted one**, and the
empirical confirmation makes the case unambiguous rather than rule-inferred.

It also validates the post-submission rule in `SUBMISSION_PLAN.md` from the other direction: every
upload *replaces* the scored artifact, which is exactly why a speculative late slot would be
destructive rather than merely useless.

## F82. Cross-family blending WORKS — and I am recording it without shipping it

F55 rejected blending because the members were near-identical (Spearman 0.9986) and a mid-point is
not on the score-optimal frontier. That reasoning implied a testable prediction: **blending should
work when the members genuinely differ.** The MLP from F80 is a different family, so I measured the
diversity first:

| pair | within-day Spearman |
|---|---|
| GBM vs GBM (same-family ensembles, F55) | 0.9986 |
| **GBM vs MLP** | **0.3428** (semiannual) / **0.3281** (annual) |
| GBM vs ridge | 0.4337 / 0.4346 |

And the prediction holds. Rank-averaging the shipped 3-seed GBM with the single-seed MLP:

| weight on MLP | semiannual delta | paired t | annual delta | paired t |
|---|---|---|---|---|
| 0.05 | +0.000516 | +5.20 | +0.000441 | +4.27 |
| 0.10 | +0.000849 | +4.06 | +0.000702 | +3.29 |
| **0.15** | **+0.001040** | **+3.20** | **+0.000839** | **+2.52** |
| 0.20 | +0.000982 | +2.20 | +0.000692 | +1.51 |
| 0.25 | +0.000678 | +1.17 | +0.000284 | +0.48 |

**w=0.15 is the maximum on BOTH fold sets**, so the weight is not cherry-picked, and the whole
0.05-0.20 range is positive and significant on both. RankICIR improves too (+0.6002 vs +0.5910
semiannual; +0.5525 vs +0.5444 annual). **This is the first thing all session that beats the shipped
configuration on the paired test.** Ridge, by contrast, is weaker (+0.0006 peak, t=1.0-2.4).

**I verified the implementation path before deciding, because a decision that hinges on effort
should rest on measured effort:**

* sklearn's MLP is **fully deterministic** — two identical fits give `max |weight diff| = 0.000e+00`.
* Its forward pass is `relu` on hidden layers and identity on the output, i.e. three matmuls.
* A **hand-rolled numpy forward pass reproduces `sklearn.predict` exactly** (`max diff = 0.000e+00`).
* So the artifact would be a small `.npz` of weights — **no pickle**, which §9.3 bans.

**Decision: do NOT ship it.** Not because the effect is unreal — it is real and replicated — but
because of the arithmetic and the shape of the change:

1. **The gain is +0.0010 RankIC, five times below the pre-registered +0.005 bar**, worth roughly
   **+0.0086 score, about one rank position**.
2. **It is the largest change to the scored artifact in the session** — a new model family, a second
   preprocessing mode (`cs_rank` for the MLP alongside `raw` for the GBM), and a hand-rolled forward
   pass inside `inference.py`. Switches 1 and 2 (drop CatBoost; 31/7 -> 63/8) each made the artifact
   *simpler* or changed two hyper-parameters. This one makes the scored file materially more complex.
3. The L63D8sb3 exception was granted for a change of that simpler kind, on **stronger** evidence
   (+0.0015/+0.0020, t=2.82/3.47). Granting a second exception for a larger change on weaker evidence
   would mean the bar no longer exists.
4. Expected value is only marginally positive: gain ~+0.0086 against a residual validation-miss risk
   that, however small, is multiplied by the cost of scoring badly again (we know what #94 feels
   like — F81).

**Recorded rather than discarded**, because it is the only measured improvement that survives the
paired test, the implementation is proven, and the code paths exist (`mlp_sub1`, `mlp_reg`,
`MLPcs_rank` predictions on both fold sets). A future session — or a human with more time than
16 hours — can pick it up deliberately instead of rediscovering it. The frozen package is untouched
and backed up at `/tmp/pkg_backup`.

## F83. One more argument for F82's decision: the code review is half the ranking

Reviewing F82 with fresh eyes, I had weighed the blend's cost only as *implementation risk*. There
is a second cost I did not state, and it is not small:

**The final ranking is "私榜排名和代码审查结果共同决定" — the code review is half of it.** Shipping
the blend would put a hand-rolled numpy MLP forward pass, a second preprocessing mode, and a
cross-family rank blend inside `inference.py`. None of that is dishonest, and all of it would be
documented — but it makes the scored file **materially harder for an organizer to audit**, and a
submission that is hard to audit is a submission that invites doubt at exactly the moment we are
being judged on trustworthiness. The current artifact is 1,097 lines of readable, uniformly-styled
code with one model family and one preprocessing path; that legibility is an asset the competition
explicitly pays for.

So the honest cost accounting for the blend is three-fold, not two:

| cost | magnitude |
|---|---|
| effect is 5x below the pre-registered bar | ~1 rank position (F82) |
| largest change to the scored file in the session | validation-miss risk x cost of a bad score |
| **reduces the auditability that half the ranking rewards** | unquantified, but real |

Against a gain of +0.0010 RankIC, that is not a trade worth making. **The decision in F82 stands,
and this is the argument that makes it comfortable rather than marginal.**

Worth noting the asymmetry for anyone revisiting this: the two changes we *did* ship both made the
artifact **simpler or nearly identical** — dropping CatBoost removed a whole model family, and
31/7 -> 63/8 changed two hyper-parameters. The one change we rejected is the one that would have
made it more complex. That is not a coincidence; it is the same "prefer the legible artifact"
principle that has governed the endgame.

## F84. The manual fallback needed a durable backup — `/tmp` is volatile

F83's manual-submission path pointed at `/tmp/pkg_backup/`. Verifying it (hash matched, tree
identical) exposed a flaw in the plan rather than in the copy: **`/tmp` is cleared on reboot on
macOS.** A fallback that only works if the machine has not restarted is not much of a fallback, and
the whole point of that section is to survive the case where *this session* is gone — which is
correlated with the machine having been restarted.

Fixed by keeping byte-identical copies in the workspace, which is durable:

    backup/submission.zip   sha256 06ea9637183133ba1c6945b1d283efcb62aa36779802ed7e1d30245502e57723
    backup/submission/      tree identical to submission/

and `SUBMISSION_PLAN.md` now names `backup/` as the authoritative copy, with `/tmp/pkg_backup/`
described as convenience only.

Small thing, but it is the same class of error the session keeps surfacing: a control that appears
to cover a failure mode but does not, because one of its assumptions was never checked. The hash
was right; the *location* was the bug.

## F85. The goal will exhaust its round budget before the submission slot — and why that is survivable

Checking the goal state against the clock:

| | |
|---|---|
| goal rounds used | **75 / 256** |
| rounds remaining | 181 |
| hours to the 00:00 slot | **15.6** |
| observed pace | ~1 min per round |
| implied pace needed to reach the slot | 5.16 min per round |

At the observed pace the goal exhausts its 181 remaining rounds in roughly **3 hours (~11:30 CST)**,
**13 hours before the submission slot**. Automatic continuation would stop well short of 00:00.

**Three independent reasons this is survivable, in order of strength:**

1. **The submission is armed as a host-level schedule, not as a goal round.** The reminder was
   created with `deliveryMode: host` and `state: scheduled`; it fires on the host at
   2026-10-10 00:00:30 +08:00 regardless of whether a goal round is in flight. Goal exhaustion
   does not disarm it.
2. **The manual path is documented and does not need an agent at all** (F83/F84): a five-step
   checklist in `SUBMISSION_PLAN.md`, the exact hash to verify, and byte-identical copies of both
   the archive and `submission/` at `backup/` in the workspace.
3. **The deadline is 2026-10-14 23:59 and only the LAST submission counts**, so there is a four-day
   window. Missing 00:00 exactly costs nothing; acting on an unverified artifact would.

**The one thing that would remove the risk entirely** is raising `maxGoalRounds`, which the goal
policy reserves for a direct human request — I cannot do it myself. The user has been told; if they
extend it, continuation simply runs through the slot and the timer becomes redundant rather than
load-bearing.

## F86. Correction to F85: rounds are not fixed-interval, so the goal is NOT certain to expire before the slot

F85 estimated that the goal's remaining rounds would run out "in roughly 3 hours (~11:30 CST), 13 hours
before the slot", on the assumption of a **~1 minute per round** pace. Observation falsifies that
assumption:

| observation | wall clock |
|---|---|
| rounds 154-160 | ~1 min apart |
| round 160 -> 161 | **17 min** |
| rounds 163 -> 164 | **6 h 25 min** |

So rounds are **not** fixed-interval — they are driven by when the session actually resumes, which
can be minutes or many hours. **F85's conclusion was directionally wrong**: 92 rounds remain with
7.8 hours to the slot, and on an observed pace that includes multi-hour gaps, **the goal may well
last through 00:00**. F85's "the goal WILL run out of rounds before 00:00" should be read as
"it may", not "it will".

**What survives from F85, and it is the part that matters:**

1. The reminder is armed with `deliveryMode: host` and fires at a wall-clock time, independent of
   goal rounds — that was verified directly from `schedule_list`, not inferred, and is unaffected.
2. The manual path (F83/F84) is documented in `SUBMISSION_PLAN.md` with the exact hash, a five-step
   checklist, and byte-identical copies at `backup/`.
3. The deadline is 2026-10-14 23:59 and only the LAST submission counts, so the four-day window makes
   exact timing irrelevant.

**The methodological point** is the one this log keeps making: F85 turned an *assumption about pace*
into a *prediction about a deadline*. The assumption was never measured — three samples would have
shown it was wrong — and the prediction was stated with more confidence than it deserved. The
mitigations were correct to put in place anyway, which is why the error cost nothing; but the
confidence was not earned. **Measure the pace before predicting from it.**

## F87. The neural net was trained ~3x too long, and fixing that DOUBLES the blend

F80 established the MLP loses to the GBM and F82 measured a small blend gain (+0.0010) that I
declined to ship on complexity grounds. Both rested on **one untuned operating point** — 60 epochs,
one architecture. Sweeping it changed the picture.

**The epoch curve is monotone DOWNWARD, which inverts the obvious intuition:**

| epochs | MLP alone (semiannual) | RankICIR |
|---|---|---|
| 10 | +0.052298 | 0.4404 |
| 15 | +0.054337 | 0.4816 |
| **20** | **+0.054466** | **0.5067** |
| 30 | +0.052566 | 0.5231 |
| 60 | +0.045600 | 0.4609 |
| 200 | +0.043646 | 0.4576 |

I had assumed the net was *under*trained and started sweeping capacity upward. The first result
killed that: (128,64) at 200 epochs scored **below** the same net at 60. It is **over**trained even
at 60 — the optimum is ~20 epochs, worth **+0.0089** over the shipped setting. The capacity sweep
was abandoned mid-flight rather than finished, because its premise was already falsified.

**And the blend gain doubles, on both fold sets:**

| MLP | semiannual delta | t | p | annual delta | t | p |
|---|---|---|---|---|---|---|
| e60 (original) | +0.001040 | +3.20 | 0.0014 | +0.000839 | +2.52 | 0.0119 |
| e30 | +0.001868 | +3.14 | 0.0017 | +0.001479 | +3.30 | 0.0010 |
| **e20** | **+0.001738** | **+3.92** | **0.0001** | **+0.001537** | **+3.38** | **0.0008** |

Three things make this credible rather than another noise excursion: **both fold sets are
significant**, **the optimal weight is 0.20 on both** (so it is not a cherry-picked knob), and the
effect size is **comparable to the L63D8sb3 switch that was shipped** (+0.0015/+0.0020,
p=0.0049/0.0005) — with *stronger* p-values. In score terms: +0.0016 RankIC x 8.60 = **+0.0138,
about 1.5 rank positions.**

**One simplification attempt failed, and it matters.** The dual preprocessing (`cs_rank` for the
net, `raw` for the trees) was the main complexity objection in F82/F83. If the tuned net worked on
`raw`, the inference change would collapse to ~20 lines on a matrix the GBM already builds. It does
not:

| mlp_e20 on `raw` | mean RankIC |
|---|---|
| semiannual | **+0.000326** |
| annual | **+0.000591** |

Essentially zero. The net needs standardised inputs, so **the dual preprocessing is unavoidable** —
the complexity objection from F82/F83 stands and must be paid, not engineered away.

**Also worth recording**: `parse_overrides` splits on commas, so a list-valued param such as
`hidden_layer_sizes=[128, 64]` **cannot be expressed on the command line** (it dies with
`--override entry '64]' is not key=value`). The sweep variants are therefore separate configs.

## F88. The submission landed — and it scored WORSE than the package we replaced

Submitted the frozen package at 2026-10-10 00:12 CST (hash verified first). Result:

| submission | RankIC | IR | s_rankic | s_ir | score |
|---|---|---|---|---|---|
| `95cc01f9` — submission #1 (LGBM + **CatBoost**) | 0.0712 | **2.8105** | 0.8497 | **0.5342** | **0.7550** |
| **`4690e013` — FROZEN package (LGBM-only 63/8)** | **0.0711** | **0.8139** | 0.8483 | **0.2775** | **0.6770** |
| `36fed7c3` — the exposure we just replaced | 0.0602 | 1.0785 | 0.7125 | 0.3115 | 0.5922 |

**The exposure is cleared** (0.5922 -> 0.6770, and the last submission is now a good artifact).
**But the frozen package scores 0.078 BELOW the package it superseded.**

**Read the decomposition, because it is the whole story:**

    0.7 * 0.8483 + 0.3 * 0.2775 = 0.6770      (frozen)
    0.7 * 0.8497 + 0.3 * 0.5342 = 0.7550      (with CatBoost)

The **RankIC term is essentially identical** (0.5938 vs 0.5948 — a 0.001 difference). The entire
0.078 gap is the **IR term**, which fell from 0.5342 to 0.2775.

**This is exactly the IR fragility documented earlier in this log, now costing real score.** Two
score series with the same mean RankIC (0.0823 in that experiment) produced engine IRs of 0.3061 vs
0.3402. Here the same thing happened on the board: RankIC matched to 0.0001 while IR differed by a
factor of **3.5**.

**What this means, and it is uncomfortable:** the decision to drop CatBoost was made on **fold
RankIC**, where CatBoost was a dead heat (paired p=0.95) and the LGBM-only config won by +0.0015 on
the paired daily test. That evidence was correct as far as it went — but **RankIC was the wrong
criterion for a 30%-weighted term**, and the board has now priced the difference at 0.078 score,
which is **~6x larger than the RankIC gain we were chasing**.

The log predicted this risk in the abstract ("IR is real but fragile; do not chase it") and then
walked into it by optimising the metric that does not carry the term. **The correct reading of the
fold evidence was not "CatBoost is neutral" but "CatBoost is neutral on RankIC and unknown on IR",
and unknown was resolved against us.**

**Open question for the next round**: whether the CatBoost package is still recoverable and should be
resubmitted (2 slots remain today, 14 across Oct 10-14). That is now the highest-value action —
higher than the MLP blend, whose measured gain (+0.0138 score) is **5.6x smaller than the 0.078
already lost**.

## F89. The CatBoost package is recoverable — and the case for resubmitting it does not hold up

F88 established that the frozen package scores 0.6770 against the superseded CatBoost package's
0.7550, and flagged resubmission as the highest-value action. Chasing that down:

**The artifact survived.** F35 kept the CatBoost blend at `/tmp/blend_submission` and it is still
there (F84 predicted `/tmp` would be cleared on reboot; it was not, this time). It is submission
#1's architecture exactly: 3x LightGBM at **31 leaves / depth 7** (the *old* config) at weight 1/6
each, plus **CatBoost seed 42** at weight 0.5. Re-zipped with the same builder and layout as the
shipped archive (9 entries, `submission/` prefix), **0 banned-token hits**, no file/network/
subprocess access, and the judge contract passes on the unpacked archive: **240/240 days**.

**Then a local head-to-head, which looked decisive and was not.** Running the same 2022 simulation
over both packages:

| package | mean_rankic | rankicir | positive_ratio |
|---|---|---|---|
| shipped 63/8 | **0.281179** | **2.2228** | 0.9833 |
| 31/7 + CatBoost | 0.168857 | 1.2403 | 0.8917 |

A 0.11 RankIC gap — apparently overwhelming. **It is worthless.** The models are trained through
**2022-12-31** and the simulation runs **2022-01-04..2022-12-30**: the entire evaluation window is
*inside the training data*. The 63/8 config has nearly twice the leaves, so it fits in-sample
better; that is all this measures. **A local simulation on a period the model was trained on cannot
compare two configs**, and I nearly acted on it. The correct use of `simulate_judge` is contract and
plumbing validation (which it does well), not model selection.

**With that removed, the evidence is a two-way disagreement:**

| | folds (11-18 walk-forward periods) | public board (1 out-of-sample period) |
|---|---|---|
| RankIC | 63/8 ahead by +0.0030 / +0.0027 | **tie** (0.0711 vs 0.0712) |
| IR | dead heat (paired p=0.9502) | CatBoost ahead by **2.00** |
| implied score | 63/8 ahead ~+0.025 (F35's arithmetic) | CatBoost ahead **+0.078** |

**Decision: keep the shipped 63/8 package; do not resubmit CatBoost.** The board's advantage rests
on a **single period's IR**, while the fold evidence spans 11-18 periods and puts the IR difference
at essentially zero (0.0014, p=0.95). F35 already computed that at the fold-measured gap the IR
term is worth **+0.00005** — the board's 2.00 gap is **1400x** that, which is the signature of a
single-period realization rather than a stable property. Resubmitting would trade a
better-evidenced RankIC advantage (70% of the score) for a worse-evidenced IR advantage (30%).

**What is genuinely learned:** the board's RankIC tie (0.0711 vs 0.0712) is itself informative — the
63/8 config's fold RankIC advantage of +0.003 **did not appear** on the public period, so the folds
are not a precise predictor either. Both estimates are noisy; the folds are simply the less noisy
one. And the honest summary of the whole episode is that **a 0.078 score swing came from the 30%
term we measured as neutral**, which is a warning about how much of this competition's outcome sits
in a quantity none of our evidence pins down.

## F90. Why our IR looks low: we were measuring it with costs the platform does not charge

The user flagged that our IR is far below the field's (ours 0.81-2.81; the board spans -1.243 to
6.4329). Investigating the engine rather than guessing produced four findings, one of which
invalidates a lot of earlier reasoning.

**1. IR is a portfolio metric, not a signal metric.**

    information_ratio = annualized_excess_return / tracking_error
    tracking_error    = std(active_return) * annualization

**2. The default optimizer is `top_k=50`, not barra.**

    selected = ranked_codes[:top_k]          # the top 50 by our score
    weight   = 1.0 / len(selected)           # equal-weighted at 2% each, ZERO elsewhere

So the engine builds a 50-name equal-weighted portfolio and measures it against HS300. **IR depends
on the HEAD of our ranking, not on the full cross-section.** Every IR number in this log before now
came from `C7_platform_guess` (barra) — a different portfolio construction entirely.

**3. The cost assumption was wrong, and it dominated everything.** `ir_lab.COSTS` charges
commission 0.00025 + slippage 0.001 = **0.125% per unit of turnover**. The engine's **default is
zero**. With turnover ~380-440 over the run, that assumption removed roughly **15 percentage points
of annual excess return**:

| config | optimizer | cost model | IR | excess | TE |
|---|---|---|---|---|---|
| 31/7+CatBoost | C5_topk50 | our COSTS | +0.4574 | +4.17% | 9.12% |
| 31/7+CatBoost | C5_topk50 | **ZERO** | **+2.0680** | **+18.92%** | 9.15% |
| 31/7+CatBoost | C7 barra | our COSTS | +0.3075 | +2.26% | 7.34% |
| 31/7+CatBoost | C7 barra | **ZERO** | **+2.5520** | **+18.79%** | 7.36% |

**Our local IR was understated by a factor of ~7.** The board's numbers (2.81, 0.81) sit in the
zero-cost range, which is consistent with the platform using the engine defaults. Every conclusion
in this log that rested on a *level* of local IR — including the "IR is fragile" observations — was
measured through a cost drag that the scorer does not apply. The *relative* comparisons survive
(costs are common to all configs), but the levels were meaningless.

**4. Locally, the two configs have the SAME IR — so F89's decision stands on stronger ground.**

| config | optimizer | IR (zero cost) | excess | TE |
|---|---|---|---|---|
| shipped 63/8 | C5_topk50 | +2.1301 | +19.68% | 9.24% |
| shipped 63/8 | C7 barra | +2.4698 | +18.46% | 7.48% |
| 31/7 seedbag3 | C5_topk50 | +2.2325 | +20.63% | 9.24% |
| 31/7 seedbag3 | C7 barra | +2.4856 | +18.55% | 7.46% |

The 63/8-vs-31/7 gap is **0.10 IR** locally. The board showed **2.00**. F89 declined to resubmit
CatBoost on the grounds that the board's IR gap was a single-period realization; the local engine
now agrees, and the decision is better supported than when it was made.

**5. The head is the strongest part of our signal — and this is the lever for IR.**

| config | full IC | top-50 return | universe return | head spread | top-50 IC |
|---|---|---|---|---|---|
| shipped 63/8 | +0.0796 | +0.19% | +0.01% | **+0.17%** | **+0.0363** |
| 31/7 seedbag3 | +0.0776 | +0.17% | +0.01% | +0.16% | +0.0318 |
| mlp_e20 alone | +0.0449 | +0.10% | +0.01% | +0.09% | +0.0145 |

Our top 50 beats the universe by **+0.17%/day** — the signal is genuinely concentrated at the head,
which is exactly where the `top_k=50` optimizer looks. This is the first direct measurement of the
quantity IR actually rewards, and it should be the target for any further improvement.

**6. The MLP blend helps BOTH scored terms, which I had doubted.** Because the net's head is much
weaker (+0.09% vs +0.17%), I expected the blend to buy full-IC at the cost of IR. The engine says
otherwise:

| config | C5_topk50 IR | excess | TE | rankIC |
|---|---|---|---|---|
| GBM alone | +1.7620 | +16.10% | 9.14% | +0.0645 |
| GBM + 0.15 MLP | +1.7976 | +16.40% | 9.12% | +0.0655 |
| **GBM + 0.20 MLP** | **+1.8683** | **+16.81%** | 9.00% | +0.0653 |

RankIC **and** IR both improve (TE falls too). In score units: RankIC +0.0008 -> +0.0069, IR +0.106
-> +0.0041, **total ~+0.011**. The blend is a net win on both terms, and the implementation is
justified on the evidence rather than on the full-IC number alone.

## F91. Retargeting at the head: a three-way blend worth ~+0.003 RankIC, and why the old target was wrong

F90 established that IR is driven by the **head** (top_k=50 equal-weighted) and that the board
applies **no transaction costs**. That reframes the objective: the thing to maximise is not
full-cross-section RankIC alone, it is

    0.70 * s_rankic  +  0.30 * s_ir        where ir depends on the top 50

So I built `tools/head_eval.py`, which scores a candidate on **both** terms in seconds — full
RankIC, the top-50 head spread, its t-statistic, and a top-100 IC — instead of the RankIC-only
screens used everywhere earlier in this log. Re-screening the members already on disk against the
head immediately produced a better answer than the MLP blend I had been about to ship.

**Shipped GBM alone:** RankIC +0.07961, head **+0.174%/day** (t=+11.42), top-50 IC +0.03626.

**Member screen (semiannual, weight on the added member):**

| member | RankIC | head/d | dRankIC | dhead |
|---|---|---|---|---|
| `lgbm_deep_csrank` w=0.25 | +0.08195 | +0.167% | **+0.00234** | −0.007% |
| `mlp_e30` w=0.15 | +0.08100 | +0.173% | +0.00139 | **−0.000%** |
| `mlp_e20` w=0.15 | +0.08056 | +0.167% | +0.00096 | −0.006% |
| `cat_sub1__fx` w=0.15 | +0.07973 | +0.172% | +0.00012 | −0.002% |
| `lgbm_rank_raw` w=0.15 | +0.07924 | **+0.175%** | −0.00037 | **+0.002%** |

Two things stand out: **`lgbm_deep_csrank` is a stronger RankIC contributor than the MLP** (a
different *recipe* beats a different *family* here), and **`mlp_e30` adds RankIC at literally zero
head cost** — the two are complementary, which the three-way sweep confirms.

**Three-way sweep (semiannual), baseline +0.08380:**

| blend | RankIC | head/d | dRankIC | dhead |
|---|---|---|---|---|
| **deep 0.15 + mlp 0.15** | **+0.08699** | +0.178% | **+0.00319** | **−0.001%** |
| deep 0.20 + mlp 0.15 | +0.08697 | +0.176% | +0.00317 | −0.003% |
| deep 0.15 + mlp 0.20 | +0.08697 | +0.178% | +0.00317 | −0.001% |
| deep 0.10 + mlp 0.20 | +0.08685 | +0.179% | +0.00305 | −0.000% |
| deep 0.00 + mlp 0.20 | +0.08564 | **+0.183%** | +0.00184 | **+0.004%** |

**`deep 0.15 + mlp 0.15` is the pick: +0.00319 RankIC for a head cost of 0.001 percentage points.**
That RankIC gain is **larger than the +0.0020 that justified the L63D8sb3 switch**, and unlike every
previous candidate it does not buy the 70% term at the expense of the 30% one. In score units the
RankIC term alone is +0.0032/0.0814 x 0.70 = **+0.0275**, with the IR term roughly neutral.

**The methodological point**: the entire search up to F90 optimised full-cross-section RankIC
because that is the headline metric. But RankIC is only 70% of the score, and the 30% term looks at
a **different part of the same ranking**. Screening on RankIC alone had been silently trading away
head quality — visible in the MLP blend, which bought +0.0016 RankIC for a measurable head loss.
**Once the screen matched the objective, a better candidate appeared within minutes from members
that were already on disk.**

## F92. The head-retargeted blend is confirmed on both fold sets AND on the engine

`deep 0.10 + mlp 0.10` was picked from the sweep as the most balanced point, then validated three
ways.

**Harness RankIC (the 70% term):**

| blend | semiannual | delta | annual | delta |
|---|---|---|---|---|
| shipped GBM alone | +0.08380 | — | +0.07702 | — |
| deep 0.15 + mlp 0.15 | +0.08699 | +0.00319 | +0.07891 | +0.00189 |
| **deep 0.10 + mlp 0.10** | **+0.08637** | **+0.00257** | **+0.07874** | **+0.00172** |
| deep 0.00 + mlp 0.15 | +0.08519 | +0.00139 | +0.07841 | +0.00139 |

**Head cost (the 30% term's input):** `deep 0.10 + mlp 0.10` is the only strong candidate whose head
spread does not fall on either fold set — **+0.004% on semiannual, −0.003% on annual**, i.e. flat.
`deep 0.15 + mlp 0.15` buys slightly more RankIC on semiannual but pays −0.011% of head on annual.

**Official engine (zero cost, `top_k=50` — the board's apparent configuration):**

| config | IR | excess | TE | rankIC |
|---|---|---|---|---|
| shipped GBM alone | +1.7620 | +16.10% | 9.14% | +0.0645 |
| **deep .10 + mlp .10** | **+1.9223** | **+17.47%** | 9.09% | **+0.0664** |
| deep .15 + mlp .15 | +1.8632 | +16.73% | 8.98% | +0.0667 |

**Both scored terms improve on the engine** — RankIC +0.0019 and IR +0.160. In score units:

    RankIC term: +0.0019 / 0.0814 * 0.70 = +0.0163
    IR term:     +0.160  / 7.6759 * 0.30 = +0.0063
    total                                = +0.0226      (~2.6 rank positions)

This is the first candidate in the entire log that improves **both** terms of the score, on **both**
fold sets, **and** on the official engine. Every earlier candidate bought one term with the other.

## F93. BUG: `--seeds` was silently ignored — a five-seed run used three seeds

Chasing why a 5-seed bagging run came back **bit-identical** to the 3-seed shipped config
(`0.08379950544016695` both times, delta `+0.00000`), I traced the seed flow.

`run_config_over_folds` built its fit arguments as:

    common = {"num_threads": num_threads}
    if seeds is not None:
        common["seeds"] = seeds                       # the CLI seeds
    selection_kwargs = {**common, "refit_full": False, **config}    # <-- config spreads LAST

Because `config` spreads **last**, the config's own `seeds: [42, 43, 44]` **overwrote** the CLI
value. `--seeds 42,43,44,45,46` therefore fitted **three** seeds, and `--seeds ...47,48` likewise —
while `report["seeds"] = seeds` recorded the CLI list, so the report *said* five. The run was a
no-op that looked like a successful experiment.

**This is the third instance of the same failure shape in this log** (F39's shallow `dict.update`,
F75's `--train-filter` no-op): a control that reports success while doing nothing, where the
evidence of "it ran" was the report rather than the result. The tell was identical each time —
**two configurations producing bit-identical output**, which for a stochastic ensemble is not a
plausible outcome but a contradiction.

**Fix**: spread `common` after `config` so the CLI wins, keeping `refit_full` last:

    selection_kwargs = {**config, **common, "refit_full": False}

**Verified non-invasive**: with no `--seeds`, the shipped config still reproduces
`mean_rankic = 0.08379950544016695` exactly, so every earlier number in this log stands.

**What it invalidates**: only the seed-count experiments I launched today — no shipped artifact, no
recorded finding, and no prior conclusion depended on a CLI `--seeds` value. The shipped model used
the config's own `seeds: [42, 43, 44]` and always did, so the artifact on the board is unaffected.

Worth noting for the record: this bug was found **only because the result was implausible**, not
because anything failed. Every check passed; the report was well-formed; the summary printed. The
same is true of F39 and F75.

## F94. Head-focused binary targets FAIL; score-shaping is weak; and a third silent no-op

**1. The leaderboard says IR is the whole game.** Top 12 all sit at IR 3.7-5.6 while we sit at
0.814 — but their RankIC (0.069-0.090) is comparable to ours (0.0711). Our full-cross-section
ranking is competitive; **the head is the binding constraint**. `docs/Problem.txt:304` confirms
`IR = Annualized Excess Return / Tracking Error`, and the optimizer is Barra-style with an active
risk penalty (so `C7_platform_guess` was the right guess, not `top_k=50`).

**2. Training directly for the head makes it WORSE.** I added two binary targets — "is this name in
the day's top 30% / top 10%?" — with everything else held at the shipped recipe:

| model | RankIC | head/d | top-50 IC |
|---|---|---|---|
| shipped GBM | **+0.08380** | **+0.179%** | **+0.03318** |
| `lgbm_top30` (binary, top 30%) | +0.03036 | +0.153% | +0.02932 |

Both terms collapse. The binary target throws away the magnitude information that makes the
regression work, and the resulting probability ranks the head no better — worse, in fact. A clean
negative: **"optimise the head" is not achievable by asking the model a head-shaped question.**

**3. Score-shaping barely moves IR, and confirms the ordering-only contract.** Sweeping shape
transforms through the Barra optimizer (zero cost):

| shape | IR | excess | TE | rankIC | failed orders |
|---|---|---|---|---|---|
| `raw` | +2.1018 | +15.57% | 7.41% | +0.0645 | 112 |
| `rank` | **+2.1018** | +15.57% | 7.41% | +0.0645 | 112 |
| `clip1_99` | +2.0704 | +15.35% | 7.41% | +0.0645 | 114 |
| `clip5_95` | **+2.1730** | +15.81% | 7.28% | +0.0644 | 129 |

`raw` and `rank` are **bit-identical**, which is the contract ("only ordering reaches the
optimizer") confirmed end-to-end. Clipping reorders only through ties and buys at most +0.07 IR —
real but small, and `clip1_99` is *worse* than raw, so it is not a reliable lever either.

**4. A THIRD silent no-op, caught by its own output.** The neutralisation rows return results
*identical to raw* — not because neutralisation does not help, but because it never ran:
`barra cols available: []`. The prediction parquets carry only `日期/代码/score/future_return`, so
`sub[cols]` was empty, `X.shape[1] == 0`, and my loop `continue`d on every day. The tell was again
**two configurations producing identical numbers** — the same signature as F39, F75 and F93. The
neutralisation question remains genuinely untested; it is not a negative result.

**Where this leaves the head problem:** the model's own head is the strongest part of our signal
(+0.179%/day) and I have not found a way to improve it — binary targets hurt, shaping is weak,
neutralisation is untested. The remaining untried levers are the blend that already measured +0.023
score (F91/F92), and a genuinely stronger base model.

## F95. CORRECTION: the CatBoost package's IR advantage is REAL and reproducible — F89 was wrong

Resubmitted the recovered CatBoost package (1 slot) to test F89's conclusion directly. Result:

| submission | RankIC | IR | s_rankic | s_ir | score | rank |
|---|---|---|---|---|---|---|
| `4690e013` LGBM-only 63/8 | 0.0711 | **0.8139** | 0.7776 | 0.3020 | 0.6350 | #73 |
| `95cc01f9` 31/7 + CatBoost | 0.0712 | **2.8105** | 0.7790 | 0.5951 | 0.7238 | — |
| **`d836c57f` 31/7 + CatBoost (resubmit)** | **0.0714** | **3.2786** | 0.7817 | 0.6639 | **0.7464** | **#35** |

**RankIC is identical across all three (0.0711 / 0.0712 / 0.0714). The entire 0.111 score gap is
the IR term.** And the CatBoost package's IR reproduced across two independent submissions
(**2.81** then **3.28**) against the LGBM-only's **0.81**.

**F89 was wrong, and the way it was wrong is the point.** F89 declined to resubmit on the grounds
that "the local engine shows the two configs have the same IR" (2.1301 vs 2.2325 — a 0.10 gap). The
board shows a **2.0-2.5** gap. So:

> **Our local engine clone does NOT reproduce the board's IR.** Every IR-based conclusion in this
> log inherited that error — including F90's "our IR was understated 7x by costs", which was
> measured on the same unreliable proxy.

**This is also why F35 was wrong**, and it is the same failure mode one level up. F35 computed that
the IR gap was worth `+0.00005` score using the **fold** gap of 0.0014, and used that to reject the
CatBoost blend. But the board's IR gap is **1400x larger**. F35 took a quantity measured on one
engine and priced it as if it held on another. **Every "IR is fragile, don't chase it" conclusion in
this log rests on a proxy that does not track the target.**

**The practical lesson, which cost one slot to learn:** when a proxy and the target disagree, the
target decides. The local engine remains useful for RankIC (which it does reproduce) and for
contract validation; it is **not** evidence about board IR.

**New standing hypothesis, now with direct support:** cross-family diversity raises IR by smoothing
the active-return series, which is exactly the effect F35 measured as ~zero on the fold engine and
dismissed. CatBoost is one such family. The MLP is another (F82 measured genuine decorrelation,
Spearman 0.34). **The next candidates should add diversity to the CatBoost base, and be judged on
board IR — not on the fold engine.**

## F96. CatBoost is the IR driver, and it stacks — score 0.7526, rank #33

Built "63/8 + CatBoost" by running the **shipped** `train.py` with `--catboost-seeds 42`
(the two `train.py` files turned out to be the same script with different defaults, and the shipped
`inference.py` already dispatches `catboost` members — so no code change was needed). Result:

| submission | LGBM cfg | CatBoost | RankIC | **IR** | score | rank |
|---|---|---|---|---|---|---|
| `4690e013` | 63/8 | no | 0.0711 | **0.8139** | 0.6350 | #73 |
| `36fed7c3` | 63/8 (earlier) | no | 0.0602 | 1.0785 | 0.5572 | — |
| `95cc01f9` | 31/7 | yes | 0.0712 | **2.8105** | 0.7238 | — |
| `d836c57f` | 31/7 | yes | 0.0714 | **3.2786** | 0.7464 | #35 |
| **`70f5adc9`** | **63/8** | **yes** | 0.0713 | **3.4477** | **0.7526** | **#33** |

**Two clean readings:**

1. **CatBoost is the IR driver.** Every CatBoost submission sits at IR 2.8-3.4; every LGBM-only
   submission sits at 0.81-1.08. It reproduces across four submissions. And it **stacks**: the
   better LGBM recipe (63/8) on top of CatBoost gives the best score so far, 0.7526.

2. **Our fold RankIC gains are NOT transferring to the board.** The 63/8 config beats 31/7 by
   +0.0030/+0.0027 in the folds, yet the board returns 0.0713 vs 0.0712 — a tie. Every one of our
   submissions lands at RankIC 0.0711-0.0714 regardless of which recipe produced it. **The board's
   RankIC appears pinned near 0.071 for this feature set**, which means the whole fold-driven
   hyper-parameter search (F67, F72, F74, F91, F92) bought real fold RankIC that the scorer does not
   see. That is worth knowing before spending more effort on that axis.

**Why CatBoost raises IR, mechanistically.** IR = mean(active)/std(active). CatBoost alone is
*weaker* (RankIC 0.0610, local IR -0.2063) yet at weight 0.5 it quadruples the blend's IR while
leaving RankIC unchanged. The only way that happens is **averaging two decorrelated signals lowers
the volatility of the active-return series without lowering its mean** — exactly the diversification
effect F35 measured as ~zero *on the fold engine* and dismissed. **The board engine sees it; ours
does not.**

**Therefore the next lever is more decorrelated members**, not more hyper-parameter tuning. The MLP
is the obvious candidate: F82 measured its decorrelation against the GBM at Spearman **0.34**, far
below the 0.9986 of same-family ensembles. The work is to add `cs_rank` + a small numpy forward pass
to `inference.py` (the MLP needs normalised inputs — F80), which is the complexity F82/F83 declined
to pay when the measured gain was only +0.011. **The CatBoost result changes that arithmetic**: if a
second diverse family repeats even half of CatBoost's IR effect, it is worth several times what the
implementation costs.

## F97. Built the third model family: MLP wired into the shipped inference path

Acting on F96's mechanism (IR responds to cross-family diversity), I added the MLP as a third
member. The work, in order:

**1. Trained the net on ALL public data** (`tools/train_mlp_member.py`): 499,799 rows, 136 features,
1,701 days, `MLPRegressor(hidden=(128,64), max_iter=30, alpha=1e-4, seed=42)` on the `rank` target
with `cs_rank`-normalised inputs. **30 epochs, not 60** — F87 showed the epoch curve is monotone
downward and the optimum sits at ~20-30.

**2. Exported the weights as `.npz`, not a pickled object.** `.npz` is a zip of `.npy` buffers, so
nothing runnable can ride along; the loader uses `allow_pickle=False`. This is what makes an sklearn
model shippable at all, given §9.3 bans serialisation formats that can carry payloads.

**3. Self-checked the forward pass**: the hand-rolled numpy relu forward pass reproduces
`sklearn.predict` with `max |diff| = 0.000e+00` on all 499,799 training rows.

**4. Added an `mlp` member kind to `inference.py`**, with a `_cs_rank` helper that reproduces
`FeatureProcessor(mode="cs_rank")` exactly (`rank(pct=True) - 0.5`, then NaN -> 0). The trees keep
the raw block; the net gets the ranked block. **This is the dual preprocessing F82/F83 declined to
pay for when the measured gain was +0.011 — the CatBoost result changed that arithmetic.**

**5. Member weights: LightGBM 0.5 (3 x 1/6) + CatBoost 0.30 + MLP 0.20.**

Verification: **0 banned tokens**, 1154 lines, `allow_pickle=False` present, judge contract
**240/240 days** on the unpacked archive. In-sample full-year RankIC 0.2396 (vs 0.2812 for
LGBM-only) — *lower*, as expected, since CatBoost and the net dilute the trees' in-sample fit. That
number is not evidence either way; F89's lesson was that in-sample local comparisons of this kind
cannot rank configs.

**Armed for 2026-10-11 00:00:30 with a pre-registered decision rule**, so the interpretation is not
chosen after seeing the result:

| outcome | reading | action |
|---|---|---|
| IR > 3.45 and score > 0.7526 | the third family helps | add more diverse families |
| IR 3.0-3.45 | neutral | the diversity hypothesis is not confirmed; look elsewhere |
| IR < 3.0 | the MLP hurts | revert to 63/8+CatBoost and resubmit it |

## F98. Candidate inventory for the remaining slots

The board's IR responds to cross-family diversity (F96), and our local engine cannot rank
candidates by it (F95), so **the only way to evaluate a candidate is to spend a slot**. That makes
it worth preparing several in advance and spending slots in order of how much each would teach us.

Built and validated (all: 0 banned tokens, judge contract passes, shipped inference path):

| candidate | members | archive sha256 | in-sample contract RankIC |
|---|---|---|---|
| **3-way** (armed @ 00:00) | LGBM 63/8 x3 @0.5, CatBoost x1 @0.3, MLP @0.2 | `949a97ce` | 0.2512 (20d) |
| **B** | LGBM 63/8 x3 @0.5, **CatBoost x3** @0.3, MLP @0.2 | `c1477382` | 0.2696 (15d) |
| **C** | LGBM 63/8 x3 @0.5, **CatBoost x3** @0.5 | `3de303bb` | 0.2411 (15d) |

B and C differ from the armed candidate in **one** respect each, so the results decompose:

* **3-way vs B** isolates *CatBoost seed-averaging* (1 seed -> 3) at fixed weights.
* **B vs C** isolates *the MLP* at fixed CatBoost (3 seeds) — the cleanest available test of
  "does a third family help", which is the hypothesis the whole direction rests on.

In-sample contract RankIC is reported only for completeness; F89 established that local
comparisons of this kind cannot rank configs, so **no decision will be based on it**.

Tooling added: `tools/make_candidate.py` assembles a package from a trained model dir, the shared
code, and the MLP member, renormalising member weights — so a new candidate is a one-line build
rather than a hand-assembled directory.

## F99. External reference: Qlib's official CSI300 benchmarks, and where we stand against them

Following the user's request to check open-source references, I pulled Qlib's published benchmark
table (CSI300, 20 runs per model, `examples/benchmarks/README.md` v0.6.3). Alpha158 is the closest
analogue to our 136-column library:

| model | IC | ICIR | Rank IC | Rank ICIR | Ann. Return | **IR** | Max DD |
|---|---|---|---|---|---|---|---|
| Linear | 0.0393 | 0.2980 | 0.0475 | 0.3546 | 0.0795 | **1.0712** | -0.1449 |
| CatBoost | 0.0503 | 0.3586 | 0.0483 | 0.3667 | 0.1080 | **1.1561** | -0.0787 |
| XGBoost | 0.0481 | 0.3659 | 0.0495 | 0.4033 | 0.1111 | **1.2915** | -0.0893 |
| **LightGBM** | 0.0475 | 0.3979 | **0.0485** | **0.4123** | 0.1143 | **1.2744** | -0.0800 |
| MLP | 0.0358 | 0.2738 | 0.0425 | 0.3221 | 0.0836 | **1.0323** | -0.1127 |
| TabNet + pretrain | 0.0344 | 0.2050 | 0.0398 | 0.3479 | 0.0827 | 1.1141 | -0.0925 |

**Three readings, and one of them is a warning about the MLP I just shipped:**

1. **We are well above the published benchmark.** Qlib's best CSI300 IR is **1.29**; we are at
   **3.45**. Different period, universe construction and cost model, so this is not a like-for-like
   comparison — but it does say our pipeline is not behind the open-source state of the art on this
   task. The field's leaders (IR 4.9-5.6) are above it too.

2. **LightGBM is the right base, and CatBoost/XGBoost are close behind.** Rank IC 0.0475-0.0495 for
   all three; the linear model is only slightly worse (0.0475), which says most of the signal is
   fairly shallow. This independently supports the GBM-centric choice.

3. **The MLP is the weakest tabular model in their table** (Rank IC 0.0425, IR 1.0323 — below even
   Linear). That is exactly what we measured locally (F80: MLP RankIC +0.0456 vs GBM +0.0838; F87:
   the epoch curve peaks at ~20). **So the third family I built is the one external evidence says is
   least likely to help**, and the pre-registered rule armed at 00:00 may well fire the "revert to
   63/8+CatBoost" branch. That is a useful prior to have *before* reading the number.

**The genuinely new lead from the table is the sequence models.** On Alpha360 (raw price/volume
sequences rather than engineered factors), GRU / ALSTM / GATs reach Rank IC 0.054-0.059 and IR
0.67-1.15, clearly above the engineered-factor tabular models on the same dataset. **Temporal
structure is where the remaining edge appears to be** — but `torch` is not installed in this
environment and a sequence model would need a new input pipeline (the panel would have to be
reshaped into per-stock windows), so this is a multi-hour direction, not a slot-away one. It is
recorded here as the most promising untried avenue if the diversity route stalls.

## F100. Added XGBoost as a fourth family, on the strength of the external IR evidence

F99's Qlib table ranks XGBoost **top of the tabular models on realized Information Ratio**
(1.2915 vs LightGBM 1.2744, CatBoost 1.1561, MLP 1.0323) at comparable Rank IC. Since the board's
IR is the binding constraint (F94/F96) and it responds to cross-family diversity, XGBoost is the
best-motivated next member — much better motivated than the MLP, which the same table puts *last*.

**Implemented in both halves of the pipeline:**

* `train.py`: `_xgboost()` fitter (lr 0.03, depth 6, min_child_weight 20, subsample 0.80,
  colsample 0.70, lambda 10.0, `tree_method="hist"`), plus `--xgboost-seeds`,
  `--xgboost-rounds` and `--xgboost-weight`. The group-weight logic now splits
  `1 - catboost_weight - xgboost_weight` across the LightGBM seeds.
* `inference.py`: an `xgboost` member kind, loading a **JSON** booster — plain text, no runnable
  payload, same reasoning as the `.npz` MLP.

Both files stay at **0 banned tokens** (1165 and 377 lines). Judge contract passes with the
XGBoost member active.

**Candidate inventory for the remaining slots** (all validated, all 0 banned tokens):

| candidate | members | archive sha256 | contract RankIC |
|---|---|---|---|
| **3-way** (armed @ 00:00) | LGBM .50, CatBoost .30, MLP .20 | `949a97ce` | 0.2512 (20d) |
| B | LGBM .50, **CatBoost x3** .30, MLP .20 | `c1477382` | 0.2696 (15d) |
| C | LGBM .50, **CatBoost x3** .50 | `3de303bb` | 0.2411 (15d) |
| **D** | LGBM .40, CatBoost .35, **XGBoost** .25 | `9798c7e1` | 0.2526 (15d) |
| **E** | LGBM .35, CatBoost .30, **XGBoost** .20, MLP .15 | `a1ff31dd` | — |

The slots now buy information, not just score: **3-way vs B** isolates CatBoost seed-averaging,
**B vs C** isolates the MLP, **D vs the 3-way** isolates XGBoost, **E vs D** isolates the MLP again
at four families. Two of those five candidates must lose, and the rule armed at 00:00 decides the
first one without hindsight.

## F101. Added a ridge member — a fifth family, at zero inference cost

Qlib's table (F99) puts **Linear at Rank IC 0.0475 against LightGBM's 0.0485** — on the same
features, a linear model is nearly as good a *ranker* as the trees while being a completely
different function class. For a metric that rewards cross-family diversity that is a better
prospect than the MLP (Rank IC 0.0425).

**It cost no inference code at all.** A ridge model is one weight matrix plus a bias, which is
exactly a 1-layer instance of the network format already implemented: `_mlp_scores` applies relu
*between* layers and never to the output, so a single-layer npz evaluates to a plain linear map.
`tools/train_linear_member.py` fits `Ridge(alpha=100.0)` on the same `cs_rank` inputs and exports
`coefs_0` / `intercepts_0` / `n_layers=[1]`, self-checking the forward pass at
`max |diff| = 0.000e+00`.

**One trap this created, caught before it bit.** Both the MLP and the ridge member carry kind
`mlp`, so `make_candidate.py`'s kind-keyed weighting would have silently split a single `mlp`
weight between them — the same "two things quietly sharing one slot" shape as F39/F75/F93. The
assembler now keys MLP-family members by **file stem** (`mlp_member` vs `linear_member`).

**Candidate F — five families**, contract passes (15d RankIC 0.2577), 0 banned tokens, 12 files,
sha256 `d8a4b1c0`:

| member | weight |
|---|---|
| LightGBM 63/8 x3 | 0.35 (0.1167 each) |
| CatBoost | 0.25 |
| XGBoost | 0.20 |
| MLP | 0.10 |
| Ridge (linear) | 0.10 |

**The full candidate set now spans five function classes**, and every adjacent pair in the
inventory isolates exactly one factor — so the remaining slots each buy a measurement rather than a
lottery ticket.

## F102. The board evaluates on data AFTER 2022 — our fold validation was never out-of-sample

Reading `docs/Problem.txt` closely for the first time on this point:

> 组织方向参赛者公开提供 2016–2022 年训练数据 … **在 2016–2022 年之外，组织方另外保留未来
> 市场数据作为隐藏评测集**，并预先划分为 Public Leaderboard 与 Private Leaderboard 两部分。
> （lines 64-83）

**The hidden evaluation set is post-2022 data.** Our public data ends 2022-12-30, so every board
score we have ever received was computed on **2023 or later**.

**This reframes a great deal of this log.** The walk-forward folds that drove the entire
hyper-parameter search (F67, F72, F74, F91, F92) run **inside 2017-2022 — i.e. inside the training
range of the final model**. They measure how well a recipe interpolates the period it is fitted on,
which is not the quantity the board scores. Specifically:

* A model fitted through 2022-12-31 and scored on 2023+ is being tested on a **different market
  regime**, not on held-out data from the same regime.
* The fold RankIC of +0.0838 and the board's +0.0713 differ by ~15%. **That gap is the
  regime-transfer penalty, and it is the thing worth optimising** — not the in-fold number.
* It explains why the board's RankIC is pinned near 0.071 across recipes whose fold RankIC differs
  by 0.003: **the fold differences are interpolation differences, and they do not survive the
  regime shift.**

**Actionable consequence**: candidates should be judged on *robustness* rather than on fold fit.
Ensembling across families (F96) is exactly such a robustness mechanism — it does not sharpen the
fit to 2017-2022, it reduces the variance of the signal — which is consistent with it being the
only lever that has actually moved the board. Any future single-model tuning should be suspect for
the same reason.

This also raises the value of the pre-registered decision rule armed at 00:00: it tests a
*variance-reduction* mechanism, which is the class of change this finding says should transfer.

## F103. Correcting my own read: RankIC is now a real gap, not just IR

Board re-pull (141 entries, us at **#32, score 0.7538**):

| # | user | RankIC | IR | score |
|---|---|---|---|---|
| 1 | Summer | 0.0900 | 4.924 | 0.9716 |
| 2 | sh | 0.0832 | 5.568 | 0.9446 |
| 3 | kai | 0.0799 | 5.257 | 0.9040 |
| 5 | 花椰菜 | 0.0829 | 4.274 | 0.8852 |
| **6** | **老鱼** | **0.0815** | **3.655** | **0.8465** |
| **32** | **Azuma (us)** | **0.0713** | **3.448** | **0.7538** |

After the CatBoost work, **our IR is no longer the main deficit**: #6 老鱼 sits at IR 3.655 against
our 3.448, essentially level, while their **RankIC is 0.0815 against our 0.0713**. So the remaining
0.093 of score between us and #6 is mostly the **RankIC** term.

**Combined with F102, this is a coherent story rather than two separate gaps.** The board scores
2023+; our folds sit inside 2017-2022. Our fold RankIC is +0.0838 and the board returns +0.0713 —
a **15% regime-transfer penalty**. #6 achieves 0.0815 on the same post-2022 data, which is close to
our *in-fold* number, i.e. **their models barely degrade across the regime shift and ours do.**

**I had been treating "IR is the whole game" as the conclusion. It was the conclusion for the
LGBM-only submissions (IR 0.81) but the CatBoost work closed that gap, and what remains is the
RankIC term — which the fold-driven search cannot address, because it optimises interpolation of
the very period the final model trains on.**

The next candidates therefore target **transfer, not fold fit**: lower capacity, stronger
regularisation, and ensembling (which reduces signal variance rather than sharpening the fit).

## F104. Candidate G: a deliberately LOWER-capacity model, to test the transfer hypothesis

F102/F103 say the remaining gap is a **regime-transfer penalty** (fold +0.0838 -> board +0.0713,
15%), not fold fit. Every tuning decision in this log went the other way — it chased fold RankIC,
which is interpolation of the period the final model trains on.

Candidate G is the opposite bet, built with the same member mix as D so the comparison isolates
capacity alone:

| | candidate D (current recipe) | candidate G (this) |
|---|---|---|
| LightGBM | 63 leaves, depth 8, l2 10, min_data 200 | **31 leaves, depth 6, l2 50, min_data 500** |
| members | LGBM .40 / CatBoost .35 / XGBoost .25 | identical |
| in-sample contract RankIC (15d) | 0.2526 | **0.2211** |

**In-sample RankIC is deliberately worse** — that is the entire point. A model that fits 2017-2022
less tightly has less to unlearn when the regime shifts. sha256 `5e75d42a`, 10 files, 0 banned
tokens, contract passes.

**This is a genuinely falsifiable bet, and it is the first candidate in the log that trades fold fit
for robustness rather than the reverse.** If G beats D on the board, the transfer hypothesis is
supported and every fold-tuned decision should be revisited. If D beats G, then capacity was never
the transfer problem and the 15% penalty has some other cause — which would itself be worth
knowing, because it would mean the fold search was never as misdirected as F102 suggests.

**Full validated candidate inventory** (7 candidates, all 0 banned tokens, all contract-passing):

| id | members | sha256 |
|---|---|---|
| 3-way (armed @00:00) | LGBM63/8 .50, CatBoost .30, MLP .20 | `949a97ce` |
| B | LGBM63/8 .50, CatBoost x3 .30, MLP .20 | `c1477382` |
| C | LGBM63/8 .50, CatBoost x3 .50 | `3de303bb` |
| D | LGBM63/8 .40, CatBoost .35, XGBoost .25 | `9798c7e1` |
| E | LGBM63/8 .35, CatBoost .30, XGBoost .20, MLP .15 | `a1ff31dd` |
| F | LGBM63/8 .35, CatBoost .25, XGBoost .20, MLP .10, Ridge .10 | `d8a4b1c0` |
| G | **LGBM31/6 .40**, CatBoost .35, XGBoost .25 | `5e75d42a` |

## F105. Compliance audit against the stated constraints, and candidate H

**Model classes are explicitly permitted.** `docs/Problem.txt:55` names "线性模型、支持向量机、
Random Forest、XGBoost、LightGBM、Learning-to-Rank"; `:73` allows "Ensemble 权重" in the submitted
artifact. So the four-family ensemble is squarely inside the rules, and `:174` even suggests
"使用 XGBoost Ranker、LightGBM LambdaRank 或其他排序学习方法直接优化截面顺序" as a legitimate
approach (we measured learning-to-rank as worse — F-noted, 0.0219 vs 0.0838 — but it is allowed).

**The 120-trading-day look-back limit is satisfied exactly, not approximately.** `docs/Problem.txt:208`
allows `X = {x_{t-L+1}, ..., x_t}` with `1 <= L <= 120`. Auditing the 136 frozen features:

* The only occurrence of "300" is **`beta_000300SH_22`** — the CSI 300 index *code*, not a window.
  It would have been easy to mistake for a 300-day window and either "fix" a non-problem or miss a
  real violation.
* After removing the index code, the maximum window-like suffix across all features is **120**.
  Nothing exceeds the limit.

**Candidate H — maximal diversification**, 11 members across 5 function classes:

| family | members | total weight |
|---|---|---|
| LightGBM 63/8 | 3 (seeds 42/43/44) | 0.35 |
| CatBoost | 3 (seeds 42/43/44) | 0.25 |
| XGBoost | 3 (seeds 42/43/44) | 0.20 |
| MLP | 1 | 0.10 |
| Ridge | 1 | 0.10 |

sha256 `899df671`, 16 files, 5.88 MiB, 0 banned tokens, contract passes (15d RankIC 0.2568).

H is the endpoint of the diversity hypothesis: **every family is seed-averaged and every family is
represented**, so if variance reduction of the active-return series is what raises the board's IR,
H should be the best of the eight. If H does *not* beat D/F, the diversity effect has saturated and
the remaining IR gap needs a different explanation.

## F106. A transfer proxy that reproduces the board's degradation, and what it says about capacity

F102 established that the board scores post-2022 data while our folds sit inside 2017-2022. To make
that testable locally I added a `longholdout` fold set — **train <= 2020-12-31, test the whole of
2021-2022 (483 days)** — which is the closest available analogue of "train through T, score a long
later period". The existing folds cannot see this: their test windows are short and sit immediately
after the training cut.

**It reproduces the board's degradation almost exactly:**

| | RankIC on short adjacent windows | RankIC on the long holdout | drop |
|---|---|---|---|
| our folds vs the board | +0.0838 (semiannual) | **+0.0713 (actual board)** | **-15%** |
| annual folds vs longholdout | +0.0770 (annual) | **+0.0615 (longholdout)** | **-20%** |

Same magnitude, same direction. **`longholdout` is a materially better proxy for the scored metric
than the folds the whole search was built on**, and it costs ~4 minutes to run.

**What it says about capacity — the transfer hypothesis is half right.** I added `lgbm_lowcap`
(31 leaves / depth 6 / l2 50 / min_data 500) as a deliberate low-capacity probe:

| config | longholdout | annual (short) |
|---|---|---|
| `lgbm_sub1_l63d8` (63 leaves, depth 8) | **+0.061471** | **+0.077019** |
| `lgbm_lowcap` (31 leaves, depth 6) | **+0.061347** | +0.074803 |
| gap | **+0.00012** | **+0.00222** |

**The capacity advantage shrinks by 18x under a long holdout** — from +0.0022 to +0.0001. So the
fold-derived preference for the bigger model is largely an artifact of short adjacent test windows,
which is exactly what F102 predicted.

**But the low-capacity model does not actually transfer *better*** — the two are tied. So the 15-20%
regime penalty is **not** caused by capacity, and candidate G's premise is only half supported:
reducing capacity removes an illusory advantage without producing a real one. That is worth knowing
before spending a slot on G, and it means **G should be deprioritised** in favour of the diversity
candidates (D/F/H), whose mechanism — variance reduction of the active-return series — is the one
that has actually moved the board.

**Also the FOURTH silent no-op, caught the same way as the other three.** My first low-capacity run
came back **bit-identical** to the shipped config (`0.061470797881860904`, 18 digits). Cause:
`--override num_leaves=31` writes a **top-level** key, but the fitter reads `config["params"]
["num_leaves"]`, so the override was recorded in the report and ignored by the model. The tell was
again **two configurations producing identical output** — F39, F75, F93, and now this. Fixed by
adding a real config entry rather than an override.

## F107. Calibrating the proxy: `longholdout` predicts the board's RankIC but NOT its IR

Ran the blends on the transfer proxy to see whether the diversity that helped the board also helps
under a long holdout.

| config | longholdout RankIC | longholdout IR | board RankIC | **board IR** |
|---|---|---|---|---|
| `lgbm_sub1_l63d8` (63/8, no CatBoost) | **+0.061471** | +0.4464 | 0.0711 | 0.814 |
| `lgbm_lowcap` (31/6) | +0.061347 | +0.4353 | — | — |
| `sub1_arch_seedbag3` (31/7 + **CatBoost**) | +0.060509 | +0.4178 | 0.0712 | **2.81** |
| `blend_lgbm_cat_xgb` (cs_rank 3-family) | +0.052125 | +0.3772 | — | — |

**Two clean calibrations, one positive and one negative:**

**Positive — `longholdout` tracks the board's RankIC.** It ranks the LGBM-only config (+0.0615) just
above the CatBoost blend (+0.0605), and the board agrees that their RankIC is a tie (0.0711 vs
0.0712). It also reproduces the *magnitude* of the degradation (F106: -20% local vs -15% board).
**This is the proxy to use for any RankIC-targeted candidate.**

**Negative — it does NOT track the board's IR, and it is confidently wrong.** The proxy puts the
CatBoost blend's IR at 0.418 against the LGBM-only's 0.446 — i.e. **CatBoost is worse** — while the
board puts it at **2.81 against 0.81**, i.e. **3.5x better**. The sign is wrong, not just the scale.

**So the local engine's IR is not a noisy version of the board's IR; it is a different quantity that
happens to share a name.** This closes the question opened in F95 with a direct measurement rather
than an inference, and it means:

* **RankIC candidates can be screened locally** on `longholdout` — cheap, and now validated.
* **IR candidates cannot be screened locally at all.** The only instrument is a board slot. That is
  why the candidate inventory (F104) was built to make each slot isolate one factor, and why the
  diversity direction must be probed by submission rather than by search.

`blend_lgbm_cat_xgb` at +0.0521 is the worst of the four on the long holdout — notably it is the one
config built on `cs_rank` preprocessing, consistent with F74's finding that `cs_rank` costs the
shipped recipe 0.013-0.016.

## F108. Screening on the validated proxy: CatBoost at a LIGHT weight beats both extremes

Now that `longholdout` is calibrated for RankIC (F107), the candidate space can be searched locally
instead of by submission. Screening the whole config library on it:

| config | longholdout RankIC |
|---|---|
| **`blend_63d8_cat35`** (63/8 + CatBoost 0.35) | **+0.061900** |
| **`blend_63d8_cat20`** (63/8 + CatBoost 0.20) | **+0.061859** |
| `lgbm_sub1_l63d8` (shipped, no CatBoost) | +0.061471 |
| `lgbm_lowcap` (31/6) | +0.061347 |
| `sub1_arch_seedbag3` (31/7 + CatBoost 0.5) | +0.060509 |
| `lgbm_rank_raw` | +0.056553 |
| `lgbm_deep_csrank` | +0.052418 |
| `blend_lgbm_cat_xgb` (cs_rank 3-family) | +0.052125 |
| `mlp_e20` | +0.041004 |
| `mlp_e30` | +0.034827 |

**Three conclusions, and the first one corrects an earlier read:**

1. **CatBoost does NOT cost RankIC — the 31/7 base did.** F107 read `sub1_arch_seedbag3`'s +0.0605 as
   evidence that CatBoost dilutes RankIC. Adding the 63/8 base isolates it: **63/8 + CatBoost at 0.20
   or 0.35 is *better* than 63/8 alone** (+0.0619 vs +0.0615). The earlier read conflated two
   changes. This is the same error shape as F95, caught this time before it cost a slot.

2. **The shipped 63/8 base is the strongest RankIC transferrer in the library**, and the MLP is by
   far the weakest (+0.041, +0.035) — independently confirming Qlib's table (F99), which puts MLP
   last among tabular models. **The armed 3-way candidate carries a 0.20 MLP weight, and on this
   evidence that is likely to cost RankIC.** The pre-registered rule will show whether the IR gain
   compensates; if it does not, the MLP comes out.

3. **Everything built on `cs_rank` transfers worse** (`lgbm_deep_csrank` +0.0524,
   `blend_lgbm_cat_xgb` +0.0521), consistent with F74's measurement that `cs_rank` costs the
   shipped recipe 0.013-0.016 on the ordinary folds. The penalty is *larger* on the transfer proxy,
   which is a new and useful detail.

**Candidate I — `blend_63d8_cat35` realised**: LightGBM 63/8 x3 at 0.65 + CatBoost x3 at 0.35.
This is the first candidate selected on the *validated* proxy rather than on the folds, and it is
the best-measured RankIC transferrer of anything built so far while still carrying the family that
produced the board's IR jump.

## F109. Closed a durability risk before it could cost the submission

Every candidate archive lived in `/tmp`, and **F84 established that `/tmp` is volatile in this
environment**. The timer armed for the 00:00 slot pointed at `/tmp/cat63mlp.zip` — so a `/tmp`
cleanup between now and midnight would have silently destroyed the submission rather than failing
loudly at build time.

**Fixed**: all 9 candidates copied to `candidates/` inside the workspace (durable), the armed
candidate's hash re-verified there (`949a97ce…` matches exactly), and the timer **deleted and
re-created** pointing at the durable path. Deleting first matters — a stale timer carrying a
superseded hash is a hazard this session already hit once.

**The re-armed prompt also carries the corrected decision rule.** The original rule was written
before F106-F108 existed and encoded the belief that "IR is the lever, RankIC is not" — which F103
already partly overturned and F108 refined further. The replacement states:

* the archive path and exact hash to verify,
* the three-way pre-registered rule keyed on IR and score,
* **which proxy is trustworthy for what** — `longholdout` for RankIC (validated), nothing for IR
  (measured wrong, sign included),
* the fallback candidate for each branch (H/D if the MLP helps, **I** if it hurts),
* and the operational failure modes (quota vs auth, `release` after a half-finished submit, TLS
  retries, `--resume` needing `--yes`).

**Nine validated candidates, all durable**: `cand_3way`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I` —
2.33 to 5.88 MiB, each 0 banned tokens and contract-passing.

## F110. Fine sweep on the transfer proxy: CatBoost ~0.35 is the RankIC optimum

F108 found CatBoost at 0.20-0.35 beat no-CatBoost. Sweeping further, and adding a `raw` XGBoost
member (the existing XGBoost config used `cs_rank`, which F108 showed transfers worse — so the
family had never been fairly evaluated):

| config | longholdout RankIC |
|---|---|
| **`blend_63d8_cat35`** (CatBoost 0.35) | **+0.061900** |
| `blend_63d8_cat20` (CatBoost 0.20) | +0.061859 |
| `blend_63d8_cat35_xgb15` (Cat 0.35 + **XGB 0.15**) | +0.061774 |
| `blend_63d8_cat50` (CatBoost 0.50) | +0.061510 |
| `lgbm_sub1_l63d8` (no CatBoost) | +0.061471 |
| `xgb_sub1_raw` (XGBoost alone, raw) | +0.059078 |

**The curve is smooth and single-peaked around CatBoost 0.20-0.35**, with 0.50 and 0.00 both worse.
XGBoost *alone* is 0.0028 below the shipped LightGBM, and adding it at 0.15 to the CatBoost blend
costs 0.0001 — so on RankIC transfer **XGBoost is a mild negative**, despite Qlib ranking it top of
the tabular models on IR (F99). That is not a contradiction: the two metrics disagree about this
family, exactly as they disagreed about CatBoost (F107).

**So candidate I is confirmed as the RankIC-transfer optimum among everything measurable**, and the
tension is now explicit:

* **RankIC (70% of the score) prefers** `I` = LightGBM 63/8 x3 @0.65 + CatBoost x3 @0.35.
* **IR (30%) prefers more families**, on the board's evidence (F96), and cannot be screened locally
  at all (F107).

**Candidate I is therefore the right default**: it takes the measured RankIC optimum while keeping
the family that produced the board's IR jump, and it drops only the MLP, which is the weakest member
on every external and internal measure we have (F99, F108). If the armed 3-way submission shows the
MLP buying IR at a RankIC cost, `I` is the direct test of whether that trade is worth taking.

## F111. Time decay re-tested under the right conditions — still negative

A time-decayed variant was rejected earlier in this log on the ordinary folds (td125 cost 0.0051).
That rejection was made on the wrong instrument: the folds test **short windows adjacent to the
training cut**, where recency weighting has nothing to offer. The board scores a **long later
period** (F102), which is precisely the case recency weighting exists for. So it deserved a re-test
on the validated proxy before being written off:

| config | longholdout RankIC |
|---|---|
| `lgbm_sub1_l63d8` (equal weights) | **+0.061471** |
| `lgbm_63d8_td500` (half-life 500 trading days) | +0.060985 |
| `lgbm_63d8_td250` (half-life 250 trading days) | +0.058804 |

**Monotone in the wrong direction, and larger decay is worse.** The original rejection stands, but
now on evidence from the instrument that can actually see the effect — which is a stronger result
than the one it replaces, and it removes recency weighting from the candidate space for good.

**Why this is worth recording even though the answer did not change.** The earlier rejection and
this one look identical in the log but are not equally trustworthy: the first could not have
detected the effect it was ruling out. Three findings in this session have had that shape —
`--seeds` (F93), `--train-filter` (F75), `--override num_leaves` (F106) — where a control reported a
result without being able to produce one. **This one is the opposite: a re-test that confirms a
conclusion the first test was not entitled to draw.**

## F112. A submission-killing bug caught during pre-flight: missing `xgboost` in requirements

Doing a final pre-flight on the armed archive, I checked `requirements.txt` against what
`inference.py` actually imports — and then against what each **candidate's members** actually need.

`submission/requirements.txt` lists numpy, pandas, pyarrow, lightgbm and catboost. **It does not
list xgboost** — but F100 added an `xgboost` member kind, and candidates D, E, F and H all carry
XGBoost members. On a judge whose environment does not already have xgboost installed:

    import xgboost as xgb   ->  ModuleNotFoundError  ->  FAILED EVALUATION

**Not a weak score — a failed evaluation, i.e. the same outcome as the two early failures at the
start of this log.** And it would have looked like a modelling result rather than a packaging bug.

**Fixed at the source, not per-candidate.** `tools/make_candidate.py` now derives the extra
requirements from the member kinds actually present:

    needed = {"xgboost": "xgboost>=2.0,<4"}

so a future candidate cannot reintroduce it. All four affected archives were rebuilt:

| candidate | new sha256 |
|---|---|
| `cand_D` | `0f010cce49d9a0e0c5e3163ea9f7e791f0df6d1b98cf07aee29d236617db4a79` |
| `cand_E` | `b6dfec7d852dc0adc683404e324b0ec03893d9200c7a76a4e4c8ecb4835edf27` |
| `cand_F` | `2599077cbeb61a6b2878b7b72de99ae19c59922ed4825ef3c2ee4c68fcc70156` |
| `cand_H` | `b16c7c0600329bb76a6ebf633256932379e9dd204510abb6100bcdb4c4bb27bb` |

Verified: each now has exactly one `xgboost` line, and the four non-XGBoost candidates
(`3way`, `B`, `C`, `I`) are unchanged at zero.

**The `mlp` member needs nothing extra** — its forward pass is pure numpy, so it adds no dependency
at all. That was verified by grepping the archive for sklearn/xgboost/scipy/torch (no matches) and
reading the `_cs_rank` / `_mlp_scores` implementations.

**How this was caught, and why it matters.** The check was "does the artifact declare everything it
uses?" — not "does it run?" It runs fine *here*, because this environment has xgboost 3.2.0
installed. **A test in an environment that already satisfies an undeclared dependency can never
detect that dependency.** The local judge simulator passed all four candidates; the contract passed;
0 banned tokens; the submission would still have failed. This is the same family as F39/F75/F93/F106
— a check that cannot fail is not a check — but this one would have cost a slot *and* produced a
misleading result rather than merely wasting effort.

## F113. The pre-registered rule fired: the MLP HURTS the board's IR badly

`cand_3way` (LightGBM 63/8 x3 @0.50 + CatBoost @0.30 + **MLP @0.20**) submitted at 00:00:

| submission | members | RankIC | **IR** | score | rank |
|---|---|---|---|---|---|
| **`70f5adc9`** (previous best) | 63/8 + CatBoost, no MLP | 0.0713 | **3.4477** | **0.7526** | **#32** |
| **`3bc8de69`** (this) | 63/8 + CatBoost + **MLP 0.20** | 0.0701 | **1.0860** | **0.6400** | **#70** |

**IR fell by 3.2x and the score by 0.113.** The rule armed before the result said exactly this
branch: *"IR < 3.0 -> the MLP HURTS. Immediately submit `cand_I.zip`, which keeps CatBoost while
dropping the MLP."*

**Everything we could measure locally predicted this, and we measured it three independent ways:**

1. **The `longholdout` proxy** (validated for RankIC): the MLP is the weakest member by far,
   **+0.041** against the 63/8's **+0.0615** — a 33% shortfall (F108).
2. **Qlib's published CSI300 benchmarks**: the MLP is the weakest tabular model in the table,
   Rank IC 0.0425 and IR **1.0323 — below even Linear** (F99).
3. **Our own fold work**: F80 measured MLP RankIC +0.0456 against the GBM's +0.0838; F87 found the
   epoch curve peaks at ~20 and decays monotonically after.

**So the diversity hypothesis is now bounded, and this is the useful part.** Cross-family diversity
raised IR when the added family was *competitive* (CatBoost: Rank IC 0.0483 in Qlib's table, and
0.0610 locally) and **destroyed** it when the added family was *much weaker* (MLP: 0.0425 / 0.0456).
**Diversity is not free — it only helps if the added member carries comparable signal.** Adding a
weak member does not smooth the active-return series; it dilutes the ranking.

**This also retroactively explains F87.** There, the MLP blend looked worth +0.011 because it was
priced on the fold engine, which F95/F107 later showed cannot see the board's IR at all. The
+0.011 was a fold-RankIC gain that did not survive; the IR cost it carried was invisible to the
instrument being used.

**Note the honest failure mode here**: I spent a slot on a candidate that three separate lines of
evidence said would fail. The value was that the pre-registered rule made the outcome unambiguous
rather than arguable — and it cost one slot of twelve.

## F114. IR scales with CatBoost WEIGHT, not with CatBoost seed count

`cand_I` (LightGBM 63/8 x3 @0.65 + **CatBoost x3 @0.35**) came back:

| submission | CatBoost config | RankIC | **IR** | score |
|---|---|---|---|---|
| **`70f5adc9`** | **x1 seed @ weight 0.50** | 0.0713 | **3.4477** | **0.7526** |
| `d078e56f` (`cand_I`) | **x3 seeds @ weight 0.35** | **0.0719** | **2.9945** | 0.7391 |
| `3bc8de69` (3-way) | x1 @ 0.30 **+ MLP 0.20** | 0.0701 | 1.0860 | 0.6400 |

**cand_I achieved our best RankIC yet (0.0719) and still lost 0.0135 of score**, because the IR term
fell by more. This is the first submission where the two terms moved in *opposite* directions, and
it is informative:

* **CatBoost weight 0.50 with ONE seed (IR 3.4477) beat weight 0.35 with THREE seeds (IR 2.9945).**
  So the IR gain is driven by the **weight**, not by seed-averaging inside the family. More CatBoost
  signal is better; a smoother CatBoost signal is not what was helping.
* **The RankIC ordering is the reverse** (0.0719 vs 0.0713) — consistent with F110's local sweep,
  which found CatBoost ~0.35 the RankIC optimum and 0.50 slightly worse. **Both instruments agreed
  about RankIC and neither predicted the IR outcome, which is the expected state given F107.**

**This supersedes F110's recommendation.** The local sweep optimised the *RankIC* optimum and that
optimum is real — but it is not the score optimum, because IR moves the other way and cannot be
screened locally. The measured score surface is:

    CatBoost 0.50, 1 seed  ->  0.7526   <- best known
    CatBoost 0.35, 3 seeds ->  0.7391
    CatBoost 0.30 + MLP    ->  0.6400

**Monotone in CatBoost weight so far**, which makes the obvious next test a *higher* weight rather
than a lower one — the opposite of what the local RankIC proxy recommends. Recorded as the standing
hypothesis for the remaining slots.

**Action taken:** with one slot left today, `candidates/cand_KNOWNGOOD_70f5adc9.zip`
(sha256 `2171047b…`, the measured 0.7526 artifact) was resubmitted to put the best-known result back
on the board. **The private board scores the LAST submission**, so leaving a 0.7391 overnight to
gamble on an untested weight would have been the wrong trade when 10 slots remain.

## F115. The board's scoring is DETERMINISTIC — verified by exact reproduction

Resubmitting the identical archive (`candidates/cand_KNOWNGOOD_70f5adc9.zip`, sha256 `2171047b…`)
as a *new* submission returned:

| submission | archive | RankIC | IR | s_rankic | s_ir | score |
|---|---|---|---|---|---|---|
| `70f5adc9` | `2171047b…` | 0.0713 | 3.4477 | 0.7817 | 0.6887 | 0.7538 |
| **`5e3e34f1`** | **`2171047b…` (same)** | **0.0713** | **3.4477** | **0.7817** | **0.6887** | **0.7538** |

**Every digit identical**, including the normalised scores. Two conclusions:

1. **The board's evaluation is deterministic.** No run-to-run noise in the metrics at all. So every
   difference between our submissions is a real property of the artifact, not sampling — which
   retroactively validates treating single submissions as measurements (F95, F96, F114).
2. **The `s_rankic` / `s_ir` normalisation did not move** between the two evaluations despite the
   field having grown to 142 entries, so the min-max span is stable over these timescales. Scores
   recorded earlier can be compared directly to later ones without a renormalisation correction.

**Board state restored**: our best-known artifact is again the last submission, at score **0.7538**.

**Standing hypothesis for the remaining slots (10 available: 3 on each of Oct 12/13/14, plus the
rest of today).** F114 showed the score surface is monotone in CatBoost weight across the three
points measured (0.00 -> 0.6350, 0.35 -> 0.7391, 0.50 -> 0.7538) while the local RankIC proxy
recommends the *opposite* direction. The obvious next probe is therefore a **higher** weight, which
the proxy cannot evaluate and only a slot can.

## F116. Score surface is monotone in CatBoost weight — probes J (0.65) and K (0.80) built

F114 established the measured score surface, and it is monotone in CatBoost weight across all three
points we have:

| CatBoost weight | submission | RankIC | IR | score |
|---|---|---|---|---|
| 0.00 (LightGBM only) | `4690e013` | 0.0711 | 0.8139 | 0.6350 |
| 0.30 **+ MLP 0.20** | `3bc8de69` | 0.0701 | 1.0860 | 0.6400 |
| 0.35 (x3 seeds) | `d078e56f` | 0.0719 | 2.9945 | 0.7391 |
| **0.50 (x1 seed)** | **`5e3e34f1`** | 0.0713 | **3.4477** | **0.7538** |

**RankIC is flat across the whole range (0.0701-0.0719) while IR spans 0.81 to 3.45.** The weight
axis is therefore almost pure IR, and the local RankIC proxy — which recommends ~0.35 — points the
opposite way. **Two probes built to extend the axis in the direction the board rewards:**

| candidate | members | sha256 | contract |
|---|---|---|---|
| `cand_J` | LightGBM 63/8 x3 @0.35 + CatBoost @**0.65** | `4914067607a42a2e…` | passes |
| `cand_K` | LightGBM 63/8 x3 @0.20 + CatBoost @**0.80** | `59e71a060f1db0ee…` | passes |

Both 0 banned tokens, 9 files. **10 slots remain** (3 each on Oct 12/13/14, plus the rest of today).
The plan: probe J, then K, and if the surface turns over, fall back to the 0.50 point which is
already measured — the archive for it is durable at `candidates/cand_KNOWNGOOD_70f5adc9.zip`.

## F117. The board's normalisation moves with the FIELD — and the score surface is monotone after all

`cand_J` (CatBoost **0.65**) came back with the **highest IR we have ever recorded — 3.7939** — but a
score of **0.7106**, *lower* than `5e3e34f1`'s 0.7602. That is impossible for a fixed min-max span:
a higher IR cannot produce a lower `s_ir`. Resolving it produced the most important measurement
detail in this log.

**The spans are min-max over the entire live leaderboard**, and I could read them off the field:

| rank | user | rankic | ir | s_rankic | s_ir |
|---|---|---|---|---|---|
| 1 | Summer | 0.0903 | 4.868 | 1.0000 | 0.6996 |
| 2 | zihan.... | 0.0782 | **7.492** | 0.8578 | **1.0000** |
| 143 | er_048 | 0.0050 | **-1.243** | 0.0000 | 0.0000 |

    rankic span [0.0050, 0.0903]      ir span [-1.243, 7.492]

Check: Summer's s_ir = (4.868 + 1.243) / 8.735 = **0.6996** — exactly the board's value. The formula
is confirmed.

**A competitor (`zihan....`) submitted IR 7.492 during the session, which widened the IR span and
pushed every existing submission's `s_ir` down.** That is why `5e3e34f1`'s score drifted 0.7538 ->
0.7602 -> and why `cand_J` scored 0.7106 despite a *better* IR. **Our score is partly other people's
submissions.**

**Recomputing every submission under ONE fixed span removes the artifact entirely:**

| submission | CatBoost | rankic | ir | s_rankic | s_ir | **score** |
|---|---|---|---|---|---|---|
| `4690e013` LightGBM only | 0.00 | 0.0711 | 0.8139 | 0.7748 | 0.2355 | 0.6130 |
| `3bc8de69` + MLP 0.20 | 0.30 | 0.0701 | 1.0860 | 0.7631 | 0.2667 | 0.6142 |
| `d078e56f` (`cand_I`) | 0.35 | 0.0719 | 2.9945 | 0.7842 | 0.4851 | 0.6945 |
| `5e3e34f1` / `70f5adc9` | 0.50 | 0.0713 | 3.4477 | 0.7772 | 0.5370 | 0.7051 |
| **`b27d9691` (`cand_J`)** | **0.65** | 0.0705 | **3.7939** | 0.7678 | **0.5766** | **0.7104** |

**Strictly monotone in CatBoost weight.** `cand_J` IS our best artifact — the reversal was pure
normalisation drift, and acting on the raw board score would have wrongly reverted a real gain.

**Two corrections to earlier findings, both now measured rather than assumed:**

1. **F115 was wrong about span stability.** I claimed the normalisation "did not move" because two
   consecutive evaluations matched — but those were minutes apart, and the span moved within hours.
   The determinism result (identical archive -> identical metrics) still holds and is unaffected;
   only the stability claim was overreached.
2. **F114's "RankIC and IR moved in opposite directions" was reading noise.** With the spans fixed,
   RankIC varies only 0.0701-0.0719 (noise-level) while IR drives everything.

**Operational rule adopted: compare submissions on raw `ir` and `rankic`, never on the board's
`score`.** The score is only meaningful at a fixed instant; the private board will renormalise
against its own field anyway, so raw metrics are the transferable quantity.

## F118. The score surface is monotone in CatBoost weight all the way to the endpoint

`cand_K` (CatBoost **0.80**) landed: **IR 4.3083** — the highest we have recorded — RankIC 0.0695.
Recomputing every submission under ONE fixed span (the field's current
`rankic [0.0050, 0.0903]`, `ir [-1.243, 7.492]`) gives a clean, strictly monotone surface:

| CatBoost weight | submission | rankic | ir | score (fixed span) |
|---|---|---|---|---|
| 0.00 (LightGBM only) | `4690e013` | 0.0711 | 0.8139 | 0.6130 |
| 0.30 (+ MLP 0.20) | `3bc8de69` | 0.0701 | 1.0860 | 0.6142 |
| 0.35 | `d078e56f` | 0.0719 | 2.9945 | 0.6945 |
| 0.50 | `5e3e34f1` | 0.0713 | 3.4477 | 0.7051 |
| 0.65 | `b27d9691` | 0.0705 | 3.7939 | 0.7104 |
| **0.80** | **`ff021f23`** | 0.0695 | **4.3083** | **0.7199** |

Linear fits over the 0.35-0.80 range:

    IR      ~ +2.858 * w + 1.993        (IR nearly quadruples across the axis)
    RankIC  ~ -0.0053 * w + 0.0739      (RankIC moves 0.0008 per 0.15 of weight)

**IR moves 12x faster than RankIC in score-relevant terms**, so the axis is worth walking to its
endpoint. Projections: w=0.90 -> IR 4.565, score 0.7252; **w=1.00 -> IR 4.851, score 0.7306.**

**The endpoint is CatBoost with NO LightGBM at all — which the local engine calls our worst
configuration.** Locally CatBoost alone has RankIC 0.0610 and a *negative* IR; the `longholdout`
proxy ranks it below everything. **The board ranks it best.** This is F107's conclusion arriving in
its sharpest form: the local IR instrument is not merely noisy here, it inverts the ordering, and
following it has cost us repeatedly (F89's rejection of CatBoost, F110's recommendation of 0.35,
F113's MLP).

**Why the mechanism is plausible rather than a fluke.** IR = mean(active)/std(active). CatBoost's
predictions are structurally smoother than LightGBM's — symmetric oblivious trees, ordered boosting,
and a lower learning rate with fewer leaves. A smoother alpha produces a less volatile active-return
series for the same mean, which is exactly what a high IR is. **It also explains why the MLP failed
so badly (F113): the MLP is not smoother, it is noisier** — a weak model's output adds variance
without adding signal, the opposite of what raises IR.

**Two probes built to find the endpoint** (2 slots remain today):
`cand_L` = CatBoost @0.90, `cand_M` = **CatBoost alone**.

## F119. The endpoint probes, built without retraining — and in-sample fit is anti-correlated

The machine went to **load average ~40** (from unrelated `rustc` builds) and the venv's scipy files
began timing out on read, so `train.py` could not import catboost at all:

    File ".../catboost/core.py", line 50, in <module>
        import scipy.sparse
    TimeoutError: [Errno 60] Operation timed out

**Workaround that avoided the blocker entirely**: `cand_L` (0.90) and `cand_M` (1.00) need the *same
trained members* as `cand_K` — only the **weights** differ. So they were built by copying
`/tmp/cat80/model` and rewriting `metadata.json`, with no training run at all. Verified the LightGBM
members are byte-identical across builds (`1492bbd2…` for seed 42 in both `/tmp/cat63` and
`/tmp/cat80`), so the substitution is sound.

One incidental finding: **the CatBoost `.cbm` files differ between builds of the same seed and
params** (`52779ba0…`, `226ba7f1…`, `8f561209…`) — CatBoost's multithreaded training is not
bit-reproducible. That does not affect validity (each is a legitimate model of the same recipe) but
it does mean a candidate cannot be reproduced byte-for-byte from `train.py` alone; the archive hash
is the only reliable identity.

**In-sample contract RankIC, against board IR, across the whole weight sweep:**

| CatBoost w | contract RankIC (in-sample) | board IR |
|---|---|---|
| 0.35 | 0.2598 | 2.9945 |
| 0.50 | — | 3.4477 |
| 0.65 | 0.2204 | 3.7939 |
| 0.80 | 0.1994 | 4.3083 |
| 0.90 | 0.1655 | *(probe)* |
| 1.00 | 0.1519 | *(probe)* |

**Monotone in opposite directions.** Every unit of in-sample fit we give up buys board IR. This is
the cleanest statement of F102/F107 in the log: **the quantity our entire early search maximised is
anti-correlated with the quantity the board rewards.** It also explains, in one line, why the
CatBoost package was rejected at F89 on local evidence and why that rejection was so costly.

**Submitting `cand_M` (CatBoost alone)** with the last slot today: the fitted surface
(`IR ≈ 2.858w + 1.993`) projects it highest, and testing the endpoint settles the shape in one slot
rather than two. `cand_L` (0.90) is built and durable if a bisection is needed.

## F120. The CatBoost-weight surface has a PEAK at 0.80 — and the last submission is now the wrong side of it

`cand_M` (CatBoost **alone**, w=1.00) came back at IR **3.4143** — well below `cand_K`'s 4.3083.
The surface turns over, so the optimum is interior. Recomputed under one fixed span:

| CatBoost w | submission | rankic | ir | score |
|---|---|---|---|---|
| 0.00 | `4690e013` | 0.0711 | 0.8139 | 0.6130 |
| 0.35 | `d078e56f` | 0.0719 | 2.9945 | 0.6945 |
| 0.50 | `5e3e34f1` | 0.0713 | 3.4477 | 0.7051 |
| 0.65 | `b27d9691` | 0.0705 | 3.7939 | 0.7104 |
| **0.80** | **`ff021f23`** | 0.0695 | **4.3083** | **0.7199** |
| 1.00 | `a8be0d8d` | 0.0679 | 3.4143 | 0.6761 |

**IR peaks at 0.80, not at the endpoint**, so the `IR ≈ 2.858w + 1.993` fit from F118 held only over
0.35-0.80 and broke down beyond it — worth noting, because that fit is what motivated testing the
endpoint rather than bisecting.

**A process failure to record honestly.** Probing the endpoint put a *worse* artifact on the board
(`a8be0d8d`, 0.6761) as the last submission, and the day's quota was then exhausted, so it could not
be reverted until the next reset. **The rule I wrote into the previous timer — "always leave the
best-known artifact on the board before stopping for the day" — was the right rule and I did not
follow it.** The correct order was: restore `cand_K` first, then spend any remaining slot on the
probe, accepting that the probe's result would only be actionable the next day.

**Why the optimum is interior, mechanistically.** Raising the CatBoost weight substitutes a smoother
alpha for a sharper one, which lowers the volatility of active returns (good for IR) but also lowers
the ranking's information (bad for both terms). At 0.80 the second effect starts to dominate. The MLP
result (F113) and this result are the same phenomenon from opposite ends: **IR rewards smoothing only
up to the point where signal is being discarded.**

**Bisection targets built and durable**: `cand_L` (0.90, sha256 `e7f604ce…`) already exists, so the
remaining search is 0.85 and 0.90 — and `cand_L` needs no training run.

## F121. Testing the smoothness mechanism directly: three CatBoost variants at the peak weight

F120 concluded that IR rewards a *smoother* alpha until signal starts being discarded, with the peak
at CatBoost weight 0.80. That makes a falsifiable prediction: **if smoothness is the mechanism, then
changing how smooth the CatBoost member itself is should move the peak.**

Three variants, all held at weight 0.80 so the weight axis is not confounded:

| candidate | CatBoost change | in-sample contract RankIC |
|---|---|---|
| `cand_K` (reference) | 400 iterations, 1 seed | 0.1994 |
| `cand_O` | **800 iterations**, 1 seed | **0.2177** |
| `cand_P` | **100 iterations**, 1 seed | **0.1585** |
| `cand_Q` | 400 iterations, **3 seeds** | *(building)* |

**The in-sample numbers order the candidates, and we now know that ordering is the wrong way
round.** F119/F120 established that in-sample contract RankIC is *anti-correlated* with board IR —
it fell monotonically 0.2598 -> 0.1519 as the board's IR climbed 2.99 -> 4.31. Applying that:

* `cand_O` (800 iters, highest in-sample fit 0.2177) is predicted **worst** on board IR.
* `cand_P` (100 iters, lowest in-sample fit 0.1585) is predicted **best** on board IR.
* `cand_K` (400 iters) sits between.

**This is a genuine prediction made before the measurement**, and it is the sharpest test available
of the smoothness mechanism — because it uses an instrument (in-sample fit) that we have measured to
run *backwards*, so a correct call here is evidence the mechanism is real rather than curve-fitting.
If `cand_P` wins, the rule "less in-sample fit -> more board IR" generalises beyond the weight axis
and becomes a usable search direction. If `cand_O` wins, the anti-correlation is specific to the
weight axis and the mechanism needs restating.

All built without retraining where possible; `cand_O` and `cand_P` needed real training runs (the
CatBoost iteration count changes the model), and the machine's load had dropped enough for catboost
to import again.

## F122. The smoothness probes are complete — a four-point prediction, made before measurement

All three variants built and validated at CatBoost weight 0.80, so the weight axis is held fixed:

| candidate | CatBoost member | in-sample contract RankIC | sha256 |
|---|---|---|---|
| `cand_O` | **800 iterations**, 1 seed | **0.2177** | `6930b502…` |
| `cand_K` | 400 iterations, 1 seed *(our best on the board)* | 0.1994 | `59e71a06…` |
| `cand_Q` | 400 iterations, **3 seeds** | 0.1803 | `d9f07ab0…` |
| `cand_P` | **100 iterations**, 1 seed | **0.1585** | `517373f8…` |

**Because in-sample fit has been measured to run backwards (F119/F120), the prediction is the
reverse of this ordering:**

| candidate | predicted board IR | reasoning |
|---|---|---|
| `cand_P` | **highest** | least in-sample fit -> least to unlearn on the post-2022 regime |
| `cand_Q` | 2nd | seed-averaging is a genuine variance reduction, not just less fitting |
| `cand_K` | 3rd | the current reference |
| `cand_O` | **lowest** | most in-sample fit, so most regime-specific structure |

**Two distinct mechanisms are being separated here, which is why the set is worth four slots:**

* **`cand_O` vs `cand_P`** varies *capacity* (trees grown) at fixed ensembling. More trees = tighter
  fit = the anti-correlation predicts worse.
* **`cand_Q` vs `cand_K`** varies *ensembling* at fixed capacity. Seed-averaging should reduce the
  variance of the alpha itself, which is the mechanism F120 proposed for IR — so `cand_Q` should beat
  `cand_K` **even though its in-sample fit is lower**, and it is the cleanest test of "smoothing
  helps" as opposed to "fitting hurts".

If `cand_P` and `cand_Q` both beat `cand_K`, then **both routes to a smoother alpha work** and the
search direction generalises. If only `cand_Q` works, the mechanism is variance reduction
specifically. If neither works, the anti-correlation is a property of the weight axis alone and F120
needs restating.

**Priority for the 6 remaining slots** (after the mandatory `cand_K` restore): `cand_Q` first — it
has the strongest mechanistic argument and the least confounded design — then `cand_P`, then the
0.85/0.90 weight bisection (`cand_N`, `cand_L`).

## F123. A six-point smoothness ladder at fixed weight — the sharpest test the session can run

Exposing CatBoost's regularisation knobs (`--catboost-depth`, `--catboost-l2`, `--catboost-lr`, all
previously hardcoded) made the smoothness axis directly controllable rather than only reachable
through the blend weight. Six probes now sit at **weight 0.80**, so the weight axis is held fixed
and only the *shape* of the CatBoost alpha varies:

| candidate | CatBoost member | in-sample contract RankIC | sha256 |
|---|---|---|---|
| `cand_O` | 800 iterations | **0.2177** | `6930b502…` |
| `cand_K` | 400 iterations *(reference, our best on the board)* | 0.1994 | `59e71a06…` |
| `cand_Q` | 400 iterations, **3 seeds** | 0.1803 | `d9f07ab0…` |
| `cand_S` | **l2_leaf_reg = 20** | 0.1767 | `bc84cc88…` |
| `cand_P` | 100 iterations | 0.1585 | `517373f8…` |
| `cand_R` | **depth 4** | **0.1569** | `a52460ce…` |

**The prediction, made before any of these is measured, is the reverse of this ordering** — because
in-sample fit was measured to run backwards against board IR (it fell 0.2598 -> 0.1519 while board IR
climbed 2.99 -> 4.31):

    cand_R  (depth 4)      predicted HIGHEST board IR   -- shallowest trees, smoothest by construction
    cand_P  (100 iters)    2nd
    cand_S  (l2 = 20)      3rd
    cand_Q  (3 seeds)      4th
    cand_K  (reference)    5th
    cand_O  (800 iters)    LOWEST

**Why this ladder is worth more than the previous probes.** Each rung changes a *different*
structural knob, so a correct prediction across several of them cannot be explained by one lucky
correlation with the weight axis:

* **`cand_R`** — shallower trees average fewer feature interactions per leaf. This is the purest
  smoothness manipulation available.
* **`cand_S`** — L2 shrinkage pulls leaf values toward zero, damping the alpha's amplitude.
* **`cand_Q`** — seed-averaging reduces the *variance* of the alpha while leaving its bias alone.
  Mechanistically this is the cleanest match to "IR = mean/std", since it attacks std directly.
* **`cand_O` / `cand_P`** — tree count, i.e. capacity at fixed shape.

**A correct prediction on `cand_R` and `cand_Q` together would establish that the board's IR is
driven by alpha smoothness as a general property**, not by a single tuning axis — and it would give a
usable search direction (smooth the member, don't sharpen it) for the remaining slots. A wrong
prediction would be equally informative: it would mean the weight axis result was specific, and the
whole "smoothness" story needs rewriting.

**6 slots remain** (3 on Oct 13, 3 on Oct 14), so the ladder cannot be walked in full. Priority:
`cand_Q` (cleanest mechanism), then `cand_R` (purest smoothness), then `cand_S`, then `cand_P`, then
the weight bisection (`cand_N` 0.85, `cand_L` 0.90).

## F124. The smoothness prediction FAILED for seed-averaging — the anti-correlation is weight-axis specific

`cand_Q` (CatBoost **3 seeds** at total weight 0.80, each 0.2667) came back:

| submission | CatBoost member | in-sample RankIC | **board IR** | board rankic |
|---|---|---|---|---|
| `f2d8797b` (`cand_K`) | 400 iters, **1 seed** @0.80 | 0.1994 | **4.3083** | 0.0695 |
| `19d56c83` (`cand_Q`) | 400 iters, **3 seeds** @0.80 total | 0.1803 | **3.4323** | 0.0705 |

**The prediction was wrong.** Lower in-sample fit did *not* produce higher board IR — it produced
**0.88 lower**. So within the CatBoost family, seed-averaging **hurts**, and the rule
"less in-sample fit -> more board IR" is **specific to the LightGBM/CatBoost weight axis**, not a
general property of the model.

**This is the branch F123 named in advance**: *"if the prediction fails, the weight-axis result was
specific and the smoothness story needs rewriting."* That is what happened, and it is worth being
precise about what survives and what does not.

**What survives:**
* The weight axis is real and large: IR 0.81 -> 4.31 as CatBoost weight goes 0.00 -> 0.80, with an
  interior peak (1.00 falls back to 3.41). Six submissions, deterministic engine.
* Cross-**family** blending raises IR substantially (LightGBM + CatBoost).
* The MLP destroys IR (1.086).

**What does NOT survive:**
* **"Smoother is better" as a general law.** Seed-averaging is the textbook variance reduction and it
  made things worse. So the mechanism is not simply `std(active)` being lowered.
* **"In-sample fit is anti-correlated with board IR" as a search heuristic.** It held across the
  weight axis and broke immediately when applied to a different knob.

**A better reading of the evidence.** The single-seed CatBoost at 0.80 beating three averaged ones at
the same total weight suggests what matters is not smoothness but **how distinctive the second family
is**. One CatBoost is a strongly opinionated, *different* model; three averaged CatCasts are closer to
each other and closer to the mean, so the blend inherits less of whatever makes CatBoost's ranking
different from LightGBM's. That is consistent with the MLP result too — the MLP was distinctive but
*weak*, and weakness is what killed it. **Diversity helps when the added family is both different and
competent; averaging inside a family reduces difference without improving competence.**

**Consequence for the remaining slots**: stop treating in-sample fit as a compass. The only axis with
a confirmed, measured, monotone effect is the **CatBoost weight**, and its peak is bracketed in
(0.80, 1.00) with 0.80 better than both 0.65 and 1.00 — so the highest-value use of a slot is the
**weight bisection** (`cand_N` 0.85, then `cand_L` 0.90), not more structural probes.

## F125. Depth 4 gives the highest IR recorded (4.6740) — the prediction was RIGHT for depth, WRONG for seeds

`cand_R` (CatBoost **depth 4** at weight 0.80) landed:

| submission | CatBoost member | in-sample RankIC | **board IR** | board rankic | board score |
|---|---|---|---|---|---|
| `f2d8797b` (`cand_K`) | depth 6, 400 iters, 1 seed | 0.1994 | 4.3083 | 0.0695 | **0.7953** |
| `19d56c83` (`cand_Q`) | depth 6, **3 seeds** | 0.1803 | 3.4323 | 0.0705 | 0.7582 |
| **`b4da96fd` (`cand_R`)** | **depth 4**, 400 iters, 1 seed | **0.1569** | **4.6740** | 0.0673 | 0.7889 |

**`cand_R` has the highest IR we have ever measured — 4.6740, above the previous best of 4.3083.**
And its in-sample fit is the *lowest* of the three. So:

* **The depth prediction was CORRECT**: the shallowest model, with the least in-sample fit, produced
  the highest board IR.
* **The seed-averaging prediction was WRONG** (F124): three averaged CatCasts produced the *lowest*
  IR despite the second-lowest in-sample fit.

**So "smoothness" is not one thing.** Two knobs that both reduce in-sample fit move board IR in
opposite directions. The reading that fits all six data points is about **overfitting, not
smoothness**:

* **Depth 4 helps** because a shallower tree generalises better to the post-2022 regime — it is a
  genuine reduction in variance *of the fitted function*.
* **Seed-averaging hurts** because it does not change how well the family generalises; it only
  dilutes the family's *distinctiveness* relative to LightGBM. The blend then inherits less of what
  makes CatBoost's ranking different — the same reason the MLP failed, except the MLP was also weak.

**This also explains why the weight axis looked like "smoothing".** Raising CatBoost's weight
substitutes a model whose *inductive bias* transfers better (oblivious trees, ordered boosting) for
one whose bias is sharper on the training regime. It was never about the amplitude of the alpha.

**The score trade-off is now explicit and close.** `cand_R` has higher IR (+0.366) and lower RankIC
(-0.0022). Under the field's current span that is -0.018 from RankIC and +0.013 from IR, i.e. `cand_K`
leads by ~0.006 on the board's score — **but the score is span-dependent (F117) and the private board
renormalises against its own field, so raw metrics are the transferable comparison.** On raw IR
`cand_R` wins; on raw RankIC `cand_K` wins.

**Quota is now exhausted for Oct 13.** Three slots remain, all on the final day (Oct 14), with the
deadline at 23:59. Plan: probe `cand_S` (l2 = 20) and a **depth 4 + l2 20 combination**, then spend
the last slot on whichever artifact has the best measured raw metrics.

## F126. The 0.80 ladder is complete — and the final day's plan

Seven probes now sit at CatBoost weight 0.80, each changing a different structural knob. Three have
been measured on the board; four have not:

| candidate | CatBoost member | in-sample RankIC | **board IR** | status |
|---|---|---|---|---|
| `cand_O` | 800 iterations | 0.2177 | — | unmeasured |
| `cand_K` | 400 iterations *(reference)* | 0.1994 | **4.3083** | measured |
| `cand_Q` | 3 seeds | 0.1803 | **3.4323** | measured — WORSE |
| `cand_S` | l2 = 20 | 0.1767 | — | unmeasured |
| `cand_P` | 100 iterations | 0.1585 | — | unmeasured |
| **`cand_T`** | **depth 4 + l2 = 20** | 0.1580 | — | unmeasured |
| **`cand_R`** | **depth 4** | **0.1569** | **4.6740** | measured — BEST IR |

**What the measured points establish.** Depth 4 raised IR by +0.366 (4.3083 -> 4.6740); seed-averaging
lowered it by -0.876 (4.3083 -> 3.4323). Both reduce in-sample fit. **So the discriminator is not
"less fit" but whether the change improves how the model's inductive bias transfers** — a shallower
tree does, averaging seeds does not (F125).

**That makes `cand_T` the highest-value remaining probe**: it stacks the one knob measured to help
(depth 4) with the one knob whose *mechanism* predicts it should help (L2 shrinkage — like depth, it
constrains the fitted function rather than diluting the family). If `cand_T` beats `cand_R`, the
"regularise the member" direction generalises and there is more room; if it does not, `cand_R` is the
endpoint.

**Final day plan (3 slots, Oct 14, deadline 23:59):**

| slot | action | rationale |
|---|---|---|
| 1 | `cand_T` (depth 4 + l2 20) | stacks the two regularisation knobs |
| 2 | `cand_S` (l2 20 alone) | isolates L2; also the fallback if T regresses |
| 3 | **the best-measured artifact, resubmitted** | the private board scores the LAST submission |

**The comparison rule for the final choice is fixed now, before the numbers arrive:** rank the
candidates by **raw IR** first, and break ties on **raw RankIC**. Raw metrics are the only transferable
quantity (F117 — the board's score is min-max normalised against a field that moves). Under that rule
`cand_R` (IR 4.6740) currently leads `cand_K` (4.3083) despite `cand_K` showing the higher board score
under today's span.

## F127. Final day, slot 1: stacking L2 on depth 4 HURTS — depth is the knob that works

`cand_T` (CatBoost **depth 4 + l2_leaf_reg 20**, w=0.80) landed:

| submission | CatBoost member | rankic | **IR** | score |
|---|---|---|---|---|
| **`b4da96fd` (`cand_R`)** | **depth 4** | 0.0673 | **4.6740** | 0.7708 |
| `db917350` (`cand_T`) | depth 4 **+ l2 = 20** | 0.0668 | 4.4336 | 0.7563 |
| `f2d8797b` (`cand_K`) | depth 6 | 0.0695 | 4.3083 | **0.7724** |
| `19d56c83` (`cand_Q`) | depth 6, 3 seeds | 0.0705 | 3.4323 | 0.7426 |

**`cand_T` is worse than `cand_R` on both raw terms**, so adding L2 shrinkage to a depth-4 model
costs 0.24 of IR. Combined with F124 (seed-averaging costs 0.88) this narrows the mechanism further:

* **Only one of the four regularisation-style knobs actually helps: reducing tree DEPTH.** L2
  shrinkage, seed-averaging and tree count all hurt.
* That is consistent with the F125 reading but sharpens it: it is not "regularisation" in general and
  not "less in-sample fit" — **depth is special because it changes the model's functional form**
  (fewer feature interactions per leaf, so a shallower, more additive alpha), whereas L2 and
  seed-averaging leave the functional form alone and only rescale or average it.

**Consequence for the remaining two slots: the productive direction is FURTHER DOWN the depth axis,
not more regularisation.** Building `cand_U` (CatBoost **depth 3**) and `cand_V` (depth 4 at weight
0.90, to check whether the weight peak moves once the member is shallower). `cand_S` (l2 alone) is
now **deprioritised** — its own mechanism has been measured to hurt, so it is no longer worth a slot.

## F128. The depth axis also peaks at 4 — and the pre-registered "rank by raw IR" rule was WRONG

`cand_U` (CatBoost **depth 3**) landed at IR **3.5490**, below depth 4's 4.6740. So the depth axis
turns over as well:

| CatBoost depth (w=0.80) | rankic | IR |
|---|---|---|
| 6 | 0.0695 | 4.3083 |
| **4** | 0.0673 | **4.6740** |
| 3 | 0.0665 | 3.5490 |

**Depth 4 is the optimum**, bracketed on both sides. Every axis the session probed has an interior
peak: blend weight peaks at 0.80, depth peaks at 4.

### Correcting the pre-registered rule, with the arithmetic that shows why it was wrong

The timer armed for today said: *"Rank all measured candidates by RAW IR first; break ties on RAW
RankIC."* **That rule is wrong**, and computing the span economics shows it decisively. The board's
normalisation spans (measured off the field, F117) are:

    rankic span 0.0853      ir span 6.845      ratio 80.2

so a unit of each is worth:

    1.0 IR      = 0.30 / 6.845 = 0.0438 score
    0.001 RankIC = 0.70 * 0.001 / 0.0853 = 0.0082 score

**RankIC is worth 187x more per unit than IR.** Ranking by raw IR therefore optimises the *smaller*
term while ignoring the larger one.

Applying the correct comparison to the two finalists:

| candidate | rankic | ir | score (current span) | score (earlier span) |
|---|---|---|---|---|
| **`cand_K`** | **0.0695** | 4.3083 | **0.7725** | **0.7953** |
| `cand_R` | 0.0673 | **4.6740** | 0.7705 | 0.7889 |

`cand_R` would need `ir_span / rankic_span < 1.1` to win — i.e. the IR span would have to be
comparable to the RankIC span, when it is actually **80x** larger. The margin is not close.

**`cand_K` also wins under two independent spans** (measured at different times with the field at 144
and 150 entries), which is the robustness check that matters given F117's finding that the score moves
with the field.

### The deviation is documented rather than hidden

I am **not** following the rule I wrote this morning, and the reason is not hindsight about which
candidate happened to score higher: the 187x-per-unit arithmetic was computable before either number
arrived, and it invalidates the rule's premise. **The pre-registered rule optimised the wrong term.**
Recording the deviation explicitly matters because a pre-registered rule's value comes from being
binding — a rule quietly discarded whenever it is inconvenient is worse than no rule at all. Here the
rule is discarded with a stated, checkable reason, and the replacement rule
(**compare expected score under the measured spans**) is written down before the final submission.

## F129. FINAL STATE — the session's last submission is in, and no slots remain

    submission  cb5ce9c3-ec07-4400-a183-afdeaa177040
    archive     candidates/cand_K.zip  sha256 59e71a060f1db0eeb919c3bf647a4f1e6bf2347384184e9154f453d1363fe785
    rankic      0.0695
    ir          4.3083
    score       0.7724      public rank #27 of 150

Quota is 3/3 with the next slot at **2026-10-15T00:00**, i.e. *after* the 2026-10-14 23:59 deadline.
**This is the final artifact** and it cannot change.

### The session in two numbers

| | session start | final |
|---|---|---|
| rank | #73 | **#27** |
| score | 0.6350 | **0.7724** |
| RankIC | 0.0711 | 0.0695 |
| **IR** | **0.8139** | **4.3083** |

**RankIC ended essentially where it started (0.0711 -> 0.0695). The entire gain was IR, 5.3x.**
Every submission this session scored RankIC in a 0.0602-0.0719 band regardless of recipe; the one
lever that moved the score was the composition of the model ensemble, and specifically the CatBoost
weight.

### The final artifact, for code review

`cand_K` = **3 x LightGBM (63 leaves, depth 8, 400 fixed rounds, seeds 42/43/44) at total weight
0.20** + **1 x CatBoost (400 iterations, depth 6, lr 0.04, l2 5.0) at weight 0.80**, blended by
cross-sectional rank, post-processed to `rank`.

* 0 banned tokens in both `inference.py` and `train.py` (word-boundary AND raw-substring scans).
* Judge contract: **240/240 days**.
* Every declared dependency is actually used, and every used dependency is declared (F112).
* Model weights are plain text / plain arrays only — LightGBM `.txt`, CatBoost `.cbm`. No pickle.
* All 136 features use a look-back of at most **120 trading days**, at the stated limit (F105).
* Model classes (LightGBM, CatBoost, ensembles) are explicitly permitted by `Problem.txt:55,73`.

### What the session established, and what it got wrong

**Established, with measurements:**
1. IR is a *portfolio* metric (`excess/TE`), not a signal metric, and the board scores post-2022 data.
2. **Our local engine cannot see the board's IR at all** — it ranks CatBoost *below* LightGBM-only
   when the board ranks it 5.3x above. Its sign is wrong (F107).
3. RankIC *can* be screened locally on a validated `longholdout` proxy.
4. The board is deterministic; the board's score is **not** comparable across time (F117).
5. Cross-family blending raises IR **only when the added family is competent**. CatBoost helped
   (+3.5 IR); the MLP destroyed it (-2.4 IR); the ridge, XGBoost and 3-seed CatBoost variants were
   all neutral-to-negative.
6. Every tuning axis has an interior peak: blend weight peaks at 0.80, depth at 4.

**Got wrong, and corrected:**
* F89 rejected CatBoost on local IR evidence — **cost 0.111 of score** until reversed.
* F35 priced the IR gap at +0.00005 using fold data; the board's gap was 1400x that.
* F110 recommended CatBoost weight 0.35 from a local RankIC sweep; the score optimum is 0.80.
* F115 claimed the normalisation span was stable; it moved within hours (F117).
* F123/F124 predicted "less in-sample fit -> higher IR" — true for depth, **false** for
  seed-averaging. The heuristic was killed rather than patched.
* F128's own pre-registered rule ("rank by raw IR") was **wrong** — RankIC is worth 187x more per
  unit, so the rule optimised the smaller term. Deviated with the arithmetic stated (F128).

**Five silent no-ops were caught** (F39, F75, F93, F106, F112), every one detected by the same tell:
**two configurations producing bit-identical output.** F112 would have caused a *failed evaluation*
rather than a weak score, and passed every local check because this machine already had xgboost.

## F130. Post-deadline check: the two failures are the OLDEST submissions, and the last one is intact

Asked to re-submit the final artifact because "some submissions had problems". Checked everything.

**The competition has ended**: `can_submit: False, reason: "ended"`, phase `ended`. **Resubmission is
impossible**, so the last submission stands — which is the outcome the user's own instruction
anticipated ("if you can't submit, it's probably fine, don't worry about it").

**Full history: 20 submissions, 2 failed, and both failures are #1 and #2 chronologically** — i.e.
the two attempts made *before* this session's work began:

    1. ffeeb8b3  failed evaluation (exit code 1)
    2. 8cbff65c  failed evaluation (exit code 1)
    3. 6077b16d  finished
    ...
    20. cb5ce9c3 finished   <-- MOST RECENT

**The most recent submission is `cb5ce9c3`, status `finished`**, so the private board will score the
artifact we intended. The two failures are historical, were refunded at the time (failed evaluations
do not consume quota), and cannot affect the final ranking because the private board scores only the
**last** submission.

**Final public standing: rank #30 of 153, rankic 0.0695, IR 4.3083, score 0.7992.**
(The score moved from 0.7724 to 0.7992 after the close purely because the field's min-max span
shifted again — F117 in action, and another reminder that only the raw `rankic`/`ir` are stable.)

Private board: `403 {"detail":"priva...}` — not visible to competitors, as documented.

**Conclusion: nothing is wrong and no action is possible or needed.**
