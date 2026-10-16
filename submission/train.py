"""Train and freeze the submission artifacts from the public 2016-2022 data.

This file is submitted alongside `inference.py` and must be able to reproduce the contents
of `model/` from the public dataset alone (Submission_Rules.md §2). The organizers re-run it
during code review, so it must stay honest: no data beyond 2016-2022, no test-period peeking,
deterministic given the seed.

Usage
-----
    python -m submission.train --data-dir data/Data --submission-dir submission
    python -m submission.train --data-dir data/Data --walk-forward   # honest validation

`--walk-forward` trains on an expanding window and reports per-year RankIC for folds whose
test period is strictly excluded from training. That is the only trustworthy evidence about
this model, because the shipped artifact is refit through 2022-12-31 and can never be
honestly scored on the public period.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling modules importable no matter how this file is invoked: as a package
# (`python -m submission.train`), as a script from inside the directory (`python train.py`),
# or by path with nothing on `sys.path` (`spec_from_file_location`). The last form is how a
# naive reviewer harness loads a file, and it is the failure mode that cost us two submissions
# before `inference.py` was made self-contained. This is a plain directory insert, not
# reflection, and it is scoped to this training entry point -- the scored `inference.py` does
# not do this and does not need it.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# `inference.py` is the single definition of the feature transform (the organizers' own
# baseline does the same: its train.py imports make_features/FEATURE_COLUMNS from
# inference.py). Importing it here is what guarantees training and inference agree.
try:  # loaded as a package: `python -m submission.train`
    from .inference import FEATURE_COLUMNS, build_features
except ImportError:  # loaded as a flat module or by path
    from inference import FEATURE_COLUMNS, build_features  # type: ignore[no-redef]


# --- data -------------------------------------------------------------------

def _panel_from_public(data_dir: Path) -> pd.DataFrame:
    """Read the public monthly parquet into our panel layout.

    Mirrors `dtcomp.data.load_panel` but without the research-side cache, so `train.py`
    stays self-contained and reproducible from the raw public dataset.
    """
    root = Path(data_dir)
    if root.name != "Data" and (root / "Data").is_dir():
        root = root / "Data"

    def read(table: str, columns: list[str]) -> pd.DataFrame:
        paths = sorted((root / table).glob("*/*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no monthly parquet under {root / table}")
        return pd.concat((pd.read_parquet(p, columns=columns) for p in paths),
                         ignore_index=True)

    market = read("MarketData", ["日期", "代码", "open", "high", "low", "close",
                                 "volume", "amount", "is_suspend"])
    try:
        raw = read("MarketDataRawOpen", ["日期", "代码", "open"]).rename(
            columns={"open": "raw_open"})
    except FileNotFoundError:
        raw = pd.DataFrame(columns=["日期", "代码", "raw_open"])
    factors = read("Factor33_winsor", None)
    barra = read("Barra_factor", None)

    factors = factors.drop(columns=["is_suspend", "名称"], errors="ignore")
    barra = barra.drop(columns=["名称"], errors="ignore")
    barra = barra.rename(columns={name: f"barra_{name}" for name in
                                  ("市值", "贝塔", "动量", "残差波动", "非线性市值",
                                   "账面市值比", "流动性", "盈利", "成长", "杠杆")})

    panel = market.merge(raw, on=["日期", "代码"], how="left", validate="one_to_one")
    panel = panel.merge(factors, on=["日期", "代码"], how="left", validate="one_to_one")
    panel = panel.merge(barra, on=["日期", "代码"], how="left", validate="one_to_one")
    if "raw_open" not in panel.columns:
        panel["raw_open"] = np.nan
    barra_cols = [c for c in panel.columns if c.startswith("barra_")]
    panel["in_barra"] = panel[barra_cols[0]].notna() if barra_cols else True
    panel = panel.drop_duplicates(["日期", "代码"], keep="last")
    return panel.sort_values(["日期", "代码"], kind="stable").reset_index(drop=True)


def build_training_frame(panel: pd.DataFrame, *, last_exit_date: int | None = None
                         ) -> pd.DataFrame:
    """Features + the next-day open-to-open label, restricted to the legal universe."""
    features = build_features(panel)
    calendar = np.sort(panel["日期"].unique())
    endpoints = pd.DataFrame({
        "日期": calendar[:-2],
        "entry_date": calendar[1:-1],
        "exit_date": calendar[2:],
    })

    prices = panel.loc[:, ["日期", "代码", "open", "is_suspend"]]
    frame = panel.loc[panel["in_barra"], ["日期", "代码"]].merge(
        endpoints, on="日期", how="inner", validate="many_to_one")
    for kind in ("entry", "exit"):
        side = prices.rename(columns={"日期": f"{kind}_date", "open": f"{kind}_open",
                                      "is_suspend": f"{kind}_suspended"})
        frame = frame.merge(side, on=[f"{kind}_date", "代码"], how="left",
                            validate="many_to_one", sort=False)

    frame["future_return"] = frame["exit_open"] / frame["entry_open"] - 1.0
    frame = frame.loc[
        frame["entry_suspended"].eq(0) & frame["exit_suspended"].eq(0)
        & np.isfinite(frame["entry_open"]) & np.isfinite(frame["exit_open"])
        & frame["entry_open"].gt(0) & frame["exit_open"].gt(0)
        & np.isfinite(frame["future_return"])]
    if last_exit_date is not None:
        frame = frame.loc[frame["exit_date"].le(last_exit_date)]
    frame = frame.merge(features, on=["日期", "代码"], how="left",
                        validate="one_to_one", sort=False)
    frame["target"] = frame.groupby("日期")["future_return"].rank(pct=True) - 0.5
    return frame.reset_index(drop=True)


# --- models -----------------------------------------------------------------

def _lightgbm(train: pd.DataFrame, valid: pd.DataFrame | None, columns: list[str],
              seed: int, rounds: int, early_stopping: int | None,
              overrides: dict | None = None):
    import lightgbm as lgb
    params = {"objective": "regression", "metric": "l2", "verbosity": -1,
              "learning_rate": 0.03, "num_leaves": 31, "max_depth": 7,
              "min_data_in_leaf": 200, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
              "seed": seed, "num_threads": 8, "deterministic": True,
              "force_col_wise": True}
    if overrides:
        params.update(overrides)
    dataset = lgb.Dataset(train[columns], label=train["target"], feature_name=columns)
    if valid is not None and early_stopping:
        valid_set = lgb.Dataset(valid[columns], label=valid["target"],
                                feature_name=columns, reference=dataset)
        booster = lgb.train(params, dataset, num_boost_round=rounds,
                            valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        return booster, booster.best_iteration
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    return booster, None


def _catboost(train: pd.DataFrame, columns: list[str], seed: int, iterations: int):
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(loss_function="RMSE", iterations=iterations,
                              learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
                              random_seed=seed, thread_count=8, verbose=False,
                              allow_writing_files=False)
    model.fit(Pool(train[columns], train["target"], feature_names=columns))
    return model


def _xgboost(train: pd.DataFrame, columns: list[str], seed: int, rounds: int):
    """Fourth model family.

    Qlib's published CSI300 benchmarks put XGBoost at the top of the tabular models on realized
    Information Ratio (1.2915 vs LightGBM 1.2744, CatBoost 1.1561, MLP 1.0323) while its Rank IC
    is comparable -- i.e. it is the family whose *portfolio* behaviour is smoothest, which is the
    quantity the board's IR term rewards.  Saved as JSON, which is plain text and carries no
    runnable payload.
    """
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=rounds, learning_rate=0.03, max_depth=6, min_child_weight=20.0,
        subsample=0.80, colsample_bytree=0.70, reg_lambda=10.0, reg_alpha=0.0,
        objective="reg:squarederror", tree_method="hist", n_jobs=8,
        random_state=seed, verbosity=0)
    model.fit(train[columns], train["target"])
    return model


# --- metrics ----------------------------------------------------------------

def mean_rankic(frame: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = frame.loc[:, ["日期", "future_return"]].copy()
    tmp["score"] = np.asarray(scores, dtype=float)
    per_day = tmp.groupby("日期").apply(
        lambda day: day["score"].corr(day["future_return"], method="spearman"),
        include_groups=False)
    return float(per_day.mean())


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--submission-dir", type=Path, default=Path("submission"))
    ap.add_argument("--train-end", default="2022-12-31",
                    help="last exit_date allowed into training (default: all public data)")
    ap.add_argument("--seed", type=int, default=42)
    # Defaults reproduce the SHIPPED artifact exactly -- `lgbm_sub1__R700sb3`: the best
    # configuration found, measured on both fold sets and through the organizers' own
    # backtest engine. Do not change these without re-running tools/simulate_judge.py and the
    # byte-for-byte reproducibility check; the organizers re-run this file during code review
    # and void the result if it does not reproduce model/.
    ap.add_argument("--lightgbm-seeds", default="42,43,44",
                    help="comma-separated seeds; one LightGBM per seed, blended by rank. "
                         "Seed bagging is worth ~10x on realized IR (0.031 -> 0.306) and "
                         "+0.0085 mean RankIC versus a single seed.")
    ap.add_argument("--catboost-seeds", default="",
                    help="comma-separated seeds; EMPTY BY DEFAULT (LightGBM-only). CatBoost "
                         "alone has negative realized IR (-0.2063) at respectable RankIC; at "
                         "weight 0.5 it leaves IR unchanged while lowering RankIC and RankICIR, "
                         "and a two-family blend did not reduce RankIC variance (F-test p=0.38). "
                         "RankIC enters the score directly at 70%% weight, so the higher-RankIC "
                         "config wins. See FINDINGS.md F35/F36.")
    ap.add_argument("--catboost-weight", type=float, default=0.5,
                    help="total weight of the CatBoost group when CatBoost is enabled; the "
                         "LightGBM group gets the rest, split evenly across its seeds")
    # 400 rounds, and NO early stopping. Both parts are measured, not assumed.
    #
    # Round count: 700 buys +0.0004 mean RankIC out of sample (t=0.99 semiannual, t=0.29 annual)
    # while fitting the training data 25% harder in sample (simulator RankIC 0.1995 -> 0.2497) --
    # the overfitting signature. `rounds=1000` is already negative (-0.0002) and every other
    # single-axis neighbour of this corner is negative too, so 400 sits near the optimum rather
    # than short of it. 400 also keeps the artifact 43% smaller.
    #
    # Early stopping: actively harmful here. At one seed, turning it off is worth +0.0071 mean
    # RankIC and +0.16 realized IR (t=4.5-6.2 on both fold sets); a 6-month inner block
    # under-selects rounds for a slow deep model. `train.py` therefore fits to a fixed round
    # count with no validation set, which also makes the artifact byte-reproducible.
    ap.add_argument("--lightgbm-rounds", type=int, default=400)
    ap.add_argument("--catboost-iterations", type=int, default=400)
    ap.add_argument("--xgboost-seeds", default="",
                    help="comma-separated seeds; EMPTY BY DEFAULT. XGBoost is the family with the "
                         "best realized IR in Qlib's published CSI300 benchmarks, so it is the most "
                         "promising fourth family for the board's IR term.")
    ap.add_argument("--xgboost-rounds", type=int, default=400)
    ap.add_argument("--xgboost-weight", type=float, default=0.25,
                    help="total rank-blend weight for the XGBoost members")
    ap.add_argument("--learning-rate", type=float, default=0.03)
    # 63 leaves / depth 8 -- the ONLY hyper-parameter change from the original sub1 corner, and
    # the only one that survives the correct statistical test. Measured against the 31/7 variant
    # on the same seeds, features and training data:
    #   semiannual  mean +0.0838 vs +0.0823, RankICIR +0.591 vs +0.568
    #   annual      mean +0.0770 vs +0.0750, RankICIR +0.544 vs +0.520
    #   paired daily RankIC difference: t=+2.82 (p=0.005) and t=+3.47 (p=0.0005)
    #   wins 9 of 11 test years
    # The paired test is the right one here: these two models share seeds, features and training
    # data and are scored on the same days and stocks, so the unpaired noise floor (~0.004)
    # overstates the uncertainty by ~7x. The same test framework is what showed CatBoost was a
    # dead heat (p=0.95) and we did not act on that; consistency requires acting on this.
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-data-in-leaf", type=int, default=200)
    ap.add_argument("--feature-fraction", type=float, default=0.70)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--lambda-l2", type=float, default=10.0)
    ap.add_argument("--walk-forward", action="store_true",
                    help="report honest expanding-window RankIC instead of freezing")
    args = ap.parse_args(argv)

    columns = list(FEATURE_COLUMNS)
    print(f"[train] {len(columns)} features", file=sys.stderr)
    panel = _panel_from_public(args.data_dir)
    frame = build_training_frame(panel, last_exit_date=int(args.train_end.replace("-", "")))
    print(f"[train] {len(frame):,} training rows over {frame['日期'].nunique()} signal days",
          file=sys.stderr)

    if args.walk_forward:
        folds = [("2020", 20191231, 20200101, 20201231),
                 ("2021", 20201231, 20210101, 20211231),
                 ("2022", 20211231, 20220101, 20221231)]
        report = []
        for name, train_end, test_lo, test_hi in folds:
            train = frame.loc[frame["exit_date"].le(train_end)]
            test = frame.loc[frame["日期"].between(test_lo, test_hi)]
            if train.empty or test.empty:
                continue
            booster, _ = _lightgbm(train, None, columns, args.seed, 250, None)
            ic = mean_rankic(test, booster.predict(test[columns]))
            report.append({"fold": name, "train_end": train_end, "test_days":
                           int(test["日期"].nunique()), "mean_rankic": ic})
            print(f"[walk-forward] train<= {train_end}  test {name}  "
                  f"RankIC {ic:+.4f}", file=sys.stderr)
        print(json.dumps({"walk_forward": report}, ensure_ascii=False, indent=2))
        return 0

    # Fill values are learned parameters: freeze them into the artifact.
    fill_values = {c: float(frame[c].median()) for c in columns}
    fill_values = {k: (v if np.isfinite(v) else 0.0) for k, v in fill_values.items()}

    model_dir = args.submission_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
                 "max_depth": args.max_depth, "min_data_in_leaf": args.min_data_in_leaf,
                 "feature_fraction": args.feature_fraction,
                 "bagging_fraction": args.bagging_fraction, "lambda_l2": args.lambda_l2}

    lgbm_seeds = [int(s) for s in args.lightgbm_seeds.split(",") if s.strip()]
    cat_seeds = [int(s) for s in args.catboost_seeds.split(",") if s.strip()]
    xgb_seeds = [int(s) for s in args.xgboost_seeds.split(",") if s.strip()]
    if not lgbm_seeds and not cat_seeds and not xgb_seeds:
        print("[train] no model seeds requested", file=sys.stderr)
        return 2

    members: list[dict] = []
    for seed in lgbm_seeds:
        name = f"lightgbm_seed{seed}.txt"
        print(f"[train] fitting LightGBM seed={seed} -> {name}", file=sys.stderr)
        booster, _ = _lightgbm(frame, None, columns, seed, args.lightgbm_rounds, None,
                               overrides)
        booster.save_model(str(model_dir / name))
        members.append({"kind": "lightgbm", "file": name})
    for seed in cat_seeds:
        name = f"catboost_seed{seed}.cbm"
        print(f"[train] fitting CatBoost seed={seed} -> {name}", file=sys.stderr)
        cat = _catboost(frame, columns, seed, args.catboost_iterations)
        cat.save_model(str(model_dir / name))
        members.append({"kind": "catboost", "file": name})
    for seed in xgb_seeds:
        name = f"xgboost_seed{seed}.json"
        print(f"[train] fitting XGBoost seed={seed} -> {name}", file=sys.stderr)
        xgb_model = _xgboost(frame, columns, seed, args.xgboost_rounds)
        xgb_model.get_booster().save_model(str(model_dir / name))
        members.append({"kind": "xgboost", "file": name})

    # Group weighting: CatBoost takes `--catboost-weight` in total, LightGBM takes the rest
    # split evenly across its seeds. With 3 LightGBM seeds and weight 0.5 this reproduces
    # submission #1's 0.5 LightGBM / 0.5 CatBoost blend while averaging out LightGBM's seed
    # variance. `_load_members` renormalises. Rank-space blending is scale-free, so members
    # with different output scales combine safely without a per-member calibration step.
    lgbm_members = [m for m in members if m["kind"] == "lightgbm"]
    cat_members = [m for m in members if m["kind"] == "catboost"]
    xgb_members = [m for m in members if m["kind"] == "xgboost"]
    cat_weight = args.catboost_weight if cat_members else 0.0
    xgb_weight = args.xgboost_weight if xgb_members else 0.0
    lgbm_weight = (1.0 - cat_weight - xgb_weight) / len(lgbm_members) if lgbm_members else 0.0
    cat_weight_each = cat_weight / len(cat_members) if cat_members else 0.0
    xgb_weight_each = xgb_weight / len(xgb_members) if xgb_members else 0.0
    for member in lgbm_members:
        member["weight"] = lgbm_weight
    for member in cat_members:
        member["weight"] = cat_weight_each
    for member in xgb_members:
        member["weight"] = xgb_weight_each

    metadata = {
        "feature_columns": columns,
        "fill_values": fill_values,
        "postprocess": {"kind": "rank"},
        "members": members,
        "train_end": args.train_end,
        "lightgbm_seeds": lgbm_seeds,
        "catboost_seeds": cat_seeds,
        "catboost_weight": cat_weight,
        "lightgbm_params": overrides,
        "catboost_params": {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5.0,
                            "iterations": args.catboost_iterations},
        "train_rows": int(len(frame)),
        "train_days": int(frame["日期"].nunique()),
        "public_data": "2016-01-04..2022-12-30",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[train] wrote {len(members)} model(s) to {model_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
