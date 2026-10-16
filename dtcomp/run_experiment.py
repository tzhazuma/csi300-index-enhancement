"""Walk-forward RankIC harness for the CSI300 cross-sectional challenge.

Usage
-----
    # one named config over the headline (annual) folds
    .venv/bin/python -m dtcomp.run_experiment --config lgbm_rank_csrank --folds annual

    # the whole built-in zoo, all fold sets
    .venv/bin/python -m dtcomp.run_experiment --config all --folds annual+semiannual

    # build a blend from cached member predictions (no retraining)
    .venv/bin/python -m dtcomp.run_experiment --blend lgbm_rank_csrank,cat_reg_csrank \
        --name blend_lgbm_cat --folds annual --blend-weights search

    # rebuild reports/leaderboard_local.md from reports/*.json
    .venv/bin/python -m dtcomp.run_experiment --rebuild-report

Protocol
--------
Two evaluation protocols, both strictly time-ordered:

``nested`` (default, honest)
    For fold ``(train_end, valid_start, valid_end)`` the *outer* training window is
    every signal date <= ``train_end``.  Its last ``inner_months`` are carved off as
    an inner validation block; the model is fitted on the remaining inner-training
    dates, early-stopped on the inner block, then refitted on the whole outer window
    at the selected complexity and scored on the outer block.  The outer block is
    never used for any selection decision, so the reported RankIC is out-of-sample.

``leaky``
    Fit on the outer window, early-stop on the outer block, report on the same block.
    This is what the organizers' ``train.py`` does.  It is kept only so the optimism
    gap can be quantified; **never** quote a ``leaky`` number as a result.

Evaluation universe
-------------------
The judge scores every member of the day's ``Barra_factor`` table, including
suspended names, and computes Spearman over all of them.  Every metric here is
therefore computed on the full ``in_barra`` universe (all rows with a finite
``future_return``), *not* on the tradability-filtered subset used for fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dtcomp.cv import daily_rankic, ic_summary, rankic_by_year
from dtcomp.data import BARRA_COLS, FACTOR33, build_labels, load_panel, tradable_mask
from dtcomp.models import CONFIGS, DATE, TARGET, FeatureProcessor, available_configs, train_model

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data"
DEFAULT_REPORTS = REPO / "reports"

# ---------------------------------------------------------------------------
# feature source: dtcomp.features (owned by feature-eng) with a safe fallback
# ---------------------------------------------------------------------------

FALLBACK_DERIVED = ("close_to_open", "high_low_range", "log_float_cap",
                    "log_total_cap", "amount_per_volume", "amount_to_floatcap")


def _fallback_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Provided columns plus a handful of legal as-of-t ratios.

    Everything here is computable from day-t close information only.
    """
    frame = panel.loc[:, [DATE, "代码", *FACTOR33, *BARRA_COLS]].copy()
    close = panel["close"].astype("float64")
    open_ = panel["open"].astype("float64")
    frame["close_to_open"] = close / open_ - 1.0
    frame["high_low_range"] = panel["high"].astype("float64") / panel["low"].astype("float64") - 1.0
    frame["log_float_cap"] = np.log(panel["float_market_cap"].astype("float64").clip(lower=1.0))
    frame["log_total_cap"] = np.log(panel["total_market_cap"].astype("float64").clip(lower=1.0))
    frame["amount_per_volume"] = panel["amount"].astype("float64") / panel["volume"].astype("float64").replace(0.0, np.nan)
    frame["amount_to_floatcap"] = panel["amount"].astype("float64") / panel["float_market_cap"].astype("float64").replace(0.0, np.nan)
    columns = [*FACTOR33, *BARRA_COLS, *FALLBACK_DERIVED]
    frame[list(columns)] = frame[list(columns)].replace([np.inf, -np.inf], np.nan)
    return frame, list(columns)


def load_feature_frame(panel: pd.DataFrame, source: str = "auto") -> tuple[pd.DataFrame, list[str], str]:
    """Return ``(frame_with_日期/代码/features, feature_columns, source_label)``.

    Contract confirmed with ``feature-eng``:
    ``build_features(panel, lookback=120) -> DataFrame`` with ``日期``, ``代码`` plus
    exactly ``FEATURE_COLUMNS``, same length and same row order as ``panel``.
    We re-key by (日期, 代码) taken from ``panel`` so a stray index cannot misalign it.
    """
    if source in ("auto", "module"):
        feature_module = None
        try:
            from dtcomp import features as feature_module  # local import: module may not exist yet
        except Exception as exc:  # pragma: no cover - depends on teammate timing
            if source == "module":
                raise
            print(f"[features] dtcomp.features unavailable ({exc.__class__.__name__}: {exc}); "
                  f"falling back to provided columns", file=sys.stderr)
        if feature_module is not None:
            build = getattr(feature_module, "build_features", None)
            columns = getattr(feature_module, "FEATURE_COLUMNS", None)
            if callable(build) and columns:
                try:
                    built = build(panel)
                    if len(built) != len(panel):
                        raise ValueError(f"build_features returned {len(built)} rows for a "
                                         f"{len(panel)}-row panel")
                    missing = [c for c in columns if c not in built.columns]
                    if missing:
                        raise KeyError(f"dtcomp.features.FEATURE_COLUMNS lists {len(missing)} "
                                       f"column(s) build_features did not return: {missing[:8]}")
                except Exception as exc:
                    # feature-eng's module is under active development; never let a
                    # broken import block the modelling harness.
                    if source == "module":
                        raise
                    print(f"[features] dtcomp.features.build_features failed "
                          f"({exc.__class__.__name__}: {exc}); falling back to provided columns",
                          file=sys.stderr)
                else:
                    frame = pd.DataFrame({DATE: panel[DATE].to_numpy(),
                                          "代码": panel["代码"].to_numpy()})
                    for column in columns:
                        frame[column] = built[column].to_numpy()
                    return frame, list(columns), "dtcomp.features"
            else:
                print("[features] dtcomp.features has no build_features/FEATURE_COLUMNS; "
                      "falling back", file=sys.stderr)
    frame, columns = _fallback_features(panel)
    return frame, columns, "fallback"


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    frame: pd.DataFrame          # all in_barra rows, features + label + is_tradable
    feature_columns: list[str]
    feature_source: str
    calendar: np.ndarray


def build_dataset(data_dir: Path, feature_source: str = "auto") -> Dataset:
    panel = load_panel(data_dir)
    labels_path = data_dir / "labels_open.parquet"
    full_labels = pd.read_parquet(labels_path) if labels_path.exists() else build_labels(panel)
    tradable = tradable_mask(full_labels)
    labels = full_labels.loc[:, [DATE, "代码", "future_return"]].copy()
    labels["is_tradable"] = tradable.to_numpy()

    features, columns, source = load_feature_frame(panel, feature_source)
    universe = panel.loc[panel["in_barra"], [DATE, "代码"]]
    frame = universe.merge(labels, on=[DATE, "代码"], how="left", validate="one_to_one")
    frame = frame.merge(features, on=[DATE, "代码"], how="left", validate="one_to_one")
    frame = frame.loc[np.isfinite(frame[TARGET].to_numpy(dtype="float64"))].copy()
    frame["is_tradable"] = frame["is_tradable"].fillna(False).astype(bool)
    frame = frame.sort_values([DATE, "代码"], kind="stable").reset_index(drop=True)
    calendar = np.sort(frame[DATE].unique())
    return Dataset(frame=frame, feature_columns=list(columns),
                   feature_source=source, calendar=calendar)


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    name: str
    train_end: int
    valid_start: int
    valid_end: int


def _last_on_or_before(calendar: np.ndarray, date: int) -> int | None:
    candidates = calendar[calendar <= date]
    return int(candidates[-1]) if len(candidates) else None


def make_folds(calendar: np.ndarray, kind: str) -> list[Fold]:
    """Expanding-window folds with explicit calendar boundaries.

    ``annual``      train_end = Dec 31 of Y, test = calendar year Y+1, Y in 2017..2021.
                    The last three reproduce the Lead's FINDINGS F2 boundaries
                    (train<=2019 -> 2020, train<=2020 -> 2021, train<=2021 -> 2022).
    ``semiannual``  train_end = Jun 30 / Dec 31, test = the following half year,
                    train_end from 2017-12 to 2022-06 (10 folds, 2018H1..2022H2).
    """
    folds: list[Fold] = []
    if kind == "annual":
        for year in range(2017, 2022):
            train_end = _last_on_or_before(calendar, year * 10000 + 1231)
            if train_end is None:
                continue
            valid_start = _last_on_or_before(calendar, year * 10000 + 1231)
            nxt = calendar[calendar > valid_start]
            if len(nxt) == 0:
                continue
            window = nxt[nxt <= (year + 1) * 10000 + 1231]
            if len(window) < 20:
                continue
            folds.append(Fold(name=f"test{year + 1}", train_end=train_end,
                              valid_start=int(window[0]), valid_end=int(window[-1])))
    elif kind == "semiannual":
        ends = [(y, m) for y in range(2017, 2023) for m in (6, 12)]
        for year, month in ends:
            day = 30 if month == 6 else 31
            train_end = _last_on_or_before(calendar, year * 10000 + month * 100 + day)
            if train_end is None:
                continue
            after = calendar[calendar > train_end]
            if len(after) == 0:
                continue
            if month == 6:
                limit = year * 10000 + 1231
                label = f"{year}H2"
            else:
                limit = (year + 1) * 10000 + 630
                label = f"{year + 1}H1"
            window = after[after <= limit]
            if len(window) < 20:
                continue
            folds.append(Fold(name=label, train_end=train_end,
                              valid_start=int(window[0]), valid_end=int(window[-1])))
    elif kind == "longholdout":
        # TRANSFER PROXY.  The board trains us through 2022-12-31 and scores us on data AFTER
        # 2022 (Problem.txt: "在 2016-2022 年之外，组织方另外保留未来市场数据作为隐藏评测集"),
        # i.e. a long holdout in a later regime.  The expanding-window folds do not reproduce
        # that shape: their test windows are short and sit immediately after the training cut.
        #
        # This fold is the closest available analogue -- train <= 2020-12-31, test the whole of
        # 2021-2022 (~485 days).  A recipe that holds up here is a better bet to hold up on 2023+
        # than one that only wins on short adjacent windows.
        cut = _last_on_or_before(calendar, 20201231)
        if cut is not None:
            after = calendar[calendar > cut]
            window = after[after <= 20221230]
            if len(window) >= 20:
                folds.append(Fold(name="test2021_2022", train_end=cut,
                                  valid_start=int(window[0]), valid_end=int(window[-1])))
    else:
        raise ValueError(f"unknown fold set {kind!r}")
    return folds


def inner_split(calendar: np.ndarray, train_end: int, inner_months: int,
                min_train_days: int = 120) -> tuple[int | None, int | None]:
    """Last ``inner_months`` of the outer training window become the inner block."""
    train_dates = calendar[calendar <= train_end]
    if len(train_dates) < min_train_days:
        return None, None
    last = pd.Timestamp(str(int(train_end)))
    cutoff = int((last - pd.DateOffset(months=inner_months)).strftime("%Y%m%d"))
    inner = train_dates[train_dates > cutoff]
    if len(inner) < 20 or len(train_dates) - len(inner) < min_train_days:
        return None, None
    return int(inner[0]), int(inner[-1])


# ---------------------------------------------------------------------------
# running one config over one fold set
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: str
    train_end: int
    valid_start: int
    valid_end: int
    train_rows: int
    valid_rows: int
    best_iteration: int | None
    rounds_used: int | None
    metrics: dict[str, float]
    seconds: float


def _score_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = frame.loc[:, [DATE, "代码", TARGET]].copy()
    out["score"] = np.asarray(scores, dtype="float64")
    return out.loc[np.isfinite(out["score"]) & np.isfinite(out[TARGET])]


def _trim_label_overlap(frame: pd.DataFrame, cutoff: int, calendar: np.ndarray,
                        ahead: int = 2) -> pd.DataFrame:
    """Training rows with ``日期 <= cutoff``, minus the last ``ahead`` signal dates.

    The label for signal date ``t`` is ``open(t+2)/open(t+1) - 1``, so the very last
    training rows would read prices from the first days of the evaluation block.  It
    is a 2-day overlap of *prices*, not of evaluation returns, but removing it costs
    two rows out of ~1500 days and makes the train/test boundary exact.
    """
    within = calendar[calendar <= cutoff]
    if len(within) <= ahead:
        return frame.iloc[0:0]
    last_clean = int(within[-(ahead + 1)])
    return frame.loc[frame[DATE] <= last_clean]


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base``, returning a new dict.

    A shallow ``dict.update`` silently *replaces* nested dicts: an override of
    ``params.min_data_in_leaf`` would wipe ``learning_rate``, ``num_leaves`` and the
    rest, leaving LightGBM's defaults in place.  That is exactly the failure this
    function exists to prevent, so nested mappings are merged key by key.
    """
    out: dict[str, Any] = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def run_config_over_folds(dataset: Dataset, config_name: str, folds: list[Fold], *,
                          protocol: str = "nested", inner_months: int = 6,
                          train_filter: str = "tradable", num_threads: int = 4,
                          seeds: list[int] | None = None,
                          overrides: dict | None = None,
                          verbose: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit ``config_name`` on every fold and return (report, outer_preds, inner_preds)."""
    config = _deep_merge(CONFIGS[config_name], overrides or {})
    # Everything the config does not pin down itself comes from the CLI.
    #
    # ORDER MATTERS.  `common` must spread AFTER `config`: with `{**common, **config}` the config's
    # own `seeds` (e.g. [42, 43, 44]) silently wins over `--seeds`, so `--seeds 42,43,44,45,46`
    # ran three seeds while the report recorded five -- a control that looked like it worked and
    # did nothing.  `refit_full` stays last so it is always False for the selection pass.
    common = {"num_threads": num_threads}
    if seeds is not None:
        common["seeds"] = seeds
    selection_kwargs = {**config, **common, "refit_full": False}
    frame = dataset.frame
    columns = dataset.feature_columns
    outer_rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    results: list[FoldResult] = []

    fit_frame = frame if train_filter == "all" else frame.loc[frame["is_tradable"]]
    for fold in folds:
        started = time.time()
        outer_train = _trim_label_overlap(fit_frame, fold.train_end, dataset.calendar)
        valid = frame.loc[(frame[DATE] >= fold.valid_start) & (frame[DATE] <= fold.valid_end)]
        if outer_train.empty or valid.empty:
            continue
        best_iteration: int | None = None
        uses_early_stopping = bool(config.get("early_stopping"))
        if protocol == "nested":
            if uses_early_stopping:
                inner_start, inner_end = inner_split(dataset.calendar, fold.train_end, inner_months)
                if inner_start is None:
                    if verbose:
                        print(f"[{config_name}] fold {fold.name}: no room for an inner block, "
                              f"skipped")
                    continue
                inner_train = fit_frame.loc[fit_frame[DATE] < inner_start]
                inner_valid = fit_frame.loc[(fit_frame[DATE] >= inner_start)
                                            & (fit_frame[DATE] <= inner_end)]
                selection = train_model(inner_train, inner_valid, feature_columns=columns,
                                        seed=42, **selection_kwargs)
                best_iteration = getattr(selection, "best_iteration", None)
                if best_iteration is None:
                    best_iteration = int(config.get("num_boost_round", 400))
                # Refit 10% longer than the selected point: the inner block is a
                # strict prefix of the outer window, so the outer fit sees slightly
                # more data per unit of complexity.
                rounds = max(int(round(best_iteration * 1.1)), 1)
                inner_pred = selection.predict(inner_valid)
                inner_block = _score_frame(inner_valid, inner_pred)
                inner_block["fold"] = fold.name
                inner_rows.append(inner_block)
            else:
                # Fixed-complexity config (what the shipped artifact uses): there is
                # nothing to select, so skip the inner fit entirely.  Use the round
                # count verbatim -- applying the 1.1 refit factor here would silently
                # train 10% more trees than the config asks for.
                rounds = int(config.get("num_boost_round", 400))

            final_kwargs = {**common, "refit_full": False, **config,
                            "num_boost_round": rounds, "early_stopping": None}
            final = train_model(outer_train, None, feature_columns=columns, seed=42,
                                **final_kwargs)
        elif protocol == "leaky":
            final = train_model(outer_train, valid, feature_columns=columns, seed=42,
                                refit_full=True, **{**common, **config})
            best_iteration = getattr(final, "best_iteration", None)
        else:
            raise ValueError(f"unknown protocol {protocol!r}")

        scores = final.predict(valid)
        block = _score_frame(valid, scores)
        block["fold"] = fold.name
        outer_rows.append(block)
        metrics = ic_summary(block)
        expected_days = int(valid[DATE].nunique())
        metrics["expected_days"] = expected_days
        metrics["degenerate_days"] = expected_days - int(metrics["days"])
        if metrics["degenerate_days"] > 0:
            # A day whose scores are constant (or all-NaN) yields an undefined
            # Spearman and used to vanish silently, inflating the reported mean.
            print(f"[{config_name}] fold {fold.name}: WARNING {metrics['degenerate_days']}"
                  f"/{expected_days} days had undefined daily RankIC "
                  f"(constant or non-finite scores) -- treat this fold as unreliable",
                  file=sys.stderr, flush=True)
        results.append(FoldResult(
            fold=fold.name, train_end=fold.train_end, valid_start=fold.valid_start,
            valid_end=fold.valid_end, train_rows=int(len(outer_train)),
            valid_rows=int(len(valid)), best_iteration=best_iteration,
            rounds_used=int(rounds), metrics=metrics,
            seconds=round(time.time() - started, 1)))
        if verbose:
            print(f"[{config_name}] fold {fold.name:8s} train<={fold.train_end} "
                  f"rankIC={metrics['mean_rankic']:+.4f} IR={metrics['rankicir']:+.3f} "
                  f"days={metrics['days']:3d}/{expected_days} rounds={rounds} "
                  f"({results[-1].seconds:.0f}s)", flush=True)

    pooled = pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame(
        columns=[DATE, "代码", TARGET, "score", "fold"])
    inner_pooled = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame(
        columns=[DATE, "代码", TARGET, "score", "fold"])

    report = _build_report(config_name, config, dataset, folds, results, pooled, protocol,
                           inner_months, train_filter, seeds)
    return report, pooled, inner_pooled


def _build_report(config_name: str, config: dict, dataset: Dataset, folds: list[Fold],
                  results: list[FoldResult], pooled: pd.DataFrame, protocol: str,
                  inner_months: int, train_filter: str,
                  seeds: list[int] | None) -> dict:
    summary = ic_summary(pooled) if len(pooled) else {}
    by_year = rankic_by_year(pooled) if len(pooled) else pd.DataFrame()
    year_map = {}
    for row in by_year.to_dict("records"):
        year_map[str(int(row["year"]))] = {k: _clean(v) for k, v in row.items() if k != "year"}
    fold_rows = []
    for result in results:
        entry = {"fold": result.fold, "train_end": result.train_end,
                 "valid_start": result.valid_start, "valid_end": result.valid_end,
                 "train_rows": result.train_rows, "valid_rows": result.valid_rows,
                 "best_iteration": result.best_iteration,
                 "rounds_used": result.rounds_used,
                 "seconds": result.seconds}
        entry.update({k: _clean(v) for k, v in result.metrics.items()})
        fold_rows.append(entry)

    year_ics = {y: v["mean_rankic"] for y, v in year_map.items()}
    worst_year = min(year_ics, key=year_ics.get) if year_ics else None
    best_year = max(year_ics, key=year_ics.get) if year_ics else None
    concentration = None
    if len(year_ics) >= 3 and summary.get("mean_rankic"):
        # share of the total per-year IC contributed by the single best year
        total = sum(year_ics.values())
        concentration = float(year_ics[best_year] / total) if total else None
    expected_days = sum(int(f.get("expected_days", f.get("days", 0))) for f in fold_rows)
    degenerate_days = sum(int(f.get("degenerate_days", 0)) for f in fold_rows)

    # The final submission model trains through 2022-12-31, so folds with the most
    # training history are the closest proxy for how it will behave on hidden data.
    # A config whose edge lives only in the short-history folds is a liability.
    recent_years = [y for y in sorted(year_ics) if int(y) >= 2021]
    recent_days = sum(int(year_map[y].get("days", 0)) for y in recent_years)
    recent_mean = None
    if recent_years and recent_days:
        recent_mean = float(sum(year_ics[y] * year_map[y].get("days", 0)
                                for y in recent_years) / recent_days)
    return {
        "config": config_name,
        "protocol": protocol,
        "inner_months": inner_months,
        "train_filter": train_filter,
        "feature_source": dataset.feature_source,
        "feature_count": len(dataset.feature_columns),
        "features": list(dataset.feature_columns),
        "seeds": seeds,
        "config_kwargs": _jsonable(config),
        # Surfaced explicitly so a shallow-merge / override bug is visible in the
        # report itself rather than only in the conclusions drawn from it: if
        # `params_keys` is smaller than the base config's, an override clobbered the
        # nested dict.
        "effective_params": _jsonable(config.get("params") or {}),
        "params_keys": len(config.get("params") or {}),
        "effective_target": config.get("target"),
        "effective_preprocess": config.get("preprocess"),
        "effective_num_boost_round": config.get("num_boost_round"),
        "effective_early_stopping": config.get("early_stopping"),
        "effective_time_decay_trading_days": config.get("time_decay_trading_days"),
        "effective_time_decay_days": config.get("time_decay_days"),
        "effective_sample_weighting": (
            f"0.5**(trading_days_ago/{config['time_decay_trading_days']})"
            if config.get("time_decay_trading_days")
            else (f"0.5**(calendar_days_ago/{config['time_decay_days']})"
                  if config.get("time_decay_days") else "none (equal weights)")),
        "folds": fold_rows,
        "summary": {k: _clean(v) for k, v in summary.items()},
        "by_year": year_map,
        "worst_year": worst_year,
        "best_year": best_year,
        "best_year_ic_share": concentration,
        "recent_years": recent_years,
        "recent_days": recent_days,
        "recent_mean_rankic": _clean(recent_mean) if recent_mean is not None else None,
        "expected_days": expected_days,
        "degenerate_days": degenerate_days,
        "reliable": degenerate_days == 0,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _clean(value):
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _clean(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# blends built from cached member predictions
# ---------------------------------------------------------------------------

def build_blend_from_cache(dataset: Dataset, members: list[str], name: str,
                           foldset: str, reports_dir: Path, *,
                           mode: str = "equal", step: float = 0.1,
                           verbose: bool = True) -> dict | None:
    """Rank-average cached member predictions.

    ``mode="equal"``   fixed equal weights (no selection at all);
    ``mode="search"``  per-fold weights fitted on that fold's *inner* validation
                       predictions only, so the outer block stays untouched.
    """
    outer = {}
    inner = {}
    for member in members:
        outer_path = reports_dir / "preds" / f"{member}__{foldset}__outer.parquet"
        if not outer_path.exists():
            print(f"[blend] missing {outer_path}; run --config {member} first", file=sys.stderr)
            return None
        outer[member] = pd.read_parquet(outer_path)
        inner_path = reports_dir / "preds" / f"{member}__{foldset}__inner.parquet"
        inner[member] = pd.read_parquet(inner_path) if inner_path.exists() else None

    from dtcomp.models import search_blend_weights

    blocks = []
    weight_log = []
    for fold in sorted(outer[members[0]]["fold"].unique()):
        merged = None
        for member in members:
            part = outer[member].loc[outer[member]["fold"] == fold, [DATE, "代码", TARGET, "score"]]
            part = part.rename(columns={"score": f"s_{member}"})
            merged = part if merged is None else merged.merge(part, on=[DATE, "代码", TARGET])
        preds = [merged[f"s_{m}"].to_numpy() for m in members]
        if mode == "search" and all(inner[m] is not None for m in members):
            imerged = None
            for member in members:
                part = inner[member].loc[inner[member]["fold"] == fold, [DATE, "代码", TARGET, "score"]]
                part = part.rename(columns={"score": f"s_{member}"})
                imerged = part if imerged is None else imerged.merge(part, on=[DATE, "代码", TARGET])
            if imerged is not None and len(imerged) > 100:
                weights = search_blend_weights([imerged[f"s_{m}"].to_numpy() for m in members],
                                               imerged[DATE].to_numpy(),
                                               imerged[TARGET].to_numpy(), step=step)
            else:
                weights = [1.0] * len(members)
        else:
            weights = [1.0] * len(members)
        from dtcomp.models import rank_average
        merged["score"] = rank_average(preds, weights)
        merged["fold"] = fold
        blocks.append(merged.loc[:, [DATE, "代码", TARGET, "score", "fold"]])
        weight_log.append({"fold": fold, "weights": {m: round(w, 4) for m, w in zip(members, weights)}})
        if verbose:
            print(f"[blend:{name}] fold {fold:8s} weights="
                  f"{ {m: round(w, 3) for m, w in zip(members, weights)} }", flush=True)

    pooled = pd.concat(blocks, ignore_index=True)
    config = {"kind": "blend", "members": members, "weight_mode": mode}
    report = _build_report(name, config, dataset, [], [], pooled, "blend-from-cache",
                           0, "tradable", None)
    report["blend_members"] = members
    report["blend_weight_mode"] = mode
    report["blend_weights"] = weight_log
    report["folds"] = [
        {"fold": fold, **{k: _clean(v) for k, v in ic_summary(
            pooled.loc[pooled["fold"] == fold]).items()}}
        for fold in sorted(pooled["fold"].unique())
    ]
    return report, pooled


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def write_report(report: dict, foldset: str, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report["foldset"] = foldset
    path = reports_dir / f"{report['config']}__{foldset}.json"
    path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def append_leaderboard(report: dict, foldset: str, reports_dir: Path) -> None:
    path = reports_dir / "leaderboard_local.md"
    if not path.exists():
        path.write_text(_LEADERBOARD_HEADER, encoding="utf-8")
    row = _leaderboard_row(report, foldset)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


_LEADERBOARD_HEADER = """# Local walk-forward leaderboard

Every row is an **out-of-sample** walk-forward result: the reported block was never in
that model's training window, and (for `protocol=nested`) never used for any selection
decision either.  `leaky` rows are marked and exist only to quantify the optimism gap --
they are not results.

Metric: mean daily Spearman RankIC over the **full 300-name Barra universe** (suspended
names included), matching the judge.  `train_filter=tradable` only restricts which rows
are used to *fit*.

Fold sets: `annual` = train<=Dec31 of Y, test = calendar year Y+1 (2018..2022);
`semiannual` = rolling half-year test blocks (2017H2..2022H2, model always fresh).

`features` is the feature source: `dtcomp.features` = the 136-column frozen library,
`fallback` = the 49 provided/derived columns.  `recent` is the mean RankIC over test
blocks in 2021+, i.e. the folds with the longest training history, which most resemble
the final submission (trained through 2022-12-31).

> ## READ THIS BEFORE SELECTING ON ANY ROW BELOW
>
> **Rank by mean RankIC, but only act on differences of about +-0.005 or more that also
> hold on the other fold set.**  Everything smaller is noise, however consistent it looks.
>
> Calibration against the only ground truth we have (the public leaderboard): the
> architecture that scored **rankic 0.0712 / ir 2.8105** on the hidden period and the one
> that scored **rankic 0.0602 / ir 1.0785** are separated here by only **+0.0011 (annual)**
> and **+0.0008 (semiannual)** mean RankIC -- paired SE 0.0007-0.0008, t = 1.1-1.5, *not*
> significant.  The folds get the RankIC **direction** right (which is what matters, since
> realized IR tracks RankIC at Spearman ~+0.88 across configs on the official engine) but
> they understate the realised gap by roughly 10x.  They cannot separate close configs.
>
> Noise floor: with ~1200-1350 evaluation days and a daily RankIC sd of ~0.14, the unpaired
> standard error on a fold mean is ~0.004; for two correlated models the *paired* standard
> error is ~0.0005-0.0012.  Picking the max of a sweep over N configs is biased upward by
> roughly 1-2 unpaired SEs.
>
> What these rows *are* good for: rejecting clearly broken configs (non-finite or constant
> scores, `(degenerate)` rows, target/model families that fail across every fold), and
> confirming large effects (e.g. `preprocess=raw` vs `cs_rank`, seed bagging) that repeat
> in every fold and every year.

**Before quoting any row, check `feature_count` and `params_keys` in its JSON.** A
`feature_count` of 49 means the run silently fell back to the provided columns instead of the
136-column library. A `params_keys` smaller than the base config's means an override
clobbered the nested `params` dict and the run used library defaults. Both failures produce
plausible-looking numbers.

| config | fold set | features | n_params | protocol | mean RankIC | RankICIR | pos. ratio | days | worst year | recent | by year |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def _leaderboard_row(report: dict, foldset: str) -> str:
    summary = report.get("summary") or {}
    years = report.get("by_year") or {}
    by_year = " ".join(f"{y}:{v['mean_rankic']:+.3f}" if v.get("mean_rankic") is not None
                       else f"{y}:n/a" for y, v in sorted(years.items()))
    def fmt(key, spec="{:.4f}"):
        value = summary.get(key)
        return spec.format(value) if value is not None else "n/a"
    recent = report.get("recent_mean_rankic")
    recent_text = f"{recent:+.4f}" if isinstance(recent, (int, float)) else "n/a"
    reliable = "" if report.get("reliable", True) else " **(degenerate)**"
    n_params = report.get("params_keys", "n/a")
    return (f"| {report['config']}{reliable} | {foldset} | "
            f"{report.get('feature_source', '?')} | {n_params} | {report['protocol']} | "
            f"{fmt('mean_rankic', '{:+.4f}')} | {fmt('rankicir', '{:+.3f}')} | "
            f"{fmt('positive_ratio', '{:.3f}')} | {summary.get('days', 'n/a')} | "
            f"{report.get('worst_year', 'n/a')} | {recent_text} | {by_year} |\n")


def rebuild_leaderboard(reports_dir: Path) -> None:
    """Recreate leaderboard_local.md from every ``reports/*__*.json`` on disk."""
    path = reports_dir / "leaderboard_local.md"
    rows = []
    for json_path in sorted(reports_dir.glob("*__*.json")):
        report = json.loads(json_path.read_text(encoding="utf-8"))
        # Prefer the foldset stored inside the report: config names may themselves
        # contain '__' (e.g. 'lgbm_rank_raw__rankz__annual.json').
        foldset = report.get("foldset") or json_path.stem.split("__")[-1]
        rows.append((report.get("summary", {}).get("mean_rankic") or -9.0,
                     _leaderboard_row(report, foldset)))
    rows.sort(key=lambda item: -item[0])
    path.write_text(_LEADERBOARD_HEADER + "".join(row for _, row in rows), encoding="utf-8")
    print(f"[report] wrote {path} with {len(rows)} row(s)")


def compare_configs(reports_dir: Path, name_a: str, name_b: str, foldset: str) -> dict | None:
    """Paired per-day comparison of two configs over the same folds.

    The headline difference (mean RankIC A - B) is not enough to choose between two
    configs: with ~1200 days and a per-day IC sd of ~0.15 the standard error of the
    *difference* is ~0.004, so anything smaller is noise.  This reports the paired
    standard error and t-statistic -- the only honest way to prefer one over another.
    """
    from dtcomp.cv import daily_rankic
    paths = {name_a: reports_dir / "preds" / f"{name_a}__{foldset}__outer.parquet",
             name_b: reports_dir / "preds" / f"{name_b}__{foldset}__outer.parquet"}
    for name, path in paths.items():
        if not path.exists():
            print(f"[compare] missing {path}", file=sys.stderr)
            return None
    blocks = {name: pd.read_parquet(path) for name, path in paths.items()}
    ic_a = daily_rankic(blocks[name_a].rename(columns={name_a: "score"}), "score")
    ic_b = daily_rankic(blocks[name_b].rename(columns={name_b: "score"}), "score")
    aligned = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    diff = aligned["a"] - aligned["b"]
    n = int(len(diff))
    mean_diff = float(diff.mean()) if n else float("nan")
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t_stat = mean_diff / se if se and np.isfinite(se) and se > 0 else None
    result = {
        "a": name_a, "b": name_b, "foldset": foldset, "days": n,
        "mean_a": float(aligned["a"].mean()) if n else None,
        "mean_b": float(aligned["b"].mean()) if n else None,
        "mean_diff": mean_diff, "paired_se": se, "t_stat": t_stat,
        "significant_5pct": bool(t_stat is not None and abs(t_stat) > 1.96),
        "fold_diffs": {},
    }
    for fold in sorted(blocks[name_a]["fold"].unique()):
        sub_a = daily_rankic(blocks[name_a].loc[blocks[name_a]["fold"] == fold]
                             .rename(columns={name_a: "score"}), "score")
        sub_b = daily_rankic(blocks[name_b].loc[blocks[name_b]["fold"] == fold]
                             .rename(columns={name_b: "score"}), "score")
        both = pd.concat([sub_a.rename("a"), sub_b.rename("b")], axis=1).dropna()
        result["fold_diffs"][fold] = {
            "a": _clean(both["a"].mean()), "b": _clean(both["b"].mean()),
            "diff": _clean((both["a"] - both["b"]).mean())}
    return _jsonable(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_overrides(spec: str | None) -> dict | None:
    """``"k=v,k2=v2"`` -> dict, values parsed as JSON with a plain-string fallback.

    Shell-friendly on purpose: a JSON object on the command line gets its quotes
    eaten by xargs/shell word splitting, which silently produced bad overrides.
    Dotted keys merge into nested dicts, so ``params.learning_rate=0.02`` reaches
    ``config["params"]["learning_rate"]`` without needing a quoted JSON blob.
    """
    if not spec:
        return None
    out: dict[str, Any] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--override entry {part!r} is not key=value")
        key, _, raw = part.partition("=")
        key = key.strip()
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.split(".")
        for step in parts[:-1]:
            node = node.setdefault(step, {})
            if not isinstance(node, dict):
                raise ValueError(f"--override key {key!r} conflicts with a scalar value")
        node[parts[-1]] = value
    return out


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="config name from dtcomp.models.CONFIGS, or 'all'")
    parser.add_argument("--blend", default=None,
                        help="comma-separated member configs; builds a blend from cached preds")
    parser.add_argument("--name", default=None, help="override the report/config name")
    parser.add_argument("--folds", default="annual",
                        help="'+'-joined fold sets: annual, semiannual (default: annual)")
    parser.add_argument("--protocol", default="nested", choices=["nested", "leaky"])
    parser.add_argument("--inner-months", type=int, default=6)
    parser.add_argument("--train-filter", default="tradable", choices=["tradable", "all"])
    parser.add_argument("--feature-source", default="auto", choices=["auto", "module", "fallback"])
    parser.add_argument("--seeds", default=None, help="comma-separated seeds for seed-bagging")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--override", default=None,
                        help="comma-separated key=value pairs merged into the config, "
                             "values parsed as JSON, e.g. "
                             "--override num_boost_round=150,early_stopping=null")
    parser.add_argument("--tag", default=None,
                        help="suffix appended to the config name in the report/pred files")
    parser.add_argument("--blend-weights", default="equal", choices=["equal", "search"])
    parser.add_argument("--compare", default=None, nargs=2, metavar=("A", "B"),
                        help="paired per-day RankIC comparison of two cached configs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--rebuild-report", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.list_configs:
        print("\n".join(available_configs()))
        return 0
    if args.rebuild_report:
        rebuild_leaderboard(args.reports_dir)
        return 0
    if args.compare:
        foldset = [p.strip() for p in args.folds.split("+") if p.strip()][0]
        result = compare_configs(args.reports_dir, args.compare[0], args.compare[1], foldset)
        if result is None:
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if not args.config and not args.blend:
        print("nothing to do: pass --config or --blend (see --help)", file=sys.stderr)
        return 2

    fold_sets = [part.strip() for part in args.folds.split("+") if part.strip()]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    dataset = build_dataset(args.data_dir, args.feature_source)
    print(f"[data] {len(dataset.frame):,} in-universe rows, "
          f"{dataset.frame[DATE].nunique()} days, {len(dataset.feature_columns)} features "
          f"from {dataset.feature_source}")

    reports_dir = args.reports_dir
    (reports_dir / "preds").mkdir(parents=True, exist_ok=True)

    for foldset in fold_sets:
        folds = make_folds(dataset.calendar, foldset)
        print(f"[folds:{foldset}] " + ", ".join(
            f"{f.name}(train<={f.train_end})" for f in folds))

    if args.blend:
        members = [m.strip() for m in args.blend.split(",") if m.strip()]
        name = args.name or ("blend_" + "_".join(members))
        for foldset in fold_sets:
            report, pooled = build_blend_from_cache(dataset, members, name, foldset,
                                                    reports_dir, mode=args.blend_weights,
                                                    verbose=not args.quiet)
            if report is None:
                continue
            pooled.to_parquet(reports_dir / "preds" / f"{name}__{foldset}__outer.parquet",
                              index=False)
            write_report(report, foldset, reports_dir)
            append_leaderboard(report, foldset, reports_dir)
            _print_summary(report, foldset)
        rebuild_leaderboard(reports_dir)
        return 0

    names = available_configs() if args.config == "all" else [args.config]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(f"unknown config(s): {unknown}; known: {available_configs()}", file=sys.stderr)
        return 2
    overrides = parse_overrides(args.override)
    if overrides and len(names) > 1:
        print("--override needs a single --config", file=sys.stderr)
        return 2

    for foldset in fold_sets:
        folds = make_folds(dataset.calendar, foldset)
        for name in names:
            run_name = f"{name}__{args.tag}" if args.tag else name
            report, pooled, inner = run_config_over_folds(
                dataset, name, folds, protocol=args.protocol,
                inner_months=args.inner_months, train_filter=args.train_filter,
                num_threads=args.num_threads, seeds=seeds, overrides=overrides,
                verbose=not args.quiet)
            report["config"] = run_name
            report["base_config"] = name
            report["cli_overrides"] = _jsonable(overrides) if overrides else None
            report["inner_months"] = args.inner_months
            pooled.to_parquet(reports_dir / "preds" / f"{run_name}__{foldset}__outer.parquet",
                              index=False)
            if len(inner):
                inner.to_parquet(reports_dir / "preds" / f"{run_name}__{foldset}__inner.parquet",
                                 index=False)
            write_report(report, foldset, reports_dir)
            append_leaderboard(report, foldset, reports_dir)
            _print_summary(report, foldset)
    rebuild_leaderboard(reports_dir)
    return 0


def _print_summary(report: dict, foldset: str) -> None:
    summary = report.get("summary") or {}
    years = report.get("by_year") or {}
    year_text = " ".join(f"{y}:{v['mean_rankic']:+.3f}" for y, v in sorted(years.items())
                         if v.get("mean_rankic") is not None)
    print(f"[summary] {report['config']} / {foldset} / {report['protocol']}: "
          f"mean={summary.get('mean_rankic')} IR={summary.get('rankicir')} "
          f"days={summary.get('days')}\n          by year: {year_text}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
