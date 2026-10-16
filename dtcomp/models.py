"""Model zoo for the CSI300 cross-sectional RankIC challenge.

Frozen public API (the Lead packages the winner through this interface)::

    model = train_model(train_frame, valid_frame, *, feature_columns, seed=42, **kwargs)
    scores = model.predict(frame)          # np.ndarray, higher = more bullish

``train_frame`` / ``valid_frame`` / ``frame`` are row-per-(date, stock) frames that
carry at least ``日期`` and ``future_return`` plus every column in ``feature_columns``.

Everything a model needs at inference time (feature preprocessing statistics, the
booster, blend weights) lives inside the returned object, so a frozen model is a
single self-contained artifact -- no refitting, no reads outside the artifact.

Design notes
------------
* **Target transforms** (``target=``): ``raw``, ``rank`` (within-day percentile rank,
  the organizers' baseline), ``rank_z`` (z-scored percentile rank), ``winsor``
  (daily 1%/99% winsorised, then scaled by the daily cross-sectional std) and
  ``zscore`` (cross-sectional z-score, clipped).  Only ``raw`` is scale-sensitive.
* **Feature preprocessing** (``preprocess=``): ``raw`` (train-median imputation),
  ``cs_z`` (within-day z-score, clipped, then zero-filled) and ``cs_rank``
  (within-day percentile rank in [-0.5, 0.5]).  The cross-sectional modes remove the
  slow drift of level variables (market cap, price) that otherwise forces a tree to
  learn regime-specific absolute thresholds.  All three are strictly causal: they
  only ever use same-day cross-sectional information.
* **Hyper-parameter selection** is the caller's job and must happen on validation
  folds only.  ``train_model`` never looks at anything beyond the two frames it is
  handed.  ``dtcomp/run_experiment.py`` implements the protocol.
* No ``eval``/``exec``/``compile``/``__import__``/``subprocess`` anywhere in this
  module: dispatch is an explicit dictionary of plain functions.

Named configs (``CONFIGS``) give a reproducible starting point; every field can be
overridden through ``train_model(..., **kwargs)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DATE = "日期"
TARGET = "future_return"

__all__ = [
    "DATE",
    "TARGET",
    "build_target",
    "FeatureProcessor",
    "train_model",
    "CONFIGS",
    "available_configs",
    "rank_average",
    "search_blend_weights",
    "daily_rankic_from_arrays",
]


# ---------------------------------------------------------------------------
# target transforms
# ---------------------------------------------------------------------------

TARGET_KINDS = ("raw", "rank", "rank_z", "winsor", "zscore", "voladj_rank",
                "top_frac", "top_decile")

# Cut for the head-focused binary targets: `top_frac` labels the day's best `TOP_FRAC` of names
# as 1.  0.30 keeps ~90 names per day, comfortably more than the ~50 the optimizer holds, so the
# model is not asked to resolve a set smaller than the portfolio it feeds.
TOP_FRAC = 0.30

# The risk-adjusted target divides the label by a trailing volatility estimate. `vol_20` is a
# frozen member of dtcomp.features.FEATURE_COLUMNS, so it is present in every dataset frame.
VOL_TARGET_COLUMN = "vol_20"


def _grouped_std(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Within-day standard deviation of ``values`` (ddof=0), aligned to frame."""
    return values.groupby(frame[DATE].to_numpy()).transform("std")


def build_target(frame: pd.DataFrame, kind: str = "rank", *,
                 target: str = TARGET, clip: float = 3.0) -> pd.Series:
    """Turn raw open-to-open returns into the regression target ``kind``.

    Parameters
    ----------
    kind:
        ``raw``      raw return, untouched (fat tails dominate the loss);
        ``rank``     within-day percentile rank in [-0.5, 0.5] (baseline recipe);
        ``rank_z``   that percentile rank mapped to a cross-sectional z-score;
        ``winsor``   daily 1%/99% winsorisation, then / daily std;
        ``zscore``   daily z-score, clipped to +/- ``clip``;
        ``voladj_rank``
                     within-day percentile rank of ``return / vol_20``.  This is the only
                     kind that REORDERS the cross-section: the other five are monotone
                     transforms of the same label, so they change the loss's tail weighting
                     but never which stock the model is told to prefer.  Measured reordering
                     is a mean within-day Spearman of 0.948 against the raw label (median
                     0.966), i.e. ~5% of the cross-section is reordered.  See FINDINGS F67.
    """
    if kind not in TARGET_KINDS:
        raise ValueError(f"unknown target kind {kind!r}; expected one of {TARGET_KINDS}")
    y = pd.to_numeric(frame[target], errors="coerce").astype("float64")
    dates = frame[DATE].to_numpy()

    if kind == "voladj_rank":
        vol = pd.to_numeric(frame[VOL_TARGET_COLUMN], errors="coerce").astype("float64")
        # A non-positive or missing volatility would flip or destroy the ordering, so those
        # rows get a NaN target and are dropped by the caller rather than silently kept.
        adjusted = y / vol.where(vol > 0)
        return adjusted.groupby(dates).rank(pct=True) - 0.5

    if kind == "top_frac":
        # HEAD-FOCUSED BINARY TARGET.  The board's IR is excess/tracking-error of a portfolio
        # built from the TOP of our ranking (Problem.txt: the optimizer maximises alpha^T w under
        # an active-risk penalty), so what IR rewards is head accuracy, not average accuracy.
        # This target asks a different question -- "is this name in the day's top 30%?" -- and a
        # binary model's probability ranks by that likelihood, concentrating resolution exactly
        # where the optimizer looks.  RankIC (70% of the score) still reads the full ordering.
        rank = y.groupby(dates).rank(pct=True)
        return (rank >= (1.0 - TOP_FRAC)).astype("float64")

    if kind == "top_decile":
        rank = y.groupby(dates).rank(pct=True)
        return (rank >= 0.90).astype("float64")

    if kind == "raw":
        return y

    if kind == "rank":
        return y.groupby(dates).rank(pct=True) - 0.5

    if kind == "rank_z":
        # Exact z-score of the within-day percentile rank: the rank is uniform on
        # {1/n, ..., 1}, so its mean and sd are closed-form.  Vectorised and
        # numerically identical to a per-group (r - mean) / sd.
        rank = y.groupby(dates).rank(pct=True)
        n = y.groupby(dates).transform("count").astype("float64")
        mean = (n + 1.0) / (2.0 * n)
        sd = np.sqrt((n * n - 1.0) / 12.0) / n
        return (rank - mean) / sd.replace(0.0, np.nan)

    if kind == "winsor":
        grouped = y.groupby(dates)
        lo = grouped.transform(lambda s: s.quantile(0.01))
        hi = grouped.transform(lambda s: s.quantile(0.99))
        clipped = y.clip(lower=lo, upper=hi)
        sd = _grouped_std(frame, clipped)
        return clipped / sd.replace(0.0, np.nan)

    # kind == "zscore"
    mean = y.groupby(dates).transform("mean")
    sd = _grouped_std(frame, y)
    return ((y - mean) / sd.replace(0.0, np.nan)).clip(-clip, clip)


# ---------------------------------------------------------------------------
# feature preprocessing
# ---------------------------------------------------------------------------

PREPROCESS_MODES = ("raw", "cs_z", "cs_rank")


@dataclass
class FeatureProcessor:
    """Causal, self-contained feature transform.

    ``fit`` only records global per-column medians (used to impute ``raw`` mode and
    any NaN that survives the cross-sectional modes).  ``transform`` never uses
    information from another day, so it is safe both in walk-forward validation and
    in the single-day inference loop.
    """

    columns: Sequence[str]
    mode: str = "cs_rank"
    clip: float = 3.0
    fill_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in PREPROCESS_MODES:
            raise ValueError(f"unknown preprocess mode {self.mode!r}; "
                             f"expected one of {PREPROCESS_MODES}")
        self.columns = list(self.columns)

    def fit(self, frame: pd.DataFrame) -> "FeatureProcessor":
        block = frame.loc[:, self.columns].replace([np.inf, -np.inf], np.nan)
        medians = block.median(numeric_only=True)
        self.fill_values = {}
        for col in self.columns:
            value = medians.get(col, np.nan)
            self.fill_values[col] = float(value) if np.isfinite(value) else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing {len(missing)} feature column(s): {missing[:8]}")
        block = frame.loc[:, self.columns].astype("float64")
        block = block.replace([np.inf, -np.inf], np.nan)
        if self.mode == "raw":
            block = block.fillna(pd.Series(self.fill_values))
        else:
            dates = frame[DATE].to_numpy()
            if self.mode == "cs_z":
                grouped = block.groupby(dates)
                mean = grouped.transform("mean")
                std = grouped.transform("std")
                block = ((block - mean) / std.replace(0.0, np.nan)).clip(-self.clip, self.clip)
            else:  # cs_rank
                block = block.groupby(dates).rank(pct=True) - 0.5
            block = block.fillna(0.0)
        return np.ascontiguousarray(block.to_numpy(dtype="float64"))

    # serialisation helpers -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "mode": self.mode,
                "clip": float(self.clip), "fill_values": dict(self.fill_values)}

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "FeatureProcessor":
        return FeatureProcessor(columns=list(payload["columns"]),
                                mode=str(payload["mode"]),
                                clip=float(payload.get("clip", 3.0)),
                                fill_values=dict(payload.get("fill_values", {})))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="float64").reshape(-1)


def rank_average(predictions: Sequence[np.ndarray],
                 weights: Sequence[float] | None = None) -> np.ndarray:
    """Percentile-rank each prediction, then take the (weighted) mean."""
    if not predictions:
        raise ValueError("rank_average needs at least one prediction vector")
    if weights is None:
        weights = [1.0] * len(predictions)
    if len(weights) != len(predictions):
        raise ValueError("weights and predictions must have the same length")
    total = float(np.sum(weights))
    if total == 0.0:
        raise ValueError("blend weights sum to zero")
    out = np.zeros(len(_as_1d(predictions[0])), dtype="float64")
    for weight, pred in zip(weights, predictions):
        series = pd.Series(_as_1d(pred))
        out += (float(weight) / total) * series.rank(pct=True).to_numpy()
    return out


def daily_rankic_from_arrays(dates: np.ndarray, y: np.ndarray,
                             score: np.ndarray) -> np.ndarray:
    """Per-day Spearman(score, y) using plain numpy (no pandas groupby overhead).

    ``dates`` need not be sorted; the returned array is ordered by ascending date.
    """
    order = np.argsort(dates, kind="stable")
    d = np.asarray(dates)[order]
    yy = _as_1d(y)[order]
    ss = _as_1d(score)[order]
    if len(d) == 0:
        return np.empty(0, dtype="float64")
    boundaries = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    out = np.empty(len(boundaries) - 1, dtype="float64")
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if b - a < 3:
            out[i] = np.nan
            continue
        ry = pd.Series(yy[a:b]).rank().to_numpy()
        rs = pd.Series(ss[a:b]).rank().to_numpy()
        ry = ry - ry.mean()
        rs = rs - rs.mean()
        denom = np.sqrt(float(ry @ ry) * float(rs @ rs))
        out[i] = float(ry @ rs) / denom if denom > 0 else np.nan
    return out


def _mean_rankic(dates: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    values = daily_rankic_from_arrays(dates, y, score)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


# ---------------------------------------------------------------------------
# model containers
# ---------------------------------------------------------------------------

class _BoosterModel:
    """Wraps a fitted gradient-boosting booster plus its feature processor."""

    def __init__(self, booster: Any, processor: FeatureProcessor, name: str,
                 backend: str) -> None:
        self.booster = booster
        self.processor = processor
        self.name = name
        self.backend = backend

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.processor.transform(frame)
        if self.backend == "lightgbm":
            scores = self.booster.predict(matrix)
        elif self.backend == "xgboost":
            import xgboost as xgb
            scores = self.booster.predict(xgb.DMatrix(matrix))
        elif self.backend == "catboost":
            scores = self.booster.predict(matrix)
        elif self.backend in ("sklearn", "linear", "mlp"):
            scores = self.booster.predict(matrix)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        return _as_1d(scores)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} name={self.name} backend={self.backend}>"


class _SeedBagModel:
    """Rank-average of the same config fitted under several seeds."""

    def __init__(self, members: Sequence[Any], name: str = "seedbag") -> None:
        self.members = list(members)
        self.name = name
        # Expose the members' selected complexity so a caller that early-stopped on
        # an inner block can refit the bag at the same complexity.  Without this the
        # harness silently fell back to the config's round *cap*, which made
        # seed-bagged and single-seed runs incomparable.
        selected = [getattr(m, "best_iteration", None) for m in self.members]
        selected = [int(s) for s in selected if s]
        self.best_iteration = int(np.median(selected)) if selected else None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return rank_average([m.predict(frame) for m in self.members])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<_SeedBagModel name={self.name} members={len(self.members)}>"


class _BlendModel:
    """Weighted rank-average of several fitted models."""

    def __init__(self, members: Sequence[Any], weights: Sequence[float],
                 name: str = "blend") -> None:
        if len(members) != len(weights):
            raise ValueError("members and weights must have the same length")
        self.members = list(members)
        self.weights = [float(w) for w in weights]
        self.name = name
        selected = [getattr(m, "best_iteration", None) for m in self.members]
        selected = [int(s) for s in selected if s]
        self.best_iteration = int(np.median(selected)) if selected else None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return rank_average([m.predict(frame) for m in self.members], self.weights)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<_BlendModel name={self.name} members={len(self.members)}>"


# ---------------------------------------------------------------------------
# per-backend fitters
# ---------------------------------------------------------------------------

def _group_sizes(dates: np.ndarray) -> np.ndarray:
    """Rows per day, assuming ``dates`` is sorted (frames are sorted upstream)."""
    order = np.argsort(dates, kind="stable")
    d = np.asarray(dates)[order]
    boundaries = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    return np.diff(boundaries)


_RELEVANCE_MAX = 63
# XGBoost caps relevance at 31 unless the exponential NDCG gain is disabled.
_RELEVANCE_MAX_XGB = 31


def _relevance_labels(y: np.ndarray, dates: np.ndarray,
                      cap: int = _RELEVANCE_MAX) -> np.ndarray:
    """Within-day integer relevance in [0, cap] -- what LambdaRank objectives need.

    Any monotone target works; we rank within the day so the transform is
    scale-free.  Ties break by position, which only affects the (arbitrary)
    gradient weighting among stocks the target itself cannot separate.
    """
    ranked = pd.Series(_as_1d(y)).groupby(np.asarray(dates)).rank(method="first")
    return np.clip(np.round(ranked.to_numpy() - 1.0), 0, cap).astype("int32")


def _time_decay_weights(dates: np.ndarray, half_life_days: float | None = None, *,
                        half_life_trading_days: float | None = None) -> np.ndarray | None:
    """Exponential recency weights: ``0.5 ** (age / half_life)``.

    A model that predicts the *future* should not weight a 2016 observation the same as
    a 2022 one.  This **re-weights** every row; it never drops any, so sample size is
    preserved and the decay interpolates smoothly.

    Two units are supported because they answer slightly different questions:

    ``half_life_trading_days``
        ``age`` counts positions in the sorted unique date list, i.e. trading days
        before the last training date.  This matches how the rest of the project
        counts time (and the 120-day look-back rule).
    ``half_life_days``
        ``age`` counts calendar days.  Roughly 1.45 calendar days per trading day on
        this data, so a 250-trading-day half-life is about a 365-calendar-day one.

    Returns ``None`` when no decay is requested, which is the identity (equal weights).
    """
    if half_life_trading_days and half_life_trading_days > 0:
        unique = np.unique(np.asarray(dates))
        position = np.searchsorted(unique, np.asarray(dates))
        age = (len(unique) - 1) - position
        return np.power(0.5, age.astype("float64") / float(half_life_trading_days))
    if half_life_days and half_life_days > 0:
        stamps = pd.to_datetime(np.asarray(dates, dtype="int64").astype(str), format="%Y%m%d")
        age = (stamps.max() - stamps).days.to_numpy(dtype="float64")
        return np.power(0.5, age / float(half_life_days))
    return None


def _fit_lightgbm(matrix: np.ndarray, y: np.ndarray, dates: np.ndarray,
                  valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                  params: Mapping[str, Any], num_boost_round: int,
                  early_stopping: int | None, group_aware: bool,
                  seed: int, weight: np.ndarray | None = None) -> tuple[Any, int | None]:
    import lightgbm as lgb

    resolved = {"verbosity": -1, "seed": seed, "num_threads": 4,
                "deterministic": True, "force_col_wise": True}
    resolved.update(dict(params))
    if group_aware:
        resolved.setdefault("objective", "lambdarank")
        resolved.setdefault("metric", "ndcg")
        resolved.setdefault("ndcg_eval_at", [10])
        resolved.setdefault("label_gain", list(range(64)))
    else:
        resolved.setdefault("objective", "regression")
        resolved.setdefault("metric", "l2")

    def dataset(x: np.ndarray, target: np.ndarray, day: np.ndarray,
                w: np.ndarray | None = None) -> Any:
        if not group_aware:
            return lgb.Dataset(x, label=target, weight=w, free_raw_data=False)
        return lgb.Dataset(x, label=_relevance_labels(target, day), group=_group_sizes(day),
                           weight=w, free_raw_data=False)

    train_set = dataset(matrix, y, dates, weight)
    callbacks: list[Any] = []
    valid_sets: list[Any] = []
    best: int | None = None
    if valid is not None and early_stopping:
        vx, vy, vday = valid
        valid_sets.append(dataset(vx, vy, vday))
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(resolved, train_set, num_boost_round=num_boost_round,
                        valid_sets=valid_sets or None, callbacks=callbacks)
    if booster.best_iteration:
        best = int(booster.best_iteration)
    return booster, best


def _fit_xgboost(matrix: np.ndarray, y: np.ndarray, dates: np.ndarray,
                 valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                 params: Mapping[str, Any], num_boost_round: int,
                 early_stopping: int | None, group_aware: bool,
                 seed: int, weight: np.ndarray | None = None) -> tuple[Any, int | None]:
    import xgboost as xgb

    resolved = {"eta": 0.04, "max_depth": 5, "subsample": 0.8,
                "colsample_bytree": 0.85, "min_child_weight": 20.0,
                "reg_lambda": 5.0, "seed": seed, "nthread": 4,
                "tree_method": "hist"}
    resolved.update(dict(params))
    if group_aware:
        resolved.setdefault("objective", "rank:pairwise")
        resolved.setdefault("eval_metric", "ndcg@10")
        resolved.setdefault("ndcg_exp_gain", False)
    else:
        resolved.setdefault("objective", "reg:squarederror")
        resolved.setdefault("eval_metric", "rmse")

    if group_aware:
        dtrain = xgb.DMatrix(matrix, label=_relevance_labels(y, dates, _RELEVANCE_MAX_XGB),
                             weight=weight)
        dtrain.set_group(_group_sizes(dates))
    else:
        dtrain = xgb.DMatrix(matrix, label=y, weight=weight)
    watchlist: list[tuple[Any, str]] = []
    if valid is not None and early_stopping:
        vx, vy, vday = valid
        dvalid = xgb.DMatrix(
            vx, label=_relevance_labels(vy, vday, _RELEVANCE_MAX_XGB) if group_aware else vy)
        if group_aware:
            dvalid.set_group(_group_sizes(vday))
        watchlist.append((dvalid, "valid"))
    booster = xgb.train(resolved, dtrain, num_boost_round=num_boost_round,
                        evals=watchlist or None,
                        early_stopping_rounds=early_stopping if watchlist else None,
                        verbose_eval=False)
    best = int(booster.best_iteration) + 1 if getattr(booster, "best_iteration", None) is not None else None
    return booster, best


def _fit_catboost(matrix: np.ndarray, y: np.ndarray, dates: np.ndarray,
                  valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                  params: Mapping[str, Any], num_boost_round: int,
                  early_stopping: int | None, group_aware: bool,
                  seed: int, weight: np.ndarray | None = None) -> tuple[Any, int | None]:
    from catboost import CatBoostRanker, CatBoostRegressor, Pool

    resolved = {"iterations": num_boost_round, "learning_rate": 0.04, "depth": 6,
                "l2_leaf_reg": 5.0, "random_seed": seed, "thread_count": 4,
                "verbose": False, "allow_writing_files": False}
    resolved.update(dict(params))
    resolved["iterations"] = num_boost_round
    if group_aware:
        resolved.setdefault("loss_function", "YetiRank")
        model = CatBoostRanker(**resolved)
        train_pool = Pool(matrix, label=y, group_id=dates, weight=weight)
        if valid is not None and early_stopping:
            vx, vy, vday = valid
            model.fit(train_pool, eval_set=Pool(vx, label=vy, group_id=vday),
                      use_best_model=True)
        else:
            model.fit(train_pool)
    else:
        resolved.setdefault("loss_function", "RMSE")
        model = CatBoostRegressor(**resolved)
        if valid is not None and early_stopping:
            vx, vy, _ = valid
            model.fit(Pool(matrix, label=y, weight=weight),
                      eval_set=Pool(vx, label=vy), use_best_model=True)
        else:
            model.fit(Pool(matrix, label=y, weight=weight))
    best_iter = model.get_best_iteration() if hasattr(model, "get_best_iteration") else None
    best = int(best_iter) + 1 if best_iter is not None else None
    return model, best


def _fit_linear(matrix: np.ndarray, y: np.ndarray, dates: np.ndarray,
                valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                params: Mapping[str, Any], num_boost_round: int,
                early_stopping: int | None, group_aware: bool,
                seed: int, weight: np.ndarray | None = None) -> tuple[Any, int | None]:
    from sklearn.linear_model import ElasticNet, Ridge

    resolved = dict(params)
    alpha = float(resolved.pop("alpha", 1.0))
    kind = str(resolved.pop("linear", "ridge"))
    if kind == "enet":
        l1_ratio = float(resolved.pop("l1_ratio", 0.2))
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000,
                           random_state=seed, **resolved)
    else:
        model = Ridge(alpha=alpha, random_state=seed, **resolved)
    model.fit(matrix, y, sample_weight=weight)
    return model, None


def _fit_mlp(matrix: np.ndarray, y: np.ndarray, dates: np.ndarray,
             valid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
             params: Mapping[str, Any], num_boost_round: int,
             early_stopping: int | None, group_aware: bool,
             seed: int, weight: np.ndarray | None = None) -> tuple[Any, int | None]:
    """Multilayer perceptron regressor -- the one model FAMILY never tested in this project.

    Every other family here is a tree ensemble or a linear model. A neural net has a different
    inductive bias (a smooth global function rather than axis-aligned splits), so it is the last
    genuinely untested direction. It goes through the same ``FeatureProcessor`` and the same causal
    folds as everything else, so any difference is attributable to the model alone.

    No early stopping, matching the linear backend: a fixed budget keeps the comparison against the
    fixed-400-round LightGBM recipe honest.
    """
    from sklearn.neural_network import MLPRegressor

    resolved = dict(params)
    hidden = tuple(resolved.pop("hidden_layer_sizes", (128, 64)))
    epochs = int(resolved.pop("max_iter", 60))
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        max_iter=epochs,
        batch_size=int(resolved.pop("batch_size", 4096)),
        learning_rate_init=float(resolved.pop("learning_rate_init", 1e-3)),
        alpha=float(resolved.pop("alpha", 1e-4)),
        early_stopping=False,
        shuffle=True,
        random_state=seed,
        **resolved,
    )
    model.fit(matrix, y)
    return model, None


_BACKENDS: dict[str, Callable[..., tuple[Any, int | None]]] = {
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "catboost": _fit_catboost,
    "linear": _fit_linear,
    "mlp": _fit_mlp,
}


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def _prepare(frame: pd.DataFrame, feature_columns: Sequence[str],
             processor: FeatureProcessor | None, target_kind: str,
             need_target: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if need_target and TARGET not in frame.columns:
        raise KeyError(f"frame is missing the target column {TARGET!r}")
    if processor is None:
        raise ValueError("a fitted FeatureProcessor is required to build a matrix")
    matrix = processor.transform(frame)
    y = build_target(frame, target_kind).to_numpy(dtype="float64") if need_target \
        else np.zeros(len(frame), dtype="float64")
    dates = frame[DATE].to_numpy()
    return matrix, y, dates


def train_model(train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None = None, *,
                feature_columns: Sequence[str] | None = None,
                seed: int = 42,
                kind: str = "lgbm_reg",
                target: str = "rank",
                preprocess: str = "cs_rank",
                params: Mapping[str, Any] | None = None,
                num_boost_round: int = 400,
                early_stopping: int | None = 40,
                seeds: Sequence[int] | None = None,
                members: Sequence[str] | None = None,
                weights: Sequence[float] | None = None,
                refit_full: bool = True,
                refit_scale: float = 1.1,
                time_decay_days: float | None = None,
                time_decay_trading_days: float | None = None,
                name: str | None = None,
                **kwargs: Any) -> Any:
    """Fit one model and return an object exposing ``.predict(frame)``.

    The signature mirrors the frozen team API.  ``feature_columns`` may be omitted
    only when ``kwargs`` carries an explicit ``processor`` (used internally when a
    blend shares one preprocessing across members).

    Parameters
    ----------
    kind:
        ``lgbm_reg``, ``lgbm_rank``, ``xgb_reg``, ``xgb_rank``, ``cat_reg``,
        ``cat_rank``, ``ridge``, ``enet`` or ``blend``.
    target / preprocess:
        see :func:`build_target` and :class:`FeatureProcessor`.
    seeds:
        when given, the same config is fitted once per seed and the predictions are
        rank-averaged.  Purely variance-reducing; no validation data is consulted.
    time_decay_trading_days / time_decay_days:
        half-life for exponential recency sample weights, counted in trading days or
        calendar days respectively.  ``None`` (both) weights every observation
        equally.  Weights come from the training dates alone, so this stays strictly
        causal, and they re-weight rather than truncate: no row is dropped.
    members / weights:
        ``kind="blend"`` only -- names of :data:`CONFIGS` entries to combine, and
        their rank-average weights (defaults to equal weights).
    """
    overrides = dict(kwargs)
    if feature_columns is None:
        feature_columns = overrides.pop("feature_columns_resolved", None)
    if feature_columns is None:
        raise ValueError("feature_columns is required")

    processor = overrides.pop("processor", None)
    if processor is None:
        processor = FeatureProcessor(columns=list(feature_columns), mode=preprocess)
        processor.fit(train_frame)

    if kind == "blend":
        return _train_blend(train_frame, valid_frame, processor,
                            members=members, weights=weights, seed=seed,
                            name=name or "blend", overrides=overrides,
                            refit_full=refit_full, refit_scale=refit_scale,
                            time_decay_days=time_decay_days,
                            time_decay_trading_days=time_decay_trading_days)

    backend, group_aware = _resolve_kind(kind)
    if seeds:
        members_fitted = [
            _fit_single(train_frame, valid_frame, processor, backend, group_aware,
                        target, params, num_boost_round, early_stopping, int(s),
                        refit_full=refit_full, refit_scale=refit_scale,
                        time_decay_days=time_decay_days,
                        time_decay_trading_days=time_decay_trading_days)
            for s in seeds
        ]
        return _SeedBagModel(members_fitted, name=name or f"{kind}[{len(seeds)} seeds]")

    return _fit_single(train_frame, valid_frame, processor, backend, group_aware,
                       target, params, num_boost_round, early_stopping, seed,
                       name=name or kind, refit_full=refit_full,
                       refit_scale=refit_scale, time_decay_days=time_decay_days,
                       time_decay_trading_days=time_decay_trading_days)


def _resolve_kind(kind: str) -> tuple[str, bool]:
    table = {
        "lgbm_reg": ("lightgbm", False),
        "lgbm_rank": ("lightgbm", True),
        "xgb_reg": ("xgboost", False),
        "xgb_rank": ("xgboost", True),
        "cat_reg": ("catboost", False),
        "cat_rank": ("catboost", True),
        "ridge": ("linear", False),
        "enet": ("linear", False),
        "mlp_reg": ("mlp", False),
    }
    if kind not in table:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {sorted(table)}")
    return table[kind]


def _fit_single(train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None,
                processor: FeatureProcessor, backend: str, group_aware: bool,
                target: str, params: Mapping[str, Any] | None,
                num_boost_round: int, early_stopping: int | None, seed: int,
                name: str | None = None, refit_full: bool = True,
                refit_scale: float = 1.1,
                time_decay_days: float | None = None,
                time_decay_trading_days: float | None = None) -> _BoosterModel:
    matrix, y, dates = _prepare(train_frame, processor.columns, processor, target, True)
    weight = _time_decay_weights(dates, time_decay_days,
                                 half_life_trading_days=time_decay_trading_days)
    valid = None
    if valid_frame is not None and len(valid_frame) and early_stopping:
        vx, vy, vday = _prepare(valid_frame, processor.columns, processor, target, True)
        valid = (vx, vy, vday)
    fitter = _BACKENDS[backend]
    if backend in ("linear", "mlp"):
        # Neither the linear nor the MLP backend early-stops: the regularisation strength and the
        # fixed epoch budget are the knobs, so they take the same simple call path.
        booster, best = fitter(matrix, y, dates, None, params or {},
                               0, None, False, seed, weight)
    else:
        booster, best = fitter(matrix, y, dates, valid, params or {},
                               num_boost_round, early_stopping, group_aware, seed, weight)
        if best is not None and valid is not None and refit_full:
            # Refit on the whole training window at the selected complexity.  The
            # caller is responsible for making sure ``valid_frame`` is *not* part of
            # the block it will report on (see dtcomp/run_experiment.py).
            booster, _ = fitter(matrix, y, dates, None, params or {},
                                max(int(round(best * refit_scale)), 1), None,
                                group_aware, seed, weight)
    model = _BoosterModel(booster, processor, name or backend, backend)
    model.best_iteration = best
    model.time_decay_days = time_decay_days
    model.time_decay_trading_days = time_decay_trading_days
    return model


def _train_blend(train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None,
                 processor: FeatureProcessor, *, members: Sequence[str] | None,
                 weights: Sequence[float] | None, seed: int, name: str,
                 overrides: Mapping[str, Any], refit_full: bool = True,
                 refit_scale: float = 1.1,
                 time_decay_days: float | None = None,
                 time_decay_trading_days: float | None = None) -> Any:
    """Fit each member config on the same frames and rank-average them.

    ``weights`` defaults to equal weights.  When ``weights`` is ``None`` and
    ``valid_frame`` is present, weights are *not* searched here -- the experiment
    harness does that on an inner validation block so the outer block stays clean.
    """
    if not members:
        raise ValueError("kind='blend' needs members=[...]")
    fitted = []
    for member in members:
        config = dict(CONFIGS[member])
        config.update(overrides.get("member_overrides", {}).get(member, {}))
        fitted.append(train_model(train_frame, valid_frame,
                                  feature_columns=processor.columns, seed=seed,
                                  processor=processor, refit_full=refit_full,
                                  refit_scale=refit_scale,
                                  time_decay_days=time_decay_days,
                                  time_decay_trading_days=time_decay_trading_days,
                                  **config))
    if weights is None:
        weights = [1.0] * len(fitted)
    return _BlendModel(fitted, weights, name=name)


def search_blend_weights(predictions: Sequence[np.ndarray], dates: np.ndarray,
                         y: np.ndarray, *, step: float = 0.1,
                         rounds: int = 3) -> list[float]:
    """Coordinate search on the simplex maximising mean daily RankIC.

    Callers must pass *inner-validation* predictions only; searching on the block
    that will be reported would make the reported number optimistic.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("no predictions to blend")
    if n == 1:
        return [1.0]
    weights = [1.0 / n] * n
    best = _mean_rankic(dates, y, rank_average(predictions, weights))
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for _ in range(rounds):
        improved = False
        for i in range(n):
            for candidate in grid:
                trial = list(weights)
                trial[i] = float(candidate)
                if sum(trial) <= 0:
                    continue
                score = _mean_rankic(dates, y, rank_average(predictions, trial))
                if np.isfinite(score) and score > best + 1e-6:
                    best, weights, improved = score, trial, True
        if not improved:
            break
    total = float(sum(weights))
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# named configs
# ---------------------------------------------------------------------------

_LGBM_BASE = {"learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
              "min_data_in_leaf": 100, "feature_fraction": 0.85,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0}

CONFIGS: dict[str, dict[str, Any]] = {
    # --- baseline recipe: raw features, within-day percentile-rank target ------
    "lgbm_rank_raw": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    # --- the same model on cross-sectionally ranked features -------------------
    "lgbm_rank_csrank": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    "lgbm_rank_csz": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "cs_z",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    # --- target ablation on the best preprocessing ----------------------------
    "lgbm_raw_csrank": {
        "kind": "lgbm_reg", "target": "raw", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    "lgbm_rankz_csrank": {
        "kind": "lgbm_reg", "target": "rank_z", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    "lgbm_winsor_csrank": {
        "kind": "lgbm_reg", "target": "winsor", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    "lgbm_zscore_csrank": {
        "kind": "lgbm_reg", "target": "zscore", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
    },
    # --- deeper / shallower capacity probes -----------------------------------
    "lgbm_shallow_csrank": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {**_LGBM_BASE, "num_leaves": 7, "max_depth": 3,
                   "min_data_in_leaf": 300, "learning_rate": 0.05},
        "num_boost_round": 600, "early_stopping": 35,
    },
    "lgbm_deep_csrank": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {**_LGBM_BASE, "num_leaves": 31, "max_depth": 8,
                   "min_data_in_leaf": 50},
        "num_boost_round": 500, "early_stopping": 35,
    },
    # --- learning-to-rank -----------------------------------------------------
    "lgbm_lambdarank": {
        "kind": "lgbm_rank", "target": "rank", "preprocess": "cs_rank",
        "params": {**_LGBM_BASE, "lambdarank_truncation_level": 30},
        "num_boost_round": 400, "early_stopping": 35,
    },
    "xgb_rank_csrank": {
        "kind": "xgb_rank", "target": "rank", "preprocess": "cs_rank",
        "params": {"eta": 0.04, "max_depth": 5, "min_child_weight": 20.0,
                   "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    "xgb_reg_csrank": {
        "kind": "xgb_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"eta": 0.04, "max_depth": 5, "min_child_weight": 20.0,
                   "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    "cat_reg_csrank": {
        "kind": "cat_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    # --- linear baselines -----------------------------------------------------
    "ridge_csrank": {
        "kind": "ridge", "target": "rank", "preprocess": "cs_rank",
        "params": {"alpha": 1.0}, "num_boost_round": 0, "early_stopping": None,
    },
    "ridge_csz": {
        "kind": "ridge", "target": "rank", "preprocess": "cs_z",
        "params": {"alpha": 10.0}, "num_boost_round": 0, "early_stopping": None,
    },
    # --- neural network: the last untested model FAMILY -----------------------
    # Everything else here is a tree ensemble or a linear model. `preprocess="raw"` to match the
    # shipped recipe exactly (F74 showed the cross-sectional modes cost 0.013-0.016 on it), and
    # `target="rank"` likewise. The MLP has no early stopping -- the epoch budget is fixed -- so
    # it is comparable to the shipped fixed-400-round LightGBM rather than to an early-stopped run.
    # --- HEAD-FOCUSED BINARY MODELS -------------------------------------------
    # The board's IR is the excess/tracking-error of a portfolio built from the TOP of our
    # ranking, and the leaderboard shows IR is where the field separates (top 12 sit at 3.7-5.6
    # while we sit at 0.8) even though their RankIC is comparable to ours.  So the head, not the
    # average, is the binding constraint.
    #
    # These two ask a binary question -- "is this name in the day's top 30% / top 10%?" -- instead
    # of regressing the full cross-section.  Everything else is held at the shipped recipe so any
    # difference is attributable to the target alone.  `objective="binary"` makes LightGBM fit
    # log-odds, which are monotone in the probability and therefore preserve the ranking.
    "lgbm_top30": {
        "kind": "lgbm_reg", "target": "top_frac", "preprocess": "raw",
        "params": {"objective": "binary", "metric": "binary_logloss",
                   "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.70,
                   "bagging_fraction": 0.80, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None, "seeds": [42, 43, 44],
    },
    "lgbm_topdec": {
        "kind": "lgbm_reg", "target": "top_decile", "preprocess": "raw",
        "params": {"objective": "binary", "metric": "binary_logloss",
                   "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.70,
                   "bagging_fraction": 0.80, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None, "seeds": [42, 43, 44],
    },
    "mlp_sub1": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "raw",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 60, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    # --- capacity / budget sweep for the neural net ---------------------------
    # `mlp_sub1` was fitted at 60 epochs on one architecture -- enough to establish that the family
    # loses to the GBM (F80), but not enough to establish how much of the gap is the family and how
    # much is an untuned budget. These four vary capacity and epochs while holding everything else
    # fixed, so the answer is attributable. All use `preprocess="cs_rank"`: F80 showed the net needs
    # normalised inputs (raw cost it 9.5x), unlike the trees.
    #
    # Note: these are separate configs rather than `--override` values because `parse_overrides`
    # splits on commas and a list value such as `[128, 64]` therefore cannot be expressed inline.
    # --- regularization sweep for the neural net ------------------------------
    # The capacity sweep was cut short by its own first result: (128,64) at 200 epochs scored
    # +0.043646, *below* the same net at 60 epochs (+0.045600). More training hurt, so the net is
    # **overfitting**, not undertrained -- which makes capacity the wrong axis and regularization
    # the right one. These four hold the architecture fixed and vary the epoch budget and the L2
    # penalty (`alpha`), the two knobs that trade fit against generalisation.
    # --- epoch axis: the trend is DOWNWARD, so follow it -----------------------
    # The sweep produced a monotone result that inverts the usual intuition:
    #
    #     30 epochs  +0.052566     60 epochs  +0.045600     200 epochs  +0.043646
    #
    # More training is strictly worse, and the 60 -> 30 step is **+0.00697**, larger than the
    # pre-registered +0.005 bar. So the optimum lies at or below 30 epochs and the earlier
    # "the net is undertrained" reading was backwards -- it is *over*trained even at 60. These
    # three bracket the low end to find where the curve turns.
    "mlp_e10": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 10, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e15": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 15, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e20": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 20, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e30": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 30, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e30_a1e2": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 30, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-2},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e60_a1e2": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 60, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-2},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_e60_a1e1": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 60, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-1},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_nn200": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (128, 64), "max_iter": 200, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_nn256_200": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (256, 128), "max_iter": 200, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_nn256_400": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (256, 128), "max_iter": 400, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    "mlp_nn512_200": {
        "kind": "mlp_reg", "target": "rank", "preprocess": "cs_rank",
        "params": {"hidden_layer_sizes": (512, 256), "max_iter": 200, "batch_size": 4096,
                   "learning_rate_init": 1e-3, "alpha": 1e-4},
        "num_boost_round": 0, "early_stopping": None,
    },
    # --- raw-preprocess siblings, for a diverse ensemble ----------------------
    # `preprocess="raw"` measurably beats the cross-sectional modes on the frozen
    # 136-column library, so the ensemble members all use it.
    "lgbm_rank_raw_lr02": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {**_LGBM_BASE, "learning_rate": 0.02},
        "num_boost_round": 900, "early_stopping": 50,
    },
    "xgb_reg_raw": {
        "kind": "xgb_reg", "target": "rank", "preprocess": "raw",
        "params": {"eta": 0.04, "max_depth": 5, "min_child_weight": 20.0,
                   "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    "cat_reg_raw": {
        "kind": "cat_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    # --- the architecture that has DIRECT evidence on the hidden period --------
    # Lifted verbatim from the shipped `submission/model/metadata.json` of board
    # submission #1 (rankic 0.0712 / ir 2.8105).  Kept here so the fold harness can
    # be asked whether it ranks this above or below the config that lost on the
    # board -- the one comparison that tells us whether the folds are trustworthy.
    "lgbm_sub1": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 7,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    "cat_sub1": {
        "kind": "cat_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5.0},
        "num_boost_round": 400, "early_stopping": 35,
    },
    # --- RECOMMENDED ----------------------------------------------------------
    # `lgbm_sub1` with the two axes the corrected coordinate sweep moved positively
    # (num_leaves 31 -> 63, max_depth 7 -> 8).  Everything else is sub1, which is the
    # only architecture with direct evidence on the hidden period.  First-class config
    # on purpose: the recommendation should not depend on getting a CLI override
    # string right.  Measured with seeds [42, 43, 44] and early_stopping=None:
    # semiannual +0.0838 (RankICIR 0.591), annual +0.0770 (0.544) -- #1 on both fold
    # sets, and nothing else measured beats it on either.
    # TRANSFER PROBE.  Identical to `lgbm_sub1_l63d8` except for capacity: half the leaves, two
    # fewer levels of depth, 5x the L2 and 2.5x the leaf minimum.  It exists to test the
    # hypothesis that our board RankIC deficit is a regime-transfer penalty rather than a fit
    # deficit -- if the board scores 2023+ (Problem.txt) and our folds sit inside 2017-2022, then
    # a model that fits the training regime less tightly has less to unlearn when the regime
    # shifts.  Measured on the `longholdout` fold set (train <= 2020-12-31, test 2021-2022).
    "lgbm_lowcap": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
                   "min_data_in_leaf": 500, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 50.0},
        "num_boost_round": 400, "early_stopping": None,
        "seeds": [42, 43, 44],
    },
    "lgbm_sub1_l63d8": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None,
        "seeds": [42, 43, 44],
    },
    # The shipped configuration, but regressing on the RISK-ADJUSTED label
    # (within-day percentile rank of return / vol_20). This is the only target ever tested
    # here that REORDERS the cross-section rather than merely reshaping the loss: the other
    # five kinds are monotone in the same label. Measured reordering is a mean within-day
    # Spearman of 0.948 vs the raw label, median 0.966 -- ~5% of names move. Everything else
    # (params, seeds, rounds, features, protocol) is identical to `lgbm_sub1_l63d8`, so a
    # difference is attributable to the target alone. See FINDINGS F67.
    "lgbm_sub1_voladj": {
        "kind": "lgbm_reg", "target": "voladj_rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None,
        "seeds": [42, 43, 44],
    },
    "sub1_arch": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1", "cat_sub1"],
        "num_boost_round": 0, "early_stopping": None,
    },
    "sub1_arch_seedbag3": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1_seedbag3", "cat_sub1"],
        "num_boost_round": 0, "early_stopping": None,
    },
    "lgbm_sub1_seedbag3": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 7,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": 35,
        "seeds": [42, 43, 44],
    },
    # --- ensembles ------------------------------------------------------------
    "blend_lgbm_cat_xgb": {
        "kind": "blend", "target": "rank", "preprocess": "cs_rank",
        "members": ["lgbm_rank_csrank", "cat_reg_csrank", "xgb_reg_csrank"],
        "num_boost_round": 0, "early_stopping": None,
    },
    # The shipped LGBM with a SMALL CatBoost weight.  `sub1_arch_seedbag3` (equal weights) gave
    # the board its IR jump but lost 0.001 of RankIC on the transfer proxy; these test whether a
    # light CatBoost weight keeps the RankIC while still supplying the cross-family diversity.
    "blend_63d8_cat20": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1_l63d8", "cat_sub1"], "weights": [0.80, 0.20],
        "num_boost_round": 0, "early_stopping": None,
    },
    "blend_63d8_cat35": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1_l63d8", "cat_sub1"], "weights": [0.65, 0.35],
        "num_boost_round": 0, "early_stopping": None,
    },
    # XGBoost on the RAW block, matching the shipped LightGBM's preprocessing.  The existing
    # `xgb_reg_csrank` uses cs_rank, and F108 showed every cs_rank config transfers worse on the
    # validated proxy (+0.052 vs +0.062), so the family could not be fairly evaluated before this.
    "xgb_sub1_raw": {
        "kind": "xgb_reg", "target": "rank", "preprocess": "raw",
        "params": {"eta": 0.03, "max_depth": 6, "min_child_weight": 20.0, "subsample": 0.8,
                   "colsample_bytree": 0.7, "reg_lambda": 10.0},
        "num_boost_round": 400, "early_stopping": None,
    },
    "blend_63d8_cat35_xgb15": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1_l63d8", "cat_sub1", "xgb_sub1_raw"],
        "weights": [0.50, 0.35, 0.15],
        "num_boost_round": 0, "early_stopping": None,
    },
    "blend_63d8_cat50": {
        "kind": "blend", "target": "rank", "preprocess": "raw",
        "members": ["lgbm_sub1_l63d8", "cat_sub1"], "weights": [0.50, 0.50],
        "num_boost_round": 0, "early_stopping": None,
    },
    # Time-decayed variants of the shipped recipe.  A decay was rejected earlier on the ordinary
    # folds (td125 cost 0.0051), but those folds test short windows adjacent to the training cut,
    # where recency weighting has little to offer.  The board scores a LONG later period (F102),
    # which is the case recency weighting is actually for -- so it is re-tested on the validated
    # `longholdout` proxy (train <= 2020-12-31, test 2021-2022) before being written off.
    "lgbm_63d8_td250": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None,
        "time_decay_trading_days": 250.0,
        "seeds": [42, 43, 44],
    },
    "lgbm_63d8_td500": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "raw",
        "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
                   "min_data_in_leaf": 200, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0},
        "num_boost_round": 400, "early_stopping": None,
        "time_decay_trading_days": 500.0,
        "seeds": [42, 43, 44],
    },
    "seedbag_lgbm_csrank": {
        "kind": "lgbm_reg", "target": "rank", "preprocess": "cs_rank",
        "params": dict(_LGBM_BASE), "num_boost_round": 500, "early_stopping": 35,
        "seeds": [42, 202, 2024],
    },
}


def available_configs() -> list[str]:
    """Sorted names of the built-in configs."""
    return sorted(CONFIGS)
