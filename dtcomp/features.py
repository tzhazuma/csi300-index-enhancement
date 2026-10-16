"""Causal, look-back-bounded feature library for the CSI300 enhancement challenge.

The single most important property of this module is that **the same pure function**
is used to build features for training (the full 2016-2022 panel) and at inference
(a rolling window of at most 120 trading days handed to ``predict()``).  Any
train/inference divergence would silently degrade the submission, so every operation
here is a pure, causal function of the frame that is passed in.

Contract
--------
``build_features(panel, lookback=120) -> pd.DataFrame``
    Returns ``panel[["日期", "代码"]]`` followed by one float column per entry of
    ``FEATURE_COLUMNS``, with the **same length and the same row order** as ``panel``.

``FEATURE_COLUMNS``
    Frozen tuple of produced feature names (all float, all numeric, Inf mapped to NaN).

``PASSTHROUGH``
    Subset of ``FEATURE_COLUMNS`` that consists of provided panel columns copied
    through unchanged; everything else is derived.

``FEATURE_FAMILIES``
    Mapping ``family -> tuple[names]`` used by the report generator.

Rules enforced by construction
------------------------------
* Every rolling window ``W`` satisfies ``1 <= W <= lookback <= 120`` rows, i.e. it can
  only ever read ``t-W+1 .. t``.  ``lookback`` clamps nominal windows (with the default
  120 and no nominal window above 120, clamping is the identity).
* Every lag ``L`` satisfies ``0 <= L <= lookback - 1``, so the longest price ratio is
  ``close_t / close_{t-119}`` -- exactly the legal 120-row window ``t-119 .. t``.
* Cross-sectional statistics are computed *inside a single 日期* and only over rows with
  ``in_barra`` true, which is exactly the platform's legal universe (that day's
  ``Barra_factor`` membership).  The ~46 non-member rows per day that ``MarketData``
  carries therefore cannot pollute the cross-section, and the training panel and the
  inference window agree by construction.
* No randomness, no global state, no file access, no dynamic dispatch.

Feature design notes
--------------------
* Back-adjusted prices only.  ``MarketDataRawOpen`` is **not** sent to ``predict()``, so
  ``raw_open`` is never referenced.  Overnight/intraday are built from the shipped
  back-adjusted ``open``/``close``, which is the correct total-return decomposition:
  ``overnight_t = open_t / close_{t-1} - 1`` and ``intraday_t = close_t / open_t - 1``.
* Absolute price *levels* are never used as features: ``sma_20``, ``ema_20``,
  ``macd_*``, ``atr_14`` are provided in price units, so they are normalised by
  ``close`` (``*_rel``) to stay invariant to the vendor's back-adjustment base.
* Daily RankIC is invariant to any per-day monotone transform of a feature, so the
  ``*_xr`` cross-sectional ranks are univariately *identical* to their raw source.
  They are kept because they remove the slow drift of the cross-sectional location
  (turnover and liquidity levels, volatility, market cap), which is a distribution-shift
  control for the tree model, not a univariate alpha.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "build_features",
    "build_extended_features",
    "FEATURE_COLUMNS",
    "EXTENDED_FEATURE_COLUMNS",
    "FEATURE_FAMILIES",
    "EXTENDED_FAMILIES",
    "PASSTHROUGH",
    "MAX_LOOKBACK",
]

MAX_LOOKBACK = 120

_ID_COLS = ("日期", "代码")

BARRA_NAMES = ("市值", "贝塔", "动量", "残差波动", "非线性市值",
               "账面市值比", "流动性", "盈利", "成长", "杠杆")

_MARKET_REQUIRED = ("open", "high", "low", "close", "volume", "amount", "is_suspend")

# Provided columns used unchanged as features.
_FACTOR_PASSTHROUGH = (
    "turn", "reversal_5", "volatility_5", "rsi_12", "cci_14",
    "kdj_k_9_3_3", "kdj_d_9_3_3", "bias_20", "beta_000300SH_22",
    "list_days", "is_st", "is_new", "is_suspend",
    "roe_avg_ttm", "roa_avg_ttm", "gross_profit_rate_ttm",
    "net_profit_rate_ttm", "debt_to_asset_lf", "current_ratio_lf",
    "pe_ttm", "pb",
)
_BARRA_PASSTHROUGH = tuple(f"barra_{name}" for name in BARRA_NAMES)

PASSTHROUGH: tuple[str, ...] = _FACTOR_PASSTHROUGH + _BARRA_PASSTHROUGH

# --------------------------------------------------------------------------- families

_F_REVERSAL = (
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "ret_119",
    "mom_skip1_5", "mom_skip1_20", "mom_skip1_60",
    "ret_5_minus_1", "ret_119_minus_20",
)

_F_VOLATILITY = (
    "vol_10", "vol_20", "vol_60", "vol_120",
    "vol_ratio_20_120",
    "downside_dev_20", "downside_dev_60", "downside_ratio_20",
    "vol_of_vol_60", "vol_of_vol_120",
    "parkinson_20", "parkinson_60",
    "ret_skew_60", "ret_skew_120",
    "max_dd_60",
    "hl_range_1", "hl_range_20", "hl_range_60",
    "atr_rel",
)

_F_OVERNIGHT = (
    "overnight_1", "intraday_1",
    "overnight_mean_5", "overnight_mean_20",
    "intraday_mean_5", "intraday_mean_20",
    "overnight_std_20", "intraday_std_20",
    "oi_diff_20",
    "gap_prev_high", "gap_prev_low", "gap_prev_high_20",
    "close_pos_day", "close_pos_day_20",
)

_F_LIQUIDITY = (
    "ln_amount", "amount_z_20", "amount_z_60",
    "amount_ratio_5_60", "amount_ratio_20_120",
    "amount_vol_60", "amount_trend_20",
    "turn_z_20", "turn_z_60", "turn_vol_20",
    "turn_ratio_5_60", "turn_ratio_20_120",
    "amihud_20", "amihud_60", "amihud_120", "illiq_ratio_20_120",
)

_F_TREND = (
    "close_over_ma5", "close_over_ma10", "close_over_ma20",
    "close_over_ma60", "close_over_ma120",
    "ma5_over_ma20", "ma20_over_ma60", "ma60_over_ma120", "ma_slope_20",
    "close_over_ema20_given", "macd_hist_rel", "macd_diff_rel",
    "high_prox_20", "high_prox_60", "high_prox_120",
    "low_prox_20", "low_prox_60",
    "range_pos_20", "range_pos_60", "range_pos_120",
)

_F_FUNDAMENTAL = (
    "earnings_yield", "book_to_price", "sales_yield",
    "ln_mktcap", "ln_floatcap", "float_ratio",
)

_F_CROSS_SECTIONAL = (
    "turn_xr", "ln_amount_xr", "amihud_20_xr", "vol_20_xr", "ln_mktcap_xr",
    "hl_range_20_xr", "earnings_yield_xr", "ret_20_xr", "close_over_ma60_xr",
    "ret_skew_60_xr", "turn_z_20_xr", "amount_ratio_5_60_xr",
)

_F_INTERACTION = (
    "mom20_over_vol20", "mom5_over_vol20", "turn_x_ret1",
    "vol_x_mktcap", "reversal_x_illiq",
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "reversal_momentum": _F_REVERSAL,
    "volatility": _F_VOLATILITY,
    "overnight_intraday": _F_OVERNIGHT,
    "liquidity": _F_LIQUIDITY,
    "price_trend": _F_TREND,
    "fundamental": _F_FUNDAMENTAL,
    "cross_sectional": _F_CROSS_SECTIONAL,
    "interaction": _F_INTERACTION,
    "provided_passthrough": PASSTHROUGH,
}

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    name for family in FEATURE_FAMILIES.values() for name in family
)

_CROSS_SECTIONAL_SOURCES = tuple(
    name[:-3] for name in _F_CROSS_SECTIONAL
)

# --------------------------------------------------------------- additive residual set
#
# Beta-hedged ("residual") reversal.  Every reversal/momentum feature in FEATURE_COLUMNS is a
# RAW return, so it carries the market component: on a day the index moves 2% the high-beta
# names all show the same "reversal", and part of the cross-sectional spread is just beta.
# These features remove it using only shipped columns -- the provided
# ``beta_000300SH_22`` and an equal-weighted return of that day's Barra members.
#
# This family is ADDITIVE.  FEATURE_COLUMNS is frozen (a model trained on exactly those
# values scored on the hidden board) and is not touched by anything below.
_F_RESIDUAL = (
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20",
    "resid_ret_1", "resid_ret_2", "resid_ret_5", "resid_ret_10",
    "resid_ret_20", "resid_ret_60",
    "resid_ret_5_minus_1", "resid_ret_20_minus_1",
    "mkt_rel_ret_5", "mkt_rel_ret_20",
    "idio_vol_20", "idio_vol_60",
    "beta_x_rev",
)

EXTENDED_FAMILIES: dict[str, tuple[str, ...]] = {
    **FEATURE_FAMILIES,
    "residual_beta": _F_RESIDUAL,
}

EXTENDED_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + _F_RESIDUAL

_RESIDUAL_SET = frozenset(_F_RESIDUAL)
_KNOWN_COLUMNS = frozenset(EXTENDED_FEATURE_COLUMNS)

# --------------------------------------------------------------------------- helpers

_EPS = 1e-12

# How many price rows a derived series already reads *before* its own date.  A rolling
# window of W rows over such a series therefore reaches W + extra rows back, and the
# competition allows at most 120 (t-119 .. t).  Clamping the window by `extra` keeps the
# true price span inside the legal window AND makes the inference-time 120-row frame
# reproduce the training value exactly (inside the frame the first return is NaN, so an
# unclamped window would silently use a different sample than training).
_SERIES_PRICE_LAG = {
    "_ret1": 1,          # close_t / close_{t-1} - 1
    "_overnight": 1,     # open_t / close_{t-1} - 1
    "_gap_high": 1,      # open_t / high_{t-1} - 1
    "_amihud": 1,        # |ret_1| / amount
    "_downside_sq": 1,   # min(ret_1, 0)^2
    "_vol5": 5,          # 5-row std of _ret1 -> reads close back to t-5
    "_resid1": 1,        # ret_1 - beta * mkt_ret_1
}


def _min_periods(window: int) -> int:
    """Half the window, at least 2 rows -- enough to be defined, not so few as to be noise."""
    return max(2, int(window) // 2)


def _barra_column(frame: pd.DataFrame, name: str) -> str | None:
    """The panel uses ``barra_<name>``; accept the platform's unprefixed name too."""
    for candidate in (f"barra_{name}", name):
        if candidate in frame.columns:
            return candidate
    return None


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Column as float64, or all-NaN when absent (never raises on an optional factor)."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64")
    return np.full(len(frame), np.nan)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype="float64")
    out = np.full(den.shape, np.nan, dtype="float64")
    ok = np.isfinite(den) & (den != 0.0)
    np.divide(np.asarray(num, dtype="float64"), den, out=out, where=ok)
    return out


# --------------------------------------------------------------------------- rolling

_ROLL_REDUCERS = {
    "mean": lambda rolling: rolling.mean(),
    "std": lambda rolling: rolling.std(),
    "min": lambda rolling: rolling.min(),
    "max": lambda rolling: rolling.max(),
    "skew": lambda rolling: rolling.skew(),
}


def _make_std_kernel(roll_var):
    """``sqrt(roll_var(ddof=1))`` with pandas' negative-variance guard."""

    def _std(values, start, end, min_periods):
        variance = roll_var(values, start, end, min_periods, ddof=1)
        with np.errstate(invalid="ignore"):
            out = np.sqrt(variance)
        out[variance < 0] = np.nan
        return out

    return _std


def _probe_reference(values, labels, window, min_periods, kind):
    """The per-group rolling value the library has always produced."""
    series = pd.Series(values)
    grouped = series.groupby(labels, sort=False)
    return grouped.transform(
        lambda s: _ROLL_REDUCERS[kind](s.rolling(window, min_periods=min_periods))
    ).to_numpy(dtype="float64")


def _build_fast_kernels():
    """Pandas' Cython rolling kernels, but only the ones verified to be bit-identical.

    Calling ``pandas._libs.window.aggregations.roll_*`` once for the whole frame with
    group-aware window bounds is ~19x faster than rolling a padded block matrix, because it
    skips pandas' per-column Python loop.  It is a private API, so every kernel is checked
    against the public per-group result on a probe that covers the awkward cases (a block
    shorter than the window, a block longer than the window, NaN at the start of a block,
    an all-NaN block, two different window/min_periods pairs).  Anything that fails the
    probe -- or any pandas where the import fails -- silently keeps the public slow path,
    so this can never change a value, only the runtime.
    """
    try:
        from pandas._libs.window import aggregations as window_aggregations
        roll_mean = window_aggregations.roll_mean
        roll_min = window_aggregations.roll_min
        roll_max = window_aggregations.roll_max
        roll_skew = window_aggregations.roll_skew
        roll_var = window_aggregations.roll_var
    except Exception:  # noqa: BLE001 - a private API we are allowed to lose
        return {}

    candidates = {
        "mean": roll_mean, "min": roll_min, "max": roll_max, "skew": roll_skew,
        "std": _make_std_kernel(roll_var),
    }

    probe_values = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,          # long block
         3.0, 1.0, 4.0, 1.0,                                        # block shorter than window
         np.nan, np.nan, np.nan, np.nan, np.nan,                    # all-NaN block
         2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0, 3.0,  # long block
         5.0],                                                       # single-row block
        dtype="float64")
    probe_labels = np.repeat(
        np.arange(5), [9, 4, 5, 13, 1])
    probe_starts = np.r_[0, np.cumsum([9, 4, 5, 13])]
    probe_pos = np.arange(len(probe_values)) - np.repeat(probe_starts, [9, 4, 5, 13, 1])
    probe_gstart = np.repeat(probe_starts, [9, 4, 5, 13, 1])
    probe_end = (np.arange(len(probe_values), dtype=np.int64) + 1)

    verified: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, kernel in candidates.items():
            try:
                for window, min_periods in ((5, 3), (12, 6), (1, 1)):
                    bounds = (probe_gstart
                              + np.maximum(0, probe_pos + 1 - window)).astype("int64")
                    got = kernel(probe_values, bounds, probe_end, min_periods)
                    want = _probe_reference(probe_values, probe_labels, window,
                                            min_periods, kind)
                    if not np.array_equal(np.asarray(got, dtype="float64"), want,
                                          equal_nan=True):
                        break
                else:
                    verified[kind] = kernel
            except Exception:  # noqa: BLE001 - any surprise means: do not use it
                continue
    return verified


_FAST_KERNELS = _build_fast_kernels()


class _BlockRoller:
    """Causal rolling/shift ops for contiguous per-stock history blocks.

    ``build_features`` sorts the frame by ``(代码, 日期)`` and marks contiguous history
    blocks in ``_segment``, so every block is one contiguous run of rows -- which means the
    frame order is already the concatenation of the blocks, with no gaps to skip.

    Rolling a window *inside* each block is then just a matter of handing pandas' Cython
    kernel the whole column once, together with per-row window bounds that restart at every
    block start (``start[i] = block_start + max(0, k + 1 - window)``, ``end[i] = i + 1``).
    That is exactly the bound array pandas itself builds for a groupby rolling, and it is
    ~19x faster than the padded-block-matrix alternative because it removes the per-column
    Python loop entirely.

    ``_FAST_KERNELS`` holds only the kernels that a probe proved bit-identical to the
    per-group result; anything else -- including ``skew`` on pandas builds whose
    single-call kernel drifts from the per-group one -- falls back to rolling a padded
    ``(max_len, n_blocks)`` NaN matrix where each block is one column and time runs down the
    index.  The fallback is slower but produces the same values, so the choice of path can
    never change the output.
    """

    def __init__(self, segment: np.ndarray) -> None:
        segment = np.asarray(segment)
        starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
        lengths = np.diff(np.r_[starts, len(segment)])
        self.n = int(len(segment))
        self.n_blocks = int(len(starts))
        self.width = int(lengths.max()) if len(lengths) else 0
        self.gid = np.repeat(np.arange(self.n_blocks), lengths)
        self.pos = np.arange(self.n) - np.repeat(starts, lengths)
        self._gstart = np.repeat(starts, lengths)
        self._end = np.arange(self.n, dtype=np.int64) + 1
        self._bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._layout: np.ndarray | None = None

    def _window_bounds(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._bounds.get(window)
        if cached is None:
            start = (self._gstart + np.maximum(0, self.pos + 1 - window)).astype("int64")
            cached = (start, self._end)
            self._bounds[window] = cached
        return cached

    def shift(self, values: np.ndarray, lag: int) -> np.ndarray:
        """Shift inside each block: pure data movement, no rolling kernel needed."""
        values = np.asarray(values, dtype="float64")
        lag = int(lag)
        if lag <= 0:
            return values.copy()
        out = np.full(self.n, np.nan, dtype="float64")
        keep = np.flatnonzero(self.pos >= lag)
        out[keep] = values[keep - lag]
        return out

    def roll(self, values: np.ndarray, window: int, kind: str = "mean",
             min_periods: int | None = None) -> np.ndarray:
        values = np.asarray(values, dtype="float64")
        window = int(window)
        mp = _min_periods(window) if min_periods is None else int(min_periods)
        mp = max(1, min(mp, window))
        kernel = _FAST_KERNELS.get(kind)
        if kernel is not None:
            start, end = self._window_bounds(window)
            return kernel(values, start, end, mp)
        return self._roll_padded(values, window, kind, mp)

    def _roll_padded(self, values: np.ndarray, window: int, kind: str,
                     mp: int) -> np.ndarray:
        """Public-API fallback: one padded column per block, one rolling call."""
        # Explicit dispatch, never dynamic attribute lookup: the competition rules ban
        # name-based dynamic dispatch outright (Submission_Rules §9.3), and a static
        # compliance scan cannot tell a benign use from an evasive one.
        reducer = _ROLL_REDUCERS.get(kind)
        if reducer is None:
            raise ValueError(f"unsupported rolling kind {kind!r}")
        if self._layout is None:
            self._layout = np.full((self.width, self.n_blocks), np.nan, dtype="float64")
        self._layout[self.pos, self.gid] = values
        rolled = reducer(pd.DataFrame(self._layout).rolling(window, min_periods=mp))
        return rolled.to_numpy(dtype="float64")[self.pos, self.gid]


def _cross_sectional_rank(frame: pd.DataFrame, values: np.ndarray,
                          member: np.ndarray) -> np.ndarray:
    """Within-day percentile rank of every column of ``values``, members only.

    This is ``groupby(日期).rank(pct=True, method="average")``.  All cross-sectional
    features go through a **single** ``groupby``: ranking 12 columns one at a time costs 12
    full groupby setups, which is one of the two most expensive steps on the 590k-row
    training panel.  A hand-rolled ``lexsort`` replacement was tried and rejected -- it was
    both slower than pandas' Cython rank and not bit-identical on ties.
    """
    values = np.asarray(values, dtype="float64")
    single = values.ndim == 1
    if single:
        values = values[:, None]
    out = np.full(values.shape, np.nan, dtype="float64")
    if not member.any():
        return out[:, 0] if single else out
    frame_sub = pd.DataFrame(values[member], columns=range(values.shape[1]))
    keys = frame.loc[member, "日期"].to_numpy()
    ranked = frame_sub.groupby(keys, sort=False).rank(pct=True).to_numpy(dtype="float64")
    out[member] = ranked
    return out[:, 0] if single else out


# ------------------------------------------------------------------- market series

def _member_market_returns(frame: pd.DataFrame, member: np.ndarray,
                           ret_1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weighted mean daily return of each day's Barra members.

    Members only, deliberately.  ``MarketData`` ships the yearly HS300 *union* (327-354
    names/day) but a judge that ships only the 300 constituents would give a different
    all-rows mean -- so a member-only definition is identical under either input shape, which
    is exactly the class of train/inference skew that is invisible locally.
    """
    days = frame["日期"].to_numpy()
    unique, inverse = np.unique(days, return_inverse=True)
    n_days = len(unique)
    valid = np.asarray(member, dtype=bool) & np.isfinite(ret_1)
    counts = np.bincount(inverse[valid], minlength=n_days).astype("float64")
    totals = np.bincount(inverse[valid], weights=ret_1[valid], minlength=n_days)
    per_day = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    return per_day, inverse.astype(np.int64)


def _rolling_growth(per_day: np.ndarray, window: int) -> np.ndarray:
    """Compounded growth of the market series over the last ``window`` days.

    A rolling product rather than ``cumprod(t)/cumprod(t-L)``: the ratio form depends on where
    the frame starts, so a 120-day inference window would not reproduce the training value.
    """
    out = np.full(len(per_day), np.nan, dtype="float64")
    window = int(window)
    if window >= 1 and len(per_day) >= window:
        gross = 1.0 + per_day
        windows = np.lib.stride_tricks.sliding_window_view(gross, window)
        out[window - 1:] = windows.prod(axis=1)
    return out


# --------------------------------------------------------------------------- builder

def build_features(panel: pd.DataFrame, lookback: int = MAX_LOOKBACK,
                   columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Build the causal feature set for every row of ``panel``.

    Parameters
    ----------
    panel
        One row per (日期, 代码) with the market columns (``open``, ``high``, ``low``,
        ``close``, ``volume``, ``amount``, ``is_suspend``), optionally the Factor33 and
        ``barra_*`` columns and ``in_barra``.  Row order is preserved.
    lookback
        Maximum number of trading-day rows any rolling feature may read (1..120).
        Nominal windows above ``lookback`` are clamped to it; with the default 120 this
        is the identity.
    columns
        Which feature columns to emit.  ``None`` (the default) means ``FEATURE_COLUMNS``,
        the frozen 136-column set, and in that case the additive ``residual_beta`` family is
        not computed at all -- so the default path is bit-identical to before that family
        existed.  Pass ``EXTENDED_FEATURE_COLUMNS`` (or any subset of it) to also get the
        beta-hedged features.

    Returns
    -------
    pd.DataFrame
        ``日期``, ``代码`` + the requested columns, float32, same row order as ``panel``.
    """
    lb = int(lookback)
    if not 1 <= lb <= MAX_LOOKBACK:
        raise ValueError(f"lookback must be in 1..{MAX_LOOKBACK}, got {lookback}")
    wanted = tuple(FEATURE_COLUMNS if columns is None else columns)
    unknown = [name for name in wanted if name not in _KNOWN_COLUMNS]
    if unknown:
        raise KeyError(f"unknown feature columns requested: {unknown}")

    missing = [c for c in _ID_COLS + _MARKET_REQUIRED if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns: {missing}")

    n = len(panel)
    original = panel.reset_index(drop=True)
    # Sort by (代码, 日期) so every per-stock rolling op sees chronological rows, then
    # reset the index so feature arrays align positionally with the sorted frame.
    order = original.sort_values(["代码", "日期"], kind="stable").index.to_numpy()
    frame = original.iloc[order].reset_index(drop=True)

    if "in_barra" in frame.columns:
        member = frame["in_barra"].fillna(False).to_numpy(dtype=bool)
    else:
        size_col = _barra_column(frame, "市值")
        member = (frame[size_col].notna().to_numpy()
                  if size_col is not None else np.ones(len(frame), dtype=bool))

    # Contiguous-history segments: a new segment starts on a stock's first row and
    # whenever a trading day is missing, so no rolling window can bridge a listing gap.
    # The frame is already sorted by (代码, 日期) and every stock's first row has a NaN
    # gap, so one global cumulative sum yields a unique id per contiguous history block.
    # The gap must be measured on the trading CALENDAR position, not on the raw 日期
    # integer: consecutive trading days differ by 1 only inside a month, so a raw diff
    # would split every month boundary into its own segment.
    calendar = np.sort(frame["日期"].unique())
    frame["_cal"] = np.searchsorted(calendar, frame["日期"].to_numpy())
    day_gap = frame.groupby("代码", sort=False)["_cal"].diff()
    frame["_segment"] = (day_gap.isna() | day_gap.ne(1)).cumsum().astype("int64")
    frame = frame.drop(columns=["_cal"])

    roller = _BlockRoller(frame["_segment"].to_numpy())

    close = _column(frame, "close")
    open_ = _column(frame, "open")
    high = _column(frame, "high")
    low = _column(frame, "low")
    amount = _column(frame, "amount")

    close_1 = roller.shift(close, 1)
    high_1 = roller.shift(high, 1)
    low_1 = roller.shift(low, 1)

    ret_1 = _safe_div(close, close_1) - 1.0
    overnight = _safe_div(open_, close_1) - 1.0
    intraday = _safe_div(close, open_) - 1.0
    ln_amount = np.log1p(np.clip(amount, 0.0, None))
    turn = _column(frame, "turn")
    ln_turn = np.log1p(np.clip(turn, 0.0, None))
    hl_range = _safe_div(high - low, close)
    close_pos = _safe_div(close - low, high - low)
    amihud = _safe_div(np.abs(ret_1), amount) * 1e9
    ln_hl2 = np.log(np.where((high > low) & (low > 0), _safe_div(high, low), np.nan)) ** 2
    downside_sq = np.minimum(ret_1, 0.0) ** 2
    gap_high = _safe_div(open_, high_1) - 1.0

    # Memoised: several features read the same rolling window of the same series, and the
    # cache is local to this call, so it cannot become cross-call state.
    _roll_cache: dict[tuple[str, int, str], np.ndarray] = {}
    _shift_cache: dict[tuple[str, int], np.ndarray] = {}

    def roll(name: str, values: np.ndarray, window: int, kind: str = "mean") -> np.ndarray:
        """Causal rolling op whose *price* span never exceeds ``lookback`` rows."""
        budget = lb - _SERIES_PRICE_LAG.get(name, 0)
        window = max(1, min(int(window), budget))
        key = (name, window, kind)
        cached = _roll_cache.get(key)
        if cached is None:
            cached = roller.roll(values, window, kind)
            _roll_cache[key] = cached
        return cached

    def shift(name: str, values: np.ndarray, lag_rows: int) -> np.ndarray:
        lag_rows = min(int(lag_rows), lb - 1)
        key = (name, lag_rows)
        cached = _shift_cache.get(key)
        if cached is None:
            cached = roller.shift(values, lag_rows)
            _shift_cache[key] = cached
        return cached

    F: dict[str, np.ndarray] = {}

    # -- reversal / momentum ------------------------------------------------------
    for L in (1, 2, 3, 5, 10, 20, 60):
        F[f"ret_{L}"] = _safe_div(close, shift("close", close, L)) - 1.0
    F["ret_119"] = _safe_div(close, shift("close", close, 119)) - 1.0
    for L in (5, 20, 60):
        F[f"mom_skip1_{L}"] = _safe_div(close_1, shift("close", close, L)) - 1.0
    F["ret_5_minus_1"] = F["ret_5"] - F["ret_1"]
    F["ret_119_minus_20"] = F["ret_119"] - F["ret_20"]

    # -- volatility ---------------------------------------------------------------
    vol = {}
    for L in (5, 10, 20, 60, 120):
        vol[L] = roll("_ret1", ret_1, L, "std")
    for L in (10, 20, 60, 120):
        F[f"vol_{L}"] = vol[L]
    F["vol_ratio_20_120"] = _safe_div(vol[20], vol[120])
    for L in (20, 60):
        F[f"downside_dev_{L}"] = np.sqrt(roll("_downside_sq", downside_sq, L))
    F["downside_ratio_20"] = _safe_div(F["downside_dev_20"], vol[20])
    F["vol_of_vol_60"] = roll("_vol5", vol[5], 60, "std")
    F["vol_of_vol_120"] = roll("_vol5", vol[5], 120, "std")
    for L in (20, 60):
        F[f"parkinson_{L}"] = np.sqrt(roll("_ln_hl2", ln_hl2, L) / (4.0 * np.log(2.0)))
    F["ret_skew_60"] = roll("_ret1", ret_1, 60, "skew")
    F["ret_skew_120"] = roll("_ret1", ret_1, 120, "skew")
    F["max_dd_60"] = _safe_div(close, roll("close", close, 60, "max")) - 1.0
    F["hl_range_1"] = hl_range
    F["hl_range_20"] = roll("_hl_range", hl_range, 20)
    F["hl_range_60"] = roll("_hl_range", hl_range, 60)
    F["atr_rel"] = _safe_div(_column(frame, "atr_14"), close)

    # -- overnight / intraday -----------------------------------------------------
    F["overnight_1"] = overnight
    F["intraday_1"] = intraday
    for L in (5, 20):
        F[f"overnight_mean_{L}"] = roll("_overnight", overnight, L)
        F[f"intraday_mean_{L}"] = roll("_intraday", intraday, L)
    F["overnight_std_20"] = roll("_overnight", overnight, 20, "std")
    F["intraday_std_20"] = roll("_intraday", intraday, 20, "std")
    F["oi_diff_20"] = F["overnight_mean_20"] - F["intraday_mean_20"]
    F["gap_prev_high"] = gap_high
    F["gap_prev_low"] = _safe_div(open_, low_1) - 1.0
    F["gap_prev_high_20"] = roll("_gap_high", gap_high, 20)
    F["close_pos_day"] = close_pos
    F["close_pos_day_20"] = roll("_close_pos", close_pos, 20)

    # -- liquidity / turnover -----------------------------------------------------
    F["ln_amount"] = ln_amount
    for L in (20, 60):
        F[f"amount_z_{L}"] = _safe_div(ln_amount - roll("_ln_amount", ln_amount, L),
                                       roll("_ln_amount", ln_amount, L, "std"))
        F[f"turn_z_{L}"] = _safe_div(ln_turn - roll("_ln_turn", ln_turn, L),
                                     roll("_ln_turn", ln_turn, L, "std"))
    F["amount_ratio_5_60"] = (roll("_ln_amount", ln_amount, 5)
                              - roll("_ln_amount", ln_amount, 60))
    F["amount_ratio_20_120"] = (roll("_ln_amount", ln_amount, 20)
                                - roll("_ln_amount", ln_amount, 120))
    F["amount_vol_60"] = roll("_ln_amount", ln_amount, 60, "std")
    F["amount_trend_20"] = (roll("_ln_amount", ln_amount, 5)
                            - roll("_ln_amount", ln_amount, 20))
    F["turn_vol_20"] = roll("_ln_turn", ln_turn, 20, "std")
    F["turn_ratio_5_60"] = roll("_ln_turn", ln_turn, 5) - roll("_ln_turn", ln_turn, 60)
    F["turn_ratio_20_120"] = roll("_ln_turn", ln_turn, 20) - roll("_ln_turn", ln_turn, 120)
    for L in (20, 60, 120):
        F[f"amihud_{L}"] = roll("_amihud", amihud, L)
    F["illiq_ratio_20_120"] = _safe_div(F["amihud_20"], F["amihud_120"])

    # -- price trend / range position ---------------------------------------------
    ma = {L: roll("close", close, L) for L in (5, 10, 20, 60, 120)}
    for L in (5, 10, 20, 60, 120):
        F[f"close_over_ma{L}"] = _safe_div(close, ma[L]) - 1.0
    F["ma5_over_ma20"] = _safe_div(ma[5], ma[20]) - 1.0
    F["ma20_over_ma60"] = _safe_div(ma[20], ma[60]) - 1.0
    F["ma60_over_ma120"] = _safe_div(ma[60], ma[120]) - 1.0
    F["ma_slope_20"] = _safe_div(ma[20], shift("_ma20", ma[20], 5)) - 1.0
    F["close_over_ema20_given"] = _safe_div(close, _column(frame, "ema_20")) - 1.0
    F["macd_hist_rel"] = _safe_div(_column(frame, "macd_hist_12_26_9"), close)
    F["macd_diff_rel"] = _safe_div(_column(frame, "macd_diff_12_26_9"), close)
    for L in (20, 60, 120):
        hi = roll("high", high, L, "max")
        lo = roll("low", low, L, "min")
        F[f"high_prox_{L}"] = _safe_div(close, hi) - 1.0
        F[f"range_pos_{L}"] = _safe_div(close - lo, hi - lo)
    for L in (20, 60):
        F[f"low_prox_{L}"] = _safe_div(close, roll("low", low, L, "min")) - 1.0

    # -- fundamentals -------------------------------------------------------------
    pe = _column(frame, "pe_ttm")
    pb = _column(frame, "pb")
    ps = _column(frame, "ps_ttm")
    F["earnings_yield"] = np.where(pe > 0, _safe_div(1.0, np.where(pe > 0, pe, np.nan)), np.nan)
    F["book_to_price"] = np.where(pb > 0, _safe_div(1.0, np.where(pb > 0, pb, np.nan)), np.nan)
    F["sales_yield"] = np.where(ps > 0, _safe_div(1.0, np.where(ps > 0, ps, np.nan)), np.nan)
    total_cap = _column(frame, "total_market_cap")
    float_cap = _column(frame, "float_market_cap")
    F["ln_mktcap"] = np.log(np.where(total_cap > 0, total_cap, np.nan))
    F["ln_floatcap"] = np.log(np.where(float_cap > 0, float_cap, np.nan))
    F["float_ratio"] = _safe_div(float_cap, total_cap)

    # -- provided passthrough -----------------------------------------------------
    for name in _FACTOR_PASSTHROUGH:
        F[name] = _column(frame, name)
    for name in BARRA_NAMES:
        column = _barra_column(frame, name)
        F[f"barra_{name}"] = (_column(frame, column) if column is not None
                              else np.full(len(frame), np.nan))

    # -- cross-sectional ranks (universe = that day's Barra membership) ------------
    sources = list(_CROSS_SECTIONAL_SOURCES)
    ranked = _cross_sectional_rank(frame, np.column_stack([F[n] for n in sources]), member)
    for j, name in enumerate(sources):
        F[f"{name}_xr"] = ranked[:, j]

    # -- interactions -------------------------------------------------------------
    F["mom20_over_vol20"] = _safe_div(F["ret_20"], vol[20] + _EPS)
    F["mom5_over_vol20"] = _safe_div(F["ret_5"], vol[20] + _EPS)
    F["turn_x_ret1"] = turn * ret_1
    F["vol_x_mktcap"] = vol[20] * F["ln_mktcap"]
    F["reversal_x_illiq"] = (-ret_1) * F["amihud_20"]

    # -- additive beta-hedged family (only when asked for) ------------------------
    if any(name in _RESIDUAL_SET for name in wanted):
        per_day, day_index = _member_market_returns(frame, member, ret_1)
        mkt_1 = per_day[day_index]

        def mkt_return(days: int) -> np.ndarray:
            window = max(1, min(int(days), lb - 1))
            return _rolling_growth(per_day, window)[day_index] - 1.0

        def mkt_skip1(days: int) -> np.ndarray:
            window = max(1, min(int(days), lb - 1) - 1)
            growth = _rolling_growth(per_day, window)
            return np.r_[np.nan, growth[:-1]][day_index] - 1.0

        beta = _column(frame, "beta_000300SH_22")
        F["mkt_ret_1"] = mkt_1
        for days in (5, 20):
            F[f"mkt_ret_{days}"] = mkt_return(days)
        for days in (1, 2, 5, 10, 20, 60):
            F[f"resid_ret_{days}"] = F[f"ret_{days}"] - beta * mkt_return(days)
        for days in (5, 20):
            F[f"mkt_rel_ret_{days}"] = F[f"ret_{days}"] - mkt_return(days)
            F[f"resid_ret_{days}_minus_1"] = (F[f"mom_skip1_{days}"]
                                              - beta * mkt_skip1(days))
        residual_1 = ret_1 - beta * mkt_1
        for days in (20, 60):
            F[f"idio_vol_{days}"] = roll("_resid1", residual_1, days, "std")
        F["beta_x_rev"] = beta * F["ret_5"]

    # -- assemble in the requested order, restore the caller's row order ----------
    missing_features = [c for c in wanted if c not in F]
    if missing_features:
        raise RuntimeError(f"internal error, features not produced: {missing_features}")

    data = np.empty((n, len(wanted)), dtype="float64")
    for j, name in enumerate(wanted):
        data[:, j] = F[name]
    np.putmask(data, ~np.isfinite(data), np.nan)

    # Cast once into a fresh float32 buffer and restore the caller's row order with a numpy
    # row-take: `DataFrame(...).iloc[inverse]` is the same result but pays a pandas take over
    # every column (0.77 s vs 0.37 s on the full panel).
    out = np.empty(data.shape, dtype="float32")
    out[:] = data
    inverse = np.argsort(order, kind="stable")
    out = out[inverse]
    result = pd.DataFrame(out, columns=list(wanted), copy=False)
    result.insert(0, "代码", frame["代码"].to_numpy()[inverse])
    result.insert(0, "日期", frame["日期"].to_numpy()[inverse])
    return result


def build_extended_features(panel: pd.DataFrame,
                            lookback: int = MAX_LOOKBACK) -> pd.DataFrame:
    """``build_features`` with the additive beta-hedged family included."""
    return build_features(panel, lookback=lookback, columns=EXTENDED_FEATURE_COLUMNS)
