"""Walk-forward validation and cross-sectional RankIC metrics.

The competition metric is mean daily Spearman correlation between the model score
and the next-day open-to-open return, over the legal universe of that day:

    RankIC(t) = spearman(score(t), future_return(t))
    RankICIR  = mean_t RankIC(t) / std_t RankIC(t)

Everything here works on a frame with at least ``日期``, ``future_return`` and a
score column, so it can be reused for validation, local test and leaderboard
replays.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd


# --- daily metrics ----------------------------------------------------------

def daily_rankic(frame: pd.DataFrame, score: str = "score",
                 target: str = "future_return") -> pd.Series:
    """Spearman(score, target) per trading day."""
    sub = frame.loc[:, ["日期", score, target]].dropna()
    if sub.empty:
        return pd.Series(dtype="float64")
    return (
        sub.groupby("日期", sort=True)
        .apply(lambda day: day[score].corr(day[target], method="spearman"),
               include_groups=False)
        .astype("float64")
    )


def ic_summary(frame: pd.DataFrame, score: str = "score",
               target: str = "future_return") -> dict[str, float]:
    """Mean RankIC, std, RankICIR and positive-IC ratio."""
    per_day = daily_rankic(frame, score, target).dropna()
    if per_day.empty:
        return {"mean_rankic": float("nan"), "std_rankic": float("nan"),
                "rankicir": float("nan"), "positive_ratio": float("nan"),
                "days": 0}
    mean = float(per_day.mean())
    std = float(per_day.std(ddof=1)) if len(per_day) > 1 else float("nan")
    return {
        "mean_rankic": mean,
        "std_rankic": std,
        "rankicir": mean / std if std and np.isfinite(std) and std > 0 else float("nan"),
        "positive_ratio": float((per_day > 0).mean()),
        "days": int(len(per_day)),
    }


def rankic_by_year(frame: pd.DataFrame, score: str = "score",
                   target: str = "future_return") -> pd.DataFrame:
    """Per-calendar-year summary, for regime-robustness inspection."""
    sub = frame.loc[:, ["日期", score, target]].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["year", "mean_rankic", "std_rankic",
                                     "rankicir", "positive_ratio", "days"])
    sub["year"] = sub["日期"] // 10000
    rows = []
    for year, chunk in sub.groupby("year", sort=True):
        row = {"year": int(year)}
        row.update(ic_summary(chunk, score, target))
        rows.append(row)
    return pd.DataFrame(rows)


def ic_decay(frame: pd.DataFrame, score: str = "score",
             target: str = "future_return") -> pd.DataFrame:
    """Mean IC per calendar month -- cheap regime/stability view."""
    sub = frame.loc[:, ["日期", score, target]].dropna().copy()
    per_day = daily_rankic(sub, score, target)
    if per_day.empty:
        return pd.DataFrame(columns=["month", "mean_rankic", "days"])
    out = per_day.rename("ic").reset_index()
    out["month"] = out["日期"] // 100
    agg = out.groupby("month")["ic"].agg(["mean", "count"])
    agg.columns = ["mean_rankic", "days"]
    return agg.reset_index()


# --- walk-forward splits ----------------------------------------------------

def walk_forward_splits(calendar: Iterable[int], *, n_folds: int = 4,
                        min_train_years: float = 2.0,
                        valid_months: int = 6) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(train_end, valid_start, valid_end, test_end)`` date cut-offs.

    Strictly time-ordered expanding-window folds.  ``valid_start`` is the first
    signal date of the validation block, ``valid_end`` its last; ``train_end`` is
    the last signal date allowed into training.  ``test_end`` bounds the data the
    fold may read at all (so a fold never sees the future).
    """
    dates = np.sort(np.asarray(list(calendar), dtype="int64"))
    if len(dates) < 100:
        raise ValueError("calendar too short for walk-forward validation")
    start_year = dates[0] // 10000
    first_train_end_year = start_year + int(np.ceil(min_train_years))
    months = np.array([d // 100 for d in dates])

    folds: list[tuple[int, int, int, int]] = []
    for fold in range(n_folds):
        train_end_year = first_train_end_year + fold
        train_end = int(dates[months < train_end_year * 100].max()) if np.any(
            months < train_end_year * 100) else None
        if train_end is None:
            continue
        valid_start_month = train_end_year * 100
        valid_end_month = valid_start_month + valid_months - 1
        valid = dates[(months >= valid_start_month) & (months <= valid_end_month)]
        if len(valid) == 0:
            continue
        folds.append((train_end, int(valid[0]), int(valid[-1]), int(valid[-1])))
    if not folds:
        raise ValueError("no walk-forward fold could be constructed")
    yield from folds


def time_split(calendar: Iterable[int], *, train_end: int,
               valid_end: int) -> dict[str, np.ndarray]:
    """A single expanding split: train <= train_end < valid <= valid_end."""
    dates = np.sort(np.asarray(list(calendar), dtype="int64"))
    return {
        "train": dates[dates <= train_end],
        "valid": dates[(dates > train_end) & (dates <= valid_end)],
        "test": dates[dates > valid_end],
    }


# --- target transforms ------------------------------------------------------

def cross_sectional_rank_target(frame: pd.DataFrame, target: str = "future_return",
                                out: str = "target") -> pd.DataFrame:
    """Map raw returns to a within-day percentile rank in [-0.5, 0.5].

    This is the baseline's target.  It is scale-free and robust to the fat tails
    of daily A-share returns, at the cost of discarding magnitude information.
    """
    frame = frame.copy()
    frame[out] = frame.groupby("日期")[target].rank(pct=True) - 0.5
    return frame


def cross_sectional_zscore(frame: pd.DataFrame, columns: Iterable[str],
                           clip: float = 3.0) -> pd.DataFrame:
    """Within-day standardisation with an optional clip (winsorisation)."""
    frame = frame.copy()
    grouped = frame.groupby("日期")
    for col in columns:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std")
        z = (frame[col] - mean) / std.replace(0.0, np.nan)
        frame[col] = z.clip(-clip, clip)
    return frame
