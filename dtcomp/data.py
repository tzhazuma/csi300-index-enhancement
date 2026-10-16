"""Panel loading, caching and label construction.

The competition ships monthly parquet files under ``<data>/<Table>/<YYYY>/<MM>/<YYYYMM>.parquet``
plus a few yearly CSVs.  Everything is small enough (590k rows x ~60 cols) to hold in a
single in-memory panel, so we build that panel once and cache it as parquet.

Public API
----------
load_panel(data_dir, last_date=None, cache=True) -> pd.DataFrame
    One row per (日期, 代码).  Columns:
      id      日期(int YYYYMMDD), 代码(str SH600000), 名称(str)
      market  open, high, low, close, volume, amount, is_suspend, raw_open
      factor  the 33 Factor33_winsor columns (``FACTOR33``)
      barra   the 10 Barra style columns (``BARRA``) renamed to ``barra_<name>``
      meta    in_barra (bool: is a true HS300 member that day)
    Sorted by (日期, 代码).

build_labels(panel) -> pd.DataFrame
    (日期, 代码) -> entry_open, exit_open, future_return with
    future_return = open(t+2)/open(t+1) - 1.

universe(panel, date) -> np.ndarray of 代码 for that signal date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- schema -----------------------------------------------------------------

MARKET_COLS = ["日期", "代码", "名称", "open", "high", "low", "close",
               "volume", "amount", "is_suspend"]

FACTOR33 = (
    "turn", "change_ratio", "daily_return", "momentum_5", "reversal_5",
    "volatility_5", "sma_20", "ema_20", "macd_diff_12_26_9",
    "macd_dea_12_26_9", "macd_hist_12_26_9", "rsi_12", "kdj_k_9_3_3",
    "kdj_d_9_3_3", "bias_20", "cci_14", "atr_14", "total_market_cap",
    "float_market_cap", "pe_ttm", "pb", "ps_ttm", "roe_avg_ttm",
    "roa_avg_ttm", "gross_profit_rate_ttm", "net_profit_rate_ttm",
    "debt_to_asset_lf", "current_ratio_lf", "beta_000300SH_22", "list_days",
    "is_suspend", "is_st", "is_new",
)

BARRA = ("市值", "贝塔", "动量", "残差波动", "非线性市值",
         "账面市值比", "流动性", "盈利", "成长", "杠杆")

BARRA_COLS = tuple(f"barra_{name}" for name in BARRA)

ID_COLS = ["日期", "代码"]
NAME_COL = "名称"

_MONTHLY = {
    "MarketData": MARKET_COLS,
    "MarketDataRawOpen": ["日期", "代码", "open"],
    "Factor33_winsor": ["日期", "代码", *FACTOR33],
    "Barra_factor": ["日期", "代码", *BARRA],
}


# --- low level readers ------------------------------------------------------

def _monthly_paths(root: Path, table: str, last_date: int | None) -> list[Path]:
    paths = sorted((root / table).glob("*/*/*.parquet"))
    if last_date is not None:
        last_month = int(str(last_date)[:6])
        paths = [p for p in paths if int(p.stem) <= last_month]
    if not paths:
        raise FileNotFoundError(f"no monthly parquet under {root / table}")
    return paths


def _read_table(root: Path, table: str, last_date: int | None) -> pd.DataFrame:
    cols = _MONTHLY[table]
    frames = []
    for path in _monthly_paths(root, table, last_date):
        frame = pd.read_parquet(path, columns=cols)
        if last_date is not None:
            frame = frame.loc[frame["日期"].le(last_date)]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out


def _dedup(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    if frame.duplicated(ID_COLS).any():
        before = len(frame)
        frame = frame.drop_duplicates(ID_COLS, keep="last")
        print(f"[data] {table}: dropped {before - len(frame)} duplicate (日期,代码) rows")
    return frame


# --- public API -------------------------------------------------------------

def load_panel(data_dir: str | Path, last_date: int | None = None,
               cache: bool = True) -> pd.DataFrame:
    """Build (or load from cache) the full 2016-2022 research panel."""
    root = Path(data_dir)
    if root.name != "Data" and (root / "Data").is_dir():
        root = root / "Data"
    if not (root / "MarketData").is_dir():
        raise FileNotFoundError(f"{root} does not look like the competition Data dir")

    cache_path = root.parent / f"panel_{last_date or 'full'}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    market = _dedup(_read_table(root, "MarketData", last_date), "MarketData")
    raw = _dedup(_read_table(root, "MarketDataRawOpen", last_date), "MarketDataRawOpen")
    factors = _dedup(_read_table(root, "Factor33_winsor", last_date), "Factor33_winsor")
    barra = _dedup(_read_table(root, "Barra_factor", last_date), "Barra_factor")

    # Factor33 carries its own is_suspend; keep the MarketData one authoritative.
    factors = factors.drop(columns=["is_suspend"], errors="ignore")

    panel = market.merge(raw.rename(columns={"open": "raw_open"}), on=ID_COLS,
                         how="left", validate="one_to_one", sort=False)
    panel = panel.merge(factors, on=ID_COLS, how="left",
                        validate="one_to_one", sort=False)
    panel = panel.merge(barra.rename(columns={n: f"barra_{n}" for n in BARRA}),
                        on=ID_COLS, how="left", validate="one_to_one", sort=False)

    panel["in_barra"] = panel[BARRA_COLS[0]].notna()
    panel = panel.sort_values(ID_COLS, kind="stable").reset_index(drop=True)
    panel["日期"] = panel["日期"].astype("int32")
    panel["代码"] = panel["代码"].astype("string")

    numeric = [c for c in panel.columns if c not in (ID_COLS + [NAME_COL])]
    for col in numeric:
        if panel[col].dtype == bool:
            continue
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")
    panel["in_barra"] = panel["in_barra"].astype(bool)

    if cache:
        try:
            panel.to_parquet(cache_path, index=False)
            print(f"[data] cached panel -> {cache_path}")
        except Exception as exc:  # cache is an optimisation, never fatal
            print(f"[data] cache write skipped: {exc}")
    return panel


def trading_calendar(panel: pd.DataFrame) -> np.ndarray:
    """Sorted unique trading dates present in MarketData."""
    return np.sort(panel["日期"].unique())


def build_labels(panel: pd.DataFrame, price: str = "open",
                 entry_offset: int = 1, exit_offset: int = 2) -> pd.DataFrame:
    """Attach entry/exit prices and the open-to-open label.

    ``price`` selects the price column: ``open`` is the back-adjusted series the
    baseline uses; ``raw_open`` is the unadjusted one.  We default to adjusted
    because a ratio of back-adjusted prices is the corporate-action-clean return.
    """
    calendar = trading_calendar(panel)
    if len(calendar) <= exit_offset:
        raise ValueError("calendar too short for the requested holding period")
    signal_dates = calendar[: -(exit_offset)]
    endpoints = pd.DataFrame({
        "日期": signal_dates,
        "entry_date": calendar[entry_offset:len(calendar) - (exit_offset - entry_offset)],
        "exit_date": calendar[exit_offset:],
    })
    if len(endpoints) != len(signal_dates):
        raise ValueError("endpoint construction produced a length mismatch")

    prices = panel.loc[:, ["日期", "代码", price, "is_suspend"]].rename(
        columns={"日期": "endpoint_date", price: "px"})
    out = endpoints.merge(
        panel.loc[:, ID_COLS].drop_duplicates(), on="日期", how="inner")
    for kind in ("entry", "exit"):
        side = prices.rename(columns={
            "endpoint_date": f"{kind}_date", "px": f"{kind}_open",
            "is_suspend": f"{kind}_suspended"})
        out = out.merge(side, on=[f"{kind}_date", "代码"], how="left",
                        validate="many_to_one", sort=False)
    out["future_return"] = out["exit_open"] / out["entry_open"] - 1.0
    out.loc[~np.isfinite(out["future_return"]), "future_return"] = np.nan
    return out


def tradable_mask(labels: pd.DataFrame) -> pd.Series:
    """Rows usable as training targets: no suspension, finite positive prices."""
    return (
        labels["entry_suspended"].eq(0)
        & labels["exit_suspended"].eq(0)
        & np.isfinite(labels["entry_open"]) & np.isfinite(labels["exit_open"])
        & labels["entry_open"].gt(0) & labels["exit_open"].gt(0)
        & labels["future_return"].notna()
    )


def universe(panel: pd.DataFrame, date: int) -> np.ndarray:
    """The legal stock universe for a signal date: Barra_factor members."""
    day = panel.loc[panel["in_barra"] & panel["日期"].eq(date), "代码"]
    return day.to_numpy()
