# core/normalize.py
from __future__ import annotations
import pandas as pd


def normalize_ts(ts, *, floor_to_minute: bool = False) -> pd.Timestamp:
    """
    Convert any datetime-like to UTC-naive pandas.Timestamp.
    Optionally floor to the minute.
    """
    t = pd.Timestamp(ts)
    if floor_to_minute:
        t = t.floor("min")
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def normalize_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Canonicalize a market-data DataFrame:
      - Ensure DatetimeIndex
      - Convert to UTC-naive
      - Sort ascending
      - Drop duplicate index (keep last)
      - Case-normalize OHLC/Volume
      - Ensure Volume exists (default 0.0)
      - Preserve extra columns
      - Do NOT apply any display offset (that belongs to the view layer)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()

    # Enforce DatetimeIndex
    out.index = pd.DatetimeIndex(out.index)

    # UTC-naive index
    idx = out.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx

    # Case-normalize columns
    rename = {}
    for c in out.columns:
        lc = str(c).lower()
        if lc == "open":   rename[c] = "Open"
        elif lc == "high": rename[c] = "High"
        elif lc == "low":  rename[c] = "Low"
        elif lc == "close": rename[c] = "Close"
        elif lc == "volume": rename[c] = "Volume"
    if rename:
        out = out.rename(columns=rename)

    # Validate OHLC presence
    for col in ("Open", "High", "Low", "Close"):
        if col not in out.columns:
            raise ValueError("DF must contain Open/High/Low/Close")

    # Ensure Volume
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    # Sort and drop dups (keep last — important for overlap merges)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Put OHLCV first, preserve extras after
    core = ["Open", "High", "Low", "Close", "Volume"]
    extras = [c for c in out.columns if c not in core]
    out = out[core + extras]

    return out