# core/market_data_hub.py
import threading
from typing import Optional, Callable, List
import pandas as pd


class MarketDataHub:
    """
    Central data store that merges authoritative history and provisional last-minute updates.
    - Canonical time = UTC-naive
    - Tracks provisional minutes
    - Applies display_offset_hours ONLY when notifying views
    """

    def __init__(self, initial_df: Optional[pd.DataFrame] = None, display_offset_hours: float = 0.0):
        self._lock = threading.RLock()

        base_df = initial_df if initial_df is not None else pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
        self._df = self._normalize_df(base_df)
        self._provisional = set()

        self._view_subs: List[object] = []
        self._fn_subs: List[Callable[[pd.DataFrame, pd.DataFrame], None]] = []

        # DISPLAY OFFSET (hours -> Timedelta). Positive = shift RIGHT (later).
        self._view_offset = pd.Timedelta(hours=float(display_offset_hours))

    # ---------- display offset API ----------

    def set_display_offset_hours(self, hours: float) -> None:
        with self._lock:
            self._view_offset = pd.Timedelta(hours=float(hours))
            self._notify_views_set_locked()

    # ---------- subscriptions ----------

    def subscribe_view(self, view: object) -> None:
        with self._lock:
            self._view_subs.append(view)
            try:
                view_df = self._df_for_view_locked()
                if hasattr(view, "set_data"):
                    view.set_data(view_df)
                elif hasattr(view, "set"):
                    view.set(view_df)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[pd.DataFrame, pd.DataFrame], None]) -> None:
        with self._lock:
            self._fn_subs.append(fn)

    # ---------- public state ----------

    def get_df(self, copy: bool = True) -> pd.DataFrame:
        with self._lock:
            return self._df.copy() if copy else self._df

    def get_clean_df(self) -> pd.DataFrame:
        with self._lock:
            if not self._provisional:
                return self._df.copy()
            mask = ~self._df.index.isin(self._provisional)
            return self._df.loc[mask].copy()

    def get_last_closed_anchor(self) -> Optional[pd.Timestamp]:
        with self._lock:
            if self._df is None or len(self._df) == 0:
                return None
            for ts in reversed(self._df.index):
                if ts not in self._provisional:
                    return ts
            return None

    # ---------- mutations (authoritative history) ----------

    def set_data(self, df: pd.DataFrame) -> None:
        df = self._normalize_df(df)
        with self._lock:
            self._df = df
            self._provisional.difference_update(df.index)
            self._notify_views_set_locked()
            self._notify_fns_locked(delta_df=df)

    def apply_delta_df(self, delta_df: pd.DataFrame) -> None:
        if delta_df is None or delta_df.empty:
            return
        d = self._normalize_df(delta_df)

        with self._lock:
            df = self._df.copy()

            # Keep all existing columns, add any new ones from d
            d = d.dropna(axis=1, how="all")
            for c in d.columns:
                if c not in df.columns:
                    df[c] = pd.NA
            d_aligned = d.reindex(columns=df.columns)

            # Union index (enlarge-safe), then apply d into df at the same rows
            if not d_aligned.index.isin(df.index).all():
                df = df.reindex(df.index.union(d_aligned.index))

            df.loc[d_aligned.index, df.columns] = d_aligned
            df.sort_index(inplace=True)
            self._df = df

            # Any authoritative row clears provisional flag
            for ts in d_aligned.index:
                self._provisional.discard(ts)

            self._notify_views_delta_locked(d_aligned)
            self._notify_fns_locked(delta_df=d_aligned)

    # ---------- mutations (provisional) ----------

    def upsert_bar(self, when, o: float, h: float, l: float, c: float, v: float, *, provisional: bool = False) -> None:
        ts = self._normalize_ts(when, floor_to_minute=False)  # adapter already floors if requested

        # Minimal row to send as delta to views (avoid double normalization)
        row = pd.DataFrame(
            {"Open": [float(o)], "High": [float(h)], "Low": [float(l)], "Close": [float(c)], "Volume": [float(v)]},
            index=pd.DatetimeIndex([ts])
        )

        with self._lock:
            df = self._df.copy()
            if "Volume" not in df.columns:
                df["Volume"] = 0.0

            df.loc[ts, ["Open", "High", "Low", "Close", "Volume"]] = [float(o), float(h), float(l), float(c), float(v)]
            df.sort_index(inplace=True)
            self._df = df

            if provisional:
                self._provisional.add(ts)
            else:
                self._provisional.discard(ts)

            self._notify_views_delta_locked(row)
            self._notify_fns_locked(delta_df=row)

    def drop_bar(self, ts) -> None:
        key = self._normalize_ts(ts, floor_to_minute=True)
        with self._lock:
            if key in self._df.index:
                self._df = self._df.drop(index=[key], errors="ignore")
            self._provisional.discard(key)

            self._notify_views_drop_or_set_locked(removed_key=key)

            removed_df = pd.DataFrame(index=pd.DatetimeIndex([key]), columns=self._df.columns)
            self._notify_fns_locked(delta_df=removed_df)

    # ---------- notifications (display offset applied here) ----------

    def _notify_views_set_locked(self) -> None:
        view_df = self._df_for_view_locked()
        for v in list(self._view_subs):
            try:
                if hasattr(v, "set_data"):
                    v.set_data(view_df)
                elif hasattr(v, "set"):
                    v.set(view_df)
            except Exception:
                pass

    def _notify_views_delta_locked(self, delta_df: pd.DataFrame) -> None:
        delta_for_view = self._shift_index(delta_df)
        for v in list(self._view_subs):
            pushed = False

            if hasattr(v, "upsert_bar"):
                try:
                    for ts, row in delta_df.iterrows():
                        ts_view = self._shift_ts(ts)
                        o = float(row["Open"])
                        h = float(row["High"])
                        l = float(row["Low"])
                        c = float(row["Close"])
                        # Volume is optional; if missing/NA, treat as 0
                        vol = float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0

                        did = False
                        # First try with volume
                        try:
                            v.upsert_bar(ts_view, o, h, l, c, vol, floor_to_minute=False)
                            did = True
                        except TypeError:
                            # Fallback: old signature without volume
                            v.upsert_bar(ts_view, o, h, l, c, floor_to_minute=False)
                            did = True

                        if not did:
                            # If neither worked, bail to bulk path below
                            raise RuntimeError("view.upsert_bar did not accept expected signatures")

                    pushed = True
                except Exception:
                    pushed = False

            if not pushed and hasattr(v, "apply_delta_df"):
                try:
                    v.apply_delta_df(delta_for_view)
                    pushed = True
                except Exception:
                    pushed = False

            if not pushed:
                try:
                    view_df = self._df_for_view_locked()
                    if hasattr(v, "set_data"):
                        v.set_data(view_df)
                    elif hasattr(v, "set"):
                        v.set(view_df)
                except Exception:
                    pass

    def _notify_views_drop_or_set_locked(self, removed_key: pd.Timestamp) -> None:
        key_view = self._shift_ts(removed_key)
        for v in list(self._view_subs):
            ok = False
            for name in ("remove_bar", "delete_bar", "drop_bar"):
                fn = getattr(v, name, None)
                if callable(fn):
                    try:
                        fn(key_view)
                        ok = True
                        break
                    except Exception:
                        pass
            if not ok:
                try:
                    view_df = self._df_for_view_locked()
                    if hasattr(v, "set_data"):
                        v.set_data(view_df)
                    elif hasattr(v, "set"):
                        v.set(view_df)
                except Exception:
                    pass

    def _notify_fns_locked(self, delta_df: pd.DataFrame) -> None:
        full = self._df
        for fn in list(self._fn_subs):
            try:
                fn(delta_df, full)
            except Exception:
                pass

    # ---------- internals ----------
    def _normalize_ts(self, ts, *, floor_to_minute: bool = False) -> pd.Timestamp:
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


    def _normalize_df(self, df: pd.DataFrame | None) -> pd.DataFrame:
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

    def _df_for_view_locked(self) -> pd.DataFrame:
        if self._view_offset == pd.Timedelta(0):
            return self._df.copy()
        out = self._df.copy()
        out.index = out.index + self._view_offset
        return out

    def _shift_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or self._view_offset == pd.Timedelta(0):
            return df
        out = df.copy()
        out.index = out.index + self._view_offset
        return out

    def _shift_ts(self, ts: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(ts) + self._view_offset