import threading
from typing import Optional, Callable, List
import pandas as pd


class MarketDataHub:
    """
    Central data store that merges authoritative history and provisional last-minute updates.
    - Tracks which timestamps are provisional so downstream consumers can ignore them.
    - Fans out updates to subscribed views (e.g., LiveGraph) and to function subscribers.
    """
    def __init__(self, initial_df: Optional[pd.DataFrame] = None):
        self._lock = threading.RLock()

        base_df = initial_df if initial_df is not None else pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
        self._df = self._normalize_df(base_df)
        self._provisional = set()

        self._view_subs: List[object] = []
        self._fn_subs: List[Callable[[pd.DataFrame, pd.DataFrame], None]] = []

    # ---------- subscriptions ----------

    def subscribe_view(self, view: object) -> None:
        """
        Subscribe a chart-like object that supports:
          - set_data(df)
          - apply_delta_df(delta_df) [optional]
          - upsert_bar(when, o,h,l,c) [optional; preferred for precision]
          - remove_bar/delete_bar/drop_bar(ts) [optional]
        On subscribe, the full dataset is pushed.
        """
        with self._lock:
            self._view_subs.append(view)
            try:
                if hasattr(view, "set_data"):
                    view.set_data(self._df)
                elif hasattr(view, "set"):
                    view.set(self._df)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[pd.DataFrame, pd.DataFrame], None]) -> None:
        """
        Subscribe a function: fn(delta_df, full_df). No immediate push on subscribe.
        """
        with self._lock:
            self._fn_subs.append(fn)

    # ---------- public state ----------

    def get_df(self, copy: bool = True) -> pd.DataFrame:
        with self._lock:
            return self._df.copy() if copy else self._df

    def get_clean_df(self) -> pd.DataFrame:
        """
        Full DataFrame excluding provisional bars.
        """
        with self._lock:
            if not self._provisional:
                return self._df.copy()
            mask = ~self._df.index.isin(self._provisional)
            return self._df.loc[mask].copy()

    def get_last_closed_anchor(self) -> Optional[pd.Timestamp]:
        """
        Return the last non-provisional timestamp (ignore the current minute’s provisional bar).
        """
        with self._lock:
            if self._df is None or len(self._df) == 0:
                return None
            # Iterate from the end
            for ts in reversed(self._df.index):
                if ts not in self._provisional:
                    return ts
            return None

    # ---------- mutations (authoritative history) ----------

    def set_data(self, df: pd.DataFrame) -> None:
        """
        Replace entire dataset with authoritative data.
        Clears provisional flags for timestamps present in df.
        Notifies subscribers with a full set.
        """
        df = self._normalize_df(df)
        with self._lock:
            self._df = df
            self._provisional.difference_update(df.index)
            self._notify_views_set_locked()
            self._notify_fns_locked(delta_df=df)

    def apply_delta_df(self, delta_df: pd.DataFrame) -> None:
        """
        Merge authoritative delta (history overlap) into the hub.
        Clears provisional flags for those bars.
        Notifies subscribers with the exact changed keys.
        """
        if delta_df is None or delta_df.empty:
            return
        d = self._normalize_df(delta_df)

        with self._lock:
            df = self._df.copy()

            # Drop truly all-NA columns in delta (OHLC should be present)
            d = d.dropna(axis=1, how="all")

            # Ensure union of columns
            for c in d.columns:
                if c not in df.columns:
                    df[c] = pd.NA
            d_aligned = d.reindex(columns=df.columns)

            # IMPORTANT: enlarge index first to avoid KeyError on .loc with new labels
            if not d_aligned.index.isin(df.index).all():
                df = df.reindex(df.index.union(d_aligned.index))

            # Upsert
            df.loc[d_aligned.index, df.columns] = d_aligned
            df.sort_index(inplace=True)
            self._df = df

            # Clear provisional flags for authoritative keys
            for ts in d_aligned.index:
                self._provisional.discard(ts)

            # Notify exact changed keys
            self._notify_views_delta_locked(d_aligned)
            self._notify_fns_locked(delta_df=d_aligned)

    # ---------- mutations (provisional) ----------

    def upsert_bar(
        self,
        when,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        *,
        provisional: bool = False,
    ) -> None:
        """
        Insert/update a single bar; mark it provisional if needed.
        Notifies subscribers with a single-row delta.
        """
        ts = pd.Timestamp(when)
        # Force UTC-naive for consistency
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

        row = pd.DataFrame(
            {"Open": [float(o)], "High": [float(h)], "Low": [float(l)], "Close": [float(c)], "Volume": [float(v)]},
            index=pd.DatetimeIndex([ts])
        )
        row = self._normalize_df(row)

        with self._lock:
            df = self._df.copy()

            # Ensure 'Volume' present and align columns
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
            row = row.reindex(columns=df.columns, fill_value=pd.NA)

            # EITHER: scalar assignment (enlargement-safe)
            df.loc[ts, ["Open", "High", "Low", "Close"]] = [
                float(o), float(h), float(l), float(c)
            ]
            # OR (equivalently, pre-union + vector assignment):
            # if ts not in df.index:
            #     df = df.reindex(df.index.union([ts]))
            # df.loc[row.index, df.columns] = row

            df.sort_index(inplace=True)
            self._df = df

            if provisional:
                self._provisional.add(ts)
            else:
                self._provisional.discard(ts)

            # Notify with the exact single-row delta
            self._notify_views_delta_locked(row)
            self._notify_fns_locked(delta_df=row)

    def drop_bar(self, ts) -> None:
        """
        Remove a single timestamped bar (used by provisional rollover).
        Notifies subscribers to remove or refresh.
        """
        key = pd.Timestamp(ts)
        with self._lock:
            if key in self._df.index:
                self._df = self._df.drop(index=[key], errors="ignore")
            self._provisional.discard(key)
            self._notify_views_drop_or_set_locked(removed_key=key)
            # Synthesize an empty delta with the removed index for fn-subs
            removed_df = pd.DataFrame(index=pd.DatetimeIndex([key]), columns=self._df.columns)
            self._notify_fns_locked(delta_df=removed_df)

    # ---------- notifications ----------

    def _notify_views_set_locked(self) -> None:
        for v in list(self._view_subs):
            try:
                if hasattr(v, "set_data"):
                    v.set_data(self._df)
                elif hasattr(v, "set"):
                    v.set(self._df)
            except Exception:
                pass

    def _notify_views_delta_locked(self, delta_df: pd.DataFrame) -> None:
        """
        Prefer precise per-bar upserts into views if available; fallback to apply_delta_df; else full set.
        This avoids LiveGraph "changed-tail" heuristics missing overlap updates.
        """
        for v in list(self._view_subs):
            pushed = False
            # 1) If view supports upsert_bar, push row-by-row (exact, robust)
            if hasattr(v, "upsert_bar"):
                try:
                    for ts, row in delta_df.iterrows():
                        v.upsert_bar(ts, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), floor_to_minute=False)
                    pushed = True
                except Exception:
                    pushed = False

            # 2) Else, try apply_delta_df with exact changed keys
            if not pushed and hasattr(v, "apply_delta_df"):
                try:
                    v.apply_delta_df(delta_df)
                    pushed = True
                except Exception:
                    pushed = False

            # 3) Fallback: full refresh
            if not pushed:
                try:
                    if hasattr(v, "set_data"):
                        v.set_data(self._df)
                    elif hasattr(v, "set"):
                        v.set(self._df)
                except Exception:
                    pass

    def _notify_views_drop_or_set_locked(self, removed_key: pd.Timestamp) -> None:
        for v in list(self._view_subs):
            ok = False
            for name in ("remove_bar", "delete_bar", "drop_bar"):
                fn = getattr(v, name, None)
                if callable(fn):
                    try:
                        fn(removed_key)
                        ok = True
                        break
                    except Exception:
                        pass
            if not ok:
                try:
                    if hasattr(v, "set_data"):
                        v.set_data(self._df)
                    elif hasattr(v, "set"):
                        v.set(self._df)
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

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        out = df.copy()
        idx = pd.DatetimeIndex(out.index)
        # Normalize to UTC-naive for consistent joins
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        out.index = idx.sort_values()

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

        for col in ("Open", "High", "Low", "Close"):
            if col not in out.columns:
                raise ValueError("DF must contain Open/High/Low/Close")
        if "Volume" not in out.columns:
            out["Volume"] = 0.0

        # Dedup index (keep last)
        out = out[~out.index.duplicated(keep="last")]
        return out