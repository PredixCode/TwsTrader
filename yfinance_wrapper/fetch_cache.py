import os
import time
import pickle
from time import sleep
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf


class FetchCache:
    """
    Persistent on-disk caching with a simple TTL and safe incremental updates
    that preserve older cached history on refresh.
    """

    def __init__(self, cache_file: str = "data/fetch_cache.pkl", max_age_seconds: int = 24 * 60 * 60):
        # Default TTL = 1 week
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_file = os.path.join(file_dir, cache_file)
        self.max_age_seconds = max_age_seconds

        # Define which intervals are intraday (used for cache-key normalization)
        self._intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

        # Overlap by interval for incremental updates (minimal but safe)
        self._overlap_by_interval: Dict[str, pd.Timedelta] = {
            "1m": pd.Timedelta(minutes=5),
            "2m": pd.Timedelta(minutes=5),
            "5m": pd.Timedelta(minutes=15),
            "15m": pd.Timedelta(minutes=30),
            "30m": pd.Timedelta(hours=1),
            "60m": pd.Timedelta(hours=2),
            "90m": pd.Timedelta(hours=3),
            "1h": pd.Timedelta(hours=2),
            "1d": pd.Timedelta(days=3),
            "5d": pd.Timedelta(days=7),
            "1wk": pd.Timedelta(weeks=2),
            "1mo": pd.Timedelta(days=7),
            "3mo": pd.Timedelta(days=14),
        }

        self._cache: Dict[Tuple, Dict] = {}
        self._ensure_cache_dir()
        self._load_cache()

    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _normalize_dataframe(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        normalized = df.copy()
        idx = pd.DatetimeIndex(normalized.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        normalized.index = idx
        normalized.sort_index(inplace=True)
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        return normalized

    def _merge_preserving_history(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Union old and new, sort by time, drop duplicate timestamps (keep latest).
        Always normalize to UTC-naive and monotonic index.
        """
        old_df = self._normalize_dataframe(old_df)
        new_df = self._normalize_dataframe(new_df)
        if old_df.empty:
            return new_df
        if new_df.empty:
            return old_df
        combined = pd.concat([old_df, new_df], axis=0)
        return self._normalize_dataframe(combined)

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "rb") as f:
                loaded = pickle.load(f)
            # Migrate legacy structure
            if isinstance(loaded, dict):
                for k, v in list(loaded.items()):
                    if isinstance(v, pd.DataFrame):
                        loaded[k] = {"data": self._normalize_dataframe(v), "fetched_at": 0.0}
                    elif isinstance(v, dict) and "data" in v:
                        v["data"] = self._normalize_dataframe(v["data"])
            self._cache = loaded
        except Exception:
            self._cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass  # best-effort cache

    def _create_cache_key(self, ticker_symbol: str, params: dict) -> Tuple:
        """
        For intraday intervals, consolidate cache keys by ignoring period/start/end
        so 1m (and other intraday) requests enrich the same cache regardless of lookback.
        """
        normalized = dict(params)

        # Optional alias normalization to avoid double-caching across synonyms
        interval = normalized.get("interval", "")
        if interval == "1h":
            normalized["interval"] = "60m"
            interval = "60m"

        if interval in self._intraday_intervals:
            # Ignore fields that only constrain fetch window, not the data granularity
            normalized.pop("period", None)
            normalized.pop("start", None)
            normalized.pop("end", None)

        # Make a stable, hashable key
        return (ticker_symbol,) + tuple(sorted(normalized.items()))

    def _needs_update(self, fetched_at: float) -> bool:
        return (time.time() - fetched_at) > self.max_age_seconds

    def _get_overlap(self, interval: str) -> pd.Timedelta:
        return self._overlap_by_interval.get(interval, pd.Timedelta(minutes=5))

    def _to_naive_utc(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    def _now_naive_utc(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _incremental_update(self, ticker: yf.Ticker, cached_df: pd.DataFrame, original_params: dict) -> pd.DataFrame:
        """
        Append only the tail since the last cached point with a small overlap.
        This preserves the older cache (important for 1m/2m/5m which don't go infinitely far back).
        """
        cached_df = self._normalize_dataframe(cached_df)
        if cached_df.empty:
            fresh = self._normalize_dataframe(self.__safe_fetch(ticker, **original_params))
            return fresh

        interval = original_params.get("interval", "1d")
        last_ts = pd.Timestamp(cached_df.index.max())
        if pd.isna(last_ts):
            fresh = self._normalize_dataframe(self.__safe_fetch(ticker, **original_params))
            return fresh

        last_ts_utc = self._to_naive_utc(last_ts)
        start_ts = last_ts_utc - self._get_overlap(interval)
        now_ts = self._now_naive_utc()

        if start_ts >= now_ts:
            return cached_df

        # Convert to a start/end query; remove period to avoid server-side truncation.
        update_params = {k: v for k, v in original_params.items() if k not in ("period", "start", "end")}
        update_params["start"] = start_ts
        update_params["end"] = now_ts

        df_new = self._normalize_dataframe(self.__safe_fetch(ticker, **update_params))
        if df_new.empty:
            return cached_df

        return self._merge_preserving_history(cached_df, df_new)

    def fetch(self, ticker: yf.Ticker, ticker_symbol: str, **params) -> pd.DataFrame:
        """
        - Always force-refresh for 1m, but MERGE with cached history (never overwrite).
        - For other intervals, use weekly TTL with incremental updates to preserve older history.
        """
        interval = params.get("interval", "")
        cache_key = self._create_cache_key(ticker_symbol, params)
        entry = self._cache.get(cache_key)
        cached_df = self._normalize_dataframe(entry["data"]) if entry else pd.DataFrame()

        # 1) Always force-refresh 1m, merging with cache (important due to ~7d limit)
        if interval == "1m":
            if not cached_df.empty:
                updated = self._incremental_update(ticker, cached_df, params)
                added = max(0, len(updated) - len(cached_df))
                print(f"[FetchCache] Forced refresh for {ticker_symbol} {params} -> incremental merge (+{added}, total {len(updated)})")
                self._cache[cache_key] = {"data": updated, "fetched_at": time.time()}
                self._save_cache()
                return updated.copy()
            else:
                fresh = self._normalize_dataframe(self.__safe_fetch(ticker, **params))
                print(f"[FetchCache] No cache for {ticker_symbol} {params} -> ({len(fresh)} rows)")
                self._cache[cache_key] = {"data": fresh, "fetched_at": time.time()}
                self._save_cache()
                return fresh.copy()

        # 2) For other intervals, standard TTL behavior
        if not cached_df.empty and entry and not self._needs_update(entry["fetched_at"]):
            print(f"[FetchCache] Cache hit for {ticker_symbol} {params} -> ({len(cached_df)} rows)")
            return cached_df.copy()

        if not cached_df.empty:
            updated = self._incremental_update(ticker, cached_df, params)
            added = max(0, len(updated) - len(cached_df))
            print(f"[FetchCache] Cache stale for {ticker_symbol} -> incremental update (+{added}, total {len(updated)})")
            self._cache[cache_key] = {"data": updated, "fetched_at": time.time()}
            self._save_cache()
            return updated.copy()

        fresh = self._normalize_dataframe(self.__safe_fetch(ticker, **params))
        print(f"[FetchCache] No cache for {ticker_symbol} {params} -> network fetch ({len(fresh)} rows)")
        self._cache[cache_key] = {"data": fresh, "fetched_at": time.time()}
        self._save_cache()
        return fresh.copy()
    
    def __safe_fetch(self, ticker, **params):
        sleep(1)
        return ticker.history(**params)