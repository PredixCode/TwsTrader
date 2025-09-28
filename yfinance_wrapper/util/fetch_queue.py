import os
import yfinance as yf
import pandas as pd
import time
import random
import pickle
from datetime import timedelta


class FetchQueue:
    """
    Manages and optimizes data fetching with persistent, on-disk caching.
    Adds TTL freshness checks and incremental updates (append newest only).
    """

    def __init__(self, cache_file="fetch_cache.pkl", max_age_seconds: int = 60):
        """
        cache_file: filename for on-disk cache (created next to this file).
        max_age_seconds: if cache entry older than this, perform incremental update.
        """
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_file = os.path.join(file_dir, cache_file)
        self.max_age_seconds = max_age_seconds

        # Small overlap by interval to avoid missing bars and handle exchange delays
        self._overlap_by_interval = {
            "1m":  pd.Timedelta(minutes=5),
            "2m":  pd.Timedelta(minutes=5),
            "5m":  pd.Timedelta(minutes=15),
            "15m": pd.Timedelta(minutes=30),
            "30m": pd.Timedelta(minutes=60),
            "60m": pd.Timedelta(hours=2),
            "90m": pd.Timedelta(hours=3),
            "1h":  pd.Timedelta(hours=2),
            "1d":  pd.Timedelta(days=3),
            "5d":  pd.Timedelta(days=7),
            "1wk": pd.Timedelta(weeks=2),
            "1mo": pd.Timedelta(days=7),
            "3mo": pd.Timedelta(days=14),
        }

        self._cache: dict = {}
        self._load_cache()  # Load cache on initialization

    def _load_cache(self):
        """Loads the cache dictionary from a pickle file and migrates legacy entries if needed."""
        if os.path.exists(self.cache_file):
            print(f"Found existing cache file: '{self.cache_file}'. Loading...")
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)

                # Migrate legacy entries (DataFrame only) to new dict structure
                migrated = 0
                for k, v in list(self._cache.items()):
                    if isinstance(v, pd.DataFrame):
                        self._cache[k] = {
                            "data": v,
                            "fetched_at": 0.0,  # unknown legacy time
                        }
                        migrated += 1
                if migrated:
                    print(f"✅ Migrated {migrated} legacy cache entries to new structure.")

                print(f"✅ Success! Loaded {len(self._cache)} items from yfinance cache.")
            except Exception as e:
                print(f"❌ Warning: Could not load yfinance cache file. Starting without cache. Error: {e}")
                self._cache = {}
        else:
            print("No cache file found. A new one will be created.")

    def _save_cache(self):
        """Saves the current cache dictionary to a pickle file."""
        print(f"💾 Saving cache to '{self.cache_file}'...")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            print("   -> Cache saved successfully.")
        except Exception as e:
            print(f"   -> ❌ Error: Could not save cache file. Error: {e}")

    def _create_cache_key(self, ticker: str, params: dict) -> tuple:
        """
        Creates a unique, order-independent key for a request based on the original
        params the caller supplied (not the internal incremental params).
        """
        return (ticker,) + tuple(sorted(params.items()))

    def _needs_update(self, fetched_at: float) -> bool:
        """Whether the entry should be refreshed based on TTL."""
        return (time.time() - fetched_at) > self.max_age_seconds

    def _get_overlap(self, interval: str) -> pd.Timedelta:
        return self._overlap_by_interval.get(interval, pd.Timedelta(minutes=5))

    def _now_like(self, idx: pd.DatetimeIndex | None) -> pd.Timestamp:
        """
        Current timestamp matching timezone of idx if available.
        """
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        if idx is not None and len(idx) > 0 and idx.tz is not None:
            return now_utc.tz_convert(idx.tz)
        return now_utc

    def _minute_interval(self, interval: str) -> bool:
        return interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

    def _incremental_update(self, ticker: yf.Ticker, cached_df: pd.DataFrame, original_params: dict) -> pd.DataFrame:
        """
        Fetches only the newest data since the last timestamp in cached_df (with small overlap),
        merges and deduplicates, preferring latest fetch on overlap.
        """
        if cached_df is None or cached_df.empty:
            # No data to increment from; fall back to full fetch with original params
            return ticker.history(**original_params)

        interval = original_params.get("interval", "1d")
        last_ts = cached_df.index.max()
        if pd.isna(last_ts):
            return ticker.history(**original_params)

        overlap = self._get_overlap(interval)
        start_ts = last_ts - overlap

        now_ts = self._now_like(cached_df.index)
        # yfinance's minute data only last 7 days; respect that cap for incremental start
        if self._minute_interval(interval):
            seven_days_ago = now_ts - pd.Timedelta(days=7)
            if start_ts < seven_days_ago:
                start_ts = seven_days_ago

        # Build update params: use start/end instead of period to get just the tail window.
        update_params = {k: v for k, v in original_params.items() if k not in ("period", "start", "end")}
        update_params["start"] = start_ts
        update_params["end"] = now_ts

        delay = random.uniform(0.6, 1.4)
        print(f"   -> Incremental fetch: {update_params} | throttling {delay:.2f}s...")
        time.sleep(delay)

        df_new = ticker.history(**update_params)

        if df_new is None or df_new.empty:
            # No new bars (market closed, etc.). Consider it up-to-date.
            print("   -> No new data returned; cache considered up-to-date.")
            return cached_df

        combined = pd.concat([cached_df, df_new])
        combined.sort_index(inplace=True)
        # Keep the latest occurrence on duplicate timestamps
        combined = combined[~combined.index.duplicated(keep='last')]
        return combined

    def fetch(self, ticker: yf.Ticker, ticker_symbol: str, **params) -> pd.DataFrame:
        """
        Fetches data with caching and freshness checks.
        - If cache exists and fresh (<= TTL), return cached.
        - If stale, perform incremental update (append newest data).
        - If no cache, fetch fresh and store.
        """
        cache_key = self._create_cache_key(ticker_symbol, params)
        entry = self._cache.get(cache_key)

        if entry is not None:
            # Normalize legacy entry shape
            if isinstance(entry, pd.DataFrame):
                entry = {"data": entry, "fetched_at": 0.0}
                self._cache[cache_key] = entry

            print(f"CACHE HIT: Found data for {ticker_symbol} with params {params} in cache.")
            if not self._needs_update(entry["fetched_at"]):
                return entry["data"].copy()

            print("   -> Cache stale (older than TTL). Incrementally updating...")
            try:
                updated_df = self._incremental_update(ticker, entry["data"], params)
                self._cache[cache_key] = {"data": updated_df, "fetched_at": time.time()}
                self._save_cache()
                return updated_df.copy()
            except Exception as e:
                print(f"   -> ❌ Incremental update failed, falling back to full fetch. Error: {e}")

        # CACHE MISS or fallback full fetch
        print(f"CACHE MISS: Fetching new data for {ticker_symbol} with params {params}...")
        delay = random.uniform(0.6, 1.4)
        print(f"   -> Throttling: Waiting for {delay:.2f} seconds...")
        time.sleep(delay)

        try:
            history = ticker.history(**params)
            if history.empty:
                print(f"   -> Warning: yfinance returned no data for this request.")
            self._cache[cache_key] = {"data": history, "fetched_at": time.time()}
            print(f"   -> Success! Data fetched and added to cache.")
            self._save_cache()
            return history.copy()
        except Exception as e:
            print(f"   -> ❌ An error occurred during fetch for {ticker_symbol}: {e}")
            return pd.DataFrame()