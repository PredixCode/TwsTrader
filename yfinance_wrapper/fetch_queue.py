import os
import yfinance as yf
import pandas as pd
import time
import random
import pickle


class FetchQueue:
    """
    Persistent on-disk caching with TTL, incremental updates, and minimal tail probes.
    """

    def __init__(self, cache_file="data/fetch_cache.pkl", max_age_seconds: int = 60):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_file = os.path.join(file_dir, cache_file)
        self.max_age_seconds = max_age_seconds

        self._overlap_by_interval = {
            "1m":  pd.Timedelta(minutes=5),
            "2m":  pd.Timedelta(minutes=5),
            "5m":  pd.Timedelta(minutes=15),
            "15m": pd.Timedelta(minutes=30),
            "30m": pd.Timedelta(hours=1),
            "60m": pd.Timedelta(hours=2),
            "90m": pd.Timedelta(hours=3),
            "1h":  pd.Timedelta(hours=2),
            "1d":  pd.Timedelta(days=3),
            "5d":  pd.Timedelta(days=7),
            "1wk": pd.Timedelta(weeks=2),
            "1mo": pd.Timedelta(days=7),
            "3mo": pd.Timedelta(days=14),
        }

        self._tail_period_by_interval = {
            "1m":  "2m",   # tiny probe
            "2m":  "5m",
            "5m":  "1d",
            "15m": "1d",
            "30m": "5d",
            "60m": "7d",
            "90m": "7d",
            "1h":  "7d",
            "1d":  "1mo",
            "5d":  "3mo",
            "1wk": "6mo",
            "1mo": "2y",
            "3mo": "5y",
        }

        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Found existing cache file: '{self.cache_file}'. Loading...")
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                migrated = 0
                for k, v in list(self._cache.items()):
                    if isinstance(v, pd.DataFrame):
                        self._cache[k] = {"data": v, "fetched_at": 0.0}
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
        print(f"💾 Saving cache to '{self.cache_file}'...")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            print("   -> Cache saved successfully.")
        except Exception as e:
            print(f"   -> ❌ Error: Could not save cache file. Error: {e}")

    def _create_cache_key(self, ticker: str, params: dict) -> tuple:
        # order-independent cache key
        return (ticker,) + tuple(sorted(params.items()))

    def _needs_update(self, fetched_at: float) -> bool:
        return (time.time() - fetched_at) > self.max_age_seconds

    def _get_overlap(self, interval: str) -> pd.Timedelta:
        return self._overlap_by_interval.get(interval, pd.Timedelta(minutes=5))

    def _minute_interval(self, interval: str) -> bool:
        return interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

    def _now_naive_utc(self) -> pd.Timestamp:
        # yfinance plays nice with naive UTC start/end
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _to_naive_utc(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            return ts
        return ts.tz_convert("UTC").tz_localize(None)

    def get_cached(self, ticker_symbol: str, **params) -> pd.DataFrame | None:
        """
        Return a copy of cached data (no network, no TTL checks). None if not cached.
        """
        cache_key = self._create_cache_key(ticker_symbol, params)
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        if isinstance(entry, pd.DataFrame):
            return entry.copy()
        return entry["data"].copy()

    def _incremental_update(self, ticker: yf.Ticker, cached_df: pd.DataFrame, original_params: dict) -> pd.DataFrame:
        if cached_df is None or cached_df.empty:
            return ticker.history(**original_params)

        interval = original_params.get("interval", "1d")

        last_ts = cached_df.index.max()
        if pd.isna(last_ts):
            return ticker.history(**original_params)

        last_ts_utc = self._to_naive_utc(last_ts)
        overlap = self._get_overlap(interval)
        start_ts = last_ts_utc - overlap

        now_ts = self._now_naive_utc()

        if self._minute_interval(interval):
            seven_days_ago = now_ts - pd.Timedelta(days=7)
            if start_ts < seven_days_ago:
                start_ts = seven_days_ago

        if start_ts >= now_ts:
            return cached_df

        update_params = {k: v for k, v in original_params.items() if k not in ("period", "start", "end")}
        update_params["start"] = start_ts
        update_params["end"] = now_ts

        delay = random.uniform(0.6, 1.4)
        print(f"   -> Incremental fetch: {update_params} | throttling {delay:.2f}s...")
        time.sleep(delay)

        df_new = ticker.history(**update_params)
        if df_new is None or df_new.empty:
            print("   -> No new data returned; cache considered up-to-date.")
            return cached_df

        combined = pd.concat([cached_df, df_new])
        combined.sort_index(inplace=True)
        combined = combined[~combined.index.duplicated(keep='last')]
        return combined

    def _tail_merge(self, cached_df: pd.DataFrame, df_tail: pd.DataFrame) -> pd.DataFrame:
        if cached_df is None or cached_df.empty:
            return df_tail if df_tail is not None else pd.DataFrame()
        if df_tail is None or df_tail.empty:
            return cached_df
        combined = pd.concat([cached_df, df_tail])
        combined.sort_index(inplace=True)
        combined = combined[~combined.index.duplicated(keep='last')]
        return combined

    def fetch(self, ticker: yf.Ticker, ticker_symbol: str, *, allow_stale: bool = False,
              force_tail: str | None = None, **params) -> pd.DataFrame:
        """
        Fetch with caching.
        - allow_stale=True: return cached immediately (no TTL checks, no network).
        - force_tail="2m" (or similar): force a small tail network fetch (period=force_tail),
          then merge into the cache entry for the ORIGINAL params.
        """
        # Cache key is always made from the ORIGINAL params (NOT force-tail ones).
        cache_key = self._create_cache_key(ticker_symbol, params)
        entry = self._cache.get(cache_key)

        # Legacy normalization
        if entry is not None and isinstance(entry, pd.DataFrame):
            entry = {"data": entry, "fetched_at": 0.0}
            self._cache[cache_key] = entry

        if allow_stale and entry is not None:
            print(f"CACHE HIT (allow_stale): {ticker_symbol} {params}")
            return entry["data"].copy()

        # If we are forcing a tail probe/update, do that now and write back under original key.
        if force_tail is not None:
            interval = params.get("interval", "1d")
            tail_params = {k: v for k, v in params.items() if k not in ("period", "start", "end")}
            tail_params["period"] = force_tail
            tail_params["interval"] = interval

            delay = random.uniform(0.6, 1.4)
            print(f"   -> Tail forced fetch: {ticker_symbol} {tail_params} | throttling {delay:.2f}s...")
            time.sleep(delay)

            try:
                df_tail = ticker.history(**tail_params)
                cached_df = entry["data"] if entry is not None else pd.DataFrame()
                merged = self._tail_merge(cached_df, df_tail)
                self._cache[cache_key] = {"data": merged, "fetched_at": time.time()}
                self._save_cache()
                return merged.copy()
            except Exception as e:
                print(f"   -> ❌ Tail forced fetch failed: {e}")
                # Fall through to normal behavior below

        # Normal cache behavior
        if entry is not None:
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
                print(f"   -> ❌ Incremental update failed. Keeping cached. Error: {e}")
                return entry["data"].copy()

        # CACHE MISS
        print(f"CACHE MISS: Fetching new data for {ticker_symbol} with params {params}...")
        delay = random.uniform(0.6, 1.4)
        print(f"   -> Throttling: Waiting for {delay:.2f} seconds...")
        time.sleep(delay)

        try:
            history = ticker.history(**params)
            if history.empty:
                print("   -> Warning: yfinance returned no data for this request.")
            self._cache[cache_key] = {"data": history, "fetched_at": time.time()}
            print("   -> Success! Data fetched and added to cache.")
            self._save_cache()
            return history.copy()
        except Exception as e:
            print(f"   -> ❌ An error occurred during fetch for {ticker_symbol}: {e}")
            return pd.DataFrame()