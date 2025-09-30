import os
import time
import pickle
from typing import Optional, Dict, Tuple, List

import pandas as pd
from ib_insync import IB, Contract, util

# --------------------------
# Fetch Cache for IB (Historical Bars)
# Mirrors the yfinance FetchCache interface/behavior
# --------------------------
class TwsCache:
    """
    Persistent on-disk caching with a simple TTL and safe incremental updates
    that preserve older cached history on refresh. Normalizes everything to
    UTC-naive DateTime index and dedupes timestamps (keep latest).
    """

    def __init__(self, cache_file: str = "data/tws_fetch_cache.pkl", max_age_seconds: int = 24 * 60 * 60):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_file = os.path.join(file_dir, cache_file)
        self.max_age_seconds = max_age_seconds

        # Intraday intervals (use unified keys & behavior)
        self._intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

        # Overlap for incremental tail refresh per interval
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

        # Sensible "max" lookbacks by interval for IB paging (approximate)
        self._max_days_for_interval: Dict[str, int] = {
            "1m": 30,    # IB typical limit for 1-min
            "2m": 60,
            "5m": 180,
            "15m": 365,
            "30m": 365,
            "60m": 365,
            "90m": 365,
            "1h": 365,
        }

        self._cache: Dict[Tuple, Dict] = {}
        self._ensure_cache_dir()
        self._load_cache()

    # --------- Cache infra ---------
    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "rb") as f:
                loaded = pickle.load(f)
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
            pass  # best-effort

    # --------- Normalization & merge ---------
    def _normalize_dataframe(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.index, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(out.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            out.index = idx
        out.sort_index(inplace=True)
        out = out[~out.index.duplicated(keep="last")]
        # Standardize column names if coming raw from ib_insync util.df
        cols_lower = {c.lower(): c for c in out.columns}
        rename_map = {}
        for raw, std in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
            if raw in cols_lower:
                rename_map[cols_lower[raw]] = std
        if rename_map:
            out.rename(columns=rename_map, inplace=True)
        # Ensure standard columns exist
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in out.columns:
                out[c] = pd.NA
        # Preserve Dividends/Splits shape like yfinance (NA for IB)
        if "Dividends" not in out.columns:
            out["Dividends"] = pd.NA
        if "Stock Splits" not in out.columns:
            out["Stock Splits"] = pd.NA
        out = out[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
        out.index.name = "Datetime"
        return out

    def _merge_preserving_history(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        old_df = self._normalize_dataframe(old_df)
        new_df = self._normalize_dataframe(new_df)
        if old_df.empty:
            return new_df
        if new_df.empty:
            return old_df
        combined = pd.concat([old_df, new_df], axis=0)
        return self._normalize_dataframe(combined)

    # --------- Keying & TTL ---------
    def _create_cache_key(self, ticker_symbol: str, params: dict) -> Tuple:
        """
        For intraday intervals, consolidate cache keys by ignoring period/start/end,
        so that 1m (and other intraday) requests enrich the same cache regardless of lookback.
        """
        normalized = dict(params)
        interval = normalized.get("interval", "")
        if interval == "1h":
            normalized["interval"] = "60m"
            interval = "60m"
        if interval in self._intraday_intervals:
            normalized.pop("period", None)
            normalized.pop("start", None)
            normalized.pop("end", None)
        # Make it minimal: include only fields that affect bar identity
        keep_fields = {"interval", "useRTH", "whatToShow"}
        normalized = {k: v for k, v in normalized.items() if k in keep_fields}
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

    # --------- IB conversions ---------
    def _interval_to_ib_bar_size(self, interval: str) -> str:
        mapping = {
            "1m": "1 min",
            "2m": "2 mins",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "60m": "1 hour",
            "90m": "90 mins",
            "1h": "1 hour",
            "1d": "1 day",
            "5d": "1 day",
            "1wk": "1 week",
            "1mo": "1 month",
            "3mo": "1 month",
        }
        return mapping.get(interval, "1 day")

    def _period_to_duration_str(self, period: str, interval: str) -> Optional[str]:
        if period == "max":
            return None  # we'll page
        mapping = {
            "1d": "1 D",
            "5d": "5 D",
            "7d": "7 D",
            "10d": "10 D",
            "14d": "14 D",
            "1mo": "1 M",
            "3mo": "3 M",
            "6mo": "6 M",
            "1y": "1 Y",
            "2y": "2 Y",
            "5y": "5 Y",
            "10y": "10 Y",
        }
        return mapping.get(period)

    # --------- IB fetchers ---------
    def _ib_fetch_chunk(
        self,
        ib: IB,
        contract: Contract,
        end_dt_utc: Optional[pd.Timestamp],
        duration_str: str,
        bar_size: str,
        useRTH: bool = False,
        whatToShow: str = "TRADES",
    ) -> pd.DataFrame:
        bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime="" if end_dt_utc is None else end_dt_utc.tz_localize("UTC").strftime("%Y%m%d %H:%M:%S"),
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=2,
            keepUpToDate=False,
        )
        df = util.df(bars)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return self._normalize_dataframe(df)

    def _ib_paged_intraday(
        self,
        ib: IB,
        contract: Contract,
        interval: str,
        total_days: int,
        useRTH: bool = False,
        whatToShow: str = "TRADES",
        chunk_days: int = 10,
        pause_sec: float = 0.3,
    ) -> pd.DataFrame:
        """
        Page backward by days to emulate 'max' or long lookbacks for intraday bars.
        """
        bar_size = self._interval_to_ib_bar_size(interval)
        end_dt = None  # now
        frames: List[pd.DataFrame] = []
        remaining = total_days
        while remaining > 0:
            days = min(remaining, chunk_days)
            df = self._ib_fetch_chunk(
                ib, contract, end_dt, duration_str=f"{days} D", bar_size=bar_size, useRTH=useRTH, whatToShow=whatToShow
            )
            if df.empty:
                break
            frames.append(df)
            oldest = df.index[0]
            end_dt = oldest - pd.Timedelta(seconds=1)
            remaining -= days
            ib.sleep(pause_sec)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return self._normalize_dataframe(out)

    def _ib_since(
        self,
        ib: IB,
        contract: Contract,
        since_utc: pd.Timestamp,
        interval: str,
        useRTH: bool = False,
        whatToShow: str = "TRADES",
    ) -> pd.DataFrame:
        """
        Fetch bars from since_utc (exclusive) to now, clipping duration to reasonable max for interval.
        """
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        if since_utc >= now_utc:
            return pd.DataFrame()
        max_days = self._max_days_for_interval.get("60m" if interval == "1h" else interval, 30)
        max_duration = pd.Timedelta(days=max_days)
        duration = min(now_utc - since_utc + pd.Timedelta(minutes=2), max_duration)
        # Prefer D over S to stay within IB norms
        total_days = max(1, int(duration.total_seconds() // 86400) + 1)
        bar_size = self._interval_to_ib_bar_size(interval)
        df = self._ib_fetch_chunk(
            ib, contract, end_dt_utc=None, duration_str=f"{total_days} D", bar_size=bar_size, useRTH=useRTH, whatToShow=whatToShow
        )
        # Filter strictly > since_utc
        df = df[df.index > since_utc]
        return self._normalize_dataframe(df)

    # --------- Incremental & public fetch ---------
    def _incremental_update(
        self,
        ib: IB,
        contract: Contract,
        cached_df: pd.DataFrame,
        original_params: dict,
    ) -> pd.DataFrame:
        cached_df = self._normalize_dataframe(cached_df)
        if cached_df.empty:
            fresh = self._normalize_dataframe(self.__safe_fetch(ib, contract, **original_params))
            return fresh

        interval = original_params.get("interval", "1d")
        last_ts = pd.Timestamp(cached_df.index.max())
        if pd.isna(last_ts):
            fresh = self._normalize_dataframe(self.__safe_fetch(ib, contract, **original_params))
            return fresh

        last_ts_utc = self._to_naive_utc(last_ts)
        start_ts = last_ts_utc - self._get_overlap(interval)

        df_new = self._ib_since(
            ib,
            contract,
            since_utc=start_ts,
            interval=interval,
            useRTH=original_params.get("useRTH", False),
            whatToShow=original_params.get("whatToShow", "TRADES"),
        )
        if df_new.empty:
            return cached_df
        return self._merge_preserving_history(cached_df, df_new)

    def fetch(self, ib: IB, contract: Contract, ticker_symbol: str, **params) -> pd.DataFrame:
        """
        - Always force-refresh for 1m via incremental merge (preserves older history).
        - For other intervals, TTL + incremental if stale, else return cached.
        - Supports period in {1d,5d,7d,1mo,3mo,6mo,1y,2y,5y,10y,max}.
        """
        interval = params.get("interval", "1d")
        cache_key = self._create_cache_key(ticker_symbol, params)
        entry = self._cache.get(cache_key)
        cached_df = self._normalize_dataframe(entry["data"]) if entry else pd.DataFrame()

        # 1) For 1m, always incremental merge into cache
        if interval == "1m":
            if not cached_df.empty:
                updated = self._incremental_update(ib, contract, cached_df, params)
                added = max(0, len(updated) - len(cached_df))
                print(f"[TwsFetchCache] Forced refresh for {ticker_symbol} {params} -> incremental merge (+{added}, total {len(updated)})")
                self._cache[cache_key] = {"data": updated, "fetched_at": time.time()}
                self._save_cache()
                return updated.copy()
            else:
                fresh = self._normalize_dataframe(self.__safe_fetch(ib, contract, **params))
                print(f"[TwsFetchCache] No cache for {ticker_symbol} {params} -> ({len(fresh)}) rows")
                self._cache[cache_key] = {"data": fresh, "fetched_at": time.time()}
                self._save_cache()
                return fresh.copy()

        # 2) TTL behavior for other intervals
        if not cached_df.empty and entry and not self._needs_update(entry["fetched_at"]):
            print(f"[TwsFetchCache] Cache hit for {ticker_symbol} {params} -> ({len(cached_df)} rows)")
            return cached_df.copy()

        if not cached_df.empty:
            updated = self._incremental_update(ib, contract, cached_df, params)
            added = max(0, len(updated) - len(cached_df))
            print(f"[TwsFetchCache] Cache stale for {ticker_symbol} -> incremental update (+{added}, total {len(updated)})")
            self._cache[cache_key] = {"data": updated, "fetched_at": time.time()}
            self._save_cache()
            return updated.copy()

        fresh = self._normalize_dataframe(self.__safe_fetch(ib, contract, **params))
        print(f"[TwsFetchCache] No cache for {ticker_symbol} {params} -> network fetch ({len(fresh)} rows)")
        self._cache[cache_key] = {"data": fresh, "fetched_at": time.time()}
        self._save_cache()
        return fresh.copy()

    # --------- Core fetch routing ---------
    def __safe_fetch(self, ib: IB, contract: Contract, **params) -> pd.DataFrame:
        """
        Emulates yfinance.history(...) interface:
          params: period, interval, useRTH, whatToShow
        """
        period = params.get("period", "7d")
        interval = params.get("interval", "1m")
        useRTH = params.get("useRTH", False)
        whatToShow = params.get("whatToShow", "TRADES")

        bar_size = self._interval_to_ib_bar_size(interval)
        duration_str = self._period_to_duration_str(period, interval)

        # Intraday "max" or long lookbacks -> paged
        if (duration_str is None) and (interval in self._intraday_intervals):
            days = self._max_days_for_interval.get("60m" if interval == "1h" else interval, 30)
            df = self._ib_paged_intraday(ib, contract, interval=interval, total_days=days, useRTH=useRTH, whatToShow=whatToShow)
            return self._normalize_dataframe(df)

        # Non-intraday "max" -> use long daily duration
        if duration_str is None:
            # daily/weekly/monthly "max"
            duration_str = "10 Y"

        # Single chunk
        df = self._ib_fetch_chunk(
            ib, contract, end_dt_utc=None, duration_str=duration_str, bar_size=bar_size, useRTH=useRTH, whatToShow=whatToShow
        )
        return self._normalize_dataframe(df)