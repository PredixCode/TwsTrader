import os
import yfinance as yf
import pandas as pd
from datetime import datetime

try:
    from yfinance_wrapper.fetch_queue import FetchQueue
except:
    from yfinance_wrapper.fetch_queue import FetchQueue


class FinanceStock:
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.queue = FetchQueue()
        self.last_fetch = None

        if not self.info:
            raise ValueError(f"Could not find data for ticker '{ticker_symbol}'. ")

    @property
    def info(self) -> dict:
        return self.ticker.info

    @property
    def name(self) -> str:
        return self.info.get('shortName')

    @property
    def symbol(self) -> str:
        return self.ticker_symbol

    @property
    def live_price(self) -> float | None:
        try:
            live_info = self.ticker.fast_info
            price = live_info.get('last_price')
            if price:
                return float(price)
            else:
                print("Could not retrieve live price. The market might be closed or data is unavailable.")
                return None
        except Exception as e:
            print(f"An error occurred while fetching the live price: {e}")
            return None

    def get_historical_data(self, period: str = "7d", interval: str = "1m", queue: FetchQueue = None) -> pd.DataFrame:
        print(f"Fetching historical data for {self.ticker_symbol} | Period: {period}, Interval: {interval}...")
        try:
            if queue is None:
                queue = self.queue
            history = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
            if history.empty:
                print("Warning: No data returned for the specified period/interval. "
                      "1-minute data is typically only available for the last 7 days.")
            self.last_fetch = history.copy()
            return history
        except Exception as e:
            print(f"An error occurred while fetching historical data: {e}")
            return pd.DataFrame()
        
    def get_max_historical_data(self, intervals: list[str]=["1m", "2m", "5m", "1h", "1d", "1mo"], queue: FetchQueue = None) -> pd.DataFrame:
        print(f"\n--- Starting comprehensive data fetch for {self.name} ---")
        if queue is None:
            queue = self.queue

        period = "max"
        now_ts = self._now_utc_naive()

        # 1) Always probe the latest 1m tail and only append if changed
        #    - Snapshot cached 1m before
        cached_1m_before = queue.get_cached(self.ticker_symbol, period=period, interval="1m")
        df_1m_after = queue.fetch(self.ticker, self.ticker_symbol,
                                  period=period, interval="1m")  # tiny network probe + merge

        changed_1m = False
        if cached_1m_before is None or cached_1m_before.empty:
            changed_1m = True if df_1m_after is not None and not df_1m_after.empty else False
        else:
            prev_last = pd.Timestamp(cached_1m_before.index.max())
            new_last = pd.Timestamp(df_1m_after.index.max()) if df_1m_after is not None and not df_1m_after.empty else prev_last
            if new_last > prev_last:
                changed_1m = True
            else:
                # same timestamp — check if the close changed (data backfill)
                prev_close = float(cached_1m_before.iloc[-1]["Close"]) if "Close" in cached_1m_before.columns else None
                new_close = float(df_1m_after.iloc[-1]["Close"]) if df_1m_after is not None and "Close" in df_1m_after.columns else None
                changed_1m = (new_close is not None and prev_close is not None and new_close != prev_close)

        data_frames = []

        # 2) If no change at 1m, return cached for all intervals (no network hits)
        if not changed_1m:
            print("No change detected in last 1m bar. Returning cached data for all intervals.")
            for interval in intervals:
                cached_df = queue.get_cached(self.ticker_symbol, period=period, interval=interval)
                # If not cached (first run), fetch once
                if cached_df is None:
                    df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
                else:
                    df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval, allow_stale=True)
                if df is not None and not df.empty:
                    data_frames.append(df)
            all_history = self.__merge_historical_data(data_frames)
            self.last_fetch = all_history.copy()
            return all_history

        # 3) 1m changed: include the updated 1m
        if df_1m_after is not None and not df_1m_after.empty:
            data_frames.append(df_1m_after)

        # 4) For other intervals, update only if their next completed bar boundary has passed.
        for interval in ["2m", "5m", "1h", "1d", "1mo"]:
            cached_df = queue.get_cached(self.ticker_symbol, period=period, interval=interval)
            if self._should_update_interval(interval, cached_df, now_ts):
                # minimally refresh using a small tail window
                tail = queue._tail_period_by_interval.get(interval, "7d")
                df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval, force_tail=tail)
            else:
                # keep cached without network
                if cached_df is None:
                    df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
                else:
                    df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval, allow_stale=True)

            if df is not None and not df.empty:
                data_frames.append(df)

        print(f"\n--- Merging all fetched data for {self.name} ---")
        all_history = self.__merge_historical_data(data_frames)
        self.last_fetch = all_history.copy()
        return all_history

    def last_fetch_to_csv(self, last_fetch: pd.DataFrame = None):
        if last_fetch is None:
            if self.last_fetch is None or not isinstance(self.last_fetch, pd.DataFrame) or self.last_fetch.empty:
                print("Last fetch is empty or invalid. Nothing to save.")
                return
        else:
            self.last_fetch = last_fetch

        safe_filename = "".join([c for c in self.name if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, "data", "csv")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{safe_filename}.csv")
        try:
            self.last_fetch.index.name = 'Datetime'
            self.last_fetch.to_csv(filename)
            full_path = os.path.abspath(filename)
            print(f"✅ Success! Data saved to: {full_path}")
            return full_path
        except Exception as e:
            print(f"❌ Error: Could not save file. Reason: {e}")

    def _now_utc_naive(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _last_completed_label(self, now_ts: pd.Timestamp, interval: str) -> pd.Timestamp:
        """
        Returns label (start time) of the last completed bar for a given interval.
        Assumes timestamps are bar-start labels as typical for yfinance.
        """
        # Minute-based
        if interval == "1m":
            return (now_ts.floor("T") - pd.Timedelta(minutes=1))
        if interval == "2m":
            base = now_ts.floor("2T")
            return base - pd.Timedelta(minutes=2)
        if interval == "5m":
            base = now_ts.floor("5T")
            return base - pd.Timedelta(minutes=5)
        if interval in ("60m", "90m", "1h"):
            base = now_ts.floor("H")
            if interval in ("60m", "1h"):
                return base - pd.Timedelta(hours=1)
            return base - pd.Timedelta(minutes=90)
        # Day/Week/Month
        if interval == "1d":
            base = now_ts.floor("D")
            return base - pd.Timedelta(days=1)
        if interval == "1wk":
            base = now_ts.floor("W-MON")  # week label; adjust as needed
            return base - pd.Timedelta(weeks=1)
        if interval == "1mo":
            # previous month start
            mstart = pd.Timestamp(now_ts.year, now_ts.month, 1)
            # last completed month starts at previous month’s first day
            prev_m = (mstart - pd.Timedelta(days=1)).replace(day=1)
            return prev_m
        # default conservative
        return now_ts - pd.Timedelta(minutes=1)

    def _should_update_interval(self, interval: str, cached_df: pd.DataFrame | None, now_ts: pd.Timestamp) -> bool:
        if cached_df is None or cached_df.empty:
            return True
        last_cached = pd.Timestamp(cached_df.index.max())
        last_completed = self._last_completed_label(now_ts, interval)
        # Update only if we can add at least one fully completed new bar
        return last_cached < last_completed

    def __merge_historical_data(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            print("Warning: No DataFrames provided to merge.")
            return pd.DataFrame()
        combined_df = pd.concat(dataframes)
        combined_df.sort_index(inplace=True)
        is_duplicate = combined_df.index.duplicated(keep='first')
        merged_df = combined_df[~is_duplicate]
        return merged_df