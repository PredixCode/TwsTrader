import os
from typing import Optional, List

import yfinance as yf
import pandas as pd

from yfinance_wrapper.fetch_cache import FetchCache


class FinanceStock:
    def __init__(self, ticker_symbol: str, queue: Optional[FetchCache] = None):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.queue = queue or FetchCache()
        self.last_fetch: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        # Best-effort short name; fallback to symbol
        try:
            info = self.ticker.info or {}
            return info.get("shortName") or self.ticker_symbol
        except Exception:
            return self.ticker_symbol

    @property
    def live_price(self) -> float | None:
        try:
            live_info = self.ticker.fast_info
            price = live_info.get("last_price")
            return float(price) if price is not None else None
        except Exception:
            return None

    def get_historical_data(
        self,
        period: str = "7d",
        interval: str = "1m",
    ) -> pd.DataFrame:
        df = self.queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
        self.last_fetch = df.copy()
        return df

    def get_accurate_max_historical_data(self) -> pd.DataFrame:
        """
        Fetch 5m, 2m, then 1m (all with period='max') and merge so higher resolution
        overwrites overlapping coarse bars. 1m:max always reloads.
        """
        print(f"[FinanceStock] --- Retrieving accurate market data ({self.name} -interval=['5m', '2m', '1m'])---")
        intervals: List[str] = ["5m", "2m", "1m"]
        frames: List[pd.DataFrame] = []
        for interval in intervals:
            df = self.get_historical_data(period="max", interval=interval)
            if not df.empty:
                frames.append(df)

        if not frames:
            self.last_fetch = pd.DataFrame()
            return self.last_fetch

        merged = self.__merge_historical_data(frames)
        self.last_fetch = merged.copy()
        return merged

    def last_fetch_to_csv(self, last_fetch: Optional[pd.DataFrame] = None) -> Optional[str]:
        if last_fetch is not None:
            self.last_fetch = last_fetch

        if self.last_fetch is None or self.last_fetch.empty:
            print("Last fetch is empty or invalid. Nothing to save.")
            return None

        safe_filename = "".join([c for c in self.name if c.isalnum() or c.isspace()]).rstrip()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, "data", "csv")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{safe_filename}.csv")

        try:
            df = self.last_fetch.copy()
            df.index.name = "Datetime"
            print(f"[FinanceStock] Saving last fetch to: {filename}")
            df.to_csv(filename)
            return os.path.abspath(filename)
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def __merge_historical_data(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        combined_df = pd.concat(dataframes, axis=0)
        combined_df.sort_index(inplace=True)
        # keep last occurrence so finer intervals (added later) overwrite coarse bars
        merged_df = combined_df[~combined_df.index.duplicated(keep="last")]
        return merged_df