import os
import yfinance as yf
import pandas as pd
from datetime import datetime

try:
    from yfinance_wrapper.util.fetch_queue import FetchQueue
except:
    from yfinance_wrapper.util.fetch_queue import FetchQueue



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
                self.last_fetch
                return float(price)
            else:
                print("Could not retrieve live price. The market might be closed or data is unavailable.")
                return None
        except Exception as e:
            print(f"An error occurred while fetching the live price: {e}")
            return None

    def get_historical_data(self, period: str = "7d", interval: str = "1m", queue: FetchQueue = None) -> pd.DataFrame:
        """
            period (str): Valid periods: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
            interval (str):Valid intervals: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo".
                            Note: 1-minute data is only available for the last 7 days.
        """
        print(f"Fetching historical data for {self.ticker_symbol} | Period: {period}, Interval: {interval}...")
        try:
            if queue is None:
                queue = self.queue
            history = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
            if history.empty:
                print(f"Warning: No data returned for the specified period/interval. "
                        "1-minute data is typically only available for the last 7 days.")
            self.last_fetch = history.copy()
            return history
        except Exception as e:
            print(f"An error occurred while fetching historical data: {e}")
            return pd.DataFrame() # Return an empty DataFrame on error

    def get_historical_data_in_range(self, start_date: str = "2024-01-01", end_date: str = datetime.now().strftime("%Y-%m-%d"), interval: str = "1d"):
        """
            start_date (str): "YYYY-MM-DD".
            end_date (str): "YYYY-MM-DD".
            interval (str): (e.g., "1m", "5m", "1h", "1d", "1wk").
        """
        print(f"Fetching data for {self.ticker_symbol} from {start_date} to {end_date} at {interval} interval...")
        try:
            # TODO: extend queueing and caching with range functionality
            history = self.ticker.history(start=start_date, end=end_date, interval=interval)
            if history.empty:
                print("Warning: No data returned for the specified date range/interval.")
            self.last_fetch = history.copy()
            return history
        except Exception as e:
            print(f"An error occurred while fetching data by range: {e}")
            return pd.DataFrame()

    def get_all_historical_data(self, queue: FetchQueue=None) -> pd.DataFrame:
        print(f"\n--- Starting comprehensive data fetch for {self.name} ---")
        period = "max"
        intervals = ["1m", "2m", "5m", "1h", "1d", "1mo"]
        data_frames = []

        if queue is None:
            queue = self.queue
        for interval in intervals:
            df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
            if not df.empty:
                data_frames.append(df)
        
        print(f"\n--- Merging all fetched data for {self.name} ---")
        all_history = self.__merge_historical_data(data_frames)
        self.last_fetch = all_history.copy()
        return all_history

    def get_all_historical_data_accurate(self, queue: FetchQueue = None) -> pd.DataFrame:
        """
        Fetches high-frequency historical data and resamples it to a consistent
        1-minute interval, creating the ideal dataset for AI training.

        This method prioritizes accuracy by using only granular intervals
        and then creating a uniform timeline from them.

        Returns:
            pd.DataFrame: A clean, 1-minute resampled DataFrame.
        """
        print(f"\n--- Starting ACCURATE data fetch for {self.name} ---")
        
        # 1. Fetch only the specified high-frequency intervals
        intervals_to_fetch = {
            "1m": "max", 
            "2m": "max", 
            "5m": "max",
        }
        data_frames = []

        if queue is None:
            queue = self.queue
            
        for interval, period in intervals_to_fetch.items():
            print(f"   -> Fetching interval: {interval} (for last {period})")
            df = queue.fetch(self.ticker, self.ticker_symbol, period=period, interval=interval)
            if not df.empty:
                data_frames.append(df)
        
        if not data_frames:
            print("   -> No data fetched. Cannot proceed.")
            return pd.DataFrame()

        # 2. Merge the raw data, prioritizing the most granular first
        print("\n--- Merging all fetched data ---")
        merged_data = self.__merge_historical_data(data_frames)
        print(f"   -> Raw merged data has {len(merged_data)} points.")

        # 3. Resample the merged data to a consistent 1-minute ('1T') frequency
        print("\n--- Resampling to a consistent 1-minute timeline ---")
        aggregation_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        resampled_data = merged_data.resample('1min').apply(aggregation_rules)

        # 4. Fill the gaps created by resampling
        #    - Forward-fill prices: If no trade happened, the price remains the same.
        #    - Fill volume with 0: If no trade happened, the volume is zero.
        resampled_data[['Open', 'High', 'Low', 'Close']] = resampled_data[['Open', 'High', 'Low', 'Close']].ffill()
        resampled_data['Volume'] = resampled_data['Volume'].fillna(0)
        resampled_data.dropna(inplace=True) # Drop any remaining NaNs at the start

        print(f"   -> Final accurate dataset has {len(resampled_data)} consistent 1-minute steps.")
        self.last_fetch = resampled_data.copy()
        return resampled_data

    def last_fetch_to_csv(self):
        """
        Saves the last fetched pandas DataFrame to a CSV file.
        Works after calling either get_historical_data() or get_historical_data_by_range().
        """
        if self.last_fetch is None or not isinstance(self.last_fetch, pd.DataFrame) or self.last_fetch.empty:
            print("Last fetch is empty or invalid. Nothing to save.")
            return

        # Sanitize the name to make it a valid filename
        safe_filename = "".join([c for c in self.name if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = f"{file_dir}\data\csv"
        filename = f"{path}\{safe_filename}.csv"
        try:
            self.last_fetch.index.name = 'Datetime'
            self.last_fetch.to_csv(filename)
            full_path = os.path.abspath(filename)
            print(f"✅ Success! Data saved to: {full_path}")
            return full_path
        except Exception as e:
            print(f"❌ Error: Could not save file. Reason: {e}")

    def __merge_historical_data(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """
        It prioritizes data from the DataFrames earlier in the list in case of
        duplicate timestamps. The final result is chronologically sorted.

        Args:
            dataframes (list[pd.DataFrame]): A list of pandas DataFrames to merge.
                                              Place the highest granularity (e.g., 1-minute)
                                              DataFrame first in the list.

        Returns:
            pd.DataFrame: A single, sorted, and deduplicated DataFrame.
        """
        if not dataframes:
            print("Warning: No DataFrames provided to merge.")
            return pd.DataFrame()
        
        combined_df = pd.concat(dataframes)
        combined_df.sort_index(inplace=True)
        is_duplicate = combined_df.index.duplicated(keep='first')
        merged_df = combined_df[~is_duplicate]        
        return merged_df


if __name__ == "__main__":
    #stock = Stock("TSLA")
    #stock.get_all_historical_data()
    #stock.last_fetch_to_csv()

    stock = FinanceStock("RHM.DE")
    print(stock.live_price)
