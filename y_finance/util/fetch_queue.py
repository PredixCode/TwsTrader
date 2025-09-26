import os
import yfinance as yf
import pandas as pd
import time
import random
import pickle


class FetchQueue:
    """
    Manages and optimizes data fetching with persistent, on-disk caching.
    """
    def __init__(self, cache_file="market/util/fetch_cache.pkl"):
        """
        Initializes the queue and loads the cache from disk if it exists.
        """
        self.cache_file = cache_file
        self._cache = {}
        self._load_cache() # Load the cache from the file upon initialization

    def _load_cache(self):
        """Loads the cache dictionary from a pickle file."""
        if os.path.exists(self.cache_file):
            print(f"Found existing cache file: '{self.cache_file}'. Loading...")
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"✅ Success! Loaded {len(self._cache)} items from the cache.")
            except Exception as e:
                print(f"❌ Warning: Could not load cache file. It might be corrupted. Starting fresh. Error: {e}")
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
        """Creates a unique, order-independent key for a request."""
        return (ticker,) + tuple(sorted(params.items()))

    def fetch(self, ticker: yf.Ticker, ticker_symbol: str, **params) -> pd.DataFrame:
        """
        Fetches data, using the cache first. If a web request is made,
        the result is saved to the on-disk cache.
        """
        cache_key = self._create_cache_key(ticker_symbol, params)

        if cache_key in self._cache:
            print(f"CACHE HIT: Found data for {ticker_symbol} with params {params} in cache.")
            return self._cache[cache_key].copy()

        print(f"CACHE MISS: Fetching new data for {ticker_symbol} with params {params}...")
        
        delay = random.uniform(0.6, 1.4)
        print(f"   -> Throttling: Waiting for {delay:.2f} seconds...")
        time.sleep(delay)

        try:
            history = ticker.history(**params)            
            if history.empty:
                print(f"   -> Warning: yfinance returned no data for this request.")

            self._cache[cache_key] = history
            print(f"   -> Success! Data fetched and added to cache.")
            self._save_cache() # Save the updated cache to disk!
            return history.copy()

        except Exception as e:
            print(f"   -> ❌ An error occurred during fetch for {ticker}: {e}")
            return pd.DataFrame()