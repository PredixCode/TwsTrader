# tws_wrapper/stock.py

import os
import math
import time
from enum import IntEnum
from datetime import datetime
from typing import Optional, List

import pandas as pd
from ib_insync import Ticker, Contract, ContractDetails
from ib_insync import Stock as IBStock

from tws_wrapper.cache import TwsCache
from tws_wrapper.connection import TwsConnection


# --------------------------
# TwsStock wrapper
# --------------------------
class TwsStock:
    def __init__(
        self,
        connection: TwsConnection,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "EUR",
        primaryExchange: str | None = None,
        market_data_type: int | None = None,
        cache: Optional[TwsCache] = None,
    ):
        self.symbol = symbol
        self.conn = connection
        if primaryExchange:
            self.contract = IBStock(symbol, exchange, currency, primaryExchange=primaryExchange)
        else:
            self.contract = IBStock(symbol, exchange, currency)
        self._qualified = False
        self.market_data_type: int | None = market_data_type
        self._ticker: Ticker | None = None
        self._min_tick: float | None = None  # cache
        self.cache = cache or TwsCache()
        self.last_fetch: Optional[pd.DataFrame] = None
        self.market_delay_min = 0

    def qualify(self) -> Contract:
        ib = self.conn.connect()
        if not self._qualified:
            try:
                [qualified] = ib.qualifyContracts(self.contract)
                self.contract = qualified
                self._qualified = True
            except ValueError as ve:
                print("[Stock] ERROR: Stock configuration invalid. Can't qualify contract!")
                raise ve
            
        return self.contract

    def get_min_tick(self) -> float:
        """
        Fetch and cache exchange minTick (min price increment) for this contract.
        """
        if self._min_tick is not None:
            return self._min_tick
        ib = self.conn.connect()
        self.qualify()
        cdetails: List[ContractDetails] = ib.reqContractDetails(self.contract)
        if not cdetails:
            self._min_tick = 0.5
        else:
            self._min_tick = float(cdetails[0].minTick or 0.5)
        return self._min_tick
    
    def get_ticker(self, market_data_type: int | None = None) -> Ticker:
        if market_data_type is not None:
            self.market_data_type = market_data_type
        ticker = self._get_ticker()
        if self.market_data_type == 3:
            self.market_delay_min = 15
        else:
            self.market_delay_min = 0
        return ticker    
    
    def get_details(self) -> ContractDetails:
        ib = self.conn.connect()
        contract = self.qualify()
        return ib.reqContractDetails(contract)[0]

    def get_latest_quote(self) -> dict:
        t = self.get_ticker()
        now_minute = pd.Timestamp.utcnow().floor("min")
        now_minute = now_minute - pd.Timedelta(minutes=self.market_delay_min)
        bid = t.bid if t.bid is not None else -1
        ask = t.ask if t.ask is not None else -1
        last = t.last if t.last is not None else -1
        mid = (bid + ask) / 2
        vol = t.volume if t.volume is not None else 0
        return {
            "minute": now_minute,
            "bid": bid,
            "ask": ask,
            "last": last,
            "mid": mid,
            "vol": vol,
        }

    # --------- Historical ---------
    def get_historical_data(
        self,
        period: str = "7d",
        interval: str = "1m",
        useRTH: bool = False,
        whatToShow: str = "TRADES",
    ) -> pd.DataFrame:
        """
        Mirror FinanceStock.get_historical_data, but fetch via IB and cache results.
        Returns a DataFrame with columns: Open, High, Low, Close, Volume
        Index is UTC-naive DatetimeIndex (sorted, deduped).
        """
        ib = self.conn.connect()
        self.qualify()

        df = self.cache.fetch(
            ib,
            self.contract,
            self.symbol,
            period=period,
            interval=interval,
            useRTH=useRTH,
            whatToShow=whatToShow,
        )
        self.last_fetch = df.copy()
        return df

    def get_accurate_max_historical_data(self, useRTH=False) -> pd.DataFrame:
        """
        Fetch 15m, 5m, 2m, then 1m with period='max' and merge, so higher resolution
        overwrites overlapping coarse bars (exactly like the yfinance wrapper).
        """
        
        intervals: List[str] = ["5m", "2m", "1m"]
        print(f"[TwsStock] --- Retrieving accurate market data ({self.symbol} - intervals={intervals}, useRTH={useRTH}) ---")
        frames: List[pd.DataFrame] = []
        for interval in intervals:
            df = self.get_historical_data(period="max", interval=interval, useRTH=useRTH)
            if not df.empty:
                frames.append(df)

        if not frames:
            self.last_fetch = pd.DataFrame()
            return self.last_fetch

        merged = self._merge_historical_data(frames)
        self.last_fetch = merged.copy()
        return merged

    def last_fetch_to_csv(self, last_fetch: Optional[pd.DataFrame] = None) -> Optional[str]:
        if last_fetch is not None:
            self.last_fetch = last_fetch

        if self.last_fetch is None or self.last_fetch.empty:
            print("Last fetch is empty or invalid. Nothing to save.")
            return None

        safe_filename = "".join([c for c in self.symbol if c.isalnum() or c.isspace()]).rstrip()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, "data", "csv")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"{safe_filename}.csv")

        try:
            df = self.last_fetch.copy()
            df.index.name = "Datetime"
            print(f"[TwsStock] Saving last fetch to: {filename}")
            df.to_csv(filename)
            return os.path.abspath(filename)
        except Exception as e:
            print(f"Error saving file: {e}")
            return None
    
    # --------- Helpers ---------
    def stream_price(self, duration_sec: int = 60) -> None:
        print(f"Streaming {self.symbol} for {duration_sec}s (marketDataType={self.market_data_type}).")
        end_time = time.time() + duration_sec
        last_printed = None
        while time.time() < end_time:
            self.conn.sleep(0.2)
            quote = self.get_latest_quote()
            if quote != last_printed:
                ts_local = datetime.now().strftime("%H:%M:%S")
                print(f"{ts_local}: {quote}")
                last_printed = quote
        print(f"Done streaming ({duration_sec}s).")
        
    # --------- Internals ---------
    def _subscribe(self, md_type: int) -> Ticker:
        ib = self.conn.connect()
        self.qualify()
        ib.reqMarketDataType(md_type)
        t = ib.reqMktData(self.contract, "", snapshot=False, regulatorySnapshot=False)
        ib.sleep(1.0)
        return t
        
    def _get_ticker(self) -> Ticker:
        def __is_valid(t: Ticker) -> bool:
            def ok(x):
                return x is not None and not (isinstance(x, float) and math.isnan(x))
            return ok(getattr(t, "last", None)) or ok(getattr(t, "bid", None)) or ok(getattr(t, "ask", None))

        def __find_market_for_ticker() -> Ticker | None:
            print("Discovering available market data types (1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen) to fetch data.")
            for md_name, md_value in zip(["LIVE","DELAYED"], [1,3]):
                ticker = self._subscribe(md_value)
                print(f"Trying market type '{md_name}'...")
                if __is_valid(ticker):
                    self.market_data_type = md_value
                    self._ticker = ticker
                    print(f"Market Type '{md_name}' returned valid data...")
                    return ticker
            return None

        if self._ticker is not None:
            return self._ticker

        if self.market_data_type is not None:
            t = self._subscribe(self.market_data_type)
            if __is_valid(t):
                self._ticker = t
                return t

        ticker = __find_market_for_ticker()
        if ticker is None:
            raise Exception("ERROR: Ticker cannot be found. This means there is no market data available for the requested Stock.")
        return ticker

    def _merge_historical_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        combined_df = pd.concat(dataframes, axis=0)
        combined_df.sort_index(inplace=True)
        merged_df = combined_df[~combined_df.index.duplicated(keep="last")]
        # Ensure standard columns/order for uniformity
        std_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in std_cols:
            if c not in merged_df.columns:
                merged_df[c] = pd.NA
        merged_df = merged_df[std_cols]
        return merged_df