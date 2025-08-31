import math
import time
from datetime import datetime

from ib_insync import Ticker, Contract
from ib_insync import Stock as IBStock

from connection import TwsConnection


from enum import IntEnum
class MarketDataType(IntEnum):
    LIVE = 1
    DELAYED = 3
    FROZEN = 2
    DELAYED_FROZEN = 4

    @classmethod
    def names(cls):
        return [e.name for e in cls]
    @classmethod
    def values(cls):
        return [e.value for e in cls]
    @classmethod
    def entries(cls):
        return [(e.name, e.value) for e in cls]


class Stock:
    """
    Ib_insync wrapper for a specified Stock, with streaming + basic trading.
    """
    def __init__(
        self,
        connection: TwsConnection,
        symbol: str,
        exchange: str = 'SMART',
        currency: str = 'EUR',
        primaryExchange: str|None = None,
        market_data_type: int|None = None ):

        self.symbol = symbol
        self.conn = connection

        if primaryExchange:
            self.contract = IBStock(symbol, exchange, currency, primaryExchange=primaryExchange)
        else:
            self.contract = IBStock(symbol, exchange, currency)

        self._qualified = False
        self.market_data_type: int|None = market_data_type  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen, None=Auto-Discover
        self._ticker: Ticker|None = None

    def _subscribe(self, md_type: int) -> Ticker:
        ib = self.conn.connect()
        self.qualify()
        ib.reqMarketDataType(md_type)
        try:
            ib.cancelMktData(self.contract)
        except Exception:
            pass
        t = ib.reqMktData(self.contract, '', snapshot=False, regulatorySnapshot=False)
        ib.sleep(1.0)
        return t

    def qualify(self) -> Contract:
        ib = self.conn.connect()
        if not self._qualified:
            [qualified] = ib.qualifyContracts(self.contract)
            self.contract = qualified
            self._qualified = True
        return self.contract

    def get_ticker(self, market_data_type: int|None = None) -> Ticker:
        def __is_valid(t: Ticker) -> bool:
            def ok(x):
                return x is not None and not (isinstance(x, float) and math.isnan(x))
            return ok(getattr(t, 'last', None)) or ok(getattr(t, 'bid', None)) or ok(getattr(t, 'ask', None))

        def __find__market_for_ticker() -> Ticker|None:
            print("Discovering available market data types (1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen) to fetch data.")
            for md_name, md_value in MarketDataType.entries():
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

        if market_data_type is not None:
            t = self._subscribe(market_data_type)
            if __is_valid(t):
                self._ticker = t
                self.market_data_type = market_data_type
                return t

        ticker = __find__market_for_ticker()
        if ticker is None:
            raise Exception("ERROR: Ticker cannot be found. This means there is no market data available for the requested Stock=(Symbol,Exchange,Currency)\n"+
                            "HINT: Either the stock parameters are invalid OR the market data for this stock is not subscribed.")
        return ticker

    def snapshot(self) -> dict:
        t = self.get_ticker()
        bid = t.bid if t.bid is not None else float('nan')
        ask = t.ask if t.ask is not None else float('nan')
        last = t.last if t.last is not None else float('nan')
        mid = (bid + ask)/2 if not math.isnan(bid) or not math.isnan(ask) else float('nan')
        return dict(symbol=self.symbol, bid=bid, ask=ask, last=last, mid=mid,
                    bidSize=t.bidSize or 0, askSize=t.askSize or 0, lastSize=t.lastSize or 0)
    
    def stream_price(self, duration_sec: int = 60) -> None:
        print(f"Streaming {self.symbol} for {duration_sec}s (marketDataType={self.market_data_type}).")

        end_time = time.time() + duration_sec
        last_printed = None
        while time.time() < end_time:
            self.conn.sleep(0.2)  # yield to event loop
            snapshot = self.snapshot()
            if snapshot != last_printed:
                ts_local = datetime.now().strftime("%H:%M:%S")
                print(f"{ts_local}: {snapshot}")
                last_printed = snapshot
        print("Done.")

if __name__ == "__main__":
    # Example usage
    with TwsConnection() as conn:
        feed = Stock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        feed.stream_price(duration_sec=5)