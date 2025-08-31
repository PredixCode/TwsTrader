import math
import time
from datetime import datetime
from ib_insync import Ticker, Contract
from ib_insync import Stock as IBStock

from connection import TwsConnection


class StockFeed:
    """
    Ib_insync wrapper for a specified Stock.
    """
    def __init__(
        self,
        connection: TwsConnection,
        symbol: str,
        exchange: str = 'SMART',
        currency: str = 'EUR',
        primaryExchange: str = None,
        market_data_type: int = None):

        # --- # 
        self.symbol = symbol
        self.conn = connection
        # Build IB contract, qualify later
        if primaryExchange:
            self.contract = IBStock(symbol, exchange, currency, primaryExchange=primaryExchange)
        else:
            self.contract = IBStock(symbol, exchange, currency)

        self._qualified = False
        self.market_data_type: int|None = market_data_type  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen, None=Auto-Discover
        self._ticker: Ticker|None = None

    def _qualify(self) -> Contract:
        ib = self.conn.connect()
        if not self._qualified:
            [qualified] = ib.qualifyContracts(self.contract)
            self.contract = qualified
            self._qualified = True
        return self.contract


    def _subscribe(self, md_type: int) -> Ticker:
        ib = self.conn.connect()
        ib.reqMarketDataType(md_type)  # 1 live, 3 delayed-streaming, 4 delayed-frozen, 2 frozen
        try:
            ib.cancelMktData(self.contract)
        except Exception:
            pass
        t = ib.reqMktData(self.contract, '', snapshot=False, regulatorySnapshot=False)
        ib.sleep(1.5)
        return t


    def get_ticker(self, market_data_type: int|None = None) -> Ticker:
        def __is_valid(t: Ticker) -> bool:
            def ok(x):
                return x is not None and not (isinstance(x, float) and math.isnan(x))
            return ok(getattr(t, 'last', None)) or ok(getattr(t, 'bid', None)) or ok(getattr(t, 'ask', None))
        
        def __find__md_for_ticker():
            print("Discovering available market data types (1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen) to fetch data.")
            for md in (1, 3, 4, 2):
                t = self._subscribe(md)
                print(f"Trying market type '{md}'...")
                if __is_valid(t):
                    self.market_data_type = md
                    self._ticker = t
                    print(f"Market Type '{md}' returned valid data...")
                    return t
        
        if self._ticker is not None:
            return self._ticker
        
        if self.market_data_type is not None:
            t = self._subscribe(self.market_data_type)
            if __is_valid(t):
                self._ticker = t
                return t
        
        if market_data_type is not None:
            t = self._subscribe(self.market_data_type)
            if __is_valid(t):
                self._ticker = t
                return t
            
        return __find__md_for_ticker()

    def stream_price(self, duration_sec: int = 60) -> None:
        ticker = self.get_ticker()
        print(f"Streaming {self.symbol} for {duration_sec}s (marketDataType={self.market_data_type}).")
        print("time (local) | last  bid/ask  mid  lastSize  bidSize/askSize")

        end_time = time.time() + duration_sec
        last_printed = None

        while time.time() < end_time:
            self.conn.sleep(0.2)  # yield to event loop

            last = ticker.last if ticker.last is not None else float('nan')
            bid = ticker.bid if ticker.bid is not None else float('nan')
            ask = ticker.ask if ticker.ask is not None else float('nan')
            mid = (bid + ask) / 2 if not math.isnan(bid) and not math.isnan(ask) else float('nan')
            last_size = ticker.lastSize or 0
            bid_size = ticker.bidSize or 0
            ask_size = ticker.askSize or 0

            snapshot = (last, bid, ask, mid, last_size, bid_size, ask_size)
            if snapshot != last_printed:
                ts_local = datetime.now().strftime("%H:%M:%S")
                print(f"{ts_local} | {last:.4f}  {bid:.4f}/{ask:.4f}  {mid:.4f}  {last_size}  {bid_size}/{ask_size}")
                last_printed = snapshot
        print("Done.")


if __name__ == "__main__":
    # Example usage
    with TwsConnection(port=7497, client_id=1) as conn:
        feed = StockFeed(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        feed.stream_price(duration_sec=60)