from datetime import datetime
from ib_insync import util
util.logToConsole()

from y_finance.stock  import FinanceStock
from tws.connection import TwsConnection
from tws.stock import TwsStock, MarketType
from tws.trader import TwsTrader


if __name__ == "__main__":
    y_finance_stock = FinanceStock("RHM.DE")
    print(y_finance_stock.live_price)

    with TwsConnection(port=7497, client_id=2) as conn:
        tws_stok = TwsStock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        #stk = TwsStock(conn, symbol="AAPL", exchange="SMART", currency="USD", primaryExchange="NASDAQ")
        ib = conn.connect()
        contract = tws_stok.qualify()

        # Make sure we know which market data type we ended up with
        # This will subscribe and set stk.market_data_type internally if needed
        ticker = tws_stok.get_ticker()
        md_type = tws_stok.market_data_type

        # Keep the event loop alive
        try:
            while True:
                ib.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping...")