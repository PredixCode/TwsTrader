from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection


# USAGE examples of wrappers and graphs
def tws_stock_stream():
    with TwsConnection(port=7496) as conn:
        feed = TwsStock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        feed.stream_price(duration_sec=60)

def tws_trade():
    with TwsConnection() as conn:
        stock = TwsStock(conn, symbol="RHM")
        trader = TwsTrader(stock)
        # Place a market buy
        trade = trader.buy(10, order_type='MKT', wait=False)
        print("BUY result:", trader.trade_summary(trade))

        # Place a limit sell
        # set limit price relative to current bid/ask
        t = stock.get_ticker()
        limit_px = (t.bid or t.last or 0) + 0.5
        trade2 = trader.sell(10, order_type='LMT', limit_price=limit_px, wait=False)
        print("SELL placed (not waiting):", trader.trade_summary(trade2))

def tws_historical():
    with TwsConnection() as conn:
        feed = TwsStock(conn, symbol="RHM")
        feed.get_accurate_max_historical_data()
        feed.last_fetch_to_csv()


if __name__ == "__main__":
    tws_stock_stream()
    #tws_trade()
    #tws_historical()
    #timezone()