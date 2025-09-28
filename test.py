import os
import multiprocessing as mp


from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection

from yfinance_wrapper.stock import FinanceStock
from ui.graph import LiveGraph


# USAGE examples of wrappers and graphs
def tws_stock_stream():
    with TwsConnection() as conn:
        feed = TwsStock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        feed.stream_price(duration_sec=60)

def tws_trade():
    with TwsConnection() as conn:
        stock = TwsStock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
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

def live_graph(stock: FinanceStock):
    mp.freeze_support()
    graph = LiveGraph(stock)
    graph.start_auto_update()

    t = graph.dataframe.index[-5]
    p = float(graph.dataframe.loc[t]['Close'])
    graph.add_trade_label(t, 'buy', price=p, text='BUY @ {:.2f}'.format(p), use_marker=False)

    graph.show(block=True)
    graph.stop_auto_update()

if __name__ == "__main__":
    stock = FinanceStock("BTC-EUR")
    print(f"Current price: {stock.live_price}")

    #tws_stock_stream()
    #tws_trade()
    live_graph(stock)
