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

    t0 = graph.dataframe.index[-10]
    p0 = float(graph.dataframe.loc[t0]['Close'])
    t1 = graph.dataframe.index[-6]
    p1 = float(graph.dataframe.loc[t1]['Close'])
    t2 = graph.dataframe.index[-1]
    p2 = float(graph.dataframe.loc[t2]['Close'])
    graph.add_trade_label(t0, 'buy', price=p0, text='BUY @ {:.2f}'.format(p0), use_marker=True)
    graph.add_trade_label(t1, 'buy', price=p1, text='BUY @ {:.2f}'.format(p1), use_marker=True)
    graph.add_trade_label(t2, 'sell', price=p2, text='SELL @ {:.2f}'.format(p2), use_marker=True)

    graph.show(block=True)
    graph.stop_auto_update()

if __name__ == "__main__":
    stock = FinanceStock("BTC-EUR")
    print(f"Current price: {stock.live_price}")

    #tws_stock_stream()
    #tws_trade()
    live_graph(stock)
