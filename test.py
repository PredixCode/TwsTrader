import os
import multiprocessing as mp


from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection

from yfinance_wrapper.stock import FinanceStock
from yfinance_wrapper.graph import  WebGraph, LightweightGraph


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

def web_graph(stock: FinanceStock):
    data = stock.get_all_historical_data()
    if data.empty:
        raise SystemExit("Cannot create visualization, no data was fetched.")
    # --- 3. Save Data to CSV ---
    path_to_csv = stock.last_fetch_to_csv()
    # --- 4. Visualize the Data ---
    visualizer = WebGraph(csv_file_path=path_to_csv)
    visualizer.plot_candlestick()

def local_graph(stock: FinanceStock):
    mp.freeze_support()
    graph = LightweightGraph(stock)
    graph.show(block=True)    # or False + input("Press Enter to exit...")

if __name__ == "__main__":
    stock = FinanceStock("RHM.DE")

    #tws_stock_stream()
    #tws_trade()
    #web_graph(stock)
    local_graph(stock)
