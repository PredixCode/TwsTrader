import os
import multiprocessing as mp

from yfinance_wrapper.stock import FinanceStock

from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection


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

def live_graph_with_tws():
    from yfinance_wrapper.stock import FinanceStock
    from ui.graph import LiveGraph
    from ui.yfinance_update_loop import YFinanceChartUpdater
    from tws_wrapper.connection import TwsConnection
    from tws_wrapper.stock import TwsStock
    from ui.tws_update_loop import TwsPriceUpdateLoop

    # 1) Chart + initial data
    finance = FinanceStock("RHM.DE")
    df_init = finance.get_accurate_max_historical_data()
    view = LiveGraph(title="RHM.DE — yfinance", initial_df=df_init)
    view.show(block=False)

    # 2) IB connect + subscription on MAIN THREAD
    conn = TwsConnection()
    ib = conn.connect()  # owns the ib_insync loop on main thread
    # If you only have delayed data, you can force: ib.reqMarketDataType(3)

    ib_stock = TwsStock(connection=conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
    tws_loop = TwsPriceUpdateLoop(stock=ib_stock, poll_secs=1.0, force_print_secs=30.0, verbose=True)
    tws_loop.prepare()  # connects + subscribes on main thread
    tws_loop.start()    # background printer thread (just reads Ticker fields)

    # 3) YF updater on main thread; pass ib.sleep to keep IB pumped
    yf_updater = YFinanceChartUpdater(
        stock=finance,
        view=view,
        period="1d",
        interval="1m",
        poll_secs=60,
        persist_csv_every=10,
        align_to_period=True,
        verbose=True,
        during_wait=lambda dt: ib.sleep(dt),  # <-- critical: pumps IB loop
    )

    try:
        yf_updater.run_forever_in_main_thread()
    finally:
        tws_loop.stop()

if __name__ == "__main__":
    #tws_stock_stream()
    #tws_trade()
    live_graph_with_tws()
