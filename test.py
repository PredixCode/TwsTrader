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

def live_graph():
    from ui.graph import LiveGraph
    from ui.yfinance_update_loop import YFinanceChartUpdater

    # 1) Fetch initial dataset for the chart
    finance = FinanceStock("RHM.DE")
    df_init = finance.get_accurate_max_historical_data()
    if df_init is None or df_init.empty:
        raise SystemExit("No data for RHM.DE")

    # 2) Create the passive chart view
    view = LiveGraph(title="RHM.DE — yfinance", initial_df=df_init)
    view.show(block=False)

    # 3) Start the yfinance updater (aligned to minute boundaries)
    updater = YFinanceChartUpdater(
        stock=finance,
        view=view,
        period="1d",
        interval="1m",
        poll_secs=60,          # 60s for 1m bars
        persist_csv_every=1,  # every 30 updates
        align_to_period=True,
        verbose=True,
    )

    # 4) Show chart; updater runs in background thread
    try:
        updater.run_forever_in_main_thread()
    finally:
        updater.stop()

if __name__ == "__main__":
    #tws_stock_stream()
    #tws_trade()
    live_graph()
