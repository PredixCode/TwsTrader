import os
import multiprocessing as mp

from yfinance_wrapper.stock import YFinanceStock

from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection


# USAGE examples of wrappers and graphs
def tws_stock_stream():
    with TwsConnection(port=7496) as conn:
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

def tws_historical():
    with TwsConnection() as conn:
        feed = TwsStock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        feed.get_accurate_max_historical_data()
        feed.last_fetch_to_csv()

def timezone():
    import pandas as pd, pytz, datetime as dt
    finance = YFinanceStock("RHM.DE")
    df = finance.get_historical_data(period="1d", interval="1m")  # as you already do
    last_ts_naive = df.index.max()
    utc = pytz.UTC
    berlin = pytz.timezone("Europe/Berlin")

    # Your cache made index UTC-naive – relocalize to UTC, then convert to Berlin
    last_ts_utc_aware = pd.Timestamp(last_ts_naive).tz_localize("UTC")
    last_ts_berlin = last_ts_utc_aware.tz_convert(berlin)

    now_utc = pd.Timestamp.now(tz="UTC")
    now_berlin = now_utc.tz_convert(berlin)

    print("Yahoo last bar UTC      :", last_ts_utc_aware)
    print("Yahoo last bar Berlin   :", last_ts_berlin)
    print("Now UTC / Berlin        :", now_utc, "/", now_berlin)
    print("Delta (Berlin, minutes) :", (now_berlin - last_ts_berlin).total_seconds()/60.0)

def live_graph_with_tws_provisional(source='tws', tz_offset_hours: float = +2.0):
    from yfinance_wrapper.stock import YFinanceStock
    from ui.graph import LiveGraph
    from ui.yfinance_update_loop import YFinanceChartUpdater
    from tws_wrapper.connection import TwsConnection
    from tws_wrapper.stock import TwsStock
    from ui.tws_update_loop import TwsProvisionalCandleUpdater, IBHistoryChartUpdater
    from ui.adapter import HubViewAdapter
    from core.market_data_hub import MarketDataHub

    """
    source:
    - 'tws'      -> IB-only history + TWS provisional current bar
    - 'yfinance' -> Yahoo history/backfill + TWS provisional current bar

    Both modes:
    - IB is connected on the main thread first.
    - TwsProvisionalCandleUpdater runs in a background thread and updates ONLY the current minute;
        on minute rollover it removes the provisional bar; the history updater (YF or IB) later
        fills the closed minute with authoritative data.
    """
    # 1) IB connect + subscribe on MAIN THREAD
    conn = TwsConnection()
    ib = conn.connect()

    ib_stock = TwsStock(connection=conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS", market_data_type=3)
    ib_stock.get_ticker()  # subscribe once on main thread
    c = ib_stock.qualify()
    details = ib.reqContractDetails(c)[0]
    print(f"Contract details of {ib_stock.symbol}")
    print("timeZone:", details.timeZoneId)
    print("tradingHours:", details.tradingHours)
    print("liquidHours:", details.liquidHours)

    # 2) Choose data source + initial history
    yf_stock = YFinanceStock("RHM.DE")  # used for name and YF mode
    stock_name = yf_stock.name.replace(" ","")
    if source == "tws":
        # IB history (minute bars up to ~29/30d); uses TwsFetchCache
        df_init = ib_stock.get_accurate_max_historical_data()
        title = f"IB - {stock_name}"
    elif source == "yfinance":
        # YF accurate max (5m→2m→1m merge); YF cache applies
        df_init = yf_stock.get_accurate_max_historical_data()
        title = f"Yahoo - {stock_name}"
    else:
        raise ValueError("source must be 'tws' or 'yfinance'.")
    
    # 3) Build hub + view + fanout
    hub = MarketDataHub(initial_df=df_init, display_offset_hours=tz_offset_hours)
    view = LiveGraph(title=title, initial_df=df_init)
    hub.subscribe_view(view)
    # Optional: function subscriber
    # hub.subscribe(lambda delta, full: print("delta:", len(delta), "full:", len(full)))

    # Use adapter so all updaters write to the hub
    hub_view = HubViewAdapter(hub)

    # 4) Start TWS provisional current-minute updater (background)
    def _on_drop_prev_minute(_view, ts):
        hub.drop_bar(ts)

    tws_candles = TwsProvisionalCandleUpdater(
        stock=ib_stock,
        view=hub_view,                     # fanout writes to hub + chart
        poll_secs=1.0,
        price_pref=("mid", "last", "bid", "ask"),
        min_change_ticks=0.0,
        verbose=True,
        on_drop_prev_minute=_on_drop_prev_minute
    )
    tws_candles.prepare()
    tws_candles.start()

    # Show chart
    view.show(block=False)

    # 4) Run the selected history updater on MAIN THREAD, pumping IB during waits
    try:
        if source == "yfinance":
            yf_updater = YFinanceChartUpdater(
                stock=yf_stock,
                view=hub_view,
                period="1d",
                interval="1m",
                whatToShow="MIDPOINT",
                useRTH=False,
                poll_secs=60,
                persist_csv_every=1,
                align_to_period=True,
                verbose=True,
                during_wait=lambda dt: ib.sleep(dt),  # pump IB while waiting
            )
            yf_updater.run_forever_in_main_thread()
        else:
            ib_updater = IBHistoryChartUpdater(
                stock=ib_stock,
                view=hub_view,
                period="max",
                interval="1m",
                poll_secs=60,
                persist_csv_every=1,
                align_to_period=True,
                verbose=True,
                during_wait=lambda dt: ib.sleep(dt),  # pump IB while waiting
            )
            ib_updater.run_forever_in_main_thread()
    finally:
        tws_candles.stop()

if __name__ == "__main__":
    #tws_stock_stream()
    #tws_trade()
    #tws_historical()
    #timezone()
    live_graph_with_tws_provisional("tws")
