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

def live_graph_with_tws_provisional(tz_offset_hours: float = +2.0):
    from ui.graph import LiveGraph
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

    ib_stock = TwsStock(connection=conn, symbol="AAPL", currency="USD", market_data_type=3)
    ib_stock.get_ticker()  # subscribe once on main thread
    c = ib_stock.qualify()
    details = ib.reqContractDetails(c)[0]
    print(f"Contract details of {ib_stock.symbol}")
    print("timeZone:", details.timeZoneId)
    print("tradingHours:", details.tradingHours)
    print("liquidHours:", details.liquidHours)
    print(details)

    # 2) Choose data source + initial history
    stock_name = details.longName
    df_init = ib_stock.get_accurate_max_historical_data()
    title = f"TWS - {stock_name}"
    
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
        hub_view.drop_bar(ts)

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
    live_graph_with_tws_provisional()
