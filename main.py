from datetime import datetime
from ib_insync import util

from connection import TwsConnection
from stock import Stock, MarketDataType
from trader import Trader
from strategy_base import Bar
from strategies.sfi_charlie import SFICharlie, SFICharlieParams


def ib_bar_to_bar(b) -> Bar:
    """
    Convert both Historical BarData (has .date) and RealTimeBar (has .time) to our Bar.
    Handles either datetime or string timestamps.
    """
    # Real-time bars have 'time'; historical bars have 'date'
    if hasattr(b, 'time'):  # RealTimeBar
        t = b.time if isinstance(b.time, datetime) else util.parseIBDatetime(str(b.time))
    else:  # Historical BarData
        # Requesting formatDate=2 (below) ensures this is a datetime already
        t = b.date if isinstance(b.date, datetime) else util.parseIBDatetime(str(b.date))
    vol = getattr(b, 'volume', None)
    return Bar(time=t, open=b.open, high=b.high, low=b.low, close=b.close, volume=vol)


def handle_decision(bar: Bar, decision, trader: Trader):
    if decision.action == 'BUY':
        print(f"{bar.time} BUY at {decision.price} ({decision.reason})")
        trade = trader.buy(quantity=10, order_type='MKT', wait=False)
        print(" -> orderId", trade.order.orderId)
    elif decision.action == 'SELL':
        print(f"{bar.time} SELL at {decision.price} ({decision.reason})")
        trade = trader.sell(quantity=10, order_type='MKT', wait=False)
        print(" -> orderId", trade.order.orderId)
    # else: HOLD; you can inspect decision.extras if needed


if __name__ == "__main__":
    params = SFICharlieParams(periods=10, multiplier=1.7, change_atr=True)

    with TwsConnection(port=7497, client_id=2) as conn:
        stk = Stock(conn, symbol="RHM", exchange="SMART", currency="EUR", primaryExchange="IBIS")
        trd = Trader(stk)
        strat = SFICharlie(params)

        ib = conn.connect()
        contract = stk.qualify()

        # Make sure we know which market data type we ended up with
        # This will subscribe and set stk.market_data_type internally if needed
        ticker = stk.get_ticker()
        md_type = stk.market_data_type

        # 1) Warmup with history (use formatDate=2 to get datetime objects)
        hist = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='5 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=2  # ensures b.date is a datetime; avoids parsing issues (like '... MET')
        )
        warm_bars = [ib_bar_to_bar(b) for b in hist]
        strat.ingest_history(warm_bars)

        # 2) Live stream:
        # - If LIVE: use real-time bars
        # - If DELAYED/DELAYED_FROZEN: use historical streaming (keepUpToDate=True)
        print(f"Running live strategy (marketDataType={md_type})... Ctrl+C to stop.")

        if md_type == MarketDataType.LIVE:
            # Real-time bars (5-second bars only)
            rt_bars = ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=False)

            def on_rt_bar(bars, hasNewBar):
                bar = ib_bar_to_bar(bars[-1])
                decision = strat.on_bar(bar)  # or strat.step(bar) if you prefer base counters
                handle_decision(bar, decision, trd)

            rt_bars.updateEvent += on_rt_bar

        else:
            # Delayed data: use historical streaming (5-minute bars here as example)
            # keepUpToDate=True turns the BarDataList into a stream (new bars appended)
            stream_bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='5 mins',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=2,
                keepUpToDate=True
            )

            def on_hist_bar(bars, hasNewBar):
                # Only act when a new bar is added (hasNewBar True)
                if not hasNewBar:
                    return
                bar = ib_bar_to_bar(bars[-1])
                decision = strat.on_bar(bar)
                handle_decision(bar, decision, trd)

            stream_bars.updateEvent += on_hist_bar

        # Keep the event loop alive
        try:
            while True:
                ib.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping...")