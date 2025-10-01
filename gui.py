from  tws_wrapper.trader import TwsTrader
from tws_wrapper.stock import TwsStock, TwsConnection
from ui.graph import LiveGraph
from tws_wrapper.connection import TwsConnection
from tws_wrapper.stock import TwsStock
from tws_wrapper.updater import IntraMinuteMarketUpdater, HistoricMarketUpdater
from core.market_data_hub import MarketDataHub


class GUI:
    def __init__(self, symbol: str = "RHM", tz_offset_hours: float = 0.0, use_regular_trading_hours=False):
        self._init_tws(symbol)
        self._init_graph(tz_offset_hours, use_regular_trading_hours)
        self._init_updater(use_regular_trading_hours)

    def _init_tws(self, symbol):
        self.connection = TwsConnection()
        self.stock = TwsStock(connection=self.connection, symbol=symbol, market_data_type=3)
        print(self.stock.details)

    def _init_graph(self, tz_offset_hours, use_regular_trading_hours):
        self.view = LiveGraph(title=f"TWS - {self.stock.details.longName}")

        initial_dataframe = self.stock.get_accurate_max_historical_data(useRTH=use_regular_trading_hours)
        self.hub = MarketDataHub(initial_df=initial_dataframe, display_offset_hours=tz_offset_hours)
        self.hub.subscribe_view(self.view)
        # Optional: anyone can subscribe to market data hub
        # hub.subscribe(lambda delta, full: print("delta:", len(delta), "full:", len(full)))

    def _init_updater(self, use_regular_trading_hours):
        def _on_drop_prev_minute(_view, ts):
            self.hub.drop_bar(ts)
        
        self.intra_minute_updater = IntraMinuteMarketUpdater(
            stock=self.stock,
            view=self.hub,
            poll_secs=1.0,
            price_pref=("mid", "last", "bid", "ask"),
            min_change_ticks=0.0,
            verbose=True,
            on_drop_prev_minute=_on_drop_prev_minute
        )

        self.historical_updater = HistoricMarketUpdater(
            stock=self.stock,
            view=self.hub,
            period="max",
            interval="1m",
            useRTH=use_regular_trading_hours,
            poll_secs=60,
            persist_csv_every=1,
            align_to_period=True,
            verbose=True,
            during_wait=lambda dt: self.connection.sleep(dt),  # pump IB while waiting
        )

    def run(self):       
        try:
            self.view.show()
            self.intra_minute_updater.start()
            self.historical_updater.run_forever_in_main_thread()
        except KeyboardInterrupt:
            return
        finally:
            self.intra_minute_updater.stop()

    def stop():
        pass


if __name__ == "__main__":
    symbol = input("Symbol: ")
    gui = GUI(symbol=symbol, tz_offset_hours=+2.0, use_regular_trading_hours=False)
    gui.run()