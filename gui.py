import logging
from typing import Optional, Tuple

from tws_wrapper.connection import TwsConnection
from tws_wrapper.stock import TwsStock
from tws_wrapper.updater import TwsHistoricUpdater, TwsIntraMinuteUpdater
from ui.chart import TradeChart
from core.market_data_hub import MarketDataHub


logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class GUI:
    """
    Simple GUI wrapper to:
      - Connect to TWS
      - Initialize the chart and market data hub
      - Stream live intra-minute updates
      - Backfill/refresh historical data
    """

    def __init__(
        self,
        symbol: str = "RHM",
        tz_offset_hours: float = 0.0,
        use_regular_trading_hours: bool = False,
        price_pref: Tuple[str, ...] = ("mid", "last", "bid", "ask"),
        intraminute_poll_secs: float = 1.0,
        historical_poll_secs: int = 60,
        verbose: bool = True,
    ) -> None:
        self.symbol = symbol
        self.tz_offset_hours = tz_offset_hours
        self.use_regular_trading_hours = use_regular_trading_hours
        self.price_pref = price_pref
        self.intraminute_poll_secs = intraminute_poll_secs
        self.historical_poll_secs = historical_poll_secs
        self.verbose = verbose

        # Will be initialized in the helpers
        self.connection: Optional[TwsConnection] = None
        self.stock: Optional[TwsStock] = None
        self.view: Optional[TradeChart] = None
        self.hub: Optional[MarketDataHub] = None
        self.intra_minute_updater: Optional[TwsIntraMinuteUpdater] = None
        self.historical_updater: Optional[TwsHistoricUpdater] = None

        self._init_tws(self.symbol)
        self._init_graph(self.tz_offset_hours, self.use_regular_trading_hours)
        self._init_updater(self.use_regular_trading_hours)

    def _init_tws(self, symbol: str) -> None:
        """Initialize TWS connection and stock contract."""
        self.connection = TwsConnection()
        self.stock = TwsStock(
            connection=self.connection,
            symbol=symbol,
            market_data_type=3,  # delayed-frozen if needed; keep original behavior
        )
        details = getattr(self.stock, "details", None)
        long_name = getattr(details, "longName", symbol) if details else symbol
        logger.info("Initialized TWS for %s (%s)", long_name, symbol)
        if details:
            logger.debug("Stock details: %s", details)

    def _init_graph(self, tz_offset_hours: float, use_regular_trading_hours: bool) -> None:
        """Build the chart view and seed it with historical data via the hub."""
        assert self.stock is not None, "Stock must be initialized before graph."

        long_name = getattr(getattr(self.stock, "details", None), "longName", self.symbol)
        self.view = TradeChart(title=f"TWS - {long_name}")

        # Seed initial data
        initial_dataframe = self.stock.get_accurate_max_historical_data(
            useRTH=use_regular_trading_hours
        )
        self.hub = MarketDataHub(
            initial_df=initial_dataframe,
            display_offset_hours=tz_offset_hours,
        )
        self.hub.subscribe_view(self.view)
        logger.info("Chart and MarketDataHub initialized (RTH=%s, tz_offset=%.2f).",
                    use_regular_trading_hours, tz_offset_hours)

    def _init_updater(self, use_regular_trading_hours: bool) -> None:
        """Configure live and historical updaters."""
        assert self.connection is not None and self.stock is not None and self.hub is not None

        def _on_drop_prev_minute(_view, ts):
            self.hub.drop_bar(ts)

        self.intra_minute_updater = TwsIntraMinuteUpdater(
            stock=self.stock,
            view=self.hub,
            poll_secs=self.intraminute_poll_secs,
            price_pref=self.price_pref,
            min_change_ticks=0.0,
            verbose=self.verbose,
            on_drop_prev_minute=_on_drop_prev_minute,
        )

        self.historical_updater = TwsHistoricUpdater(
            stock=self.stock,
            view=self.hub,
            period="max",
            interval="1m",
            useRTH=use_regular_trading_hours,
            poll_secs=self.historical_poll_secs,
            persist_csv_every=1,
            align_to_period=True,
            verbose=self.verbose,
            during_wait=lambda dt: self.connection.sleep(dt),  # pump IB while waiting
        )
        logger.info("Updaters initialized (live=%ss, hist=%ss).",
                    self.intraminute_poll_secs, self.historical_poll_secs)

    def run(self) -> None:
        """Start streaming and historical refresh; block in main thread."""
        assert self.view is not None
        assert self.intra_minute_updater is not None
        assert self.historical_updater is not None

        try:
            logger.info("Starting GUI...")
            self.view.show()
            self.intra_minute_updater.start()
            self.historical_updater.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop updaters and close the connection."""
        logger.info("Shutting down...")

        if self.intra_minute_updater is not None:
            try:
                self.intra_minute_updater.stop()
            except Exception as e:
                logger.exception("Error stopping intra-minute updater: %s", e)

        if self.historical_updater is not None:
            try:
                self.historical_updater.stop()
            except Exception as e:
                logger.exception("Error stopping historical updater: %s", e)

        if self.connection is not None:
            try:
                self.connection.disconnect()
            except Exception as e:
                logger.exception("Error closing TWS connection: %s", e)

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        symbol = input("Symbol [default RHM]: ").strip() or "RHM"
    except EOFError:
        symbol = "RHM"

    gui = GUI(symbol=symbol, tz_offset_hours=+2.0, use_regular_trading_hours=False, verbose=False)
    gui.run()