from ib_insync import util
util.logToConsole()

from tws.connection import TwsConnection
from unified.bot import UnifiedTradingBot, BotConfig, SymbolMapping

if __name__ == "__main__":
    # Map Yahoo symbol to IB contract details
    mapping = SymbolMapping(
        yf_symbol="RHM.DE",
        ib_symbol="RHM",
        exchange="SMART",
        currency="EUR",
        primaryExchange="IBIS"
    )

    cfg = BotConfig(
        mapping=mapping,
        quantity=10,
        use_limit_orders=True,
        limit_offset=0.2,       # e.g., 0.2 EUR above bid for BUY / below ask for SELL
        tif="DAY",
        outsideRTH=False,
        history_period="7d",
        history_interval="1m",
        market_data_type=None,   # None => auto-discover via your TwsStock logic
        wait_for_fills=False,
        max_position=10
    )

    with TwsConnection(host="127.0.0.1", port=7497, client_id=2, timeout=5) as conn:
        bot = UnifiedTradingBot(conn, cfg)

        # Single cycle
        bot.run_once()

        # Or run continuously
        # bot.run_loop(poll_seconds=1)