# main.py
from ib_insync import util
util.logToConsole()

from tws_wrapper.connection import TwsConnection
from unified.bot import UnifiedTradingBot, BotConfig, SymbolMapping
from unified.strategy.sfi_charlie import SfiCharlieStrategy, SfiCharlieConfig

if __name__ == "__main__":
    mapping = SymbolMapping(
        yf_symbol="RHM.DE",
        ib_symbol="RHM",
        exchange="SMART",
        currency="EUR",
        primaryExchange="IBIS"
    )

    bot_cfg = BotConfig(
        mapping=mapping,
        quantity=10,
        use_limit_orders=True,
        limit_offset=0.2,
        tif="DAY",
        outsideRTH=False,
        history_period="7d",
        history_interval="1m",
        market_data_type=None,   # auto-discover
        wait_for_fills=False,
        max_position=100,
        session_window=10000,
        price_epsilon=0.0,
        evaluate_on_close=True,  # match TradingView behavior
        tp_exit_all=True
    )

    strat_cfg = SfiCharlieConfig(
        atr_period=10,
        multiplier=1.7,
        use_true_atr=True,
        source="ohlc4",
        tp_atr_length=14,
        tp_multiplier=1.0,
        evaluate_on_close=True
    )
    strategy = SfiCharlieStrategy(strat_cfg)

    with TwsConnection(host="127.0.0.1", port=7497, client_id=2, timeout=5) as conn:
        bot = UnifiedTradingBot(conn, bot_cfg, strategy)
        bot.run_loop()