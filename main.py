# main.py
from ib_insync import util
util.logToConsole()

from tws_wrapper.connection import TwsConnection
from gui import GUI
from core.strategy.sfi_charlie import SfiCharlieStrategy, SfiCharlieConfig

if __name__ == "__main__":
    try:
        symbol = input("Symbol [default RHM]: ").strip() or "RHM"
    except EOFError:
        symbol = "RHM"

    gui = GUI(symbol=symbol, tz_offset_hours=+2.0, use_regular_trading_hours=False, verbose=False)
    gui.run()

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