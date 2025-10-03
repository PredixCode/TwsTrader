# main.py
from gui.app import TradeApp


if __name__ == "__main__":
    try:
        symbol = input("Symbol [default RHM]: ").strip() or "RHM"
    except EOFError:
        symbol = "RHM"

    app = TradeApp(symbol=symbol, tz_offset_hours=+2.0, use_regular_trading_hours=False, verbose=True)
    app.run()