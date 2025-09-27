# unified/bot.py
from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd

from y_finance.stock import FinanceStock
from tws.stock import TwsStock
from tws.trader import TwsTrader
from tws.connection import TwsConnection


@dataclass
class SymbolMapping:
    yf_symbol: str           # e.g. "RHM.DE"
    ib_symbol: str           # e.g. "RHM"
    exchange: str = "SMART"  # IB exchange router
    currency: str = "USD"    # "EUR" for European listings, etc.
    primaryExchange: Optional[str] = None  # e.g. "IBIS", "NASDAQ"


@dataclass
class BotConfig:
    mapping: SymbolMapping
    quantity: int = 1
    use_limit_orders: bool = False
    limit_offset: float = 0.0           # absolute offset for limits (e.g., 0.2 EUR/USD)
    tif: str = "DAY"
    outsideRTH: bool = False
    history_period: str = "7d"          # yfinance period
    history_interval: str = "1m"        # yfinance interval
    market_data_type: Optional[int] = None  # IB Market Data Type (None = auto discover)
    account: Optional[str] = None
    wait_for_fills: bool = False
    max_position: int = 100             # simple risk cap


class UnifiedTradingBot:
    def __init__(self, conn: TwsConnection, config: BotConfig):
        self.conn = conn
        self.cfg = config

        # Yahoo Finance side
        self.finance = FinanceStock(config.mapping.yf_symbol)

        # IB side
        self.stock = TwsStock(
            connection=conn,
            symbol=config.mapping.ib_symbol,
            exchange=config.mapping.exchange,
            currency=config.mapping.currency,
            primaryExchange=config.mapping.primaryExchange,
            market_data_type=config.market_data_type
        )
        self.trader = TwsTrader(self.stock)

    # --- Data ---
    def fetch_history(self) -> pd.DataFrame:
        """
        Fetch historical data from yfinance using your FinanceStock wrapper.
        """
        df = self.finance.get_historical_data(
            period=self.cfg.history_period,
            interval=self.cfg.history_interval
        )
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple indicators (SMA crossover) on historical data.
        """
        if df is None or df.empty:
            return df
        out = df.copy()
        price_col = "Close" if "Close" in out.columns else "close"
        out["sma_fast"] = out[price_col].rolling(20, min_periods=5).mean()
        out["sma_slow"] = out[price_col].rolling(50, min_periods=10).mean()
        return out

    # --- Position ---
    def get_position(self) -> int:
        """
        Get current IB position for this contract (shares).
        """
        ib = self.conn.connect()
        pos = 0
        for p in ib.positions():
            if p.contract.conId == self.stock.contract.conId:
                pos = int(p.position)
                break
        return pos

    # --- Strategy ---
    def decide(self, df: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
        """
        Very simple SMA crossover decision:
        - BUY if fast > slow
        - SELL if fast < slow
        - HOLD otherwise
        """
        if df is None or df.empty:
            return "HOLD"
        last = df.iloc[-1]
        if pd.isna(last.get("sma_fast")) or pd.isna(last.get("sma_slow")):
            return "HOLD"
        if last["sma_fast"] > last["sma_slow"]:
            return "BUY"
        elif last["sma_fast"] < last["sma_slow"]:
            return "SELL"
        return "HOLD"

    # --- Trading ---
    def place_trade(self, side: Literal["BUY", "SELL"]) -> None:
        qty = self.cfg.quantity
        if qty <= 0:
            print("Quantity <= 0. Skipping.")
            return

        if self.cfg.use_limit_orders:
            # Use latest bid/ask/last to place a limit slightly favorable by limit_offset
            t = self.stock.get_ticker()
            ref = (t.bid if side == "BUY" else t.ask)
            if ref is None:
                ref = t.last or 0
            px = ref + self.cfg.limit_offset if side == "BUY" else ref - self.cfg.limit_offset
            px = float(px)
            if side == "BUY":
                trade = self.trader.buy(
                    qty, order_type="LMT", limit_price=px,
                    tif=self.cfg.tif, outsideRTH=self.cfg.outsideRTH,
                    account=self.cfg.account, wait=self.cfg.wait_for_fills
                )
            else:
                trade = self.trader.sell(
                    qty, order_type="LMT", limit_price=px,
                    tif=self.cfg.tif, outsideRTH=self.cfg.outsideRTH,
                    account=self.cfg.account, wait=self.cfg.wait_for_fills
                )
        else:
            if side == "BUY":
                trade = self.trader.buy(
                    qty, order_type="MKT",
                    tif=self.cfg.tif, outsideRTH=self.cfg.outsideRTH,
                    account=self.cfg.account, wait=self.cfg.wait_for_fills
                )
            else:
                trade = self.trader.sell(
                    qty, order_type="MKT",
                    tif=self.cfg.tif, outsideRTH=self.cfg.outsideRTH,
                    account=self.cfg.account, wait=self.cfg.wait_for_fills
                )
        print(f"Placed {side} order:", TwsTrader.trade_summary(trade))

    # --- Orchestration ---
    def run_once(self) -> None:
        """
        One full cycle: ensure market data, fetch history, compute indicators,
        decide, risk-check, then place a trade if needed.
        """
        # Ensure IB contract is qualified and market data type discovered
        self.stock.get_ticker()

        hist = self.fetch_history()
        hist = self.compute_indicators(hist)
        decision = self.decide(hist)
        pos = self.get_position()
        print(f"Decision: {decision} | Current position: {pos}")

        # Simple absolute position cap
        if decision == "BUY" and pos + self.cfg.quantity > self.cfg.max_position:
            print("Risk cap reached. Skipping BUY.")
            return
        if decision == "SELL" and -pos + self.cfg.quantity > self.cfg.max_position:
            print("Risk cap reached. Skipping SELL.")
            return

        if decision in ("BUY", "SELL"):
            self.place_trade(decision)

    def run_loop(self, poll_seconds: int = 1):
        """
        Keep polling every poll_seconds seconds.
        """
        ib = self.conn.connect()
        print("Starting unified bot loop. Ctrl+C to stop.")
        try:
            while True:
                self.run_once()
                ib.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("Stopped by user.")