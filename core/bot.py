# unified/bot.py
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import pandas as pd
import math

from yfinance_wrapper.stock import YFinanceStock
from tws_wrapper.stock import TwsStock
from tws_wrapper.trader import TwsTrader
from tws_wrapper.connection import TwsConnection

from core.strategy.base import BaseStrategy, StrategySignal


@dataclass
class SymbolMapping:
    yf_symbol: str           # e.g., "RHM.DE"
    ib_symbol: str           # e.g., "RHM"
    exchange: str = "SMART"
    currency: str = "USD"
    primaryExchange: Optional[str] = None  # e.g., "IBIS", "NASDAQ"


@dataclass
class BotConfig:
    mapping: SymbolMapping
    quantity: int = 1
    use_limit_orders: bool = False
    limit_offset: float = 0.0
    tif: str = "DAY"
    outsideRTH: bool = False
    # History fetched once:
    history_period: str = "7d"
    history_interval: str = "1m"  # keep 1m to match the strategy calc
    market_data_type: Optional[int] = None
    account: Optional[str] = None
    wait_for_fills: bool = False
    max_position: int = 100
    # Session cache control
    session_window: Optional[int] = None  # keep only last N rows if set (e.g., 10_000)
    price_epsilon: float = 0.0            # minimal change to treat as "new price"
    evaluate_on_close: bool = True        # run strategy only on bar close (default)
    tp_exit_all: bool = True              # on TP, close full position instead of 'quantity'


class UnifiedTradingBot:
    """
    Unified bot with pluggable strategy:
    - Fetches yfinance 1m history once (OHLC).
    - Appends minute OHLC bars built from IB snapshot() prices.
    - Polls once per second, acting only when price changes.
    - Runs the strategy on bar close (or intrabar if evaluate_on_close=False).
    """
    def __init__(self, conn: TwsConnection, config: BotConfig, strategy: BaseStrategy):
        self.conn = conn
        self.cfg = config
        self.strategy = strategy

        # Yahoo Finance side
        self.finance = YFinanceStock(config.mapping.yf_symbol)

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

        # Session state
        self.session_df: Optional[pd.DataFrame] = None  # OHLCV (at least OHLC)
        self._history_initialized: bool = False
        self._last_price: Optional[float] = None

        # 1-minute bar builder state
        self._cur_minute: Optional[pd.Timestamp] = None
        self._cur_open: Optional[float] = None
        self._cur_high: Optional[float] = None
        self._cur_low: Optional[float] = None
        self._cur_close: Optional[float] = None

    # ---------- Initialization ----------
    def ensure_initialized(self) -> None:
        # Ensures contract + market data type + minTick cache
        self.stock.get_ticker()
        _ = self.stock.get_min_tick()
        if not self._history_initialized:
            self._init_history()
            self.strategy.warmup(self.session_df)

    def _init_history(self) -> None:
        df = self.finance.get_historical_data(
            period=self.cfg.history_period,
            interval=self.cfg.history_interval
        )
        if df is None or df.empty:
            self.session_df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        else:
            cols = {c.lower(): c for c in df.columns}
            need = ["open", "high", "low", "close"]
            if not all(c in cols for c in need):
                rename = {}
                if "open" in cols:  rename |= {df.columns[df.columns.str.lower() == "open"][0]: "Open"}
                if "high" in cols:  rename |= {df.columns[df.columns.str.lower() == "high"][0]: "High"}
                if "low" in cols:   rename |= {df.columns[df.columns.str.lower() == "low"][0]: "Low"}
                if "close" in cols: rename |= {df.columns[df.columns.str.lower() == "close"][0]: "Close"}
                df = df.rename(columns=rename)
            else:
                df = df.rename(columns={
                    df.columns[df.columns.str.lower() == "open"][0]: "Open",
                    df.columns[df.columns.str.lower() == "high"][0]: "High",
                    df.columns[df.columns.str.lower() == "low"][0]: "Low",
                    df.columns[df.columns.str.lower() == "close"][0]: "Close",
                })
            self.session_df = df[["Open", "High", "Low", "Close"]].copy()

        self._history_initialized = True

        # Seed last price from history but start a fresh live bar
        if self.session_df is not None and not self.session_df.empty:
            last_close = float(self.session_df["Close"].iloc[-1])
            self._last_price = last_close

        # Start fresh on first live tick
        self._cur_minute = None
        self._cur_open = self._cur_high = self._cur_low = self._cur_close = None

    def _safe_price(self, x) -> Optional[float]:
        if x is None: return None
        if isinstance(x, float) and math.isnan(x): return None
        return float(x)

    def _snapshot_price(self) -> Optional[float]:
        snap = self.stock.get_latest_quote()
        price = self._safe_price(snap.get("mid"))
        if price is None:
            price = self._safe_price(snap.get("last"))
        if price is None:
            bid = self._safe_price(snap.get("bid"))
            ask = self._safe_price(snap.get("ask"))
            if bid is not None and ask is not None:
                price = (bid + ask) / 2.0
            elif bid is not None:
                price = bid
            elif ask is not None:
                price = ask
        return price

    def update_with_snapshot(self, wait_until_change: bool = False) -> Tuple[bool, bool]:
        """
        Returns (did_update, bar_closed).
        did_update -> session_df mutated
        bar_closed -> a minute bar was closed and a new one started
        """
        ib = self.conn.connect()
        eps = float(self.cfg.price_epsilon or 0.0)

        def changed(new_p: Optional[float], last_p: Optional[float]) -> bool:
            if new_p is None: return False
            if last_p is None: return True
            return abs(new_p - last_p) > eps

        while True:
            p = self._snapshot_price()

            tz = getattr(self.session_df.index, "tz", None) if self.session_df is not None else None
            now_minute = pd.Timestamp.now(tz=tz).floor("min")

            if changed(p, self._last_price):
                self._last_price = p
                bar_closed = self._update_ohlc_with_price(p)
                return True, bar_closed

            # No price change: still roll minute on boundary so bars close
            if self._cur_minute is None or now_minute > self._cur_minute:
                seed = p if p is not None else self._last_price
                if seed is not None:
                    bar_closed = self._update_ohlc_with_price(seed)
                    return True, bar_closed

            if not wait_until_change:
                return False, False

            ib.sleep(1.0)    

    def _update_ohlc_with_price(self, price: float) -> bool:
        """
        Update current minute bar with new price; start a new bar when minute changes.
        Returns True if a bar was closed (i.e., a new minute started).
        """
        tz = getattr(self.session_df.index, "tz", None) if self.session_df is not None else None
        now = pd.Timestamp.now(tz=tz).floor("min")  # minute key
        bar_closed = False

        if self._cur_minute is None:
            # Start bar
            self._start_new_bar(now, price)
            return False

        if now == self._cur_minute:
            # Update current bar
            self._cur_high = max(self._cur_high, price) if self._cur_high is not None else price
            self._cur_low = min(self._cur_low, price) if self._cur_low is not None else price
            self._cur_close = price
            # Reflect into session_df row
            self._upsert_row(now, self._cur_open, self._cur_high, self._cur_low, self._cur_close)
        else:
            # Previous minute is closed; ensure it is in session_df with final values
            self._upsert_row(self._cur_minute, self._cur_open, self._cur_high, self._cur_low, self._cur_close)
            bar_closed = True
            # Start new bar with current price
            self._start_new_bar(now, price)

        # Optional rolling window
        if self.cfg.session_window and self.session_df is not None and len(self.session_df) > self.cfg.session_window:
            self.session_df = self.session_df.iloc[-self.cfg.session_window :].copy()

        return bar_closed

    def _start_new_bar(self, minute_key: pd.Timestamp, price: float) -> None:
        self._cur_minute = minute_key
        self._cur_open = price
        self._cur_high = price
        self._cur_low = price
        self._cur_close = price
        self._upsert_row(minute_key, price, price, price, price)

    def _upsert_row(self, idx: pd.Timestamp, o: float, h: float, l: float, c: float) -> None:
        if self.session_df is None:
            self.session_df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        # Insert or update the row at index idx
        self.session_df.loc[idx, ["Open", "High", "Low", "Close"]] = [o, h, l, c]
        self.session_df.sort_index(inplace=True)

    # ---------- Trading helpers ----------
    def _round_to_tick(self, px: float) -> float:
        tick = self.stock.get_min_tick()
        # Round to nearest valid multiple
        steps = round(px / tick)
        return round(steps * tick, 6)

    def _has_active_order(self) -> bool:
        ib = self.conn.connect()
        for tr in ib.openTrades():
            if tr.contract.conId == self.stock.contract.conId:
                st = (tr.orderStatus.status or "").lower()
                if st in ("presubmitted", "submitted", "pendingsubmit"):
                    return True
        return False

    def get_position(self) -> int:
        ib = self.conn.connect()
        pos = 0
        for p in ib.positions():
            if p.contract.conId == self.stock.contract.conId:
                pos = int(p.position)
                break
        return pos

    def place_trade(self, side: Literal["BUY", "SELL"], qty: int, reason: str = "") -> None:
        if qty <= 0:
            return

        # Avoid spamming new orders if one is already working
        if self._has_active_order():
            print("Open order exists for this contract; skipping new order.")
            return

        if self.cfg.use_limit_orders:
            t = self.stock.get_ticker()
            ref = (t.bid if side == "BUY" else t.ask)
            if ref is None:
                ref = t.last or 0.0
            px = float(ref + self.cfg.limit_offset) if side == "BUY" else float(ref - self.cfg.limit_offset)
            px = self._round_to_tick(px)
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

    # ---------- Orchestration ----------
    def run_loop(self):
        self.ensure_initialized()
        ib = self.conn.connect()
        print("Starting bot loop (1s polling; acting on price changes, strategy on bar close). Ctrl+C to stop.")

        try:
            while True:
                did_update, bar_closed = self.update_with_snapshot(wait_until_change=True)
                if not did_update:
                    ib.sleep(1.0)
                    continue

                # Log minimally so you can see progress
                print(f"[{pd.Timestamp.now():%H:%M:%S}] Bar {'closed' if bar_closed else 'updated'}. Last price={self._last_price}")

                if self.cfg.evaluate_on_close and not bar_closed:
                    continue

                signal: StrategySignal = self.strategy.on_bar(self.session_df)
                if signal.action == "HOLD":
                    continue

                pos = self.get_position()
                qty = self.cfg.quantity

                if signal.is_take_profit and self.cfg.tp_exit_all:
                    qty = abs(pos) if abs(pos) > 0 else qty

                if signal.action == "BUY" and pos > 0 and not signal.is_take_profit:
                    continue
                if signal.action == "SELL" and pos < 0 and not signal.is_take_profit:
                    continue

                if signal.action == "BUY" and pos + qty > self.cfg.max_position:
                    print("Risk cap reached. Skipping BUY.")
                    continue
                if signal.action == "SELL" and -pos + qty > self.cfg.max_position:
                    print("Risk cap reached. Skipping SELL.")
                    continue

                if signal.is_take_profit and pos == 0:
                    continue

                if signal.is_take_profit and self.cfg.tp_exit_all and abs(pos) > 0:
                    qty = abs(pos)

                self.place_trade(signal.action, qty, reason=signal.reason)

        except KeyboardInterrupt:
            print("Stopped by user.")