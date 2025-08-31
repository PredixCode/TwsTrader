# sfi_charlie.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from strategy_base import Strategy, Bar, Decision


@dataclass
class SFICharlieParams:
    periods: int = 10            # ATR Period
    multiplier: float = 1.7      # ATR Multiplier
    change_atr: bool = True      # True ATR (RMA) vs SMA(TR)
    show_signals: bool = True    # Not used for plotting here, but left for completeness
    highlight_trend: bool = True # Not used for plotting here


class SFICharlie(Strategy):
    """
    Python port of the 'S F I CHARLIE' TradingView script's core signal logic.
    - Signals:
        BUY  when trend flips from -1 to +1
        SELL when trend flips from +1 to -1
    - Profit event 'BOOK1' when price reaches entry +/- ATR(14)
    """
    name = "SFI Charlie"

    def __init__(self, params: SFICharlieParams|None = None) -> None:
        super().__init__()
        self.p = params or SFICharlieParams()

        # Internal state
        self._prev_close: float|None = None

        # ATR (for entry logic)
        self._tr_win: Deque[float] = deque(maxlen=self.p.periods)  # for SMA init
        self._atr: float|None = None

        # ATR(14) for profit target
        self._atr14_len: int = 14
        self._tr14_win: Deque[float] = deque(maxlen=self._atr14_len)
        self._atr14: float|None = None

        # Supertrend-like lines
        self._up: float|None = None
        self._dn: float|None = None

        # Trend: +1 up, -1 down
        self._trend: int = 1
        self._prev_trend: int = 1

        # Profit target tracking
        self._buy_entry_price: float|None = None
        self._sell_entry_price: float|None = None
        self._buy_book1_done: bool = False
        self._sell_book1_done: bool = False

        # Warmup requirements: enough bars to stabilize ATR and the recursive up/dn
        self.warmup_bars: int = max(self.p.periods, self._atr14_len) + 3

    def reset(self) -> None:
        self.__init__(params=self.p)

    @staticmethod
    def _tr(high: float, low: float, prev_close: float|None) -> float:
        if prev_close is None:
            return high - low
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

    @staticmethod
    def _sma(values: Deque[float]) -> float|None:
        if len(values) == 0:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _rma(prev_rma: float|None, value: float, length: int, seed_sma: float|None) -> float|None:
        """
        Wilder's RMA used by Pine ta.atr:
            RMA(x, n) = (prevRMA*(n-1) + x) / n
        The first value uses SMA as a seed.
        """
        if prev_rma is None:
            return seed_sma
        return (prev_rma * (length - 1) + value) / length

    def _compute_atrs(self, bar: Bar) -> None:
        tr = self._tr(bar.high, bar.low, self._prev_close)

        # Main ATR
        self._tr_win.append(tr)
        sma_seed = self._sma(self._tr_win)
        if self.p.change_atr:
            # True ATR (RMA)
            self._atr = self._rma(self._atr, tr, self.p.periods, sma_seed)
        else:
            # SMA(TR)
            self._atr = sma_seed

        # ATR(14) for profit target
        self._tr14_win.append(tr)
        sma_seed_14 = self._sma(self._tr14_win)
        self._atr14 = self._rma(self._atr14, tr, self._atr14_len, sma_seed_14)

    def _update_trend_lines(self, src: float, close: float, prev_close: float|None) -> None:
        """
        Reproduce the Pine series logic:
          up  = src - Multiplier * atr
          up1 = nz(up[1], up)
          up := close[1] > up1 ? max(up, up1) : up

          dn  = src + Multiplier * atr
          dn1 = nz(dn[1], dn)
          dn := close[1] < dn1 ? min(dn, dn1) : dn
        """
        if self._atr is None:
            # not enough data yet
            return

        up = src - (self.p.multiplier * self._atr)
        dn = src + (self.p.multiplier * self._atr)

        # previous values with nz(..., current)
        up1 = self._up if self._up is not None else up
        dn1 = self._dn if self._dn is not None else dn

        # close[1] is prev_close
        if prev_close is not None:
            up = max(up, up1) if prev_close > up1 else up
            dn = min(dn, dn1) if prev_close < dn1 else dn

        self._up = up
        self._dn = dn

    def _update_trend_and_signals(self, close: float) -> tuple[bool, bool]:
        """
        Update trend per Pine:
          trend := trend == -1 and close > dn1 ? 1 :
                   trend == 1 and close < up1 ? -1 : trend
        """
        buy_signal = False
        sell_signal = False

        up1 = self._up
        dn1 = self._dn

        # Use previous trend to compute next
        trend = self._trend
        if dn1 is not None and up1 is not None:
            if trend == -1 and close > dn1:
                trend = 1
            elif trend == 1 and close < up1:
                trend = -1

        self._prev_trend = self._trend
        self._trend = trend

        # Cross detection
        buy_signal = (self._trend == 1 and self._prev_trend == -1)
        sell_signal = (self._trend == -1 and self._prev_trend == 1)

        # Track entry prices for Book1
        if buy_signal:
            self._buy_entry_price = close
            self._buy_book1_done = False
        if sell_signal:
            self._sell_entry_price = close
            self._sell_book1_done = False

        return buy_signal, sell_signal

    def _check_book1(self, close: float) -> str|None:
        """
        Check profit target events (Book1) using ATR(14).
        """
        if self._atr14 is None:
            return None

        # Long side
        if (self._buy_entry_price is not None) and (not self._buy_book1_done):
            target = self._buy_entry_price + (1.0 * self._atr14)
            if close >= target:
                self._buy_book1_done = True
                return "BOOK1_LONG"

        # Short side
        if (self._sell_entry_price is not None) and (not self._sell_book1_done):
            target = self._sell_entry_price - (1.0 * self._atr14)
            if close <= target:
                self._sell_book1_done = True
                return "BOOK1_SHORT"

        return None

    def on_bar(self, bar: Bar) -> Decision:
        # Prepare series values
        src = (bar.open + bar.high + bar.low + bar.close) / 4.0

        # 1) Update ATRs
        self._compute_atrs(bar)

        # 2) Update supertrend-like lines using previous close
        self._update_trend_lines(src=src, close=bar.close, prev_close=self._prev_close)

        # 3) Update trend and detect signals
        buy, sell = self._update_trend_and_signals(bar.close)

        # 4) Profit event
        book1 = self._check_book1(bar.close)

        # Update prev_close at the very end to reflect "close[1]" semantics next bar
        self._prev_close = bar.close

        # Warmup gating
        warm = (len(self._tr_win) >= self.p.periods) and (len(self._tr14_win) >= self._atr14_len)
        if not warm:
            return Decision(action='HOLD', reason='WARMUP')

        if buy and self.p.show_signals:
            return Decision(action='BUY', price=bar.close, reason='TREND_FLIP_UP', extras={'book1': book1})
        if sell and self.p.show_signals:
            return Decision(action='SELL', price=bar.close, reason='TREND_FLIP_DOWN', extras={'book1': book1})

        # No change in direction: HOLD, but expose current trend, lines, and book1 event
        return Decision(
            action='HOLD',
            reason='NO_SIGNAL',
            extras={
                'trend': self._trend,
                'up': self._up,
                'dn': self._dn,
                'book1': book1
            }
        )