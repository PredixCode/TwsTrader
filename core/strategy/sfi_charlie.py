# unified/strategy/sfi_charlie.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySignal


SourceType = Literal["ohlc4", "close"]


def _rma(series: pd.Series, length: int) -> pd.Series:
    # Wilder's RMA is EMA with alpha = 1/length and adjust=False
    return series.ewm(alpha=1.0 / float(length), adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def _compute_src(df: pd.DataFrame, source: SourceType) -> pd.Series:
    if source == "ohlc4":
        return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    return df["Close"]  # "close"


@dataclass
class SfiCharlieConfig:
    atr_period: int = 10
    multiplier: float = 1.7
    use_true_atr: bool = True
    source: SourceType = "ohlc4"
    # Profit-taking (“Book1”) settings
    tp_atr_length: int = 14
    tp_multiplier: float = 1.0
    # Evaluate strictly on bar close for non-repainting behavior
    evaluate_on_close: bool = True


class SfiCharlieStrategy(BaseStrategy):
    """
    Python port of the provided Pine v6 script (core signals).
    - Trend flips when close crosses previous final bands (up1/dn1).
    - buySignal = trend == 1 and trend[1] == -1
      sellSignal = trend == -1 and trend[1] == 1
    - Take-profit: 1 * ATR(14) from entry in the direction of the trade.
      On trigger, emits a TP signal (which the bot will treat as an exit order).
    State (position/entry/tp) is managed inside the strategy instance.
    """

    def __init__(self, cfg: Optional[SfiCharlieConfig] = None) -> None:
        self.cfg = cfg or SfiCharlieConfig()
        # Internal state for TP tracking
        self.position: int = 0               # -1 short, 0 flat, +1 long
        self.entry_price: Optional[float] = None
        self.tp_level: Optional[float] = None
        self.tp_fired: bool = False

    # ---------- Core calc ----------
    def _calc_bands_and_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with:
          'up', 'dn', 'trend' (1 or -1), 'buySignal', 'sellSignal', 'atr', 'tpATR'
        Mirrors Pine logic; computed iteratively to respect recursive dependencies.
        """
        out = pd.DataFrame(index=df.index)
        src = _compute_src(df, self.cfg.source)
        tr = _true_range(df["High"], df["Low"], df["Close"])
        atr_true = _rma(tr, self.cfg.atr_period)
        atr_sma = tr.rolling(self.cfg.atr_period, min_periods=1).mean()
        atr = atr_true if self.cfg.use_true_atr else atr_sma
        out["atr"] = atr

        # For TP we use ATR(14) like the script (true ATR)
        out["tpATR"] = _rma(tr, self.cfg.tp_atr_length)

        n = len(df)
        up = np.full(n, np.nan, dtype=float)
        dn = np.full(n, np.nan, dtype=float)
        trend = np.full(n, 1, dtype=int)  # var int trend = 1

        close = df["Close"].values
        src_vals = src.values
        atr_vals = atr.values

        for i in range(n):
            if np.isnan(src_vals[i]) or np.isnan(atr_vals[i]):
                # leave NaNs; trend stays whatever default is
                continue

            base_up = src_vals[i] - self.cfg.multiplier * atr_vals[i]
            base_dn = src_vals[i] + self.cfg.multiplier * atr_vals[i]

            if i == 0:
                up[i] = base_up
                dn[i] = base_dn
                trend[i] = 1
                continue

            up1 = up[i - 1]
            dn1 = dn[i - 1]

            # up := close[1] > up1 ? max(up, up1) : up
            if close[i - 1] > up1:
                up[i] = max(base_up, up1)
            else:
                up[i] = base_up

            # dn := close[1] < dn1 ? min(dn, dn1) : dn
            if close[i - 1] < dn1:
                dn[i] = min(base_dn, dn1)
            else:
                dn[i] = base_dn

            # trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
            prev_trend = trend[i - 1]
            if prev_trend == -1 and close[i] > dn1:
                trend[i] = 1
            elif prev_trend == 1 and close[i] < up1:
                trend[i] = -1
            else:
                trend[i] = prev_trend

        out["up"] = up
        out["dn"] = dn
        out["trend"] = trend

        # Signals: buy when -1 -> 1, sell when 1 -> -1
        trend_shift = out["trend"].shift(1)
        out["buySignal"] = (out["trend"] == 1) & (trend_shift == -1)
        out["sellSignal"] = (out["trend"] == -1) & (trend_shift == 1)

        return out

    # ---------- Strategy lifecycle ----------
    def warmup(self, df: pd.DataFrame) -> None:
        # Reset state and backfill initial trend context from history
        self.position = 0
        self.entry_price = None
        self.tp_level = None
        self.tp_fired = False
        # Run calc once to sync internal idea of current trend (no orders placed here)
        _ = self._calc_bands_and_trend(df)

    def on_bar(self, df: pd.DataFrame) -> StrategySignal:
        """
        Evaluate on the latest available bar (by default, bar-close).
        Emits:
          - BUY  on trend flip up
          - SELL on trend flip down
          - TP   when 1x ATR(14) profit target is hit from the last entry
        """
        if df is None or df.empty:
            return StrategySignal(action="HOLD", reason="no data")

        calc = self._calc_bands_and_trend(df)
        last = calc.iloc[-1]
        close = float(df["Close"].iloc[-1])

        # Entry/flip logic
        if bool(last["buySignal"]):
            # Enter/flip long
            self.position = 1
            self.entry_price = close
            self.tp_level = self.entry_price + self.cfg.tp_multiplier * float(last["tpATR"])
            self.tp_fired = False
            return StrategySignal(action="BUY", is_take_profit=False, price=close, reason="trend flip up")

        if bool(last["sellSignal"]):
            # Enter/flip short
            self.position = -1
            self.entry_price = close
            self.tp_level = self.entry_price - self.cfg.tp_multiplier * float(last["tpATR"])
            self.tp_fired = False
            return StrategySignal(action="SELL", is_take_profit=False, price=close, reason="trend flip down")

        # TP logic (Book1)
        if self.position == 1 and self.entry_price is not None and self.tp_level is not None and not self.tp_fired:
            if close >= self.tp_level:
                self.tp_fired = True
                return StrategySignal(action="SELL", is_take_profit=True, price=close, reason="TP long (Book1)")

        if self.position == -1 and self.entry_price is not None and self.tp_level is not None and not self.tp_fired:
            if close <= self.tp_level:
                self.tp_fired = True
                return StrategySignal(action="BUY", is_take_profit=True, price=close, reason="TP short (Book1)")

        return StrategySignal(action="HOLD", reason="no signal")