# strategy_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float|None = None


@dataclass
class Decision:
    action: str                  # 'BUY' | 'SELL' | 'HOLD' (or Action enum value)
    size: float|int|None = None  # let the runner/strategy decide sizing; None means "no suggestion"
    price: float|None = None     # suggested execution price (e.g., close)
    reason: str|None = None
    extras: dict[str, Any]|None = None

    # Convenience factories
    @staticmethod
    def buy(size: float|int|None = None, price: float|None = None, reason: str|None = None, extras: dict[str, Any]|None = None) -> "Decision":
        return Decision(action=Action.BUY.value, size=size, price=price, reason=reason, extras=extras)

    @staticmethod
    def sell(size: float|int|None = None, price: float|None = None, reason: str|None = None, extras: dict[str, Any]|None = None) -> "Decision":
        return Decision(action=Action.SELL.value, size=size, price=price, reason=reason, extras=extras)

    @staticmethod
    def hold(reason: str|None = None, extras: dict[str, Any]|None = None) -> "Decision":
        return Decision(action=Action.HOLD.value, reason=reason, extras=extras)


class Strategy(ABC):
    """
    Base strategy interface. It is purely signal-generating. No broker/IB logic here.
    The runner (or caller) wires Strategy + Stock + Trader.

    Typical usage:
      strat = MyStrategy()
      strat.ingest_history(historical_bars)  # warms up
      for bar in live_bars:
          decision = strat.step(bar)         # updates internal state + returns Decision
    """
    name: str = "BaseStrategy"
    warmup_bars: int = 0  # minimum bars needed before signals can be reliable

    def __init__(self) -> None:
        # State tracked by the base
        self._is_warm: bool = False
        self._bars_seen: int = 0
        self._last_bar: Bar|None = None
        self._last_decision: Decision|None = None
        # Optional parameter store for subclasses (not enforced)
        self._params: dict[str, Any] = {}

    # ---------- Lifecycle hooks ----------

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state (subclass must implement).
        Subclasses should call super()._base_reset() early in reset().
        """
        ...

    def _base_reset(self) -> None:
        """Base helper to reset shared counters/state; call from subclass reset()."""
        self._is_warm = False
        self._bars_seen = 0
        self._last_bar = None
        self._last_decision = None

    def on_start(self) -> None:
        """Optional hook when a runner starts live processing."""
        return None

    def on_stop(self) -> None:
        """Optional hook when a runner stops."""
        return None

    # ---------- Core processing ----------

    @abstractmethod
    def on_bar(self, bar: Bar) -> Decision:
        """
        Consume one bar and return a Decision.
        Caller feeds historical bars first to warm up, then live bars.
        Subclasses must be side-effectful (update their internal state).
        """
        ...

    def step(self, bar: Bar) -> Decision:
        """
        Wrapper around on_bar that also maintains base counters and warmup state.
        Runners should prefer calling step() for live processing.
        """
        self._last_bar = bar
        decision = self.on_bar(bar)
        self._last_decision = decision
        self._bars_seen += 1
        if not self._is_warm and self._bars_seen >= int(getattr(self, "warmup_bars", 0)):
            self._is_warm = True
        return decision

    def on_bars(self, bars: Iterable[Bar]) -> Decision|None:
        """
        Convenience: process a batch of bars (e.g., historical), returning the last Decision.
        """
        last: Decision|None = None
        for b in bars:
            last = self.step(b)
        return last

    def ingest_history(self, bars: list[Bar]) -> None:
        """
        Optional helper: warm the strategy with historical bars (no trading).
        Resets the strategy first.
        """
        self.reset()
        self.on_bars(bars)
        # Warmup is determined by bars_seen vs warmup_bars
        # If enough history was provided, _is_warm becomes True inside step()

    # ---------- Introspection ----------

    def is_warm(self) -> bool:
        return self._is_warm

    def bars_seen(self) -> int:
        return self._bars_seen

    def last_bar(self) -> Bar|None:
        return self._last_bar

    def last_decision(self) -> Decision|None:
        return self._last_decision

    # ---------- Parameter helpers (optional) ----------

    def set_params(self, **kwargs: Any) -> None:
        """
        Store arbitrary parameters (subclasses may read from self._params).
        No behavior is enforced by the base.
        """
        self._params.update(kwargs)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    # ---------- Persistence helpers (optional) ----------

    def get_state(self) -> dict[str, Any]:
        """
        Return serializable state (subclasses should extend).
        Use this to checkpoint and resume strategies.
        """
        return {
            "name": self.name,
            "warmup_bars": self.warmup_bars,
            "is_warm": self._is_warm,
            "bars_seen": self._bars_seen,
            "params": dict(self._params),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore state (subclasses should extend).
        Call after reset() if you want to resume from a checkpoint.
        """
        self._is_warm = bool(state.get("is_warm", False))
        self._bars_seen = int(state.get("bars_seen", 0))
        self._params = dict(state.get("params", {}))