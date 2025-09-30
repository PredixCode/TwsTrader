# unified/strategy/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Protocol, Dict, Any
import pandas as pd


Action = Literal["BUY", "SELL", "HOLD"]


@dataclass
class StrategySignal:
    action: Action                    # BUY, SELL, or HOLD
    is_take_profit: bool = False      # True when signal is a TP/“Book” action
    price: Optional[float] = None     # reference price (e.g., close)
    reason: str = ""                  # human-readable explanation
    extra: Dict[str, Any] = None      # any additional details


class BaseStrategy(Protocol):
    """
    Minimal interface all strategies must implement.
    The bot will call:
      - warmup(history_df) once after history is loaded
      - on_bar(session_df) each time a bar is closed (or intrabar if configured)
    """
    def warmup(self, df: pd.DataFrame) -> None: ...
    def on_bar(self, df: pd.DataFrame) -> StrategySignal: ...