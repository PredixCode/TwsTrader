import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple

import pandas as pd

from tws_wrapper.stock import TwsStock
from ui.graph import LiveGraph


class TwsProvisionalCandleUpdater:
    """
    Provisional last-candle updater driven by TWS:
    - Touches ONLY the current minute bar.
    - On each poll (~5–15s), if price changed meaningfully, updates O/H/Low/Close and upserts to LiveGraph.
    - Never edits closed minutes; yfinance will supply/overwrite those every minute.

    Notes:
    - Call TwsStock.get_ticker() on the main thread before prepare()/start().
    - Keep ib_insync pumped in the main thread (e.g., during_wait=lambda dt: ib.sleep(dt) in your YF loop).
    """

    def __init__(
        self,
        stock: TwsStock,
        view: LiveGraph,
        *,
        poll_secs: float = 5.0,                      # ~5–15s as you prefer
        price_pref: Tuple[str, ...] = ("mid", "last", "bid", "ask"),
        min_change_ticks: float = 0.0,               # 0.0 -> update on any change; 1.0 -> >= 1 tick
        verbose: bool = True,
        daemon: bool = True,
        log_every_secs: Optional[float] = 30.0,      # force a log line even if unchanged
    ) -> None:
        self.stock = stock
        self.view = view
        self.poll_secs = max(0.25, float(poll_secs))
        self.price_pref = tuple(price_pref) if price_pref else ("mid", "last", "bid", "ask")
        self.min_change_ticks = float(min_change_ticks)
        self.verbose = verbose
        self.daemon = daemon
        self.log_every_secs = None if log_every_secs is None else max(1.0, float(log_every_secs))

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prepared: bool = False

        # Active minute state
        self._minute_key: Optional[pd.Timestamp] = None
        self._o: Optional[float] = None
        self._h: Optional[float] = None
        self._l: Optional[float] = None
        self._c: Optional[float] = None

        # Change detection
        self._last_pushed_close: Optional[float] = None
        self._last_log_ts: float = 0.0

        # tz for chart index
        try:
            self._tz = getattr(self.view.dataframe.index, "tz", None)
        except Exception:
            self._tz = None

        # tick size for epsilon
        try:
            self._min_tick = float(self.stock.get_min_tick())
        except Exception:
            self._min_tick = 0.01

    # ---------- lifecycle ----------

    def prepare(self) -> None:
        """
        Must be called on main thread AFTER TwsStock.get_ticker() subscribed the ticker.
        """
        self._prepared = True
        if self.verbose:
            print("[TWS-Provisional] Prepared (ticker subscribed; ready to update chart).")

    def start(self) -> None:
        if self.is_running():
            return
        if not self._prepared:
            raise RuntimeError("Call prepare() on main thread after TwsStock.get_ticker().")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TWS-Provisional-Loop", daemon=self.daemon)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    # ---------- internals ----------

    def _run(self) -> None:
        try:
            # Immediate first pass
            self._tick(force=True)
            while not self._stop.is_set():
                time.sleep(self.poll_secs)
                if self._stop.is_set():
                    break
                self._tick(force=False)
        except Exception as e:
            print(f"[TWS-Provisional] ERROR in loop: {e!r}")

    def _tick(self, *, force: bool) -> None:
        snap = self.stock.snapshot()  # {'symbol','bid','ask','last','mid','bidSize','askSize','lastSize'}
        price = self._pick_price(snap)
        if price is None:
            # Still log occasionally so you know it’s alive
            self._maybe_log(snap, tag="no-price")
            return

        now_minute = pd.Timestamp.now(tz=self._tz).floor("min")

        # Switch to current minute if needed
        if self._minute_key is None or now_minute > self._minute_key:
            self._minute_key = now_minute
            self._o = self._h = self._l = self._c = float(price)
            self._last_pushed_close = None
            # Seed the new bar immediately
            self._push_bar(log_reason="seed")
            return

        # Same minute: update H/L/C
        changed = False
        if self._h is None or price > self._h:
            self._h = float(price); changed = True
        if self._l is None or price < self._l:
            self._l = float(price); changed = True

        # Close updates with epsilon
        if self._c is None or self._price_changed(price, self._c):
            self._c = float(price); changed = True

        # Push if changed enough
        if changed:
            self._push_bar(log_reason="update")
        else:
            self._maybe_log(snap, tag="unchanged")

    def _push_bar(self, log_reason: str) -> None:
        # Only push if close changed meaningfully from last push
        if self._last_pushed_close is not None and not self._price_changed(self._c, self._last_pushed_close):
            self._maybe_log(None, tag="skip-dup")
            return

        try:
            # Upsert only OHLC (compatible with your current LiveGraph)
            self.view.upsert_bar(
                when=self._minute_key,
                o=float(self._o),
                h=float(self._h),
                l=float(self._l),
                c=float(self._c),
                floor_to_minute=False
            )
            self._last_pushed_close = float(self._c)

            # Log a concise line with OHLC and a key price snapshot
            ts_local = datetime.now().strftime("%H:%M:%S")
            print(f"{ts_local} [TWS-Provisional:{log_reason}] {self._minute_key} "
                  f"O={self._o} H={self._h} L={self._l} C={self._c}")

        except TypeError as te:
            # If your LiveGraph signature differs, fall back to minimal call
            try:
                self.view.upsert_bar(self._minute_key, float(self._o), float(self._h), float(self._l), float(self._c))
                self._last_pushed_close = float(self._c)
            except Exception as e:
                print(f"[TWS-Provisional] upsert fallback failed: {e!r}")
        except Exception as e:
            print(f"[TWS-Provisional] upsert error: {e!r}")

    def _maybe_log(self, snap: Optional[Dict[str, Any]], tag: str) -> None:
        if not self.verbose or self.log_every_secs is None:
            return
        now = time.time()
        if (now - self._last_log_ts) < self.log_every_secs:
            return
        self._last_log_ts = now
        if snap is None:
            print(f"[TWS-Provisional] {tag}")
            return
        ts_local = datetime.now().strftime("%H:%M:%S")
        b = snap.get("bid"); a = snap.get("ask"); l = snap.get("last"); m = snap.get("mid")
        bs = int(snap.get("bidSize") or 0); as_ = int(snap.get("askSize") or 0); ls = int(snap.get("lastSize") or 0)
        sym = snap.get("symbol")
        print(f"{ts_local} [TWS-Provisional:{tag}] {sym} bid={b}({bs}) ask={a}({as_}) last={l}({ls}) mid={m}")

    def _pick_price(self, snap: Dict[str, Any]) -> Optional[float]:
        for k in self.price_pref:
            v = self._safe_float(snap.get(k))
            if v is not None:
                return v
        bid = self._safe_float(snap.get("bid"))
        ask = self._safe_float(snap.get("ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return bid or ask

    def _price_changed(self, a: Optional[float], b: Optional[float]) -> bool:
        if a is None or b is None:
            return True
        if self.min_change_ticks <= 0:
            return a != b
        eps = self._min_tick * self.min_change_ticks
        return abs(float(a) - float(b)) >= eps

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            xf = float(x)
            if xf != xf:  # NaN
                return None
            return xf
        except Exception:
            return None