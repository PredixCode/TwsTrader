import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from tws_wrapper.stock import TwsStock


class TwsPriceUpdateLoop:
    """
    Lightweight TWS price printer that:
      - Requires prepare() to be called on the main thread (connect + subscribe)
      - Polls snapshot() periodically
      - Prints only on changes (or every `force_print_secs`)
      - Runs in its own thread so it won't block your yfinance main-thread pump

    IMPORTANT:
      Call prepare() on the MAIN THREAD before start(). This avoids ib_insync
      event loop errors in background threads.
    """

    def __init__(
        self,
        stock: TwsStock,
        *,
        poll_secs: float = 1.0,
        force_print_secs: Optional[float] = None,   # print even if unchanged every N seconds (None to disable)
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True,
        daemon: bool = True,
    ) -> None:
        self.stock = stock
        self.poll_secs = max(0.1, float(poll_secs))
        self.force_print_secs = None if force_print_secs is None else max(0.5, float(force_print_secs))
        self.on_update = on_update
        self.verbose = verbose
        self.daemon = daemon

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._last_print_ts: float = 0.0
        self._prepared: bool = False

    # ---------- Main-thread preparation ----------

    def prepare(self) -> None:
        """
        Must be called on the main thread:
          - Ensures IB connection is established
          - Subscribes to market data and caches ticker
        """
        # This calls TwsConnection.connect() internally (main thread),
        # discovers/selects a market data type, and stores _ticker
        self.stock.get_ticker()
        self._prepared = True
        if self.verbose:
            print("[TWS-Loop] Prepared (connected + subscribed).")

    # ---------- Threaded loop control ----------

    def start(self) -> None:
        if self.is_running():
            return
        if not self._prepared:
            raise RuntimeError("TwsPriceUpdateLoop.start() called before prepare(). Call prepare() on the main thread first.")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TWS-Price-Loop", daemon=self.daemon)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    # ---------- Internals ----------

    def _run(self) -> None:
        """
        Poll snapshot() periodically. No IB connect calls in this thread.
        """
        try:
            # Immediate first print
            self._tick(force_print=True)

            while not self._stop.is_set():
                time.sleep(self.poll_secs)
                if self._stop.is_set():
                    break
                self._tick(force_print=False)

        except Exception as e:
            print(f"[TWS-Loop] ERROR in loop: {e!r}")

    def _tick(self, *, force_print: bool) -> None:
        snap = self.stock.snapshot()  # {'symbol','bid','ask','last','mid','bidSize','askSize','lastSize'}
        now = time.time()

        should_print = False
        if self._last_snapshot is None:
            should_print = True
        elif not self._equal_snapshots(snap, self._last_snapshot):
            should_print = True
        elif self.force_print_secs is not None and (now - self._last_print_ts) >= self.force_print_secs:
            should_print = True

        if should_print:
            self._print_snapshot(snap)
            self._last_print_ts = now
            self._last_snapshot = snap

        if self.on_update:
            try:
                self.on_update(snap)
            except Exception as cb_ex:
                if self.verbose:
                    print(f"[TWS-Loop] on_update callback error: {cb_ex!r}")

    def _print_snapshot(self, snap: Dict[str, Any]) -> None:
        ts_local = datetime.now().strftime("%H:%M:%S")
        bid = snap.get("bid")
        ask = snap.get("ask")
        last = snap.get("last")
        mid = snap.get("mid")
        bsz = int(snap.get("bidSize") or 0)
        asz = int(snap.get("askSize") or 0)
        lsz = int(snap.get("lastSize") or 0)
        sym = snap.get("symbol")
        print(f"{ts_local} [TWS] {sym} bid={bid} ({bsz}) ask={ask} ({asz}) last={last} ({lsz}) mid={mid}")

    @staticmethod
    def _equal_snapshots(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        def f(x):
            try:
                return float(x)
            except Exception:
                return None

        for key in ("bid", "ask", "last", "mid"):
            aa = f(a.get(key))
            bb = f(b.get(key))
            if aa is None and bb is None:
                continue
            if aa is None or bb is None:
                return False
            if abs(aa - bb) > 1e-9:
                return False

        for key in ("bidSize", "askSize", "lastSize"):
            if int(a.get(key) or 0) != int(b.get(key) or 0):
                return False

        return True