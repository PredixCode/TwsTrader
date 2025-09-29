import time
import threading
import traceback
from typing import Optional, Callable, Tuple
import pandas as pd

from yfinance_wrapper.stock import FinanceStock
from ui.graph import LiveGraph


class YFinanceChartUpdater:
    """
    Periodically fetches yfinance data and pushes deltas into a LiveGraph.

    Two modes:
      - Threaded: start()/stop() manage a background thread (may stall if UI hogs GIL).
      - Main-thread pump: call run_forever_in_main_thread() after view.show(block=False).

    Features:
      - Immediate first fetch
      - Optional alignment to time boundaries
      - Exponential backoff on errors
      - Verbose logs
    """

    def __init__(
        self,
        stock: FinanceStock,
        view: LiveGraph,
        *,
        period: str = "1d",
        interval: str = "1m",
        poll_secs: int = 60,
        persist_csv_every: int = 30,
        align_to_period: bool = True,
        on_update: Optional[Callable[[pd.DataFrame], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        daemon: bool = True,
        verbose: bool = True,
    ) -> None:
        self.stock = stock
        self.view = view

        self.period = period
        self.interval = interval
        self.poll_secs = max(1, int(poll_secs))
        self.persist_csv_every = max(0, int(persist_csv_every))
        self.align_to_period = bool(align_to_period)
        self.on_update = on_update
        self.on_error = on_error
        self.verbose = verbose

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fetch_lock = threading.Lock()
        self._updates = 0
        self._daemon = daemon

    # ---------- Threaded mode ----------

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_threaded, name="YF-Chart-Updater", daemon=self._daemon)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def _run_threaded(self) -> None:
        backoff = self.poll_secs

        while not self._stop.is_set():
            try:
                delay = self._next_delay()
                self._wait(delay)
                if self._stop.is_set():
                    break

                self._tick()
                backoff = self.poll_secs

            except Exception as e:
                self._handle_error(e)
                backoff = min(backoff * 2, max(5 * self.poll_secs, 300))
                self._wait(backoff)

    # ---------- Main-thread pump mode ----------

    def run_forever_in_main_thread(self) -> None:
        """
        Call this on the main thread AFTER showing the chart with block=False.
        This avoids any UI-thread/GIL starvation problems.
        """
        backoff = self.poll_secs

        try:
            while not self._stop.is_set():
                delay = self._next_delay()
                self._wait(delay)  # uses Event.wait, responsive to stop
                if self._stop.is_set():
                    break

                try:
                    self._tick()
                    backoff = self.poll_secs
                except Exception as e:
                    self._handle_error(e)
                    backoff = min(backoff * 2, max(5 * self.poll_secs, 300))
                    self._wait(backoff)
        except KeyboardInterrupt:
            if self.verbose:
                print("[YF-Updater] Stopped by user.")

    # ---------- Shared internals ----------

    def _tick(self) -> None:
        t0 = time.time()
        try:
            with self._fetch_lock:
                recent_df = self.stock.get_historical_data(period=self.period, interval=self.interval)

            if recent_df is None or recent_df.empty:
                if self.verbose:
                    print(f"[YF-Updater] No data returned.")
                return

            recent_df = recent_df.sort_index()

            delta_count, merged_count = self._compute_delta_sizes(self.view.dataframe, recent_df)

            # Push to chart
            self.view.apply_delta_df(recent_df)
            self.view.scroll_to_latest()

            self._updates += 1
            if self.persist_csv_every and (self._updates % self.persist_csv_every == 0):
                try:
                    self.stock.last_fetch_to_csv(self.view.dataframe)
                except Exception as ex:
                    if self.verbose:
                        print(f"[YF-Updater] Persist CSV failed: {ex}")

            t1 = time.time()
            if self.verbose:
                print(f"[YF-Updater] tick "
                      f"delta={delta_count}, merged_len={merged_count}, "
                      f"elapsed={t1 - t0:.2f}s")

            if self.on_update:
                try:
                    self.on_update(recent_df)
                except Exception as cb_ex:
                    if self.verbose:
                        print(f"[YF-Updater] on_update callback error: {cb_ex}")

        except Exception as e:
            self._handle_error(e)

    def _compute_delta_sizes(self, current: pd.DataFrame, recent: pd.DataFrame) -> Tuple[int, int]:
        try:
            if current is None or current.empty:
                merged = recent.copy()
            else:
                merged = (
                    pd.concat([current, recent], axis=0)
                    .sort_index()
                )
                merged = merged[~merged.index.duplicated(keep="last")]

            last_idx = current.index.max() if (current is not None and len(current)) else None
            if last_idx is not None and last_idx in merged.index:
                changed = merged.loc[merged.index >= last_idx]
            else:
                changed = recent

            return len(changed), len(merged)
        except Exception:
            return -1, -1

    def _next_delay(self) -> float:
        if not self.align_to_period:
            return float(self.poll_secs)
        return self._secs_to_next_multiple(self.poll_secs)

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._stop.wait(timeout=float(seconds))

    def _handle_error(self, e: BaseException) -> None:
        if self.on_error:
            try:
                self.on_error(e)
            except Exception:
                pass
        print("[YF-Updater] ERROR:", repr(e))
        traceback.print_exc()

    @staticmethod
    def _secs_to_next_multiple(step: int) -> float:
        now = time.time()
        rem = now % step
        delay = step - rem
        if delay < 0.01:
            delay += step
        return float(delay)