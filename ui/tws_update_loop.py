import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple

import pandas as pd

from tws_wrapper.stock import TwsStock
from ui.graph import LiveGraph

class IBHistoryChartUpdater:
    """
    IB-driven historical updater for LiveGraph.
    - Runs on MAIN THREAD.
    - Periodically calls TwsStock.get_historical_data(period, interval) which uses TwsFetchCache
      and increments the tail with overlap.
    - Pushes only OHLC to the chart via view.apply_delta_df(...).
    - Optionally persists last_fetch to CSV.
    - during_wait(dt) is called so you can ib.sleep(dt) to pump IB events while waiting.
    """

    def __init__(
        self,
        stock: TwsStock,
        view: object,
        *,
        period: str = "max",
        interval: str = "1m",
        whatToShow="TRADES", useRTH=True,
        poll_secs: float = 60.0,
        persist_csv_every: int = 1,
        align_to_period: bool = True,
        verbose: bool = True,
        during_wait: Optional[Callable[[float], None]] = None,
        overlap_minutes: int = 5,      # send only the last N minutes each tick
        fallback_tail_rows: int = 500, # safety if index math fails
    ):
        self.stock = stock
        self.view = view
        self.period = period
        self.interval = interval
        self.whatToShow = whatToShow
        self.useRTH = useRTH
        self.poll_secs = max(1.0, float(poll_secs))
        self.persist_csv_every = max(0, int(persist_csv_every))
        self.align_to_period = bool(align_to_period)
        self.verbose = verbose
        self.during_wait = during_wait
        self.overlap_minutes = max(0, int(overlap_minutes))
        self.fallback_tail_rows = max(1, int(fallback_tail_rows))
        self._loop_count = 0

    def _wait(self, seconds: float):
        if self.during_wait is not None:
            remaining = float(seconds)
            step = 0.2
            while remaining > 0:
                dt = step if remaining >= step else remaining
                self.during_wait(dt)
                remaining -= dt
        else:
            time.sleep(seconds)

    def _seconds_to_next_minute(self) -> float:
        now = pd.Timestamp.now()
        nxt = (now + pd.Timedelta(minutes=1)).floor("min")
        return max(0.0, (nxt - now).total_seconds())

    def _log(self, msg: str):
        if self.verbose:
            print(f"[IBHistoryUpdater] {msg}")

    def run_forever_in_main_thread(self):
        # First tick immediately; optionally align afterward
        self._tick_once()
        while True:
            try:
                if self.align_to_period:
                    wait = self._seconds_to_next_minute()
                    wait = max(0.1, wait)  # small buffer
                else:
                    wait = self.poll_secs
                self._wait(wait)
                self._tick_once()
            except KeyboardInterrupt:
                self._log("Interrupted; exiting.")
                break
            except Exception as e:
                self._log(f"ERROR: {e!r}")
                self._wait(2.0)

    def _tick_once(self):
        self._loop_count += 1
        df = self.stock.get_historical_data(period=self.period, interval=self.interval, whatToShow=self.whatToShow, useRTH=self.useRTH)
        if df is None or df.empty:
            self._log("No data returned.")
            return

        # Only OHLCV to the chart
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]

        # Send only a small overlap window to minimize work
        delta = ohlcv
        try:
            if self.overlap_minutes > 0 and len(ohlcv) > 1:
                end_ts = ohlcv.index[-1]
                start_ts = end_ts - pd.Timedelta(minutes=self.overlap_minutes)
                delta = ohlcv.loc[ohlcv.index >= start_ts]
            else:
                # Fallback to tail rows
                delta = ohlcv.tail(self.fallback_tail_rows)
        except Exception:
            delta = ohlcv.tail(self.fallback_tail_rows)

        # LiveGraph incremental apply
        self.view.apply_delta_df(delta)

        if self.persist_csv_every and (self._loop_count % self.persist_csv_every == 0):
            self.stock.last_fetch_to_csv(df)

        if self.verbose:
            last = ohlcv.index[-1]
            self._log(f"Updated history -> rows={len(df)} last={last} interval={self.interval} period={self.period} pushed={len(delta)}")


class TwsProvisionalCandleUpdater:
    """
    TWS-driven provisional last-candle:
      - Updates ONLY the current minute bar on each poll (e.g., every 5–15s) when price changes.
      - On minute rollover, DESTROYS the prior minute provisional bar; yfinance will later replace it with authoritative data.
      - Never touches closed minutes.

    Usage notes:
      - Subscribe on MAIN THREAD first: stock.get_ticker()
      - Keep ib_insync pumped in main thread (e.g., during_wait=lambda dt: ib.sleep(dt) in your YF loop).
    """

    def __init__(
        self,
        stock: TwsStock,
        view: object,
        *,
        poll_secs: float = 1,
        price_pref: Tuple[str, ...] = ("mid", "last", "bid", "ask"),
        min_change_ticks: float = 0.0,  # 0.0: any change; 1.0: >= 1 tick
        verbose: bool = True,
        daemon: bool = True,
        log_every_secs: Optional[float] = 30.0,
        on_drop_prev_minute: Optional[Callable[[object, pd.Timestamp], None]] = None,
    ) -> None:
        self.stock = stock
        self.view = view
        self.poll_secs = max(0.25, float(poll_secs))
        self.price_pref = tuple(price_pref) if price_pref else ("mid", "last", "bid", "ask")
        self.min_change_ticks = float(min_change_ticks)
        self.verbose = verbose
        self.daemon = daemon
        self.log_every_secs = None if log_every_secs is None else max(1.0, float(log_every_secs))
        self.on_drop_prev_minute = on_drop_prev_minute

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._prepared: bool = False

        # Active minute state
        self._minute_key: Optional[pd.Timestamp] = None
        self._o: Optional[float] = None
        self._h: Optional[float] = None
        self._l: Optional[float] = None
        self._c: Optional[float] = None
        self._cum_vol_seed: Optional[float] = None  # cumulative day volume at minute start

        # Change detection
        self._last_pushed_close: Optional[float] = None
        self._last_log_ts: float = 0.0

        # tick size for epsilon
        try:
            self._min_tick = float(self.stock.get_min_tick())
        except Exception:
            self._min_tick = 0.01

    # ---------- lifecycle ----------

    def prepare(self) -> None:
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
            self._maybe_log(snap, tag="no-price")
            return

        now_minute = pd.Timestamp.utcnow().floor("min")

        # Minute rollover: destroy previous provisional bar, seed new current minute
        if self._minute_key is None or now_minute > self._minute_key:
            prev_minute = self._minute_key
            if prev_minute is not None:
                self._drop_prev_minute(prev_minute)

            self._minute_key = now_minute
            self._o = self._h = self._l = self._c = float(price)

            # Seed cumulative volume at minute start; minute volume starts at 0
            self._cum_vol_seed = self._cum_volume_from_snap(snap)
            self._v = 0.0
            self._last_pushed_close = None

            self._push_bar(log_reason="seed")
            return

        # Same minute: update H/L/C
        changed = False
        if self._h is None or price > self._h:
            self._h = float(price); changed = True
        if self._l is None or price < self._l:
            self._l = float(price); changed = True
        if self._c is None or self._price_changed(price, self._c):
            self._c = float(price); changed = True

        # Recompute minute volume from cumulative
        cur_cum = self._cum_volume_from_snap(snap)
        if cur_cum is not None and self._cum_vol_seed is not None:
            minute_vol = max(0.0, float(cur_cum) - float(self._cum_vol_seed))
            if self._v is None or minute_vol != self._v:
                self._v = minute_vol
                changed = True

        if changed or force:
            self._push_bar(log_reason="update")
        else:
            self._maybe_log(snap, tag="unchanged")

    def _push_bar(self, log_reason: str) -> None:
        if self._last_pushed_close is not None and not self._price_changed(self._c, self._last_pushed_close):
            self._maybe_log(None, tag="skip-dup")
            return
        try:
            self.view.upsert_bar(
                when=self._minute_key,
                o=float(self._o),
                h=float(self._h),
                l=float(self._l),
                c=float(self._c),
                v=float(self._v),
                floor_to_minute=False
            )
            self._last_pushed_close = float(self._c)
            if self.verbose:
                ts_local = datetime.now().strftime("%H:%M:%S")
                print(f"{ts_local} [TWS-Provisional:{log_reason}] {self._minute_key} "
                      f"O={self._o} H={self._h} L={self._l} C={self._c} V={self._v}")
        except TypeError:
            # Minimal positional fallback if your signature differs
            try:
                self.view.upsert_bar(self._minute_key, float(self._o), float(self._h), float(self._l), float(self._c), float(self._v))
                self._last_pushed_close = float(self._c)
            except Exception as e:
                print(f"[TWS-Provisional] upsert fallback failed: {e!r}")
        except Exception as e:
            print(f"[TWS-Provisional] upsert error: {e!r}")

    def _drop_prev_minute(self, ts: pd.Timestamp) -> None:
        """
        Destroy the provisional bar for the previous minute.
        Tries a few LiveGraph methods, else falls back to dropping from dataframe.
        """
        try:
            # User-provided hook takes precedence
            if self.on_drop_prev_minute is not None:
                self.on_drop_prev_minute(self.view, ts)
                self._log_drop(ts, source="custom")
                return

            # Common method names on custom LiveGraph
            for name in ("remove_bar", "delete_bar", "drop_bar"):
                fn = getattr(self.view, name, None)
                if callable(fn):
                    fn(ts)
                    self._log_drop(ts, source=name)
                    return

            # Fallback: drop from dataframe
            df = getattr(self.view, "dataframe", None)
            if df is not None and ts in df.index:
                new_df = df.drop(index=[ts], errors="ignore")
                # If LiveGraph exposes a setter, prefer it
                if callable(getattr(self.view, "set_dataframe", None)):
                    self.view.set_dataframe(new_df)
                else:
                    # Last resort: mutate and hope the view observes it
                    self.view.dataframe = new_df
                    # If a refresh/replot method exists, call it
                    for name in ("refresh", "replot", "redraw"):
                        fn = getattr(self.view, name, None)
                        if callable(fn):
                            fn()
                            break
                self._log_drop(ts, source="dataframe-drop")
                return

            # If nothing worked, at least log
            print(f"[TWS-Provisional] WARN: Could not drop provisional bar {ts} (no supported method).")
        except Exception as e:
            print(f"[TWS-Provisional] drop_prev_minute error for {ts}: {e!r}")

    def _cum_volume_from_snap(self, snap: Dict[str, Any]) -> Optional[float]:
        """
        Return cumulative day volume if available.
        Prefer RTVolume totalVolume when present, else fall back to snapshot 'vol'.
        IB RTVolume format: "price;size;time;totalVolume;VWAP;singleTrade"
        """
        rv = snap.get("rtVolume")
        if rv:
            try:
                parts = str(rv).split(";")
                if len(parts) >= 4:
                    return float(parts[3])
            except Exception:
                pass
        v = snap.get("vol")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _log_drop(self, ts: pd.Timestamp, *, source: str) -> None:
        if self.verbose:
            ts_local = datetime.now().strftime("%H:%M:%S")
            print(f"{ts_local} [TWS-Provisional:drop-{source}] removed provisional bar {ts}")

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
        def pos(x) -> Optional[float]:
            try:
                xf = float(x)
                if xf > 0 and xf == xf:  # positive and not NaN
                    return xf
            except Exception:
                pass
            return None

        # Prefer mid if both bid/ask are positive
        bid = pos(snap.get("bid"))
        ask = pos(snap.get("ask"))
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            if mid > 0:
                return mid

        # Fall back through preferred keys but only accept positive numbers
        for k in self.price_pref:
            v = pos(snap.get(k))
            if v is not None:
                return v

        # As a very last resort, try either side if positive
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return None

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