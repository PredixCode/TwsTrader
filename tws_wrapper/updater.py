import asyncio
import threading
import time
import pandas as pd

from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple
from concurrent.futures import Future


from tws_wrapper.stock import TwsStock



class TwsHistoricUpdater:
    """
    IB-driven historical updater.
    - Can run on MAIN THREAD (run()) or in a BACKGROUND THREAD (start()).
    - Periodically calls TwsStock.get_historical_data(period, interval) which uses TwsFetchCache
      and increments the tail with overlap.
    - Pushes only OHLC to the chart via view.apply_delta_df(...).
    - Optionally persists last_fetch to CSV.

    Threading notes:
      - Use start()/stop() for background operation.
      - run() remains available for synchronous use (backwards compatible).
    """

    def __init__(
        self,
        stock: TwsStock,
        view: object,
        *,
        period: str = "max",
        interval: str = "1m",
        whatToShow: str = "TRADES",
        useRTH: bool = False,
        poll_secs: float = 60.0,
        persist_csv_every: int = 1,
        align_to_period: bool = True,
        verbose: bool = True,
        overlap_minutes: int = 5,       # send only the last N minutes each tick
        fallback_tail_rows: int = 500,  # safety if index math fails
        daemon: bool = True,            # thread daemon flag
        thread_name: str = "TwsHistoricUpdater-Loop",
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
        self.overlap_minutes = max(0, int(overlap_minutes))
        self.fallback_tail_rows = max(1, int(fallback_tail_rows))

        self.daemon = bool(daemon)
        self.thread_name = thread_name

        self._loop_count = 0
        self._running = False  # logical running flag
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------- public API ----------
    def start(self) -> None:
        if self.is_running():
            return
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=self.thread_name,
            daemon=self.daemon,
        )
        self._thread.start()

    def run(self) -> None:
        # Synchronous mode – ensure a loop in this thread too (usually main already has one,
        # but this keeps behavior consistent if someone calls run() from another thread).
        if self.is_running():
            return
        self._stop_evt.clear()
        self._running = True
        try:
            self._run_loop()
        finally:
            self._running = False
            self._stop_evt.set()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """
        Signal the loop to stop and join the thread if needed.
        """
        self._running = False
        self._stop_evt.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def is_running(self) -> bool:
        t = self._thread
        return bool(self._running and t and t.is_alive())

    # ---------- internals ----------
    def _run_loop(self) -> None:
        # Ensure an asyncio event loop exists for this thread, required by ib_insync.util.run()
        self._ensure_event_loop()

        try:
            self._tick_once()
            while not self._stop_evt.is_set():
                try:
                    if self.align_to_period:
                        wait = max(0.1, self._seconds_to_next_minute() + 1) #1s buffer
                    else:
                        wait = self.poll_secs
                    self._wait(wait)
                    if self._stop_evt.is_set():
                        break
                    self._tick_once()
                except KeyboardInterrupt:
                    self._log("Interrupted; exiting.")
                    break
                except Exception as e:
                    self._log(f"ERROR: {e!r}")
                    self._wait(2.0)
        finally:
            self._running = False

    def _ensure_event_loop(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    def _tick_once(self) -> None:
        self._loop_count += 1
        try:
            # Run the (blocking) stock.get_historical_data call ON THE IB LOOP
            df = self._call_on_ib_loop(
                self.stock.get_historical_data,
                period=self.period,
                interval=self.interval,
                whatToShow=self.whatToShow,
                useRTH=self.useRTH,
            )
        except Exception as e:
            self._log(f"historical fetch error: {e!r}")
            self._wait(2.0)
            return

        if df is None or df.empty:
            self._log("No data returned.")
            return

        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]

        try:
            if self.overlap_minutes > 0 and len(ohlcv) > 1:
                end_ts = ohlcv.index[-1]
                start_ts = end_ts - pd.Timedelta(minutes=self.overlap_minutes)
                delta = ohlcv.loc[ohlcv.index >= start_ts]
            else:
                delta = ohlcv.tail(self.fallback_tail_rows)
        except Exception:
            delta = ohlcv.tail(self.fallback_tail_rows)

        try:
            self.view.apply_delta_df(delta)
        except Exception as e:
            self._log(f"apply_delta_df error: {e!r}")

        if self.persist_csv_every and (self._loop_count % self.persist_csv_every == 0):
            try:
                self.stock.last_fetch_to_csv(df)
            except Exception as e:
                self._log(f"CSV persist error: {e!r}")

        if self.verbose:
            try:
                last = ohlcv.index[-1]
                self._log(
                    f"Updated history -> rows={len(df)} last={last} "
                    f"interval={self.interval} period={self.period} pushed={len(delta)}"
                )
            except Exception:
                pass

    def _call_on_ib_loop(self, fn, *args, **kwargs):
        """
        Execute a synchronous function that internally uses ib_insync on the IB loop thread
        and return its result to this worker thread.
        """
        # Get the IB instance from your TwsConnection wrapper
        ib = getattr(self.stock.conn, "ib", None)
        if ib is None or not hasattr(ib, "loop"):
            # Fallback: last resort, just call here (not recommended)
            return fn(*args, **kwargs)

        fut: Future = Future()

        def runner():
            try:
                res = fn(*args, **kwargs)
                fut.set_result(res)
            except Exception as e:
                fut.set_exception(e)

        try:
            ib.loop.call_soon_threadsafe(runner)
        except Exception:
            # As a backup, execute inline
            return fn(*args, **kwargs)

        return fut.result()

    def _wait(self, seconds: float) -> None:
        """
        Wait with responsiveness to stop events.
        """
        remaining = float(seconds)
        step = 0.2
        while remaining > 0 and not self._stop_evt.is_set():
            dt = step if remaining >= step else remaining
            time.sleep(dt)
            remaining -= dt

    def _seconds_to_next_minute(self) -> float:
        now = pd.Timestamp.now()
        nxt = (now + pd.Timedelta(minutes=1)).floor("min")
        return max(0.0, (nxt - now).total_seconds())

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[HistoricMarketUpdater] {msg}")



class TwsIntraMinuteUpdater:
    """
    TWS-driven provisional last-candle:
      - Updates ONLY the current minute bar on each poll (e.g., every 5–15s) when price changes.
      - On minute rollover, DESTROYS the prior minute provisional bar; yfinance will later replace it with authoritative data.
      - Never touches closed minutes.

    Usage notes:
      - Subscribe on MAIN THREAD first: stock.get_ticker()
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
    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TwsIntraMinuteUpdater-Loop", daemon=self.daemon)
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
            print(f"[TwsIntraMinuteUpdater] ERROR in loop: {e!r}")

    def _tick(self, *, force: bool) -> None:
        quote = self.stock.get_latest_quote()  # {'minute','bid','ask','last','mid'}
        price = self._pick_price(quote)
        if price is None:
            self._maybe_log(quote, tag="no-price")
            return

        now_minute = quote["minute"]

        # Minute rollover: destroy previous provisional bar, seed new current minute
        if self._minute_key is None or now_minute > self._minute_key:
            prev_minute = self._minute_key
            if prev_minute is not None:
                self._drop_prev_minute(prev_minute)

            self._minute_key = now_minute
            self._o = self._h = self._l = self._c = float(price)

            # Seed cumulative volume at minute start; minute volume starts at 0
            self._cum_vol_seed = self._cum_volume_from_snap(quote)
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
        cur_cum = self._cum_volume_from_snap(quote)
        if cur_cum is not None and self._cum_vol_seed is not None:
            minute_vol = max(0.0, float(cur_cum) - float(self._cum_vol_seed))
            if self._v is None or minute_vol != self._v:
                self._v = minute_vol
                changed = True

        if changed or force:
            self._push_bar(log_reason="update")
        else:
            self._maybe_log(quote, tag="unchanged")

    def _push_bar(self, log_reason: str) -> None:
        ts = self._minute_key
        o = float(self._o); h = float(self._h); l = float(self._l); c = float(self._c); v = float(self._v)

        # If nothing changed vs last pushed close, log a compact duplicate note and return
        if self._last_pushed_close is not None and not self._price_changed(self._c, self._last_pushed_close):
            if self.verbose and self.log_every_secs is not None:
                ts_local = datetime.now().strftime("%H:%M:%S")
                sym = getattr(self.stock, "symbol", None) or "?"
                print(f"{ts_local} [IntraMinuteUpdater:duplicate] {sym} {ts} C={c}")
            return

        self.view.upsert_bar(ts, o, h, l, c, v, provisional=True)
        self._last_pushed_close = c
        if self.verbose:
            ts_local = datetime.now().strftime("%H:%M:%S")
            sym = getattr(self.stock, "symbol", None) or "?"
            print(f"{ts_local} [IntraMinuteUpdater:{log_reason}] {sym} {ts} O={o} H={h} L={l} C={c} V={v}")
        

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

            print(f"[TwsIntraMinuteUpdater] WARN: Could not drop provisional bar {ts} (no supported method).")
        except Exception as e:
            print(f"[TwsIntraMinuteUpdater] drop_prev_minute error for {ts}: {e!r}")

    def _cum_volume_from_snap(self, quote: Dict[str, Any]) -> Optional[float]:
        """
        Return cumulative day volume if available.
        Prefer RTVolume totalVolume when present, else fall back to quote 'vol'.
        IB RTVolume format: "price;size;time;totalVolume;VWAP;singleTrade"
        """
        rv = quote.get("rtVolume")
        if rv:
            try:
                parts = str(rv).split(";")
                if len(parts) >= 4:
                    return float(parts[3])
            except Exception:
                pass
        v = quote.get("vol")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _log_drop(self, ts: pd.Timestamp, *, source: str) -> None:
        if self.verbose:
            ts_local = datetime.now().strftime("%H:%M:%S")
            print(f"{ts_local} [IntraMinuteUpdater:drop-{source}] removed provisional bar {ts}")

    def _maybe_log(self, quote: Optional[Dict[str, Any]], tag: str) -> None:
        if not self.verbose or self.log_every_secs is None:
            return
        now = time.time()
        if (now - self._last_log_ts) < self.log_every_secs:
            return
        self._last_log_ts = now

        ts_local = datetime.now().strftime("%H:%M:%S")
        sym = getattr(self.stock, "symbol", None) or (quote.get("symbol") if quote else None) or "?"
        cur_min = str(self._minute_key) if self._minute_key is not None else "NA"

        if quote is None:
            print(f"{ts_local} [IntraMinuteUpdater:{tag}] {sym} minute={cur_min}")
            return

        b = quote.get("bid"); a = quote.get("ask"); l = quote.get("last"); m = quote.get("mid")
        bs = int(quote.get("bidSize") or 0); as_ = int(quote.get("askSize") or 0); ls = int(quote.get("lastSize") or 0)
        print(f"{ts_local} [IntraMinuteUpdater:{tag}] {sym} minute={cur_min} bid={b}({bs}) ask={a}({as_}) last={l}({ls}) mid={m}")

    def _pick_price(self, quote: Dict[str, Any]) -> Optional[float]:
        def pos(x) -> Optional[float]:
            try:
                xf = float(x)
                if xf > 0 and xf == xf:  # positive and not NaN
                    return xf
            except Exception:
                pass
            return None

        # Prefer mid if both bid/ask are positive
        bid = pos(quote.get("bid"))
        ask = pos(quote.get("ask"))
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            if mid > 0:
                return mid

        # Fall back through preferred keys but only accept positive numbers
        for k in self.price_pref:
            v = pos(quote.get(k))
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