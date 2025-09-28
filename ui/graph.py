import time
import threading
import random
from typing import Optional, List, Dict
import pandas as pd
from lightweight_charts import Chart

from yfinance_wrapper.stock import FinanceStock


class LiveGraph:
    def __init__(self, stock: FinanceStock) -> None:
        # Get stock data
        self.stock = stock
        df = self.stock.get_accurate_max_historical_data()
        if df is None or df.empty:
            raise ValueError("No stock data returned, can't construct chart.")
        self.dataframe = df.sort_index()
        self.stock.last_fetch_to_csv()

        # Runtime state for auto-update
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Series + trade markers state
        self.candles = None
        self._markers: List[Dict] = []
        self._markers_lock = threading.Lock()

        # Build chart
        self.__construct_chart()

    def show(self, block: bool = True) -> None:
        self.chart.show(block=block)

    def __construct_chart(self) -> Chart:
        self.chart = Chart(title=self.stock.name, maximize=True)

        self.chart.topbar.visible = False
        self.chart.layout(background_color="#0e1117", text_color="#d1d5db")
        self.chart.options = {
            "timeScale": {
                "timeVisible": True,
                "secondsVisible": False,
                "rightOffset": 0,
                "barSpacing": 1.0,
                "fixLeftEdge": False,
                "fixRightEdge": False,
                "lockVisibleTimeRangeOnResize": False
            },
            "rightPriceScale": {
                "autoScale": True,
                "scaleMargins": {"top": 0.05, "bottom": 0.05},
                "entireTextOnly": False,
                "borderVisible": True
            },
            "handleScale": {
                "mouseWheel": True,
                "pinch": True,
                "axisPressedMouseMove": {"time": True, "price": True},
            },
            "handleScroll": {"mouseWheel": True, "pressedMouseMove": True},
            "crosshair": {"mode": 1},
            "grid": {
                "vertLines": {"visible": True, "color": "#2a2e39", "style": 0},
                "horzLines": {"visible": True, "color": "#2a2e39", "style": 0}
            }
        }

        # PREFERRED: create/own the candlestick series and set data explicitly
        if hasattr(self.chart, "add_candlestick_series"):
            self.candles = self.chart.add_candlestick_series()
            self.candles.set_data(self._df_to_lw_bars(self.dataframe))
        else:
            # Fallback to chart.set (older wrappers), try to discover the series if exposed
            self.chart.set(self.dataframe)
            self.candles = getattr(self.chart, "candlestick_series", None)

        # Ensure full content is visible initially
        ts = self.chart.time_scale() if callable(self.chart.time_scale) else self.chart.time_scale
        try:
            ts.fit_content()
        except Exception:
            pass

        # Apply any existing markers (none initially)
        self._apply_markers()

        return self.chart

    # -------- Auto-update --------

    def start_auto_update(
        self,
        interval_sec: int = 60,
        period: str = "1d",
        interval: str = "1m",
        persist_csv_every: int = 30
    ) -> None:
        if self._update_thread and self._update_thread.is_alive():
            return  # already running

        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            args=(interval_sec, period, interval, persist_csv_every),
            daemon=True,
        )
        self._update_thread.start()

    def stop_auto_update(self) -> None:
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=5)

    def _update_loop(self, interval_sec: int, period: str, interval: str, persist_csv_every: int) -> None:
        updates = 0
        backoff = interval_sec

        time.sleep(backoff)  # __init__ already fetched; wait until next tick
        while not self._stop_event.is_set():
            try:
                recent_df = self.stock.get_historical_data(period=period, interval=interval)
                if recent_df is None or recent_df.empty:
                    time.sleep(backoff)
                    continue

                recent_df = recent_df.sort_index()
                merged = (
                    pd.concat([self.dataframe, recent_df], axis=0)
                    .sort_index()
                )
                merged = merged[~merged.index.duplicated(keep="last")]

                last_idx = self.dataframe.index.max() if len(self.dataframe) else None
                delta_df = merged if last_idx is None else merged.loc[merged.index > last_idx]

                if not delta_df.empty:
                    self.dataframe = merged
                    try:
                        self.stock.last_fetch_to_csv(self.dataframe)
                    except Exception:
                        pass

                    pushed = False
                    # Prefer updating the owned series if possible
                    try:
                        if self.candles and hasattr(self.candles, "update"):
                            for bar in self._df_to_lw_bars(delta_df):
                                self.candles.update(bar)
                            pushed = True
                    except Exception:
                        pushed = False

                    if not pushed:
                        # Fallback to bulk reset on chart
                        try:
                            if hasattr(self.chart, "update") and callable(self.chart.update):
                                self.chart.update(delta_df)
                            else:
                                # final fallback
                                if self.candles and hasattr(self.candles, "set_data"):
                                    self.candles.set_data(self._df_to_lw_bars(self.dataframe))
                                else:
                                    self.chart.set(self.dataframe)
                        except Exception:
                            # last resort
                            try:
                                self.chart.set(self.dataframe)
                            except Exception:
                                pass

                    # Re-apply markers after any reset
                    self._apply_markers()

                    # Keep view pinned to the latest candle
                    ts = self.chart.time_scale() if callable(self.chart.time_scale) else self.chart.time_scale
                    for mname in ("scroll_to_real_time", "scrollToRealTime"):
                        m = getattr(ts, mname, None)
                        if callable(m):
                            try:
                                m()
                                break
                            except Exception:
                                pass

                    updates += 1
                    if persist_csv_every and updates % persist_csv_every == 0:
                        try:
                            self.stock.last_fetch_to_csv(self.dataframe)
                        except Exception:
                            pass

                backoff = interval_sec

            except Exception:
                backoff = min(backoff * 2, max(5 * interval_sec, 300))

            time.sleep(backoff)

    def add_trade_label(
        self,
        when,
        side: str,
        price: Optional[float] = None,
        text: Optional[str] = None,
        *,
        use_marker: bool = False,        # markers are hover-only; leave False if you just want visible labels
        show_price_label: bool = True,   # horizontal line with text at the trade price
        show_price_stub: bool = True,    # short segment at the candle to anchor visually
        stub_bars: int = 2,              # length of the short segment
        line_style: str = 'dashed',      # 'solid' | 'dashed' | 'dotted'
        line_width: int = 1,
    ) -> None:
        """
        Adds a clear trade annotation with a price-anchored label:
          - optional marker (hover text only)
          - horizontal line at the trade price with text label (always visible)
          - short trend line segment at the trade bar to make it feel anchored next to the candle
        """
        if when is None:
            if len(self.dataframe) == 0:
                return
            when = self.dataframe.index.max()

        time_dt = self._to_marker_time(when)
        side_norm = str(side).lower()
        is_buy = side_norm.startswith("b")
        color = "#3b82f6" if is_buy else "#f59e0b"

        # Build label text
        if text is None:
            text = 'BUY' if is_buy else 'SELL'
            if price is not None:
                text = f"{text} @ {float(price):.2f}"

        # 1) Optional classic marker (hover-only text; omit if you don't need it)
        if use_marker:
            marker = {
                "time": time_dt,
                "position": "aboveBar" if not is_buy else "belowBar",
                "color": color,
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": text,
            }
            with self._markers_lock:
                self._markers.append(marker)
            try:
                self.chart.marker(**marker)
            except Exception:
                try:
                    m2 = dict(marker)
                    m2["time"] = time_dt.isoformat()
                    self.chart.marker(**m2)
                except Exception:
                    pass

        # 2) Always-visible price-anchored label via horizontal line
        if show_price_label and price is not None:
            try:
                # Draw a horizontal line with a visible label (title)
                self.chart.horizontal_line(
                    price=float(price),
                    color=color,
                    width=line_width,
                    style=line_style,
                    text=text,
                    axis_label_visible=True
                )
            except Exception:
                pass

        # 3) Short segment at the candle time to make it feel attached to that bar
        if show_price_stub and price is not None and stub_bars > 0:
            try:
                end_time = self._offset_time_by_bars(time_dt, stub_bars)
                self.chart.trend_line(
                    start_time=time_dt,
                    start_value=float(price),
                    end_time=end_time,
                    end_value=float(price),
                    color=color,
                    width=line_width,
                    style=line_style,
                    round=False
                )
            except Exception:
                pass

    def _offset_time_by_bars(self, time_dt, bars: int):
        """
        Returns the timestamp bars steps to the right within the existing index.
        Falls back to the original time if we can't find a clean offset.
        """
        try:
            # Convert to Timestamp for lookup
            t = pd.Timestamp(time_dt)
            idx = self.dataframe.index
            if t in idx:
                pos = idx.get_loc(t)
                new_pos = min(pos + max(bars, 0), len(idx) - 1)
                return idx[new_pos].to_pydatetime()
        except Exception:
            pass
        return time_dt

    def add_random_trade_labels(self, count: int = 20, seed: Optional[int] = 42) -> None:
        """
        Scatter random buy/sell markers across existing candles for testing.
        Uses datetime from the DataFrame index (no unix conversion).
        """
        if len(self.dataframe) == 0 or count <= 0:
            return

        rng = random.Random(seed)
        idx = list(self.dataframe.index)
        n = len(idx)
        if n == 0:
            return

        picks = [idx[rng.randrange(n)] for _ in range(count)]
        for t in picks:
            side = "buy" if rng.random() < 0.5 else "sell"
            self.add_trade_label(t, side=side, text=("B" if side == "buy" else "S"))

    # -------- Internals for markers --------

    def _apply_markers(self) -> None:
        """
        Re-apply stored markers after any chart.set / data reset.
        Ensures chronological order and correct time type (datetime/str).
        """
        with self._markers_lock:
            markers = list(self._markers)

        # Normalize times and sort chronologically
        norm_markers = []
        for m in markers:
            m2 = dict(m)
            m2["time"] = self._to_marker_time(m2["time"])
            norm_markers.append(m2)
        norm_markers.sort(key=lambda m: m["time"])

        # Clear then add one by one
        self._clear_markers_on_chart()
        for m in norm_markers:
            try:
                self.chart.marker(**m)
            except Exception:
                try:
                    m2 = dict(m)
                    m2["time"] = m2["time"].isoformat()
                    self.chart.marker(**m2)
                except Exception:
                    pass

    def _clear_markers_on_chart(self) -> None:
        """
        Use the wrapper's clear_markers() API if available.
        """
        for name in ("clear_markers", "clearMarkers"):
            fn = getattr(self.chart, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

    @staticmethod
    def _to_marker_time(when):
        """
        Return a Python datetime (tz-aware if available) for marker placement.
        This matches what lightweight-charts-python expects.
        """
        if isinstance(when, pd.Timestamp):
            # preserve tz if present; convert to python datetime
            return when.to_pydatetime()
        # datetime-like
        if hasattr(when, "isoformat"):
            return when
        # numeric epoch -> convert to UTC datetime
        if isinstance(when, (int, float)):
            from datetime import datetime, timezone as _tz
            return datetime.fromtimestamp(float(when), tz=_tz.utc)
        # strings -> parse via pandas, then to python datetime
        try:
            ts = pd.Timestamp(when)
            return ts.to_pydatetime()
        except Exception:
            # fallback to "now"
            from datetime import datetime
            return datetime.now()