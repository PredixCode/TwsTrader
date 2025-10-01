import threading
from typing import Optional, List, Dict, Iterable
import pandas as pd
from lightweight_charts import Chart

from core.normalize import normalize_df


class LiveGraph:
    """
    Passive chart view for OHLC data (1-minute or higher timeframe).
    - No internal fetching or threads.
    - Accepts initial DataFrame and incremental updates from the outside.
    - Keeps an internal copy of the DataFrame for convenience and labels.
    """

    def __init__(self, title: str, initial_df: pd.DataFrame) -> None:
        """
        Args:
            title: Chart title.
            initial_df: DataFrame indexed by datetime-like, with columns Open/High/Low/Close
                        (case-insensitive; will be normalized).
        """
        self._chart_lock = threading.RLock()

        # Data
        self.dataframe = normalize_df(initial_df)

        # Series + trade markers state
        self.chart: Optional[Chart] = None
        self.candles = None
        self.volume = None
        self._markers: List[Dict] = []
        self._markers_lock = threading.Lock()

        # Build chart and render initial data
        self._construct_chart(title)
        self.set_data(self.dataframe)

    # ---------- Public API ----------

    def show(self, block: bool = True) -> None:
        with self._chart_lock:
            if self.chart:
                self.chart.show(block=block)

    def set_data(self, df: pd.DataFrame) -> None:
        with self._chart_lock:
            self.dataframe = df
            if self.candles and hasattr(self.candles, "set_data"):
                self.candles.set_data(self._df_to_lw_bars(df))
            elif self.chart and hasattr(self.chart, "set"):
                self.chart.set(df)

            # NEW: push volume data if series exists
            if self.volume and hasattr(self.volume, "set_data") and "Volume" in df.columns:
                try:
                    self.volume.set_data(self._df_to_volume_bars(df))
                except Exception:
                    pass

            self._apply_markers_locked()

    def upsert_bar(
        self,
        when,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float | None = None,   # NEW
        *,
        floor_to_minute: bool = True
    ) -> None:
        """
        Insert or update a single bar by timestamp.
        Args:
            when: datetime-like index key (will be normalized).
            o,h,l,c: OHLC values to set.
            floor_to_minute: if True, floor timestamp to minute for 1m bars.
        """
        ts = pd.Timestamp(when)
        if floor_to_minute:
            ts = ts.floor("min")

        with self._chart_lock:
            if "Volume" not in self.dataframe.columns:
                self.dataframe["Volume"] = 0.0

            vals = [float(o), float(h), float(l), float(c)]
            if v is not None:
                self.dataframe.loc[ts, ["Open", "High", "Low", "Close", "Volume"]] = vals + [float(v)]
            else:
                self.dataframe.loc[ts, ["Open", "High", "Low", "Close"]] = vals

            self.dataframe.sort_index(inplace=True)

            # Update candle series
            bar = {
                "time": ts.to_pydatetime(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }

            pushed = False
            if self.candles and hasattr(self.candles, "update"):
                try:
                    self.candles.update(bar)
                    pushed = True
                except Exception:
                    pushed = False

            # Volume update (only if provided and series exists)
            if v is not None and self.volume and hasattr(self.volume, "update"):
                try:
                    self.volume.update({"time": ts.to_pydatetime(), "value": float(v)})
                except Exception:
                    # If live update failed, fall back to full redraw
                    try:
                        if hasattr(self.volume, "set_data"):
                            self.volume.set_data(self._df_to_volume_bars(self.dataframe))
                    except Exception:
                        pass

            if not pushed:
                # Fallback to bulk reset for candles (and volume will be correct via set_data)
                try:
                    if self.candles and hasattr(self.candles, "set_data"):
                        self.candles.set_data(self._df_to_lw_bars(self.dataframe))
                    elif self.chart and hasattr(self.chart, "set"):
                        self.chart.set(self.dataframe)
                except Exception:
                    pass

    def drop_bar(self, when, *, floor_to_minute=True) -> None:
        ts = pd.Timestamp(when)
        if floor_to_minute:
            ts = ts.floor("min")
        with self._chart_lock:
            if self.dataframe is None or self.dataframe.empty or ts not in self.dataframe.index:
                return
            self.dataframe = self.dataframe.drop(index=[ts])
            # Re-render both series
            try:
                if self.candles and hasattr(self.candles, "set_data"):
                    self.candles.set_data(self._df_to_lw_bars(self.dataframe))
                elif self.chart and hasattr(self.chart, "set"):
                    self.chart.set(self.dataframe)
            except Exception:
                pass
            if self.volume and hasattr(self.volume, "set_data") and "Volume" in self.dataframe.columns:
                try:
                    self.volume.set_data(self._df_to_volume_bars(self.dataframe))
                except Exception:
                    pass
            self._apply_markers_locked()

    def add_trade_label(
        self,
        when,
        side: str,
        price: Optional[float] = None,
        text: Optional[str] = None,
        *,
        use_marker: bool = False,        # markers are hover-only; leave False to rely on visible lines/segments
        show_price_label: bool = True,   # horizontal line with label at trade price
        show_price_stub: bool = True,    # short segment at candle time
        stub_bars: int = 2,
        line_style: str = 'dashed',
        line_width: int = 1,
    ) -> None:
        """
        Visible, price-anchored label at trade time:
          - optional marker (hover-only)
          - horizontal line at price with label
          - short segment starting at the candle time
        """
        with self._chart_lock:
            if self.dataframe is None or len(self.dataframe) == 0:
                return

            if when is None:
                when = self.dataframe.index.max()

            time_dt = self._to_marker_time(when)
            side_norm = str(side).lower()
            is_buy = side_norm.startswith("b")
            color = "#3b82f6" if is_buy else "#f59e0b"

            if text is None:
                base = 'BUY' if is_buy else 'SELL'
                text = f"{base} @ {float(price):.2f}" if price is not None else base

            # 1) Optional classic marker (hover-only)
            if use_marker:
                marker = {
                    "time": time_dt,
                    "position": "belowBar" if is_buy else "aboveBar",
                    "color": color,
                    "shape": "arrowUp" if is_buy else "arrowDown",
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

            # 2) Horizontal price line with label
            if show_price_label and price is not None:
                try:
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

            # 3) Short segment anchored at the candle time
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

    def scroll_to_latest(self) -> None:
        # Manual-only method; call this from UI/user action if you want to jump right.
        with self._chart_lock:
            self._scroll_to_latest_locked()

    # ---------- Internals ----------

    def _construct_chart(self, title: str) -> None:
        with self._chart_lock:
            self.chart = Chart(title=title, maximize=True)

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
                    "lockVisibleTimeRangeOnResize": False,
                    "shiftVisibleRangeOnNewBar": False,  # prevents auto-shift on new bars
                },
                "rightPriceScale": {
                    "autoScale": False,  # prevents auto price rescale
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

            # Create candlestick series and keep handle
            if hasattr(self.chart, "add_candlestick_series"):
                self.candles = self.chart.add_candlestick_series()
            else:
                # Fallback for older wrappers will rely on chart.set()
                self.candles = getattr(self.chart, "candlestick_series", None)

            if hasattr(self.chart, "add_histogram_series"):
                try:
                    # Basic volume histogram; adjust options if you want separate scale/pane
                    self.volume = self.chart.add_histogram_series()
                except Exception:
                    self.volume = None

    @staticmethod
    def _df_to_lw_bars(df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to a list of dicts for lightweight-charts.
        """
        bars: List[Dict] = []
        if df is None or df.empty:
            return bars
        for ts, row in df.iterrows():
            bars.append({
                "time": pd.Timestamp(ts).to_pydatetime(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            })
        return bars
    
    def _df_to_volume_bars(self, df: pd.DataFrame):
        bars = []
        if df is None or df.empty or "Volume" not in df.columns:
            return bars
        for ts, row in df.iterrows():
            try:
                bars.append({
                    "time": pd.Timestamp(ts).to_pydatetime(),
                    "value": float(row["Volume"]),
                })
            except Exception:
                continue
        return bars

    def _apply_markers_locked(self) -> None:
        """
        Re-apply stored markers after any chart.set / data reset.
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
        self._clear_markers_on_chart_locked()
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

    def _clear_markers_on_chart_locked(self) -> None:
        for name in ("clear_markers", "clearMarkers"):
            fn = getattr(self.chart, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

    def _fit_or_scroll_to_latest_locked(self) -> None:
        """
        No-op: previously fit content or scrolled to latest.
        Left in place for compatibility but intentionally does nothing.
        """
        return

    def _scroll_to_latest_locked(self) -> None:
        # Retained ONLY for manual calls via scroll_to_latest()
        ts = self.chart.time_scale() if callable(self.chart.time_scale) else self.chart.time_scale
        for mname in ("scroll_to_real_time", "scrollToRealTime"):
            m = getattr(ts, mname, None)
            if callable(m):
                try:
                    m()
                    break
                except Exception:
                    pass

    def _offset_time_by_bars(self, time_dt, bars: int):
        """
        Returns the timestamp bars steps to the right within the existing index.
        Falls back to the original time if we can't find a clean offset.
        """
        try:
            t = pd.Timestamp(time_dt)
            idx = self.dataframe.index
            if t in idx:
                pos = idx.get_loc(t)
                new_pos = min(pos + max(bars, 0), len(idx) - 1)
                return idx[new_pos].to_pydatetime()
        except Exception:
            pass
        return time_dt

    @staticmethod
    def _to_marker_time(when):
        """
        Return a Python datetime (tz-aware if available) for marker placement.
        """
        if isinstance(when, pd.Timestamp):
            return when.to_pydatetime()
        if hasattr(when, "isoformat"):
            return when
        if isinstance(when, (int, float)):
            from datetime import datetime, timezone as _tz
            return datetime.fromtimestamp(float(when), tz=_tz.utc)
        try:
            ts = pd.Timestamp(when)
            return ts.to_pydatetime()
        except Exception:
            from datetime import datetime
            return datetime.now()