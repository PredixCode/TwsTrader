import os
import pandas as pd
import plotly.graph_objects as go
from lightweight_charts import Chart


from yfinance_wrapper.stock import FinanceStock


class WebGraph:
    """
    A class to create financial visualizations from market data stored in CSV files.
    """    
    def __init__(self, csv_file_path: str):
        """
        Initializes the Visualizer by loading and preparing data from a CSV file.

        Args:
            csv_file_path (str): The full path to the market data CSV file.
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"The specified file was not found: {csv_file_path}")

        self.csv_path = csv_file_path
        self.stock_name = os.path.basename(csv_file_path).replace('.csv', '')
        self.fig = None
        
        try:
            # Load the data, critically setting the 'Datetime' column as a parsed index
            self.data = pd.read_csv(
                csv_file_path, 
                index_col='Datetime', 
                parse_dates=True
            )
            print(f"Visualizer initialized for '{self.stock_name}'. Loaded {len(self.data)} data points.")
        except Exception as e:
            print(f"❌ Error loading or parsing CSV file: {e}")
            self.data = pd.DataFrame()

    def plot_candlestick(self):
        """
        Generates and saves an interactive candlestick chart from the loaded data.
        """
        if self.data.empty:
            print("Data is empty, cannot generate plot.")
            return

        print(f"Generating candlestick chart for {self.stock_name}...")

        # Create the main Candlestick trace
        candlestick_trace = go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='Price'
        )

        # Create the Figure object with our trace
        self.fig = go.Figure(data=[candlestick_trace])

        # Update the layout for a professional look and feel
        self.fig.update_layout(
            title=f'{self.stock_name} Candlestick Chart',
            yaxis_title='Stock Price',
            xaxis_title='Date',
            # Hide the rangeslider for a cleaner look
            xaxis_rangeslider_visible=False,
            # Use a dark theme popular in trading applications
            template='plotly_dark',
            font=dict(family="Arial, sans-serif", size=12, color="white")
        )

        # remove non-trading periods from the x-axis
        self.__remove_non_trading_periods()
        self.fig.show()
        self.__save_plot()

    def __remove_non_trading_periods(self):
        self.fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # Hides weekends
                dict(pattern="hour", bounds=[16, 9.5]) # Hides non-trading hours (4 PM to 9:30 AM)
            ]
        )
        return self.fig

    def __save_plot(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = f"{file_dir}\charts"
        os.makedirs(output_dir, exist_ok=True)
        fig_title = self.fig.layout["title"]["text"]
        output_path = os.path.join(output_dir, f"{fig_title}.html")
        self.fig.write_html(output_path)
        print(f"✅ Success! Interactive chart saved to: {os.path.abspath(output_path)}")



class LightweightGraph:
    def __init__(self, stock: FinanceStock) -> None:
        # Get stock data
        self.stock = stock
        df = self.stock.get_all_historical_data()
        if df is None or df.empty:
            raise ValueError("No stock data returned, can't construct chart.")
        self.dataframe = df
        self.stock.last_fetch_to_csv()
        
        # Build chart
        self.__construct_chart()

    def show(self, block: bool = True) -> None:
        self.chart.show(block=block)

    def  __construct_chart(self) -> Chart:
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

        # Set data first
        self.chart.set(self.dataframe)

        # IMPORTANT: time_scale is a function → call it to get the API object
        ts = self.chart.time_scale() if callable(self.chart.time_scale) else self.chart.time_scale

        # Try canonical "show all" first
        try:
            ts.fit_content()
            return self.chart
        except Exception:
            pass

        # Fallbacks for different wrapper method names/APIs
        def call_first(obj, names, *args, **kwargs):
            for name in names:
                m = getattr(obj, name, None)
                if callable(m):
                    return m(*args, **kwargs)
            raise AttributeError(f"None of {names} exist on {obj}")

        # Fallback by logical range
        n = len(self.dataframe)
        try:
            call_first(
                ts,
                ["set_visible_logical_range", "setVisibleLogicalRange"],
                {"from": 0, "to": max(0, n - 1)},
            )
            return self.chart
        except Exception:
            pass

        # Fallback by timestamp range
        try:
            t0 = self.dataframe.index.min()
            t1 = self.dataframe.index.max()
            if hasattr(t0, "to_pydatetime"):
                t0, t1 = t0.to_pydatetime(), t1.to_pydatetime()
            call_first(
                ts,
                ["set_visible_range", "setVisibleRange"],
                {"from": t0, "to": t1},
            )
        except Exception:
            # If everything fails, you’ll still have the default view.
            pass
