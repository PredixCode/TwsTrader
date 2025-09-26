import pandas as pd
import plotly.graph_objects as go
import os

class FinanceGraph:
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

    def __save_plot(self, output_dir: str="market/charts/"):
        os.makedirs(output_dir, exist_ok=True)
        fig_title = self.fig.layout["title"]["text"]
        output_path = os.path.join(output_dir, f"{fig_title}.html")
        self.fig.write_html(output_path)
        print(f"✅ Success! Interactive chart saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    from yfinance.stock import FinanceStock

    # --- 1. Setup ---
    stock = FinanceStock("RHM.DE")
    # --- 2. Data Preparation ---
    data = stock.get_all_historical_data()
    if data.empty:
        raise SystemExit("Cannot create visualization, no data was fetched.")
    # --- 3. Save Data to CSV ---
    path_to_csv = stock.last_fetch_to_csv()
    # --- 4. Visualize the Data ---
    visualizer = FinanceGraph(csv_file_path=path_to_csv)
    visualizer.plot_candlestick()