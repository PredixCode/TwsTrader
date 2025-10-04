import pandas as pd


class TradeBot:
    def __init__(self):
        self.dataframe = None

    def set_data(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe