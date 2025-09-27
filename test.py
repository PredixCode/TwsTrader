# YFINANCE
#from y_finance.stock import FinanceStock
#stock = FinanceStock("RHM.DE")
#print(stock.live_price)

# GRAPH
from lightweight_charts import Chart
from y_finance.stock import FinanceStock
chart = Chart()
stock = FinanceStock("RHM.DE")
df = stock.get_historical_data()
chart.set(df)
chart.show(block=True)