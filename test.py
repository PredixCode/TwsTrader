import multiprocessing as mp

from y_finance.stock import FinanceStock
from y_finance.graph import  LightweightGraph

def local_graph(stock):
    graph = LightweightGraph(stock)
    graph.show(block=True)    # or False + input("Press Enter to exit...")

if __name__ == "__main__":
    stock = FinanceStock("RHM.DE")
    mp.freeze_support()
    local_graph(stock)
