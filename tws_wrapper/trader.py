import time
from ib_insync import MarketOrder, LimitOrder, Trade

from tws_wrapper.connection import TwsConnection
from tws_wrapper.stock import TwsStock


class TwsTrader:
    def __init__(self, stock: TwsStock, timeout:float = 30.0):
        self.stock = stock
        self.stock.get_ticker()
        self.timeout:int = timeout

    def buy(self,
        quantity: float|int, order_type: str = 'MKT', limit_price: float|None = None,
        tif: str = 'DAY', outsideRTH: bool = False, account: str|None = None, wait: bool = True
        ) -> Trade:
        """
        Place a BUY order. order_type: 'MKT' or 'LMT'.
        If order_type == 'LMT', provide limit_price.
        """
        return self._place_order(
            action='BUY', quantity=quantity, order_type=order_type, limit_price=limit_price,
            tif=tif, outsideRTH=outsideRTH, account=account,
            wait=wait)


    def sell( self,
        quantity: float|int, order_type: str = 'MKT', limit_price: float|None = None,
        tif: str = 'DAY', outsideRTH: bool = False, account: str|None = None, wait: bool = True
        ) -> Trade:
        """
        Place a SELL order. order_type: 'MKT' or 'LMT'.
        If order_type == 'LMT', provide limit_price.
        """
        return self._place_order(
            action='SELL', quantity=quantity, order_type=order_type, limit_price=limit_price,
            tif=tif, outsideRTH=outsideRTH, account=account, 
            wait=wait)


    def cancel_all_for_contract(self):
        ib = self.stock.conn.connect()
        for tr in ib.trades():
            if tr.contract.conId == self.stock.contract.conId and not tr.isDone():
                ib.cancelOrder(tr.order)

    def _place_order(
        self,
        action: str,                         # 'BUY' or 'SELL'
        quantity: float|int,
        order_type: str = 'MKT',            # 'MKT' or 'LMT'
        limit_price: float|None = None,     # required if LMT
        tif: str = 'DAY',                   # e.g., 'DAY', 'GTC'
        outsideRTH: bool = False,
        account: str|None = None,           # if None, use first managed account
        wait: bool = True,
        ) -> Trade:
        """
        Core order placement. Returns the Trade object.
        If wait=True, waits up to `timeout` seconds for completion (filled/cancelled).
        """
        if action not in ('BUY', 'SELL'):
            raise ValueError("action must be 'BUY' or 'SELL'")
        if quantity is None or float(quantity) <= 0:
            raise ValueError("quantity must be a positive number")

        ib = self.stock.conn.connect()
        contract = self.stock.qualify()

        order_type = order_type.upper()
        if order_type == 'MKT':
            order = MarketOrder(action, quantity, tif=tif, outsideRth=outsideRTH)
        elif order_type == 'LMT':
            if limit_price is None:
                raise ValueError("limit_price must be provided for LMT orders")
            order = LimitOrder(action, quantity, limit_price, tif=tif, outsideRth=outsideRTH)
        else:
            raise ValueError("order_type must be 'MKT' or 'LMT'")

        # Resolve account if not provided
        acct = account
        if acct is None:
            accounts = ib.managedAccounts()
            if accounts:
                acct = accounts[0]
        if acct is not None:
            order.account = acct

        trade: Trade = ib.placeOrder(contract, order)

        if wait:
            self._wait_for_trade(trade)
        return trade

    def _wait_for_trade(self, trade: Trade) -> None:
        """
        Wait until trade is done (filled/cancelled) or timeout.
        Raises TimeoutError if not done in time.
        """
        deadline = time.time() + self.timeout
        while not trade.isDone() and time.time() < deadline:
            self.stock.conn.sleep(0.2)

        if not trade.isDone():
            raise TimeoutError(f"Order {trade.order.orderId} not done within {self.timeout}s")

    @staticmethod
    def trade_summary(trade: Trade) -> dict:
        """
        Compact dict with the most relevant fields.
        """
        os = trade.orderStatus
        return {
            'orderId': trade.order.orderId,
            'status': os.status,
            'filled': os.filled,
            'remaining': os.remaining,
            'avgFillPrice': os.avgFillPrice,
            'lastFillPrice': getattr(os, 'lastFillPrice', None)
        }