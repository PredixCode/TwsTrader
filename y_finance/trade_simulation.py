import pandas as pd

class TradeSimulation:
    """
    A trading simulation environment for backtesting strategies.
    """
    def __init__(self, historical_data: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee: float = 1.0):
        if 'Close' not in historical_data.columns:
            raise ValueError("Historical data must contain a 'Close' column.")
        
        self.data = historical_data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # --- State Variables ---
        self.current_step = 0
        self.current_balance = initial_balance
        self.shares_held = 0.0
        self.total_fees_paid = 0.0
        self.trade_history = []
        
        print(f"Trader initialized. Starting Balance: €{self.initial_balance:,.2f}, Transaction Fee: €{self.transaction_fee:.2f}")

    def _get_current_price(self) -> float:
        """Returns the closing price at the current time step."""
        return self.data['Close'].iloc[self.current_step]

    @property
    def portfolio_value(self) -> float:
        """Calculates the total current value of the portfolio (cash + assets)."""
        return self.current_balance + (self.shares_held * self._get_current_price())

    @property
    def is_insolvent(self) -> bool:
        """
        Checks if the agent is in an unrecoverable state (no shares and not enough cash to buy).
        The smallest possible action is a buy, which requires at least the transaction fee and 1€ in funds.
        """
        return self.shares_held == 0 and self.current_balance < self.transaction_fee + 1 

    @property
    def total_steps(self) -> int:
        return len(self.data)

    def buy(self, amount_in_currency: float):
        """
        Executes a buy order with a specific amount of currency.
        """
        
        if not self.is_buy_valid(amount_in_currency):
            return False

        price = self._get_current_price()
        total_cost = amount_in_currency + self.transaction_fee

        self.current_balance -= total_cost
        shares_bought = amount_in_currency / price
        self.shares_held += shares_bought
        self.total_fees_paid += self.transaction_fee
        
        log_entry = {
            "step": self.current_step,
            "action": "BUY",
            "price": price,
            "shares": shares_bought,
            "cost": total_cost,
            "portfolio_value": self.portfolio_value
        }
        self.trade_history.append(log_entry)
        #print(f"Step {self.current_step}: BOUGHT {shares_bought:.4f} shares @ €{price:,.2f} "
                #f"| New Balance: €{self.current_balance:,.2f}"
                #f"| Total Fees Paid: €{self.total_fees_paid:,.2f}")
        return True
    
    def is_buy_valid(self, amount_in_currency: float) -> bool:
        if amount_in_currency <= 0:
            return False

        total_cost = amount_in_currency + self.transaction_fee

        if total_cost > self.current_balance:
            #print(f"Step {self.current_step}: BUY FAILED - Insufficient funds. "
                    #f"Need €{total_cost:,.2f}, have €{self.current_balance:,.2f}")
            return False
        return True

    def sell(self, amount_in_shares: float):
        """
        Executes a sell order with a specific number of shares.
        """

        if not self.is_sell_valid(amount_in_shares):
            return False
        
        price = self._get_current_price()
        revenue = amount_in_shares * price
        net_revenue = revenue - self.transaction_fee

        self.shares_held -= amount_in_shares
        self.current_balance += net_revenue
        self.total_fees_paid += self.transaction_fee

        log_entry = {
            "step": self.current_step,
            "action": "SELL",
            "price": price,
            "shares": amount_in_shares,
            "revenue": net_revenue,
            "portfolio_value": self.portfolio_value
        }
        self.trade_history.append(log_entry)
        #print(f"Step {self.current_step}: SOLD {amount_in_shares:.4f} shares @ €{price:,.2f} "
                #f"| New Balance: €{self.current_balance:,.2f}")
        return True

    def is_sell_valid(self, amount_in_shares: float) -> bool:
        if amount_in_shares > self.shares_held or amount_in_shares == 0:
            #print(f"Step {self.current_step}: SELL FAILED - Not enough shares. "
                    #f"Trying to sell {amount_in_shares:.4f}, have {self.shares_held:.4f}")
            return False

        #price = self._get_current_price()
        #revenue = amount_in_shares * price

        #if revenue < self.transaction_fee:
            # This sale would lose money, so it's an invalid action.
            #print(f"Step {self.current_step}: SELL FAILED - Sell is not profitable. "
                #f"Would create {revenue - self.transaction_fee:.4f} in losses")
            #return False
        return True

    def hold(self):
        """
        Represents the action of doing nothing at the current step.
        """
        #print(f"Step {self.current_step}: HOLD")
        return True

    def next_step(self) -> bool:
        """
        Advances the simulation to the next time step.

        Returns:
            bool: True if the simulation can continue, False if it has ended.
        """
        if self.current_step >= len(self.data) - 1:
            print("\n--- End of historical data reached. Simulation finished. ---")
            return False
        
        self.current_step += 1
        return True

    def print_summary(self):
        """Prints a final summary of the trading simulation's performance."""
        final_value = self.portfolio_value
        profit_loss = final_value - self.initial_balance
        profit_percent = (profit_loss / self.initial_balance) * 100

        print("\n" + "="*40)
        print("          TRADING SIMULATION SUMMARY")
        print("="*40)
        print(f"Initial Balance:    €{self.initial_balance:15,.2f}")
        print(f"Final Portfolio Value: €{final_value:15,.2f}")
        print(f"Total Fees Paid:      €{self.total_fees_paid:15,.2f}")
        print("-" * 40)
        print(f"Net Profit/Loss:      €{profit_loss:15,.2f}")
        print(f"Return on Investment: {profit_percent:14,.2f}%")
        print("="*40)