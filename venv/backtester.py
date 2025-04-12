import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

class Backtester:
    def __init__(self, initial_balance=1000, fee_rate=0.0006):
        """
        Initialize the backtester with starting capital and fee structure
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate  # 0.06% trading fee
        
    def run_backtest(self, predictions, prices, dates=None):
        """
        Run backtest simulation based on model predictions
        
        Args:
            predictions: Array of predicted classes (0=Sell, 1=Hold, 2=Buy)
            prices: Price data corresponding to each prediction
            dates: Optional datetime index for the backtest periods
        
        Returns:
            Dictionary of backtest results
        """
        # Initialize portfolio state
        balance = self.initial_balance
        btc = 0
        portfolio_values = []
        positions = []  # 'cash' or 'btc'
        trade_history = []
        
        # Track metrics
        trades = 0
        last_action = None
        
        # Run simulation
        for i in range(len(predictions)):
            price = prices[i]
            action = predictions[i]
            
            # Convert prediction class (0=Sell, 1=Hold, 2=Buy)
            if action == 2 and balance > 0:  # BUY
                btc_amount = (balance * (1 - self.fee_rate)) / price
                trade_details = {
                    'timestamp': dates[i] if dates is not None else i,
                    'action': 'BUY',
                    'price': price,
                    'amount': btc_amount,
                    'value': balance,
                    'fee': balance * self.fee_rate
                }
                trade_history.append(trade_details)
                
                btc = btc_amount
                balance = 0
                trades += 1
                last_action = 'BUY'
                positions.append('btc')
                
            elif action == 0 and btc > 0:  # SELL
                sale_value = btc * price
                balance_after_fee = sale_value * (1 - self.fee_rate)
                
                trade_details = {
                    'timestamp': dates[i] if dates is not None else i,
                    'action': 'SELL',
                    'price': price,
                    'amount': btc,
                    'value': sale_value,
                    'fee': sale_value * self.fee_rate
                }
                trade_history.append(trade_details)
                
                balance = balance_after_fee
                btc = 0
                trades += 1
                last_action = 'SELL'
                positions.append('cash')
            else:
                # Hold current position
                positions.append('btc' if btc > 0 else 'cash')
            
            # Calculate portfolio value at this step
            portfolio_value = balance + (btc * price)
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = ((final_value - self.initial_balance) / self.initial_balance)
        total_return_pct = total_return * 100
        
        # Trade frequency (trades per period)
        trade_freq = trades / len(predictions)
        
        # Calculate returns and volatility
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        daily_return_mean = np.mean(returns)
        daily_return_std = np.std(returns)
        
        # Annualize based on hourly data (adjust if using different timeframe)
        # Assuming 24 hours * 365 days
        periods_per_year = 24 * 365
        annual_return = (1 + daily_return_mean) ** periods_per_year - 1
        annual_volatility = daily_return_std * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        
        # Return results as dictionary
        results = {
            'portfolio_values': portfolio_values,
            'positions': positions,
            'trade_history': trade_history,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'trade_count': trades,
            'trade_frequency': trade_freq,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'dates': dates
        }
        
        return results
    
    def print_results(self, results):
        """
        Print backtest performance metrics
        """
        print("\n======= BACKTEST RESULTS =======")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Number of Trades: {results['trade_count']}")
        print(f"Trade Frequency: {results['trade_frequency']*100:.2f}%")
        print(f"Annualized Return: {results['annual_return']*100:.2f}%")
        print(f"Annualized Volatility: {results['annual_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']*100:.2f}%")
        print("===============================")
    
    def plot_portfolio_performance(self, results):
        """
        Plot portfolio value over time with buy/sell markers
        """
        portfolio_values = results['portfolio_values']
        trade_history = results['trade_history']
        
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio value
        if results.get('dates') is not None:
            plt.plot(results['dates'], portfolio_values, label='Portfolio Value')
            
            # Add buy/sell markers
            buy_dates = [trade['timestamp'] for trade in trade_history if trade['action'] == 'BUY']
            buy_values = [trade['value'] for trade in trade_history if trade['action'] == 'BUY']
            
            sell_dates = [trade['timestamp'] for trade in trade_history if trade['action'] == 'SELL']
            sell_values = [trade['value'] for trade in trade_history if trade['action'] == 'SELL']
            
            plt.scatter(buy_dates, buy_values, color='green', marker='^', s=100, label='Buy')
            plt.scatter(sell_dates, sell_values, color='red', marker='v', s=100, label='Sell')
            
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.plot(portfolio_values, label='Portfolio Value')
            
            # Add buy/sell markers using indices
            buy_indices = [trade['timestamp'] for trade in trade_history if trade['action'] == 'BUY']
            buy_values = [trade['value'] for trade in trade_history if trade['action'] == 'BUY']
            
            sell_indices = [trade['timestamp'] for trade in trade_history if trade['action'] == 'SELL']
            sell_values = [trade['value'] for trade in trade_history if trade['action'] == 'SELL']
            
            plt.scatter(buy_indices, buy_values, color='green', marker='^', s=100, label='Buy')
            plt.scatter(sell_indices, sell_values, color='red', marker='v', s=100, label='Sell')
        
        # Add reference line for initial balance
        plt.axhline(y=self.initial_balance, color='gray', linestyle='--', label='Initial Balance')
        
        # Add labels and legend
        plt.title('Portfolio Performance')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, results):
        """
        Plot drawdown over time
        """
        portfolio_values = results['portfolio_values']
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        
        plt.figure(figsize=(12, 4))
        
        if results.get('dates') is not None:
            plt.plot(results['dates'], drawdowns * 100)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.plot(drawdowns * 100)
        
        plt.title('Portfolio Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1f}%'))
        
        # Fill area between 0 and drawdown line
        plt.fill_between(
            range(len(drawdowns)) if results.get('dates') is None else results['dates'], 
            0, 
            drawdowns * 100, 
            color='red', 
            alpha=0.3
        )
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test backtester independently with synthetic data
    import asyncio
    from data_loader import fetch_data
    from feature_engineering import engineer_features, normalize_features, create_labels, prepare_sequences
    from model import train_model
    
    async def test_backtester():
        # Get data and train model
        df = await fetch_data()
        df = engineer_features(df)
        df, features = normalize_features(df)
        df = create_labels(df)
        X, y = prepare_sequences(df, features)
        
        model, history, X_test, y_test, y_pred_classes, y_true_classes = train_model(X, y, epochs=5)
        
        # Extract test period prices for backtesting
        # Note: This assumes test data is at the end of the original dataframe
        test_prices = df['flow_total'].iloc[-len(y_pred_classes):].values
        test_dates = df.index[-len(y_pred_classes):]
        
        # Run backtest
        bt = Backtester(initial_balance=1000)
        results = bt.run_backtest(y_pred_classes, test_prices, test_dates)
        
        # Show results
        bt.print_results(results)
        bt.plot_portfolio_performance(results)
        bt.plot_drawdown(results)
    
    asyncio.run(test_backtester())