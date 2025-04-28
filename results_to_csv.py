import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioResultsCSV:
    """A class to generate a CSV file with portfolio weights, look-ahead returns, and cumulative returns."""
    
    def __init__(self, backtester, dates):
        """
        Initialize the CSV generator.
        
        Parameters:
        - backtester: PortfolioBacktester instance after running backtest
        - dates: pandas Series or list of dates corresponding to returns in backtester
        """
        self.backtester = backtester
        self.dates = pd.to_datetime(dates) if isinstance(dates, (pd.Series, list)) else dates
        self.asset_names = backtester.asset_names
        # Ensure dates align with returns
        if len(self.dates) != backtester.returns.shape[0]:
            raise ValueError("Length of dates must match number of returns rows")
    
    def generate_csv(self, output_file="portfolio_results.csv"):
        """
        Generate a CSV file with start date, end date, weights, look-ahead return, and cumulative return.
        """
        # Initialize lists to store data
        start_dates = []
        end_dates = []
        weights_data = []
        look_ahead_returns = []
        cumulative_returns = []
        
        # Track the date index for out-of-sample periods
        date_idx = self.backtester.window_size
        first_test_date_idx = self.backtester.window_size
        
        if not hasattr(self.backtester, 'portfolio_weights') or not self.backtester.portfolio_weights:
            raise ValueError("No portfolio weights available. Run backtester first.")
        
        weights_array = np.array(self.backtester.portfolio_weights)
        
        # Calculate daily portfolio returns from first test date onward
        daily_port_returns = []
        current_weight_idx = 0
        weight_period_end = first_test_date_idx + self.backtester.look_ahead
        
        for i in range(first_test_date_idx, len(self.backtester.returns)):
            if i >= weight_period_end:
                current_weight_idx += 1
                weight_period_end += self.backtester.look_ahead
                if current_weight_idx >= len(weights_array):
                    break
            
            current_weights = weights_array[current_weight_idx]
            daily_return = self.backtester.returns[i] @ current_weights
            daily_port_returns.append(daily_return)
        
        # Convert to numpy array and calculate cumulative returns
        daily_port_returns = np.array(daily_port_returns)
        cumulative_products = np.cumprod(1 + daily_port_returns)
        
        # Iterate through weights to build results
        current_cum_idx = 0
        for i, weights in enumerate(weights_array):
            if date_idx >= len(self.dates):
                break
                
            # Start and end dates
            start_date = self.dates[date_idx]
            end_date_idx = min(date_idx + self.backtester.look_ahead - 1, len(self.dates) - 1)
            end_date = self.dates[end_date_idx]
            
            # Look-ahead return (for the current period only)
            look_ahead_indices = range(date_idx, end_date_idx + 1)
            if look_ahead_indices:
                look_ahead_ret = self.backtester.returns[look_ahead_indices] @ weights
                look_ahead_cum_ret = np.prod(1 + look_ahead_ret) - 1
            else:
                look_ahead_cum_ret = 0
            
            # Cumulative return (from first test date to current end date)
            period_length = end_date_idx - date_idx + 1
            if current_cum_idx + period_length <= len(cumulative_products):
                cum_ret = cumulative_products[current_cum_idx + period_length - 1] - 1
            else:
                cum_ret = cumulative_products[-1] - 1 if len(cumulative_products) > 0 else 0
            
            # Store results
            start_dates.append(start_date)
            end_dates.append(end_date)
            weights_data.append(weights)
            look_ahead_returns.append(look_ahead_cum_ret)
            cumulative_returns.append(cum_ret)
            
            # Update indices
            date_idx += self.backtester.look_ahead
            current_cum_idx += period_length
        
        # Create DataFrame
        weights_df = pd.DataFrame(weights_data, columns=[f"{name}_weight" for name in self.asset_names])
        results_df = pd.DataFrame({
            'Start_Date': start_dates,
            'End_Date': end_dates,
            'Look_Ahead_Return': look_ahead_returns,
            'Cumulative_Return': cumulative_returns
        })
        
        # Concatenate weights
        results_df = pd.concat([results_df, weights_df], axis=1)
        
        # Format dates
        results_df['Start_Date'] = pd.to_datetime(results_df['Start_Date']).dt.strftime('%Y-%m-%d')
        results_df['End_Date'] = pd.to_datetime(results_df['End_Date']).dt.strftime('%Y-%m-%d')
        
        # Validate weight sums
        weight_cols = [col for col in results_df.columns if "_weight" in col]
        weight_sums = results_df[weight_cols].sum(axis=1)
        if any(weight_sums > 1 + 1e-6):
            print("Warning: Some weight sums exceed 1:", weight_sums[weight_sums > 1])
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        
        return results_df