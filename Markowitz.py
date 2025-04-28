import numpy as np
import pandas as pd
import cvxpy as cp


class PortfolioBacktester:
    """A class to perform rolling window backtesting for portfolio optimization."""
    
    def __init__(self, returns, window_size=252, look_ahead=21, risk_free_rate=0.02/252):
        """
        Initialize the backtester.
        
        Parameters:
        - returns: pandas DataFrame or numpy array of asset returns (T x n)
        - window_size: in-sample window size (e.g., 252 days)
        - look_ahead: out-of-sample period (e.g., 21 days)
        - risk_free_rate: daily risk-free rate (not used in Markowitz optimization)
        """
        # Convert returns to numpy array if DataFrame
        if isinstance(returns, pd.DataFrame):
            self.returns = returns.values
            self.asset_names = returns.columns.tolist()
        else:
            self.returns = np.array(returns)
            self.asset_names = [f"Asset_{i}" for i in range(self.returns.shape[1])]
        
        self.window_size = window_size
        self.look_ahead = look_ahead
        self.risk_free_rate = risk_free_rate  # Kept for compatibility, not used
        self.T, self.n = self.returns.shape
        self.portfolio_returns = []
        self.portfolio_weights = []
        self.metrics = {}
    
    def optimize_portfolio(self, in_sample):
        """
        Optimize portfolio weights using Markowitz mean-variance optimization.
        Maximizes expected return with annualized variance <= 0.01.
        
        Parameters:
        - in_sample: numpy array of in-sample returns (window_size x n)
        
        Returns:
        - weights: optimized portfolio weights or None if optimization fails
        """
        mu = np.mean(in_sample, axis=0)  # Expected returns
        cov = np.cov(in_sample.T)  # Covariance matrix
        
        # CVXPY optimization: Maximize expected return
        weights = cp.Variable(self.n)
        portfolio_return = mu @ weights
        portfolio_variance = cp.quad_form(weights, cov * 250)  # Annualized
        
        # Objective and constraints
        objective = cp.Maximize(portfolio_return)
        constraints = [
            cp.sum(weights) <= 1,  # Weights sum to 1
            weights >= 0,  # No short-selling
            portfolio_variance <= 0.0289  # Annualized variance <= 0.01
        ]
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status != cp.OPTIMAL:
                print(f"Optimization failed, status: {problem.status}")
                return None
            return weights.value
        except cp.SolverError:
            print("Solver error, skipping...")
            return None
    
    def run(self):
        """Run the rolling window backtest."""
        self.portfolio_returns = []
        self.portfolio_weights = []
        
        # Rolling window loop
        for t in range(self.window_size, self.T - self.look_ahead + 1, self.look_ahead):
            # In-sample data
            in_sample = self.returns[t - self.window_size:t]
            
            # Optimize portfolio
            weights = self.optimize_portfolio(in_sample)
            if weights is None:
                continue
            
            # Out-of-sample returns
            out_sample = self.returns[t:t + self.look_ahead]
            port_ret = out_sample @ weights
            
            # Store results
            self.portfolio_weights.append(weights)
            self.portfolio_returns.extend(port_ret)
        
        # Compute performance metrics
        self.calculate_metrics()
        
        return self.portfolio_returns, self.portfolio_weights, self.metrics
    
    def calculate_metrics(self):
        """Calculate performance metrics for the portfolio."""
        portfolio_returns = np.array(self.portfolio_returns)
        if len(portfolio_returns) == 0:
            self.metrics = {
                'Sharpe Ratio': np.nan,
                'Cumulative Return': np.nan,
                'Annualized Volatility': np.nan,
                'Max Drawdown': np.nan
            }
            return
        
        self.metrics = {
            'Sharpe Ratio': ((np.mean(portfolio_returns) - self.risk_free_rate) / 
                             np.std(portfolio_returns) * np.sqrt(250)),
            'Cumulative Return': np.prod(1 + portfolio_returns) - 1,
            'Annualized Volatility': np.std(portfolio_returns) * np.sqrt(250),
            'Max Drawdown': np.max(1 - np.cumprod(1 + portfolio_returns) / 
                                  np.maximum.accumulate(np.cumprod(1 + portfolio_returns)))
        }
    
    def get_results_summary(self):
        """Return a formatted summary of the backtest results."""
        summary = "Portfolio Backtest Results:\n"
        summary += f"Assets: {', '.join(self.asset_names)}\n"
        summary += f"Window Size: {self.window_size} periods\n"
        summary += f"Look-Ahead Period: {self.look_ahead} periods\n"
        summary += "\nPerformance Metrics:\n"
        for key, value in self.metrics.items():
            summary += f"{key}: {value:.4f}\n"
        summary += "\nSample Weights (last window):\n"
        if self.portfolio_weights:
            last_weights = np.round(self.portfolio_weights[-1], 4)
            for asset, weight in zip(self.asset_names, last_weights):
                summary += f"{asset}: {weight:.4f}\n"
        else:
            summary += "No weights available (optimization failed).\n"
        return summary

    