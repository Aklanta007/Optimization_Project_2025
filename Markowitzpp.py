import numpy as np
import pandas as pd
import cvxpy as cp

class PortfolioBacktester:
    """A class to perform rolling window backtesting for Markowitz++ portfolio optimization."""
    
    def __init__(self, returns, window_size=252, look_ahead=21, risk_free_rate=0.02/252):
        """
        Initialize the backtester with Markowitz++ parameters.
        
        Parameters:
        - returns: pandas DataFrame or numpy array of asset returns (T x n)
        - window_size: in-sample window size (e.g., 252 days)
        - look_ahead: out-of-sample period (e.g., 21 days)
        - risk_free_rate: daily risk-free rate
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
        self.risk_free_rate = risk_free_rate
        self.T, self.n = self.returns.shape
        self.portfolio_returns = []
        self.portfolio_weights = []
        self.metrics = {}
        
        # Markowitz++ parameters from Section 5.3
        self.gamma_hold = 0.8
        self.gamma_trade = 0.2
        self.risk_target = 0.17 # Increased to target 0.1 annualized volatility
        self.c_min = 0.0
        self.c_max = 1.00
        self.w_min = 0.01 # Relaxed to allow larger short positions
        self.w_max = 1   # Relaxed to allow larger long positions
        self.L_tar = 1.5
        self.z_min = 0
        self.z_max = 0.3
        self.T_tar = 20.0
        self.gamma_risk = 0.01  # Reduced to lower risk penalty
        self.gamma_lev = 0.0005
        self.gamma_turn = 0.0025
        self.rho = None  # Set dynamically as 20th percentile of abs(return forecast)
        self.rho_factor = 0.100
        
        # Assume simple holding and trading costs (can be extended with real data)
        self.kappa_short = np.ones(self.n) * 0.0001  # Borrow cost for short positions
        self.kappa_borrow = 0.0001  # Borrow cost for cash
        self.kappa_spread = np.ones(self.n) * 0.00005  # Half bid-ask spread
    
    def optimize_portfolio(self, in_sample, pre_weights=None):
        """
        Optimize portfolio weights using Markowitz++ with CVXPY.
        
        Parameters:
        - in_sample: numpy array of in-sample returns (window_size x n)
        - pre_weights: previous portfolio weights for trading costs (n,)
        
        Returns:
        - weights: optimized portfolio weights or None if optimization fails
        """
        mu = np.mean(in_sample, axis=0)  # Expected returns
        cov = np.cov(in_sample.T)  # Covariance matrix
        n = self.n
        
        # CVXPY variables
        w = cp.Variable(n)  # Asset weights
        c = cp.Variable()   # Cash weight
        z = cp.Variable(n)  # Trade vector
        T = cp.Variable()   # Turnover
        L = cp.Variable()   # Leverage
        
        # Mean return and uncertainty
        mean_return = w @ mu + self.risk_free_rate * c
        if self.rho is None:
            self.rho = np.percentile(np.abs(mu), 20)  # 20th percentile
        return_uncertainty = self.rho * cp.norm(w, 1)
        mean_return_wc = mean_return - return_uncertainty
        
        # Risk (using Cholesky for numerical stability)
        L_chol = np.linalg.cholesky(cov * 252)  # Annualized
        risk = cp.norm(L_chol @ w, 2)
        risk_uncertainty = self.rho_factor * cp.norm(w, 1)
        risk_wc = risk + risk_uncertainty
        
        # Holding costs
        holding_cost = (self.kappa_short @ cp.pos(-w) + 
                       self.kappa_borrow * cp.pos(-c))
        
        # Trading costs
        if pre_weights is None:
            pre_weights = np.zeros(n)
        z = w - pre_weights
        trading_cost = self.kappa_spread @ cp.abs(z)
        T = 0.5 * cp.norm(z, 1)
        L = cp.norm(w, 1)
        
        # Objective
        objective = cp.Maximize(
            mean_return_wc -
            self.gamma_hold * holding_cost -
            self.gamma_trade * trading_cost -
            self.gamma_risk * cp.pos(risk_wc - self.risk_target) -
            self.gamma_lev * cp.pos(L - self.L_tar) -
            self.gamma_turn * cp.pos(T - self.T_tar)
        )
        
        # Constraints
        constraints = [
            cp.sum(w) + c <= 1,  # Budget constraint
            self.w_min <= w, w <= self.w_max,  # Weight limits
            self.c_min <= c, c <= self.c_max,  # Cash limits
            self.z_min <= z, z <= self.z_max,  # Trade limits
            L <= self.L_tar,  # Leverage limit
            T <= self.T_tar,  # Turnover limit
            # risk_wc <= self.risk_target  # Commented out to allow higher volatility
        ]
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status != cp.OPTIMAL:
                print(f"Optimization failed, status: {problem.status}")
                return None
            return w.value
        except cp.SolverError:
            print("Solver error, skipping...")
            return None
    
    def run(self):
        """Run the rolling window backtest."""
        self.portfolio_returns = []
        self.portfolio_weights = []
        pre_weights = None
        
        # Rolling window loop
        for t in range(self.window_size, self.T - self.look_ahead + 1, self.look_ahead):
            # In-sample data
            in_sample = self.returns[t - self.window_size:t]
            
            # Optimize portfolio
            weights = self.optimize_portfolio(in_sample, pre_weights)
            if weights is None:
                continue
            
            # Out-of-sample returns
            out_sample = self.returns[t:t + self.look_ahead]
            port_ret = out_sample @ weights
            
            # Store results
            self.portfolio_weights.append(weights)
            self.portfolio_returns.extend(port_ret)
            pre_weights = weights
        
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
                             np.std(portfolio_returns) * np.sqrt(252)),
            'Cumulative Return': np.prod(1 + portfolio_returns) - 1,
            'Annualized Volatility': np.std(portfolio_returns) * np.sqrt(252),
            'Max Drawdown': np.max(1 - np.cumprod(1 + portfolio_returns) / 
                                  np.maximum.accumulate(np.cumprod(1 + portfolio_returns)))
        }
    
    def get_results_summary(self):
        """Return a formatted summary of the backtest results."""
        summary = "Markowitz++ Portfolio Backtest Results:\n"
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