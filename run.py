import pandas as pd
import argparse
from results_to_csv import PortfolioResultsCSV

# Command-line argument parser
parser = argparse.ArgumentParser(description="Run Markowitz portfolio optimization with specified strategy.")
parser.add_argument(
    "--strategy",
    choices=["Markowitz++", "Markowitz", "Equal"],
    required=True,
    help="Specify the Markowitz strategy to use (Markowitz1, Markowitz2, or Markowitz3)"
)
args = parser.parse_args()

# Dynamically import the specified Markowitz module
try:
    if args.strategy == "Markowitz++":
        from Markowitzpp import PortfolioBacktester
    elif args.strategy == "Markowitz":
        from Markowitz import PortfolioBacktester
    elif args.strategy == "Equal":
        from Equal import PortfolioBacktester
except ImportError:
    print(f"Error: Could not import module for strategy '{args.strategy}'. Ensure the module exists and is correctly named.")
    exit(1)

# Configuration
csv_file = "Datasets/backtest_rev.csv"  # Path to your CSV file
window_size = 250 * 3  # In-sample window size (e.g., 3 years)
look_ahead = 250       # Out-of-sample period (e.g., 1 year)
risk_free_rate = 0.02 / 250  # Daily risk-free rate
output_csv = f"portfolio_results_{args.strategy}.csv"  # Strategy-specific output CSV

# Load returns data from CSV
try:
    returns_df = pd.read_csv(csv_file, index_col=0)  # Assume first column is date/index
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file}' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: CSV file '{csv_file}' is empty.")
    exit(1)

# Extract dates
dates = returns_df.index

# Initialize and run backtester
backtester = PortfolioBacktester(
    returns=returns_df,
    window_size=window_size,
    look_ahead=look_ahead,
    risk_free_rate=risk_free_rate
)
portfolio_returns, portfolio_weights, metrics = backtester.run()

# Generate CSV with weights and returns
csv_generator = PortfolioResultsCSV(backtester, dates)
results_df = csv_generator.generate_csv(output_file=output_csv)
print(f"Results saved to '{output_csv}'")

# Print results
print(backtester.get_results_summary())