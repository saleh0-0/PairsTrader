from DataCollector import load_data, validate_data, remove_long_nan_columns, remove_row_nan
from PairsTrader import PairsTradingAnalyzer
import pandas as pd


# Load data
df = pd.read_csv("stocks_prices.csv", index_col='Date', parse_dates=True)

stock1 = df["2270.SR"]
stock2 = df["1202.SR"]

# Initialize analyzer
analyzer = PairsTradingAnalyzer()

# Calculate cointegration between two stocks
p_value = analyzer.calculate_cointegration(stock1, stock2)
print(f"Cointegration p-value between {stock1.name} and {stock2.name}: {p_value:.8f}")

# Calculate cointegration between all
#coint_matrix = analyzer.calculate_cointegration(stock1, stock2).replace(0, pd.NA)
#coint_matrix.to_csv("coint_matrix.csv")

#stock1, stock2 = coint_matrix.stack().idxmin()
#p_value = coint_matrix.loc[stock1, stock2]

#print(f"Cointegration p-value between {stock1} and {stock2}: {p_value:.8f}")





# Calculate hedge ratio
hedge_ratio = analyzer.calculate_hedge_ratio(stock1, stock2)

print(f"Hedge Ratio between {stock1.name} and {stock2.name} is: {hedge_ratio:.4f}")


# Calculate spread and z-score
spread = analyzer.calculate_spread(stock1, stock2, hedge_ratio)
zscore = analyzer.calculate_spread_zscore(spread, True, window=256)

print(f"Spread between {stock1.name} and {stock2.name} calculated.")


# Generate positions based on z-score thresholds
# 95% confidence intervals for entry and exit
positions = analyzer.positions_from_thresholds(stock1, stock2, hedge_ratio, zscore, entry_threshold=1.96, exit_threshold=0.6, window_size=252)
print("Positions generated based on z-score thresholds.")

# ATR-based backtest
returns1, signals1, _ = analyzer.rolling_backtest_atr(stock1, stock2, hedge_ratio, spread, z_scores=zscore, k=2.0)
metrics1 = analyzer.evaluate_strategy(returns1, signals1)

print("ATR-based Backtest Metrics:")
for key, value in metrics1.items():
    print(f"{key}: {value:.4f}")

#print(stock1.name)


# Fixed z-score backtest
returns2, signals2 = analyzer.rolling_backtest(stock1, stock2, hedge_ratio, z_scores=zscore, entry_threshold=1.25, exit_threshold=0.4)
metrics2 = analyzer.evaluate_strategy(returns2, signals2)
print("Backtest Metrics:")
for key, value in metrics2.items():
    print(f"{key}: {value:.4f}")

positions = analyzer.generate_positions(signals2, hedge_ratio, stock1, stock2)
#print(positions.tail())