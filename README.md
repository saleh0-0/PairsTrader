# Pairs Trading and Data Collector

## Goal
This project provides two Python modules:

1. **DataCollector** — Downloads and cleans historical stock price data from Yahoo Finance.  
2. **PairsTrader** — Performs statistical analysis and backtesting for pairs trading strategies, including cointegration testing, hedge ratio calculation, spread analysis, z-score signals, and strategy evaluation.

Together, they let you fetch price data, identify cointegrated stock pairs, and test trading strategies.

---

## How to Run

### 1. Install requirements
Make sure you have Python 3.8+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare tickers
- Create a CSV file (e.g., `company_sympol_db.csv`) with your stock tickers in a column (no `.SR` or exchange suffix).
- The DataCollector script will append `.SR` automatically for Saudi market symbols.

### 3. Download and clean price data
```bash
python DataCollector.py
```
This will:
- Download historical prices from Yahoo Finance.
- Remove columns/rows with excessive missing data.
- Save the cleaned data to `stocks_prices.csv`.

### 4. Run the example analysis
You can run the provided example:
```bash
python run_example.py
```
This will:
- Load `stocks_prices.csv`.
- Select two stocks (`2270.SR` and `1202.SR` in the example).
- Calculate cointegration, hedge ratio, spread, and z-score.
- Generate positions based on thresholds.
- Run two backtests (ATR-based and fixed z-score).
- Print performance metrics.

---

## How to Use in Your Own Code

### 1. Fetch Data
```python
from DataCollector import load_data, save_data

tickers = ["AAPL", "MSFT"]
df = load_data(tickers, start_date="2023-01-01", end_date="2023-12-31")
save_data(df, "my_prices.csv")
```

### 2. Analyze and Backtest
```python
import pandas as pd
from PairsTrader import PairsTradingAnalyzer

df = pd.read_csv("my_prices.csv", index_col="Date", parse_dates=True)
stock1 = df["AAPL"]
stock2 = df["MSFT"]

analyzer = PairsTradingAnalyzer()
hedge_ratio = analyzer.calculate_hedge_ratio(stock1, stock2)
spread = analyzer.calculate_spread(stock1, stock2, hedge_ratio)
zscore = analyzer.calculate_spread_zscore(spread, avoid_lookahead=True)

positions = analyzer.positions_from_thresholds(
    stock1, stock2, hedge_ratio, zscore,
    entry_threshold=2.0, exit_threshold=0.5
)
```

---

## Notes
- **Data source**: Yahoo Finance via `yfinance`.  
- **Key outputs**:
  - `stocks_prices.csv` — cleaned price data.
  - Backtest metrics printed to console.
- The scripts can be adapted to any Yahoo Finance tickers.
