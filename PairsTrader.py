# pairs_trading.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from DataCollector import load_data, validate_data, remove_long_nan_columns


class PairsTradingAnalyzer:

    def calculate_cointegration(self, Stock1:pd.Series = None, Stock2: pd.Series = None, df: pd.DataFrame = None) -> float:
        """
        Calculate the cointegration between two stocks or all pairs in a DataFrame.

        If Stock1 and Stock2 are provided, it calculates the cointegration between them.
        If df is provided, it calculates the cointegration for all pairs in the DataFrame.

        Parameters:
            Stock1 (pd.Series): Prices of the first stock.
            Stock2 (pd.Series): Prices of the second stock.
            df (pd.DataFrame): DataFrame containing prices of multiple stocks.
        
        Returns:
            pd.DataFrame: DataFrame containing p-values of cointegration tests between all pairs of stocks.
            p_value (float): Cointegration p-value between Stock1 and Stock2 if they are provided.
        """

        if df is not None:
            banks = df.columns[1:]
            n = len(banks)
            coint_matrix = pd.DataFrame(np.zeros((n, n)), index=banks, columns=banks)

            for i in range(n):
                for j in range(i + 1, n):
                    score, p_value, _ = coint(df[banks[i]].dropna(), df[banks[j]].dropna())
                    coint_matrix.loc[banks[i], banks[j]] = p_value
                    coint_matrix.loc[banks[j], banks[i]] = p_value
            return coint_matrix
        elif Stock1 is not None and Stock2 is not None:
            score, p_value, _ = coint(Stock1.dropna(), Stock2.dropna())
            return p_value
        else:
            raise ValueError("Either Stock1 and Stock2 or df must be provided for cointegration calculation.")


    def visualize_cointegration(self, coint_matrix) :
        plt.figure(figsize=(10, 8))
        sns.heatmap(coint_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"label": "Cointegration p-value"})
        plt.title("Cointegration Matrix")
        plt.show()


    def calculate_hedge_ratio(self, stock1: pd.Series, stock2: pd.Series) -> float:
        """
        Calculate the hedge ratio between two stocks using Ordinary Least Squares (OLS) regression.

        Parameters:
            stock1 (pd.Series): Prices of the first stock.
            stock2 (pd.Series): Prices of the second stock.
        
        Returns:
            float: Hedge ratio between the two stocks.
        """

        if stock1.empty or stock2.empty:
            raise ValueError("Both stock1 and stock2 must contain data to calculate hedge ratio.")


        # Align on common index and drop missing/inf
        df = pd.concat([stock1.rename("y"), stock2.rename("x")], axis=1, join="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit()

        return model.params["x"]


    def calculate_spread(self, stock1: pd.Series, stock2:pd.Series, hedge_ratio: float) -> pd.Series:
        """
        Calculate the spread between two stocks using the hedge ratio.
        
        Parameters:
            stock1 (pd.Series): Prices of the first stock.
            stock2 (pd.Series): Prices of the second stock.
            hedge_ratio (float): Hedge ratio between the two stocks.
        Returns:
            pd.Series: Spread between the two stocks.
        """
        if stock1.empty or stock2.empty:
            raise ValueError("Both stock1 and stock2 must contain data to calculate spread.")
        if hedge_ratio is None:
            raise ValueError("Hedge ratio must be provided to calculate spread.")
        

        spread = stock1 - hedge_ratio * stock2
        return spread


    def calculate_spread_zscore(self, spread: pd.Series, avoid_lookahead: bool = False, window: int = 252) -> pd.Series:
        """
        Calculate the z-score of the spread between two stocks.

        Parameters:
            spread (pd.Series): Spread between the two stocks.
            avoid_lookahead (bool): 
                - If False (default): use full-sample mean and std (may introduce look-ahead bias).
                - If True: use rolling mean/std to avoid look-ahead bias.
            window (int): Rolling window size for look-ahead avoidance.

        Returns:
            pd.Series: Z-score of the spread.
        """
        if spread.empty:
            raise ValueError("Spread must contain data to calculate z-score.")

        if avoid_lookahead:
            mean = spread.rolling(window=window, min_periods=window).mean()
            std = spread.rolling(window=window, min_periods=window).std()
            zscore = (spread - mean) / std
        else:
            mean = spread.mean()
            std = spread.std()
            zscore = (spread - mean) / std

        return zscore.replace([np.inf, -np.inf], np.nan)


    def positions_from_thresholds(self, stock1: pd.Series, stock2: pd.Series, hedge_ratio: float, z_scores: pd.Series, entry_threshold: float, exit_threshold: float = 0.0, window_size: int = 252) -> pd.DataFrame:
        """
        Generate long/short positions based on z-score thresholds.

        Entry rules:
            z >  +entry_threshold  -> signal = -1  (short spread = short A, long beta*B)
            z <  -entry_threshold  -> signal = +1  (long  spread = long  A, short beta*B)

        Exit rule:
            |z| < exit_threshold   -> signal = 0   (flat)

        Otherwise, hold previous signal.

        Parameters:
            stock1 (pd.Series): Price series of the first stock (A).
            stock2 (pd.Series): Price series of the second stock (B).
            hedge_ratio (float): Static hedge ratio beta (A ≈ alpha + beta*B).
            z_scores (pd.Series): z-score series (ideally rolling, no look-ahead).
            entry_threshold (float): Absolute z threshold to enter a trade.
            exit_threshold (float): Absolute z threshold to flatten a trade.
            window_size (int): Warm-up bars before allowing signals (match your z window).

        Returns:
            pd.DataFrame: Positions for both stocks (A column = stock1.name, B column = stock2.name).
                        Values are in spread units: +1 means long A / short beta*B; -1 is the reverse.
        """
        if stock1.empty or stock2.empty:
            raise ValueError("stock1 and stock2 must contain data.")
        if not isinstance(hedge_ratio, (int, float)):
            raise TypeError("hedge_ratio must be a float (static beta).")

        # Align everything on a common index
        common_index = z_scores.index.intersection(stock1.index).intersection(stock2.index)
        if common_index.empty:
            raise ValueError("No overlapping dates between inputs.")

        z = z_scores.reindex(common_index)
        signals = pd.Series(index=common_index, dtype=float)

        # Build signals using your backtest’s logic
        for i in range(min(window_size, len(common_index)), len(common_index)):
            date = common_index[i]
            z_today = z.iat[i]

            if pd.isna(z_today):
                # Hold if z is NaN
                signals.iat[i] = signals.iat[i - 1] if i > 0 else 0.0
                continue

            if z_today > entry_threshold:
                signals.iat[i] = -1.0
            elif z_today < -entry_threshold:
                signals.iat[i] = +1.0
            elif abs(z_today) < exit_threshold:
                signals.iat[i] = 0.0
            else:
                signals.iat[i] = signals.iat[i - 1] if i > 0 else 0.0

        signals.ffill(inplace=True)
        signals.fillna(0.0, inplace=True)

        # Convert signals to per-leg positions using static beta
        positions = pd.DataFrame(index=common_index, columns=[stock1.name, stock2.name], dtype=float)
        positions[stock1.name] = signals
        positions[stock2.name] = -signals * float(hedge_ratio)
        positions.fillna(0.0, inplace=True)

        return positions


    def rolling_backtest(self, stock1: pd.Series, stock2: pd.Series, hedge_ratio: float ,z_scores: pd.Series, entry_threshold: float, exit_threshold: float = 0, window_size: int = 252) -> tuple:
        """
        Perform a rolling backtest using z-scores for entry and exit signals.
        
        Parameters:
            stock1 (pd.Series): Prices of the first stock.
            stock2 (pd.Series): Prices of the second stock.
            hedge_ratio (float): Hedge ratio between the two stocks.
            z_scores (pd.Series): Z-scores of the spread for additional filtering.
            entry_threshold (float): Z-score threshold for entering positions.
            exit_threshold (float): Z-score threshold for exiting positions.
            window_size (int): Size of the rolling window for z-score calculation.
        Returns:
            tuple: (strategy_returns, signals)
        """
        signals = pd.Series(index=z_scores.index, dtype=float)

        for i in range(window_size, len(stock1)):
            z_today = z_scores.iloc[i]
            date = z_scores.index[i]

            if z_today > entry_threshold:
                signals.at[date] = -1
            elif z_today < -entry_threshold:
                signals.at[date] = 1
            elif abs(z_today) < exit_threshold:
                signals.at[date] = 0
            else:
                signals.at[date] = signals.iloc[i - 1]

        signals.ffill(inplace=True)

        returns_A = stock1.pct_change()
        returns_B = stock2.pct_change()
        hedge_ratio = hedge_ratio
        spread_return = returns_A - hedge_ratio * returns_B
        strategy_returns = signals.shift(1) * spread_return

        return strategy_returns.dropna(), signals.dropna()


    def rolling_backtest_atr(self, stock1: pd.Series, stock2: pd.Series, hedge_ratio: float, spread: pd.Series, z_scores: pd.Series = None, k: float = 2.0, window_size: int = 30, exit_z: float= 0) -> tuple:
        """
        Perform a rolling backtest using Average True Range (ATR) for entry and exit signals.

        Parameters:
            stock1 (pd.Series): Prices of the first stock.
            stock2 (pd.Series): Prices of the second stock.
            hedge_ratio (float): Hedge ratio between the two stocks.
            spread (pd.Series): Spread between two stocks.
            z_scores (pd.Series): Z-scores of the spread for additional filtering.
            k (float): Multiplier for ATR to set entry thresholds.
            window_size (int): Size of the rolling window for ATR calculation.
            exit_z (float): Z-score threshold for exiting positions.
        Returns:
            tuple: (strategy_returns, signals, thresholds_df)
        """

        rolling_mean = spread.rolling(window=window_size).mean()
        spread_diff = spread.diff().abs()
        atr = spread_diff.rolling(window=window_size).mean()

        upper_thresh = rolling_mean + k * atr
        lower_thresh = rolling_mean - k * atr

        signals = pd.Series(index=spread.index, dtype=float)

        for i in range(window_size, len(spread)):
            date = spread.index[i]
            s = spread.iloc[i]

            if s > upper_thresh.iloc[i]:
                signals.at[date] = -1
            elif s < lower_thresh.iloc[i]:
                signals.at[date] = 1
            elif z_scores is not None and abs(z_scores.iloc[i]) < exit_z:
                signals.at[date] = 0
            else:
                signals.at[date] = signals.iloc[i - 1]

        signals.ffill(inplace=True)

        returns_A = stock1.pct_change()
        returns_B = stock2.pct_change()
        
        spread_return = returns_A - hedge_ratio * returns_B
        strategy_returns = signals.shift(1) * spread_return

        thresholds_df = pd.DataFrame({
            "Upper Threshold": upper_thresh,
            "Lower Threshold": lower_thresh
        })

        return strategy_returns.dropna(), signals.dropna(), thresholds_df


    def evaluate_strategy(self, returns: pd.Series, signals: pd.Series) -> dict:
        """
        Evaluate the performance of a trading strategy based on returns and signals.
        Parameters:
            returns (pd.Series): Series of returns from the strategy.
            signals (pd.Series): Series of trading signals.
        Returns:
            dict: Dictionary containing performance metrics.
        """
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else np.nan
        max_drawdown = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).sub(1).min()
        num_trades = (signals.diff() != 0).sum()
        win_rate = (returns[returns > 0].count()) / returns.count()

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Number of Trades": num_trades
        }


    def generate_positions(self, signals: pd.Series, hedge_ratio: float, stock1: pd.Series, stock2: pd.Series) -> pd.DataFrame:
        """
        Convert signals into position sizes for A and B based on hedge ratio.

        Parameters:
            signals (pd.Series): Series of trading signals.
            hedge_ratio (float): Hedge ratio between the two stocks.
            stock1 (pd.Series): Prices of the first stock.
            stock2 (pd.Series): Prices of the second stock.

        Returns:
            pd.DataFrame: DataFrame containing positions for both stocks.
        """
        if not isinstance(stock1, pd.Series) or not isinstance(stock2, pd.Series):
            raise TypeError("stock1 and stock2 must be pandas Series.")
        
        # Align index between signals and both stock price series
        common_index = signals.index.intersection(stock1.index).intersection(stock2.index)

        # Reindex and forward-fill signals to maintain positions until changed
        sig = signals.reindex(common_index).ffill()

        positions = pd.DataFrame(index=common_index, columns=[stock1.name, stock2.name], dtype=float)
        positions[stock1.name] = sig
        positions[stock2.name] = -sig * hedge_ratio
        positions.fillna(0.0, inplace=True)
        return positions



if __name__ == "__main__":

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
