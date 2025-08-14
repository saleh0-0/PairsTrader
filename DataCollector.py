# Yahoo finance stock price historical data collector
import pandas as pd
import yfinance as yf
import datetime as dt
import time
import random

def load_data(
    tickers,
    start_date=None,
    end_date=None,
    retries=3,
    delay=2,
    use_long_names=False
) -> pd.DataFrame:
    """
    Load stock data from Yahoo Finance with retries and rate limit handling.

    Parameters:
        tickers (str or list): Stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        retries (int): Number of retries on failure.
        delay (int): Delay in seconds between retries.
        use_long_names (bool): Whether to fetch long company names from Yahoo.

    Returns:
        pd.DataFrame: DataFrame with stock prices.
    """
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (dt.datetime.now() - dt.timedelta(days=730)).strftime("%Y-%m-%d")

    if isinstance(tickers, str):
        tickers = [tickers]

    try:
        print(f"Downloading data for: {tickers}")
        stock_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            progress=True,
            threads=True
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        if retries > 0:
            time.sleep(delay)
            return load_data(tickers, start_date, end_date, retries - 1, delay * 2, use_long_names)
        else:
            raise

    close_prices = {}

    if isinstance(stock_data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                label = ticker
                if use_long_names:
                    label = get_company_name(ticker)
                close_prices[label] = stock_data.xs('Close', level=1, axis=1)[ticker]
            except Exception:
                print(f"Skipping {ticker} due to missing data.")
    else:
        try:
            ticker = tickers[0]
            label = ticker
            if use_long_names:
                label = get_company_name(ticker)
            close_prices[label] = stock_data['Close']
        except Exception:
            print(f"Skipping {ticker} due to missing data.")

    df = pd.DataFrame(close_prices)
    df.index.name = 'Date'
    df.reset_index(inplace=True)

    return df


def get_company_name(ticker: str) -> str:
    """
    Fetch the long company name from Yahoo Finance using yfinance.

    Parameters:
        ticker (str): Ticker symbol.

    Returns:
        str: Long company name or the ticker if name is not found.
    """
    try:
        time.sleep(random.uniform(2, 4))  # Delay to avoid rate limit
        return yf.Ticker(ticker).info.get("longName", ticker)
    except Exception as e:
        print(f"Failed to get name for {ticker}: {e}")
        return ticker


def save_data(df, filename="stocks_prices.csv") -> None:
    """
    Save the DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the file to save the DataFrame.

    Returns:
        None
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def is_valid_ticker(ticker) -> bool:
    try:
        df = yf.download(ticker, period="5d")
        return not df.empty
    except:
        return False


def validate_data(df) -> pd.DataFrame:
    """
    Validate the DataFrame to ensure it contains valid stock data, and remove any rows with NaN values.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        df (pd.DataFrame): Validated DataFrame with NaN values removed.
    """

    if df.empty:
        print("DataFrame is empty. No data to validate.")
        return df

    # Check for NaN values and drop them
    df = df.dropna()

    if df.empty:
        print("DataFrame is empty after removing NaN values. No valid data to return.")
        return df
    
    return df

# remove columns with more than 10 continuous NaN values
def remove_long_nan_columns(df, threshold=10) -> pd.DataFrame:
    """
    Remove columns with more than a specified number of continuous NaN values.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        threshold (int): Maximum number of continuous NaN values allowed.
    Returns:
        pd.DataFrame: DataFrame with long NaN columns removed.
    """
    if df.empty:
        print("DataFrame is empty. No columns to process.")
        return df
    
    df = df.dropna(axis=1, thresh=len(df) - threshold)

    return df


def remove_row_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with any NaN values from the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
    Returns:
        pd.DataFrame: DataFrame with rows containing NaN values removed.
    """
    if df.empty:
        print("DataFrame is empty. No rows to process.")
        return df
    
    df = df.dropna(axis=0, how='any')

    return df


if __name__ == "__main__":
    #tickers = ["1120.SR", "1180.SR", "1111.SR", "1140.SR", "1150.SR"]
    tickers = pd.read_csv("company_sympol_db.csv")["الرمز"].tolist()
    tickers = [f"{ticker}.SR" for ticker in tickers]

    # Load data
    print("Loading data...")
    stocks_prices = load_data(tickers, use_long_names=False)

    stocks_prices = remove_long_nan_columns(stocks_prices, threshold=10)
    stocks_prices = remove_row_nan(stocks_prices)

    #df = load_data(tickers, use_long_names=True)
    #df = validate_data(df)  # Set to True if you want company names

    if not stocks_prices.empty:
        save_data(stocks_prices)
        print("Data loading and saving completed.")
    else:
        print("No data to save.")