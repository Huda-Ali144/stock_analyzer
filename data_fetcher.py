import yfinance
import yfinance as yf
import pandas as pd
def fetch_info(ticker: str) -> dict:
    
    #Return the metadata dict for a given stock ticker.
    symbol = ticker.strip().upper()
    stock  = yf.Ticker(symbol)
    return stock.info

def fetch_history(ticker: str,
                  period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
    
    #Return a DataFrame of historical price data.

    symbol = ticker.strip().upper()
    stock  = yf.Ticker(symbol)
    return stock.history(period=period, interval=interval)