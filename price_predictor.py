# stock_analyzer/price_predictor.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)

class RandomForestPredictor:
    def __init__(self, ticker: str, n_estimators: int = 100, random_state: int = 42):
        self.ticker = ticker
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.mse = None
        self.r2 = None

    def fetch_data(self, period: str = '2y', interval: str = '1d'):
        df = yf.download(self.ticker, period=period, interval=interval)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Target'] = df['Close'].shift(-1)
        self.df = df.dropna()

    def feature_engineering(self):
        df = self.df.copy()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['DayOfWeek'] = df.index.dayofweek
        df = df.dropna()

        self.X = df[['Close', 'MA5', 'MA10', 'DayOfWeek']]
        self.y = df['Target']

    def train_test(self, test_size: float = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, shuffle=False
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        self.mse = mean_squared_error(y_test, preds)
        self.r2 = r2_score(y_test, preds)

        logging.info(f"Mean Squared Error: {self.mse:.2f}")
        logging.info(f"R² Score: {self.r2:.2f}")

        self._plot_results(y_test, preds)

    def _plot_results(self, y_test, preds):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual')
        ax.plot(preds, label='Predicted')
        ax.legend()
        ax.set_title("Random Forest Stock Price Prediction")
        plt.tight_layout()
        plt.show()

    def get_plot_figure(self):
        """Return plot figure for Streamlit or export use."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        preds = self.model.predict(X_test)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual')
        ax.plot(preds, label='Predicted')
        ax.legend()
        ax.set_title("Random Forest Stock Price Prediction")
        return fig

    def predict_next_day(self) -> float:
        last_row = self.X.iloc[-1].values.reshape(1, -1)
        return self.model.predict(last_row)[0]
