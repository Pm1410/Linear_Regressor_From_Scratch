import yfinance as yf
import pandas as pd
import numpy as np


df = yf.download(
    "BTC-USD",
    start="2016-03-01",
    end="2026-03-01"
)

df.columns = df.columns.get_level_values(0)
print(df.head)
df.to_csv("data/raw/raw.csv")

close_prices=df["Close"]
df["Returns"] = close_prices[1:] / close_prices[:-1] - 1