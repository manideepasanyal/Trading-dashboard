import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)
symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', '    qnccf]

all_data = []

for symbol in symbols:
    print(f"Fetching data for {symbol}...")  #sanity check

<<<<<<< HEAD
    # never allow more than one symbol at a time
=======
>>>>>>> 43c28c9 (Update app and ETL, add Streamlit config)
    df = yf.download([symbol], period='1d', interval='1m', auto_adjust=True)

    if df.empty:
        print(f"No data for {symbol}, skipping.")
        continue

    print("df.columns BEFORE reset:", df.columns)

    df = df.reset_index()
    df['symbol'] = symbol
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']

    print("Cleaned columns:", df.columns)

    all_data.append(df)


df_all = pd.concat(all_data, ignore_index=True)

print(df_all.head())
print(df_all.columns)

df_all.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']

df_all.to_sql('market_data', engine, if_exists='append', index=False)
