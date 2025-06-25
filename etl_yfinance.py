import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Connect to your PostgreSQL DB (replace with your actual username if needed)
engine = create_engine('postgresql://manideepasanyal@localhost:5432/imcdb')

symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

all_data = []

for symbol in symbols:
    print(f"Fetching data for {symbol}...")  # Sanity check

    # 🔒 Explicit: never allow more than one symbol at a time
    df = yf.download([symbol], period='1d', interval='1m', auto_adjust=True)

    if df.empty:
        print(f"No data for {symbol}, skipping.")
        continue

    print("df.columns BEFORE reset:", df.columns)

    df = df.reset_index()
    df['symbol'] = symbol
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']

    print("✅ Cleaned columns:", df.columns)

    all_data.append(df)


# Combine into one DataFrame
df_all = pd.concat(all_data, ignore_index=True)

# Print to confirm structure
print(df_all.head())
print(df_all.columns)

# Force rename to match SQL table exactly
df_all.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']

# Upload to SQL
df_all.to_sql('market_data', engine, if_exists='append', index=False)
