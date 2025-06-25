# ------------------ dashboard/app.py ------------------
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import requests
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine('postgresql://manideepasanyal@localhost:5432/imcdb')

st.set_page_config(page_title="Real-Time Market Dashboard", layout="wide")

st.title("📈 Real-Time Market Intelligence Dashboard")

symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
symbol = st.selectbox("Choose a stock symbol:", symbols)

#fetch data
query = f"""
SELECT * FROM market_data
WHERE symbol = '{symbol}'
ORDER BY datetime DESC
LIMIT 500
"""
df = pd.read_sql(query, engine)
df = df.sort_values(by='datetime')

#moving averages
df['ma_5'] = df['close'].rolling(window=5).mean()
df['ma_20'] = df['close'].rolling(window=20).mean()

#plot with moving averages
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma_5'], mode='lines', name='5-period MA'))
fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma_20'], mode='lines', name='20-period MA'))

fig.update_layout(title=f"{symbol} Intraday Price with Moving Averages",
                  xaxis_title='Time',
                  yaxis_title='Price ($)',
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# KPIs
if not df.empty:
    latest_price = df['close'].iloc[-1]
    latest_volume = df['volume'].iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("Latest Price", f"${latest_price:.2f}")
    col2.metric("Volume", f"{int(latest_volume):,}")
else:
    st.warning("No data available for selected symbol.")

# sentiments
st.markdown("---")
st.subheader(f"📰 Recent News for {symbol}")

api_key = os.getenv("NEWS_API_KEY")
news_url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&apiKey={api_key}&language=en&pageSize=5"

try:
    response = requests.get(news_url)
    articles = response.json().get("articles", [])

    if articles:
        rows = []
        for article in articles:
            title = article['title']
            published = article['publishedAt'][:19].replace("T", " ")
            polarity = TextBlob(title).sentiment.polarity
            sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            rows.append({"Time": published, "Headline": title, "Sentiment": sentiment})

        df_news = pd.DataFrame(rows)
        st.dataframe(df_news)
    else:
        st.info("No news articles found.")
except Exception as e:
    st.error("Failed to load news. Check your API key or connection.")
