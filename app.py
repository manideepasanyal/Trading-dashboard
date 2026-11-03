# ------------------ app.py: Real-Time Market Intelligence Dashboard ------------------
# Original sections (intraday chart + moving averages, price/volume KPIs, news sentiment)
# are kept. New sections: daily price & risk, forecasting, model comparison, feature importance.
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from textblob import TextBlob

from src.config import SYMBOLS, get_secret
from src.db import init_schema, load_daily_prices
from src.features import add_features
from src.risk import annualised_volatility, historical_var, max_drawdown

st.set_page_config(page_title="Real-Time Market Dashboard", layout="wide")


def secret(name: str):
    """st.secrets first (Streamlit Cloud), then environment / local secrets file."""
    try:
        return st.secrets[name]
    except Exception:
        return get_secret(name)


@st.cache_resource
def get_db():
    url = secret("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL is not set. Copy .streamlit/secrets.toml.example to "
                 ".streamlit/secrets.toml and fill it in (see README).")
        st.stop()
    engine = create_engine(url, pool_pre_ping=True)
    init_schema(engine)          # idempotent: creates any missing tables
    return engine


engine = get_db()

st.title("Real-Time Market Intelligence Dashboard")
symbol = st.selectbox("Choose a stock symbol:", SYMBOLS)
if st.sidebar.button("Refresh data"):       # query results are cached for 5 minutes
    st.cache_data.clear()


# ----------------------------------------------------------------- data access
@st.cache_data(ttl=300)
def load_intraday(sym: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT * FROM market_data WHERE symbol = :s ORDER BY datetime DESC LIMIT 500"),
        engine, params={"s": sym})
    return df.sort_values("datetime")


@st.cache_data(ttl=300)
def load_daily(sym: str) -> pd.DataFrame:
    return load_daily_prices(engine, sym)


@st.cache_data(ttl=300)
def load_model_data(sym: str) -> dict:
    """Latest classification run and latest regression run for this symbol."""
    res = pd.read_sql(text("SELECT run_id, run_at, model, task, split, metric, value "
                           "FROM model_results WHERE symbol = :s"), engine, params={"s": sym})
    out = {"metrics": {}, "run_id": {}, "pred": {}, "importance": pd.DataFrame()}
    for task, model in (("classification", "xgboost"), ("regression", "lstm")):
        sub = res[res["task"] == task]
        if sub.empty:
            continue
        run_id = sub.sort_values("run_at").iloc[-1]["run_id"]
        out["run_id"][task] = run_id
        out["metrics"][task] = sub[sub["run_id"] == run_id]
        out["pred"][model] = pd.read_sql(
            text("SELECT date, actual_return, predicted_return, prob_up FROM predictions "
                 "WHERE run_id = :r AND symbol = :s AND model = :m ORDER BY date"),
            engine, params={"r": run_id, "s": sym, "m": model})
    if "classification" in out["run_id"]:
        out["importance"] = pd.read_sql(
            text("SELECT feature, importance FROM feature_importance "
                 "WHERE run_id = :r AND symbol = :s AND model = 'xgboost' ORDER BY importance"),
            engine, params={"r": out["run_id"]["classification"], "s": sym})
    return out


# ------------------------------------------------- 1. intraday (original section)
df = load_intraday(symbol)
if df.empty:
    st.info("No intraday data yet. Run `python etl_yfinance.py` to collect 1-minute bars.")
else:
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["ma_5"], mode="lines", name="5-period MA"))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["ma_20"], mode="lines", name="20-period MA"))
    fig.update_layout(title=f"{symbol} Intraday Price with Moving Averages",
                      xaxis_title="Time", yaxis_title="Price ($)", legend=dict(x=0, y=1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Latest Price", f"${df['close'].iloc[-1]:.2f}")
    col2.metric("Volume", f"{int(df['volume'].iloc[-1]):,}")

# ------------------------------------------------------- 2. daily price and risk
st.markdown("---")
st.subheader(f"Daily price & risk: {symbol}")
daily = load_daily(symbol)
if daily.empty:
    st.info("No daily history yet. Run `python -m src.data_pipeline` first.")
else:
    feat = add_features(daily)
    years = st.slider("Years of history to show", 1, 5, 2)
    shown = feat[feat["date"] >= feat["date"].max() - pd.Timedelta(days=365 * years)]

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=shown["date"], y=shown["close"], mode="lines", name="Close"))
    for w in (5, 10, 20, 50):
        f1.add_trace(go.Scatter(x=shown["date"], y=shown[f"ma_{w}"], mode="lines", name=f"{w}-day MA"))
    f1.update_layout(title=f"{symbol} daily close with moving averages",
                     xaxis_title="Date", yaxis_title="Price ($, split/dividend adjusted)",
                     legend=dict(x=0, y=1))
    st.plotly_chart(f1, use_container_width=True)

    returns = feat["ret_1d"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Volatility (20d, annualised)", f"{annualised_volatility(returns):.1%}")
    k2.metric("Max drawdown (shown period)", f"{max_drawdown(shown['close']):.1%}")
    k3.metric("1-day 95% historical VaR", f"{historical_var(returns):.2%}")
    k4.metric("RSI (14)", f"{feat['rsi_14'].iloc[-1]:.0f}")
    st.caption("VaR = the one-day loss exceeded on only 5% of past days. It describes history, "
               "not the worst case.")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=shown["date"], y=shown["vol_5d"] * (252 ** 0.5), mode="lines",
                            name="5-day vol (ann.)"))
    f2.add_trace(go.Scatter(x=shown["date"], y=shown["vol_20d_annualised"], mode="lines",
                            name="20-day vol (ann.)"))
    f2.update_layout(title="Rolling volatility (annualised)", yaxis_tickformat=".0%",
                     xaxis_title="Date", legend=dict(x=0, y=1))
    st.plotly_chart(f2, use_container_width=True)

# ------------------------------------------------------------- 3. forecasting
st.markdown("---")
st.subheader(f"Forecasting: {symbol}")
models = load_model_data(symbol)
if not models["run_id"]:
    st.info("No trained models yet. Run `python -m src.train_xgb` and `python -m src.train_lstm`.")
else:
    xgb_pred = models["pred"].get("xgboost", pd.DataFrame())
    lstm_pred = models["pred"].get("lstm", pd.DataFrame())

    a, b = st.columns(2)
    live_x = xgb_pred[xgb_pred["actual_return"].isna()] if not xgb_pred.empty else xgb_pred
    if not live_x.empty:
        row = live_x.iloc[-1]
        a.metric("XGBoost: probability next day is UP", f"{row['prob_up']:.1%}",
                 help=f"Made from data up to the close of {row['date']}. About 50% means no signal.")
    live_l = lstm_pred[lstm_pred["actual_return"].isna()] if not lstm_pred.empty else lstm_pred
    if not live_l.empty:
        row = live_l.iloc[-1]
        b.metric("LSTM: forecast next-day return", f"{row['predicted_return']:+.2%}",
                 help=f"Made from data up to the close of {row['date']}.")

    if not lstm_pred.empty:
        t = lstm_pred.dropna(subset=["actual_return"])
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=t["date"], y=t["actual_return"], mode="lines", name="Actual"))
        f3.add_trace(go.Scatter(x=t["date"], y=t["predicted_return"], mode="lines", name="LSTM predicted"))
        f3.update_layout(title="LSTM: actual vs predicted next-day return (held-out test period)",
                         yaxis_tickformat=".1%", xaxis_title="Date (day the prediction was made)",
                         legend=dict(x=0, y=1))
        st.plotly_chart(f3, use_container_width=True)

    if not xgb_pred.empty:
        t = xgb_pred.dropna(subset=["actual_return"])
        colours = ["green" if r > 0 else "red" for r in t["actual_return"]]
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=t["date"], y=t["prob_up"], mode="lines", name="P(up)",
                                line=dict(color="lightgray")))
        f4.add_trace(go.Scatter(x=t["date"], y=t["prob_up"], mode="markers",
                                marker=dict(color=colours, size=5), name="Outcome (green = went up)"))
        f4.add_hline(y=0.5, line_dash="dot")
        f4.update_layout(title="XGBoost: predicted probability of an up day (walk-forward, out-of-sample)",
                         yaxis_range=[0, 1], xaxis_title="Date (day the prediction was made)",
                         legend=dict(x=0, y=1))
        st.plotly_chart(f4, use_container_width=True)

# ------------------------------------------------------- 4. model comparison
    st.markdown("---")
    st.subheader("Model performance comparison")
    st.caption("Classification and regression metrics answer different questions and are shown "
               "separately. Only directional accuracy appears in both. Baselines are the yardstick: "
               "a model only means something if it beats them out of sample.")

    if "classification" in models["metrics"]:
        c = models["metrics"]["classification"].pivot_table(
            index="model", columns="metric", values="value")
        cols = [m for m in ("accuracy", "precision", "recall", "f1", "directional_accuracy",
                            "roc_auc", "share_up_days") if m in c.columns]
        st.markdown("**Classification: will the next day be up?** (XGBoost, walk-forward validation)")
        st.dataframe(c[cols].style.format("{:.3f}", na_rep="n/a"), use_container_width=True)
    if "regression" in models["metrics"]:
        r = models["metrics"]["regression"]
        r = r[r["split"] == "test"].pivot_table(index="model", columns="metric", values="value")
        cols = [m for m in ("mae", "rmse", "directional_accuracy") if m in r.columns]
        st.markdown("**Regression: next-day return** (LSTM, held-out test period; lower MAE/RMSE is better)")
        st.dataframe(r[cols].style.format("{:.5f}", na_rep="n/a"), use_container_width=True)
    st.warning("These are historical, out-of-sample statistics, not evidence of a profitable strategy. "
               "They ignore trading costs, slippage and position sizing. Next-day direction is close to "
               "a coin flip for liquid stocks, so results near the baselines are normal.")

# ------------------------------------------------------ 5. feature importance
    imp = models["importance"]
    if not imp.empty:
        st.markdown("---")
        st.subheader("XGBoost feature importance")
        f5 = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h"))
        f5.update_layout(xaxis_title="Importance (gain-based, final model)", height=450)
        st.plotly_chart(f5, use_container_width=True)
        st.caption("Importance shows what the model used, not what causes returns. With noisy "
                   "targets, the ranking can change between runs.")


# ------------------------------------------------ 6. news and sentiment (original)
@st.cache_data(ttl=600)
def fetch_news(sym: str, key: str):
    resp = requests.get("https://newsapi.org/v2/everything",
                        params={"q": sym, "sortBy": "publishedAt", "apiKey": key,
                                "language": "en", "pageSize": 5}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("articles", [])


st.markdown("---")
st.subheader(f"Recent News for {symbol}")
api_key = secret("NEWS_API_KEY")
if not api_key:
    st.info("Set NEWS_API_KEY in .streamlit/secrets.toml to enable news sentiment.")
else:
    try:
        articles = fetch_news(symbol, api_key)
        if articles:
            rows = []
            for article in articles:
                title = article["title"]
                polarity = TextBlob(title).sentiment.polarity
                sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
                rows.append({"Time": article["publishedAt"][:19].replace("T", " "),
                             "Headline": title, "Sentiment": sentiment})
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No news articles found.")
    except Exception as exc:
        st.error(f"Failed to load news ({type(exc).__name__}). Check your API key or connection.")
