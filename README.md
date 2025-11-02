# 📊 Real-Time Market Intelligence Dashboard

A trading-focused analytics dashboard for real-time equity market monitoring. Built to simulate the tools a Junior Sales Trading Analyst might use at a firm like IMC Trading.

##  Features

- Live price data (1-minute resolution) from Yahoo Finance via `yfinance`
- PostgreSQL backend for structured storage
- Streamlit dashboard with interactive charts and KPIs
- Moving averages for intraday technical analysis
- News sentiment analysis using NewsAPI + TextBlob NLP

##  Tech Stack

| Tool        | Purpose                              |
|-------------|---------------------------------------|
| Python      | Data extraction and ETL               |
| PostgreSQL  | Time-series database                  |
| Streamlit   | Interactive dashboard (BI layer)      |
| Plotly      | Advanced visualization                |
| SQLAlchemy  | Database interface                    |
| NewsAPI     | Financial news headlines              |
| TextBlob    | Sentiment analysis (NLP)              |

##  Folder Structure

imc-dashboard/
├── dashboard/
│ └── app.py # Streamlit dashboard
├── scripts/
│ └── etl_yfinance.py # Python ETL for market data
├── sql/
│ └── schema.sql # PostgreSQL schema
├── .env # API keys (not committed)
├── requirements.txt
├── README.md

##  Usage

1. Clone this repo
2. Set up PostgreSQL and create a database named `imcdb`
3. Run the ETL script to populate data:

```bash
python scripts/etl_yfinance.py
```
4. Start the dashboard:


streamlit run dashboard/app.py
 Environment Setup
Store your API key in a .env file:


NEWS_API_KEY=your_news_api_key_here
 Make sure .env is in your .gitignore.

 Why This Project Matters
This project simulates how a junior sales trader might:

Monitor live prices with technical overlays

Visualize market trends by symbol

Interpret news sentiment to inform trading decisions

Combine data, SQL, and visualization in a single BI tool

📜 License
MIT License


<img width="1394" alt="Screenshot 2025-06-26 at 01 17 17" src="https://github.com/user-attachments/assets/1c2283cd-cd6e-4ef8-bf06-eb81a8510205" />
<img width="1375" alt="Screenshot 2025-06-26 at 01 18 00" src="https://github.com/user-attachments/assets/39152b13-ac89-4f37-9856-a16fad5d7f13" />

