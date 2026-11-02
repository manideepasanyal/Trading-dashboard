-- PostgreSQL schema. Safe to run repeatedly (IF NOT EXISTS everywhere).
-- Applied automatically by `python -m src.data_pipeline` (see src/db.py).

-- Original 1-minute intraday table (used by the existing dashboard charts).
-- The primary key makes re-running etl_yfinance.py harmless: no duplicate bars.
CREATE TABLE IF NOT EXISTS market_data (
    symbol    TEXT             NOT NULL,
    datetime  TIMESTAMPTZ      NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    BIGINT,
    PRIMARY KEY (symbol, datetime)
);

-- Daily OHLCV history used for feature engineering and modelling.
-- One row per (symbol, trading day); the primary key is what prevents duplicates.
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol       TEXT             NOT NULL,
    date         DATE             NOT NULL,
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       BIGINT           NOT NULL,
    ingested_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, date),
    CHECK (close > 0 AND open > 0 AND high > 0 AND low > 0),
    CHECK (high >= low),
    CHECK (volume >= 0)
);

-- Evaluation metrics in "long" format: one row per (run, symbol, model, split, metric).
-- Written by the training scripts, read by the dashboard. Never hand-edited.
CREATE TABLE IF NOT EXISTS model_results (
    id       BIGSERIAL PRIMARY KEY,
    run_id   TEXT             NOT NULL,
    run_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),
    symbol   TEXT             NOT NULL,
    model    TEXT             NOT NULL,   -- 'baseline_*', 'xgboost', 'lstm'
    task     TEXT             NOT NULL,   -- 'classification' or 'regression'
    split    TEXT             NOT NULL,   -- e.g. 'walk_forward' or 'test'
    metric   TEXT             NOT NULL,
    value    DOUBLE PRECISION
);

-- Per-day predictions for charting actual vs predicted (test period only).
CREATE TABLE IF NOT EXISTS predictions (
    run_id            TEXT             NOT NULL,
    symbol            TEXT             NOT NULL,
    model             TEXT             NOT NULL,
    date              DATE             NOT NULL,   -- the day the prediction is *made* (features known at its close)
    actual_return     DOUBLE PRECISION,            -- realised return of the NEXT trading day (NULL for the latest day)
    predicted_return  DOUBLE PRECISION,            -- regression models
    prob_up           DOUBLE PRECISION,            -- classification models
    PRIMARY KEY (run_id, symbol, model, date)
);

CREATE TABLE IF NOT EXISTS feature_importance (
    run_id      TEXT             NOT NULL,
    symbol      TEXT             NOT NULL,
    model       TEXT             NOT NULL,
    feature     TEXT             NOT NULL,
    importance  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, symbol, model, feature)
);
