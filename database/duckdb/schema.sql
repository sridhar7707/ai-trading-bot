CREATE TABLE IF NOT EXISTS price_history (
    symbol  VARCHAR,
    date    DATE,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE,
    volume  BIGINT,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_date   DATE PRIMARY KEY,
    portfolio_value DOUBLE,
    cash_balance    DOUBLE,
    health_score    DOUBLE,
    max_drawdown    DOUBLE,
    sharpe_ratio    DOUBLE
);

CREATE TABLE IF NOT EXISTS recommendation_history (
    symbol                  VARCHAR,
    prediction_date         DATE,
    recommendation          VARCHAR,
    confidence              DOUBLE,
    prev_recommendation     VARCHAR DEFAULT NULL,
    change_reason           VARCHAR DEFAULT NULL,
    price_at_recommendation DOUBLE  DEFAULT NULL,
    actual_return           DOUBLE  DEFAULT NULL,
    resolved                BOOLEAN DEFAULT FALSE,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, prediction_date)
);

-- Migration: add columns to existing DBs (no-op when already present)
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS prev_recommendation     VARCHAR DEFAULT NULL;
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS change_reason           VARCHAR DEFAULT NULL;
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS price_at_recommendation DOUBLE  DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_history(symbol, date);
CREATE INDEX IF NOT EXISTS idx_rec_symbol ON recommendation_history(symbol)
