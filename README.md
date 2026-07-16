# Company Ranking / Research Pipeline

A Python system that ranks companies by combining:
1. Business fundamentals (SEC EDGAR XBRL data)
2. Price/momentum behavior
3. News attention & sentiment (volume-normalized, not raw mention counts)
4. Structured commercial-event extraction (via the Claude API)
5. Valuation & financial risk

It outputs a transparent, explainable composite score, and separately
supports training a walk-forward ML model to test whether any of this
actually predicts sector-relative outperformance historically.

## Quick start

```bash
pip install -r requirements.txt
python main.py demo
```

`demo` runs entirely on synthetic data — no API keys or network access
needed. It exists so you can see every stage of the pipeline (feature
building → composite scoring → explanation → walk-forward ML → held-out
predictions) run successfully before touching real data.

## Running on real data

```bash
export SEC_USER_AGENT="Your Name your_email@example.com"   # required by SEC
export NEWS_API_KEY="..."                                    # https://newsapi.org
export ANTHROPIC_API_KEY="..."                                # for event extraction
python main.py rank
```

This hits three external services:
- `data.sec.gov` — free, no key, but requires a real User-Agent header
- `newsapi.org` — free tier is limited (100 req/day, no historical depth);
  fine for prototyping, not for a serious backtest
- `api.anthropic.com` — used to turn news text into structured,
  materiality-scored events instead of generic sentiment

Price data comes from stooq.com (free, no key) with an optional yfinance
fallback.

## Backtesting

`backtest/` contains portfolio construction (top-decile, equal-weight,
monthly rebalance), transaction cost modeling (don't skip this — it's
what separates a real backtest from a fantasy), and evaluation metrics
(Sharpe, max drawdown, excess return vs benchmark).

Use `backtest/evaluation.compare_models()` to build the "did this data
source actually help" comparison table:

| Model | ROC AUC | Top-decile excess return | Sharpe | Max drawdown |
|---|---|---|---|---|
| Baseline (fundamentals + price only) | ... | ... | ... | ... |
| + News volume | ... | ... | ... | ... |
| + NLP events | ... | ... | ... | ... |

Don't assume news or event data improves the model — measure it.

## What this is and isn't

- This is a research/ranking tool, not a trading bot. It doesn't place
  orders.
- The composite score's weights (`config.py`) are a reasonable starting
  point, not validated truths. Backtest before trusting them.
- Point-in-time correctness matters: fundamentals only use data that
  was actually *filed* by the as-of date (see `data/fundamentals.py`),
  and features should never peek at future news or prices.
- Known bias traps this code tries to guard against: look-ahead bias,
  survivorship bias (you must supply a universe that includes delisted
  names for a real backtest), data leakage, and news-source duplication.
  It does NOT currently handle survivorship bias for you — that
  requires a historical index-membership dataset this project doesn't
  ship with.
- Nothing here guarantees you'll find "the next big company." It's a
  disciplined way to combine evidence, not a crystal ball.

## Project layout

```
config.py             # API keys, universe, feature windows, scoring weights
universe.py           # ticker/company-name alias table (avoids "ON", "ALL", "AI" collisions)
data/
  prices.py           # stooq/yfinance price history
  fundamentals.py      # SEC EDGAR XBRL company facts, point-in-time safe
  news.py             # NewsAPI fetch + near-duplicate removal
  sec_filings.py      # 8-K filing index as a high-reliability event source
features/
  price_features.py    # momentum, volatility, distance from 52w high
  fundamental_features.py  # growth, margins, leverage, valuation
  news_features.py     # acceleration, z-score attention, source weighting
  text_features.py     # FinBERT sentiment + Claude-based event extraction
models/
  scoring_model.py      # transparent weighted composite score + explanations
  train_model.py        # walk-forward logistic regression baseline
  predict.py            # apply a saved model to the current universe
backtest/
  portfolio.py          # top-decile equal-weight construction
  transaction_costs.py  # turnover-based cost modeling
  evaluation.py          # Sharpe, drawdown, model comparison table
main.py                 # `python main.py demo` or `python main.py rank`
```
