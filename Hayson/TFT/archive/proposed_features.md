| **Feature**                   | **Source / API**                      | **Feature Type**   | **TFT Input Group**                   |
| ----------------------------- | ------------------------------------- | ------------------ | ------------------------------------- |
| OHLC + Volume                 | `yfinance`                            | Past observed      | `time_varying_unknown_reals`          |
| Bid / Ask                     | `yfinance` (Ticker.info)              | Past observed      | `time_varying_unknown_reals`          |
| Technical Indicators          | `compute_ta.py` (e.g. SMA, RSI, MACD) | Past observed      | `time_varying_unknown_reals`          |
| Sector, Market Cap            | `yfinance.info`                       | Static             | `static_categoricals`, `static_reals` |
| Earnings Dates                | `yfinance` (limited)                  | Known future       | `time_varying_known_categoricals`     |
| Dividends / Splits            | `yfinance`                            | Known future       | `time_varying_known_categoricals`     |
| Holidays / Market Closures    | `yfinance` (limited)                  | Known future       | `time_varying_known_categoricals`     |
| News Embeddings               | NewsAPI.org + BERT                    | Past observed      | `time_varying_unknown_reals`          |
| News Sentiment (optional)     | NewsAPI.org or similar                | Past observed      | `time_varying_unknown_reals`          |
| CPI, FEDFUNDS, UNRATE, T10Y2Y | FRED (`fredapi`)                      | Known future reals | `time_varying_known_reals`            |
