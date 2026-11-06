# Volatility forecasting with PyTorch

Forecasting S&P 500 realized volatility with an LSTM trained on a **custom QLIKE
loss**, tracked with **MLflow**, with a **Streamlit** dashboard for the results.

## Task

Predict the forward 5-day annualized realized volatility of S&P 500 log returns,
using only information available at prediction time: lagged returns, trailing
realized-volatility windows (5/10/21 days), Parkinson high-low range volatility,
mean absolute returns and high-low range features.

## Data

Daily ^GSPC OHLC, ~15 years, pulled from the Yahoo Finance chart endpoint by
`data.py` and cached to `data/spx.csv` (committed, so the repo runs offline).
Chronological split: train <= 2021, validation 2022-2023, test 2024 onward.

## Model

- `model.py`: 2-layer LSTM (hidden 64) with a small MLP head and softplus
  output, predicting variance (positivity by construction).
- `losses.py`: custom **QLIKE loss** (Patton, 2011), a strictly consistent
  scoring rule for volatility forecasts, plus a regime-weighted MSE.
- `baselines.py`: naive 21-day rolling volatility and RiskMetrics EWMA
  (lambda = 0.94), evaluated on the same test window for honest comparison.

## Results (held-out test period)

All numbers below are produced by `train.py` and logged to MLflow; see
`results/metrics.json` for the exact run.

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| LSTM (QLIKE loss) | 0.0817 | 0.0524 | 0.447 |
| EWMA (RiskMetrics) | 0.0894 | 0.0580 | 0.602 |
| Naive 21-day rolling | 0.0933 | 0.0580 | 0.684 |

The LSTM beats both baselines on all three metrics.

![Test forecast](results/test_forecast.png)

## Reproduce

```bash
pip install -r requirements.txt
python train.py          # trains, evaluates, logs to MLflow (sqlite backend)
mlflow ui --backend-store-uri sqlite:///mlflow.db
streamlit run app.py     # results dashboard
```
