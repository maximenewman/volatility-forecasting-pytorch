"""Streamlit dashboard for the volatility forecasting results.

Run: streamlit run app.py
Reads results/metrics.json and results/test_predictions.csv produced by train.py.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent / "results"

st.set_page_config(page_title="S&P 500 Volatility Forecast", layout="wide")
st.title("S&P 500 realized-volatility forecast (PyTorch LSTM)")

metrics = json.loads((RESULTS / "metrics.json").read_text())
preds = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["date"])

st.caption(
    f"Test period {metrics['test_period'][0]} to {metrics['test_period'][1]} - "
    f"target: forward {metrics['params']['horizon']}-day annualized realized volatility"
)

cols = st.columns(len(metrics["metrics"]))
for col, (name, m) in zip(cols, metrics["metrics"].items()):
    col.metric(name, f"RMSE {m['rmse']:.4f}", f"MAE {m['mae']:.4f} | QLIKE {m['qlike']:.4f}")

tab1, tab2 = st.tabs(["Forecast vs realized", "Model config"])
with tab1:
    st.line_chart(preds.set_index("date")[["realized_vol", "lstm_vol", "ewma"]])
    st.image(str(RESULTS / "val_loss.png"), caption="Validation QLIKE per epoch")
with tab2:
    st.json(metrics["params"])
    st.markdown(
        "Custom loss: **QLIKE** (Patton, 2011), a strictly consistent scoring "
        "rule for volatility forecasts, trained in variance space with a "
        "softplus output head."
    )
