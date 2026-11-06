"""Feature engineering for volatility forecasting.

Target: forward 5-day realized volatility of log returns (annualized).
Features are built only from information available at time t (no lookahead).
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5",
    "vol_5", "vol_10", "vol_21",
    "parkinson_5", "parkinson_21",
    "absret_5", "absret_21",
    "range_5",
]


def build_frame(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    logp = np.log(df["Close"])
    ret = logp.diff()
    out["ret_1"] = ret
    out["ret_2"] = ret.shift(1)
    out["ret_3"] = ret.shift(2)
    out["ret_5"] = logp - logp.shift(5)
    out["vol_5"] = ret.rolling(5).std() * np.sqrt(252)
    out["vol_10"] = ret.rolling(10).std() * np.sqrt(252)
    out["vol_21"] = ret.rolling(21).std() * np.sqrt(252)
    # Parkinson (1980) high-low range volatility estimator
    hl = np.log(df["High"] / df["Low"]) ** 2
    out["parkinson_5"] = np.sqrt(hl.rolling(5).mean() / (4 * np.log(2))) * np.sqrt(252)
    out["parkinson_21"] = np.sqrt(hl.rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252)
    out["absret_5"] = ret.abs().rolling(5).mean() * np.sqrt(252)
    out["absret_21"] = ret.abs().rolling(21).mean() * np.sqrt(252)
    out["range_5"] = (df["High"].rolling(5).max() - df["Low"].rolling(5).min()) / df["Close"]
    # Target: annualized forward horizon-day realized volatility
    fwd = ret.shift(-1)
    out["target"] = fwd.rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(252)
    return out.dropna()


def make_sequences(frame: pd.DataFrame, lookback: int = 20):
    """Return X (N, lookback, F), y (N,), dates (N,)."""
    X_all = frame[FEATURE_COLS].values.astype(np.float32)
    y_all = frame["target"].values.astype(np.float32)
    dates = frame.index
    X, y, d = [], [], []
    for i in range(lookback - 1, len(frame)):
        X.append(X_all[i - lookback + 1 : i + 1])
        y.append(y_all[i])
        d.append(dates[i])
    return np.stack(X), np.array(y), np.array(d)
