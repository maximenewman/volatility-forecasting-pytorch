"""Simple volatility baselines for honest comparison."""
import numpy as np
import pandas as pd


def naive_rolling(frame: pd.DataFrame, window: int = 21) -> np.ndarray:
    """Predict forward vol = trailing `window`-day realized vol."""
    return frame["vol_21"].values


def ewma_riskmetrics(returns: pd.Series, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA variance forecast, aligned to frame index."""
    var = (returns.fillna(0.0) ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var.reindex(returns.index).values) * np.sqrt(252)
