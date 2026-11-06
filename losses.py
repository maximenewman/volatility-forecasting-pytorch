"""Custom loss functions for volatility forecasting."""
import torch


class QLikeLoss(torch.nn.Module):
    """QLIKE loss: mean(x - log x - 1) with x = y_true / y_pred (both variances).

    QLIKE is a strictly consistent scoring rule for volatility estimation
    (Patton, 2011); unlike MSE it ranks competing volatility forecasts
    correctly even with a noisy realized-volatility proxy.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_true / y_pred.clamp_min(1e-8)
        return (x - torch.log(x.clamp_min(1e-8)) - 1.0).mean()


class WeightedMSELoss(torch.nn.Module):
    """MSE weighted toward high-volatility regimes (they matter more in risk work)."""

    def __init__(self, weight_scale: float = 1.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        w = 1.0 + self.weight_scale * y_true / y_true.mean().clamp_min(1e-8)
        return (w * (y_pred - y_true) ** 2).mean()
