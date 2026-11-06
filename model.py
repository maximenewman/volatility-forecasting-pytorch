"""LSTM volatility forecaster in PyTorch."""
import torch


class VolLSTM(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Softplus(),  # variance must be positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)
