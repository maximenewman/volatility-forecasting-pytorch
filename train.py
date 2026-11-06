"""Train the LSTM volatility forecaster with MLflow tracking.

Run: python train.py
Everything is reproducible: fixed seed, chronological split, baselines logged
alongside the model so the MLflow run tells the full story.
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

import baselines
import data as data_mod
import features as feat_mod
from losses import QLikeLoss
from model import VolLSTM

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PARAMS = dict(
    lookback=20,
    hidden=64,
    layers=2,
    dropout=0.1,
    lr=1e-3,
    batch_size=256,
    max_epochs=100,
    patience=12,
    seed=42,
    loss="qlike",
    horizon=5,
    train_end="2021-12-31",
    val_end="2023-12-31",
)


def qlike(y_pred_var: np.ndarray, y_true_var: np.ndarray) -> float:
    x = y_true_var / np.maximum(y_pred_var, 1e-8)
    return float(np.mean(x - np.log(np.maximum(x, 1e-8)) - 1.0))


def main() -> None:
    random.seed(PARAMS["seed"])
    np.random.seed(PARAMS["seed"])
    torch.manual_seed(PARAMS["seed"])

    df = data_mod.download_spx()
    frame = feat_mod.build_frame(df, horizon=PARAMS["horizon"])
    X, y, dates = feat_mod.make_sequences(frame, lookback=PARAMS["lookback"])

    dates = pd.to_datetime(dates)
    train_end = pd.Timestamp(PARAMS["train_end"])
    val_end = pd.Timestamp(PARAMS["val_end"])
    tr = dates <= train_end
    va = (dates > train_end) & (dates <= val_end)
    te = dates > val_end

    # normalize features with train-period statistics only
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xn = (X - mu) / sd
    y_var = (y ** 2).astype(np.float32)  # model predicts variance

    def tensors(mask):
        return (
            torch.from_numpy(Xn[mask]),
            torch.from_numpy(y_var[mask]),
        )

    Xtr, ytr = tensors(tr)
    Xva, yva = tensors(va)
    Xte, yte = tensors(te)

    model = VolLSTM(X.shape[-1], PARAMS["hidden"], PARAMS["layers"], PARAMS["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])
    loss_fn = QLikeLoss()

    mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).parent / "mlflow.db"))
    mlflow.set_experiment("spx-volatility")
    with mlflow.start_run(run_name="lstm-qlike"):
        mlflow.log_params(PARAMS)
        mlflow.log_params({"train_n": int(tr.sum()), "val_n": int(va.sum()), "test_n": int(te.sum())})

        best_val, best_state, bad = np.inf, None, 0
        history = []
        for epoch in range(1, PARAMS["max_epochs"] + 1):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), PARAMS["batch_size"]):
                idx = perm[i : i + PARAMS["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva), yva))
            history.append(val_loss)
            mlflow.log_metric("val_qlike", val_loss, step=epoch)
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PARAMS["patience"]:
                    break
        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_qlike", best_val)
        mlflow.log_metric("epochs_trained", len(history))

        # --- evaluation on the held-out test period ---
        model.eval()
        with torch.no_grad():
            pred_var = model(Xte).numpy()
        pred_vol = np.sqrt(np.maximum(pred_var, 0.0))
        true_vol = np.sqrt(yte.numpy())

        # baselines aligned to test dates
        frame_test = frame.loc[dates[te]]
        base_naive = frame_test["vol_21"].values
        rets = np.log(df["Close"]).diff()
        base_ewma_full = pd.Series(baselines.ewma_riskmetrics(rets), index=df.index)
        base_ewma = base_ewma_full.reindex(dates[te]).values

        metrics = {}
        for name, pv in [("lstm", pred_vol), ("naive_21d", base_naive), ("ewma_riskmetrics", base_ewma)]:
            metrics[name] = {
                "rmse": float(np.sqrt(np.mean((pv - true_vol) ** 2))),
                "mae": float(np.mean(np.abs(pv - true_vol))),
                "qlike": qlike(pv ** 2, true_vol ** 2),
            }
        for name, m in metrics.items():
            for k, v in m.items():
                mlflow.log_metric(f"test_{k}_{name}", v)

        # --- artifacts ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates[te], true_vol, label="realized (fwd 5d)", lw=1)
        ax.plot(dates[te], pred_vol, label="LSTM forecast", lw=1)
        ax.plot(dates[te], base_ewma, label="EWMA baseline", lw=1, alpha=0.7)
        ax.legend(); ax.set_title("Test period: S&P 500 volatility forecast")
        fig.tight_layout()
        plot_path = RESULTS / "test_forecast.png"
        fig.savefig(plot_path, dpi=120)
        mlflow.log_artifact(str(plot_path))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history)
        ax.set_xlabel("epoch"); ax.set_ylabel("val QLIKE")
        ax.set_title("Validation loss")
        fig.tight_layout()
        loss_path = RESULTS / "val_loss.png"
        fig.savefig(loss_path, dpi=120)
        mlflow.log_artifact(str(loss_path))
        plt.close(fig)

        pd.DataFrame(
            {"date": dates[te], "realized_vol": true_vol, "lstm_vol": pred_vol,
             "naive_21d": base_naive, "ewma": base_ewma}
        ).to_csv(RESULTS / "test_predictions.csv", index=False)

        summary = {
            "params": PARAMS,
            "best_val_qlike": best_val,
            "epochs_trained": len(history),
            "test_period": [str(dates[te].min().date()), str(dates[te].max().date())],
            "metrics": metrics,
        }
        (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(RESULTS / "metrics.json"))
        print(json.dumps(metrics, indent=2))
        print("test period:", summary["test_period"], "epochs:", len(history))


if __name__ == "__main__":
    main()
