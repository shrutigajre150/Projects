"""Forecast accuracy metrics."""
import numpy as np
import pandas as pd


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mape(y, f):
    return float(np.mean(np.abs((y - f) / y)) * 100)


def wape(y, f):
    """Weighted absolute percentage error: total absolute error / total actuals."""
    return float(np.sum(np.abs(y - f)) / np.sum(np.abs(y)) * 100)


def bias(y, f):
    """Forecast bias as % of actuals; positive = over-forecasting."""
    return float(np.sum(f - y) / np.sum(y) * 100)


def mase(y, f, scale):
    """Mean absolute scaled error vs. in-sample seasonal-naive MAE."""
    return float(np.mean(np.abs(y - f)) / scale)


def summarize(df: pd.DataFrame, scales: dict) -> pd.DataFrame:
    """df has columns id, model, y, yhat. Returns overall metrics by model.

    MASE is computed per series (each with its own scale) and averaged.
    """
    rows = []
    for model, g in df.groupby("model", sort=False):
        y, f = g["y"].to_numpy(), g["yhat"].to_numpy()
        mase_by_series = [
            mase(s["y"].to_numpy(), s["yhat"].to_numpy(), scales[sid])
            for sid, s in g.groupby("id")
        ]
        rows.append({
            "model": model,
            "WAPE_%": wape(y, f),
            "MAPE_%": mape(y, f),
            "MAE": mae(y, f),
            "RMSE": rmse(y, f),
            "MASE": float(np.mean(mase_by_series)),
            "Bias_%": bias(y, f),
        })
    return pd.DataFrame(rows)


def by_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, sid), g in df.groupby(["model", "id"], sort=False):
        y, f = g["y"].to_numpy(), g["yhat"].to_numpy()
        rows.append({"model": model, "id": sid, "WAPE_%": wape(y, f),
                     "MAE": mae(y, f), "Bias_%": bias(y, f)})
    return pd.DataFrame(rows)
