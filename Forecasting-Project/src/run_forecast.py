"""End-to-end pipeline: load data, backtest all models, forecast the next 52 weeks.

Usage (from the project root):
    python src/run_forecast.py

Outputs go to reports/metrics/ and reports/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import by_series, summarize
from models import all_models, MeanEnsemble, ml_models
from plots import plot_future, plot_holdout, plot_model_comparison

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "walmart_sales_weekly.csv"
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"

HORIZON = 52            # long-range holdout, as in the original R project
ROLL_HORIZON = 13       # quarterly planning horizon
ROLL_FOLDS = 4          # expanding-window origins covering the last 52 weeks

# Walmart's holiday weeks for the forecast year (from the Kaggle test calendar).
FUTURE_HOLIDAYS = pd.to_datetime(
    ["2012-11-23", "2012-12-28", "2013-02-08", "2013-09-06"])


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[["id", "Date", "Weekly_Sales", "IsHoliday"]]
    return df.sort_values(["id", "Date"]).reset_index(drop=True)


def seasonal_naive_scale(train):
    """Per-series in-sample MAE of the seasonal naive forecast (MASE denominator)."""
    scales = {}
    for sid, g in train.groupby("id"):
        s = g.sort_values("Date")["Weekly_Sales"].to_numpy()
        scales[sid] = float(np.mean(np.abs(s[52:] - s[:-52])))
    return scales


def backtest(df, cutoff, horizon):
    """Train on data up to `cutoff`, forecast the next `horizon` weeks."""
    dates = np.sort(df["Date"].unique())
    test_dates = dates[(dates > cutoff)][:horizon]
    train = df[df["Date"] <= cutoff]
    test = df[df["Date"].isin(test_dates)].reset_index(drop=True)
    out = []
    for model in all_models():
        model.fit(train)
        yhat = model.predict(test[["id", "Date", "IsHoliday"]])
        out.append(pd.DataFrame({"model": model.name, "id": test["id"],
                                 "Date": test["Date"], "y": test["Weekly_Sales"],
                                 "yhat": yhat}))
    return pd.concat(out, ignore_index=True), seasonal_naive_scale(train)


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load()
    dates = np.sort(df["Date"].unique())
    n_series, n_weeks = df["id"].nunique(), len(dates)
    print(f"{n_series} series x {n_weeks} weeks "
          f"({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()})")

    # 1) 52-week holdout (single origin)
    cutoff = dates[-HORIZON - 1]
    hold, scales = backtest(df, cutoff, HORIZON)
    hold_summary = summarize(hold, scales).sort_values("WAPE_%")
    hold_summary.to_csv(METRICS / "holdout_52wk_overall.csv", index=False)
    by_series(hold).to_csv(METRICS / "holdout_52wk_by_series.csv", index=False)
    hold.to_csv(METRICS / "holdout_52wk_predictions.csv", index=False)

    # 2) Rolling-origin backtest: 4 x 13-week folds, expanding window
    folds = []
    for k in range(ROLL_FOLDS):
        c = dates[-HORIZON - 1 + k * ROLL_HORIZON]
        res, sc = backtest(df, c, ROLL_HORIZON)
        res["fold"] = k + 1
        folds.append((res, sc))
    roll = pd.concat([r for r, _ in folds], ignore_index=True)
    # MASE per fold uses that fold's own training scale; then average folds
    roll_fold_summ = pd.concat(
        [summarize(r, sc).assign(fold=i + 1) for i, (r, sc) in enumerate(folds)])
    roll_summary = summarize(roll, folds[-1][1])  # WAPE/MAE/RMSE pooled over folds
    roll_summary["MASE"] = roll_summary["model"].map(
        roll_fold_summ.groupby("model")["MASE"].mean())
    roll_summary["WAPE_fold_sd"] = roll_summary["model"].map(
        roll_fold_summ.groupby("model")["WAPE_%"].std())
    roll_summary = roll_summary.sort_values("WAPE_%")
    roll_summary.to_csv(METRICS / "rolling_13wk_overall.csv", index=False)
    roll_fold_summ.to_csv(METRICS / "rolling_13wk_by_fold.csv", index=False)

    # 3) Refit the ensemble on all history and forecast the next 52 weeks
    future_dates = pd.date_range(pd.Timestamp(dates[-1]) + pd.Timedelta(weeks=1),
                                 periods=HORIZON, freq="7D")
    future = pd.DataFrame([(sid, d) for sid in sorted(df["id"].unique())
                           for d in future_dates], columns=["id", "Date"])
    future["IsHoliday"] = future["Date"].isin(FUTURE_HOLIDAYS)
    ens = MeanEnsemble(ml_models()).fit(df)
    future["forecast"] = ens.predict(future)
    # Empirical 80% interval from the ensemble's rolling-backtest errors (y / yhat)
    r = roll[roll["model"] == ens.name]
    lo, hi = np.quantile(r["y"] / r["yhat"], [0.10, 0.90])
    future["lower_80"], future["upper_80"] = future["forecast"] * lo, future["forecast"] * hi
    future.to_csv(METRICS / "future_52wk_forecast.csv", index=False)

    # Figures
    plot_holdout(df, hold, FIGURES / "holdout_forecast_vs_actual.png")
    plot_model_comparison(hold_summary, roll_summary, FIGURES / "model_comparison_wape.png")
    plot_future(df, future, FIGURES / "future_52wk_forecast.png")

    pd.set_option("display.width", 160)
    fmt = {"WAPE_%": "{:.2f}", "MAPE_%": "{:.2f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}",
           "MASE": "{:.3f}", "Bias_%": "{:+.2f}", "WAPE_fold_sd": "{:.2f}"}
    for title, t in (("52-week holdout", hold_summary), ("Rolling 4 x 13-week backtest", roll_summary)):
        print(f"\n{title}")
        print(t.to_string(index=False, formatters={k: v.format for k, v in fmt.items() if k in t}))
    print(f"\n80% interval multipliers from backtest errors: [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
