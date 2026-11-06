"""Feature engineering for weekly multi-series sales forecasting.

Every feature is either known in advance (calendar, holiday flags) or comes from
at least 52 weeks in the past (lag_52). That makes the models "direct": any week
within a 52-week horizon can be forecast without feeding predictions back in,
and nothing from the forecast window leaks into training.

Exogenous columns in the raw data (CPI, fuel price, temperature, markdowns) are
deliberately left out: they are not known at forecast time, and the markdown
fields are missing before Nov 2011.
"""
import numpy as np
import pandas as pd

FOURIER_K = 3
PERIOD = 52.1775  # average number of weeks per year


def _weeks_to_christmas(dates: pd.Series) -> pd.Series:
    xmas = pd.to_datetime(dates.dt.year.astype(str) + "-12-25")
    return ((xmas - dates).dt.days / 7.0).round()


def add_calendar_features(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    d = out["Date"]
    out["t"] = (d - origin).dt.days / 7.0
    woy = d.dt.isocalendar().week.astype(float)
    for k in range(1, FOURIER_K + 1):
        out[f"sin_{k}"] = np.sin(2 * np.pi * k * woy / PERIOD)
        out[f"cos_{k}"] = np.cos(2 * np.pi * k * woy / PERIOD)
    out["week_of_year"] = woy
    out["month"] = d.dt.month.astype(float)
    hol = out["IsHoliday"].astype(bool)
    out["is_holiday"] = hol.astype(float)
    out["is_thanksgiving"] = (hol & (d.dt.month == 11)).astype(float)
    out["is_superbowl"] = (hol & (d.dt.month == 2)).astype(float)
    out["is_laborday"] = (hol & (d.dt.month == 9)).astype(float)
    # Christmas falls in a different Walmart week each year, so lag_52 misaligns
    # it; an explicit "weeks until Christmas" feature lets models re-align it.
    wtx = _weeks_to_christmas(d)
    for w in (0, 1, 2, 3):
        out[f"xmas_minus_{w}"] = (wtx == w).astype(float)
    return out


def add_lag_features(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add same-week-last-year sales (lag_52) looked up from `history` only."""
    hist = history[["id", "Date", "Weekly_Sales"]].copy()
    hist["Date"] = hist["Date"] + pd.Timedelta(weeks=52)
    hist = hist.rename(columns={"Weekly_Sales": "lag_52"})
    return df.merge(hist, on=["id", "Date"], how="left")


def feature_columns(include_lag: bool, include_trend: bool = True) -> list:
    cols = [f"sin_{k}" for k in range(1, FOURIER_K + 1)]
    cols += [f"cos_{k}" for k in range(1, FOURIER_K + 1)]
    cols += ["week_of_year", "month", "is_holiday", "is_thanksgiving",
             "is_superbowl", "is_laborday"]
    cols += [f"xmas_minus_{w}" for w in (0, 1, 2, 3)]
    if include_trend:
        cols.append("t")
    if include_lag:
        cols.append("lag_52_scaled")
    return cols
