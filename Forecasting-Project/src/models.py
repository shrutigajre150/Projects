"""Forecasting models.

Each model exposes fit(train) and predict(future) on long-format frames with
columns id, Date, Weekly_Sales (+ IsHoliday). The ML models are *global*: one
model is trained across all series on scale-normalised sales, so the series
share seasonal and holiday patterns. Each series' scale is the mean of its
training sales; predictions are multiplied back by that scale.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import add_calendar_features, add_lag_features, feature_columns

SEED = 42


class SeasonalNaive:
    """Forecast = sales in the same week one year earlier."""
    name = "Seasonal naive"

    def fit(self, train):
        self.history = train
        return self

    def predict(self, future):
        f = add_lag_features(future, self.history)
        return f["lag_52"].to_numpy()


class SeasonalNaiveGrowth:
    """Seasonal naive scaled by each series' year-over-year growth over the
    last 13 weeks of training data."""
    name = "Seasonal naive + YoY growth"

    def fit(self, train):
        self.history = train
        self.growth = {}
        for sid, g in train.groupby("id"):
            g = g.sort_values("Date")
            recent = g["Weekly_Sales"].iloc[-13:].sum()
            last_year = g["Weekly_Sales"].iloc[-65:-52].sum()
            self.growth[sid] = recent / last_year
        return self

    def predict(self, future):
        f = add_lag_features(future, self.history)
        return (f["lag_52"] * f["id"].map(self.growth)).to_numpy()


class GlobalML:
    """Global regression model on scale-normalised sales."""

    def __init__(self, name, estimator, include_lag, include_trend, one_hot_ids):
        self.name = name
        self.estimator = estimator
        self.include_lag = include_lag
        self.include_trend = include_trend
        self.one_hot_ids = one_hot_ids

    def _design(self, df, history):
        X = add_calendar_features(df, self.origin)
        X = add_lag_features(X, history)
        X["scale"] = X["id"].map(self.scale)
        X["lag_52_scaled"] = X["lag_52"] / X["scale"]
        cols = feature_columns(self.include_lag, self.include_trend)
        M = X[cols].copy()
        if self.one_hot_ids:
            for sid in self.ids:
                M[f"id_{sid}"] = (X["id"] == sid).astype(float)
        else:
            M["id_code"] = X["id"].map({s: i for i, s in enumerate(self.ids)})
        return X, M

    def fit(self, train):
        self.history = train
        self.origin = train["Date"].min()
        self.ids = sorted(train["id"].unique())
        self.scale = train.groupby("id")["Weekly_Sales"].mean().to_dict()
        X, M = self._design(train, train)
        if self.include_lag:
            keep = X["lag_52"].notna().to_numpy()
            X, M = X[keep], M[keep]
        y = X["Weekly_Sales"] / X["scale"]
        self.estimator.fit(M, y)
        return self

    def predict(self, future):
        X, M = self._design(future, self.history)
        return self.estimator.predict(M) * X["scale"].to_numpy()


class MeanEnsemble:
    def __init__(self, members, name="Ensemble (mean of ML models)"):
        self.members, self.name = members, name

    def fit(self, train):
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, future):
        return np.mean([m.predict(future) for m in self.members], axis=0)


def ml_models():
    ridge = GlobalML(
        "Ridge (trend + Fourier seasonality)",
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        include_lag=False, include_trend=True, one_hot_ids=True)
    rf = GlobalML(
        "Random Forest",
        RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                              random_state=SEED, n_jobs=-1),
        include_lag=True, include_trend=False, one_hot_ids=False)
    hgb = GlobalML(
        "Gradient Boosting",
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      max_leaf_nodes=15, min_samples_leaf=5,
                                      l2_regularization=1.0, random_state=SEED),
        include_lag=True, include_trend=False, one_hot_ids=False)
    return [ridge, rf, hgb]


def all_models():
    """Fresh, unfitted instances of every model in the comparison."""
    ml = ml_models()
    ens = MeanEnsemble(ml_models())
    return [SeasonalNaive(), SeasonalNaiveGrowth(), *ml, ens]
