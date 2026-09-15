# ============================================================================
# Predicting Next-Day Stock Direction: SVM vs Random Forest vs XGBoost
# Multi-ticker, walk-forward validated classification pipeline
# ============================================================================
#
# What this script does (mapped to the resume bullets it's meant to support):
#
# 1. Feature engineering from OHLC data: Open-Close, High-Low, RSI, MACD,
#    and rolling volatility.
# 2. Hyperparameter tuning (GridSearchCV) for SVM (kernel/C/gamma), and
#    baseline comparisons against Random Forest and XGBoost.
# 3. Walk-forward (rolling-origin) validation across multiple tickers,
#    instead of a single static train/test split.
#
# Note on expectations: next-day direction from price-derived features alone
# sits close to a coin flip (this is well documented in the finance ML
# literature — markets are close to efficient at this horizon). The value of
# this script is the *pipeline* (features, tuning, validation methodology),
# not a claim that it reliably beats the market. Treat accuracy numbers
# accordingly and don't read too much into a few points above 50%.
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

import ta  # technical analysis indicators (RSI, MACD, etc.)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TICKERS = ["PLTR", "AAPL", "NVDA", "MSFT", "AMZN"]
PERIOD = "5y"
N_WALK_FORWARD_SPLITS = 5   # number of rolling train/test folds per ticker
TEST_FOLD_SIZE = 60         # trading days held out per fold (~3 months)


# ----------------------------------------------------------------------
# Step 1: Feature engineering
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer predictors from raw OHLCV data."""
    df = df.copy()

    # Simple price-action features
    df["Open-Close"] = df["Open"] - df["Close"]
    df["High-Low"] = df["High"] - df["Low"]

    # Momentum: RSI (14-day)
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # Trend: MACD (12/26/9)
    macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    # Volatility: rolling std of daily returns (10-day), annualization not
    # needed since it's used as a relative feature, not reported directly
    df["Return_1d"] = df["Close"].pct_change()
    df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()

    return df


FEATURE_COLS = [
    "Open-Close", "High-Low", "RSI", "MACD", "MACD_Signal", "MACD_Diff", "Volatility_10d",
]


# ----------------------------------------------------------------------
# Step 2: Target variable
# ----------------------------------------------------------------------
def build_target(df: pd.DataFrame) -> pd.Series:
    """+1 if tomorrow's close > today's close, else 0."""
    return pd.Series(
        np.where(df["Close"].shift(-1) > df["Close"], 1, 0),
        index=df.index,
    )


# ----------------------------------------------------------------------
# Step 3: Model candidates + hyperparameter grids
# ----------------------------------------------------------------------
def get_model_grids():
    """
    Returns a dict of {name: (pipeline, param_grid)}.
    SVM features are scaled since SVC is distance/kernel based; tree
    ensembles don't need scaling but a scaler in the pipeline is harmless.
    """
    models = {}

    models["SVM"] = (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
        {
            "clf__kernel": ["linear", "rbf", "poly", "sigmoid"],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
        },
    )

    models["RandomForest"] = (
        Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 5],
        },
    )

    models["XGBoost"] = (
        Pipeline([("clf", XGBClassifier(
            random_state=42, eval_metric="logloss"
        ))]),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.01, 0.1],
        },
    )

    return models


# ----------------------------------------------------------------------
# Step 4: Walk-forward validation
# ----------------------------------------------------------------------
def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str,
                           pipeline: Pipeline, param_grid: dict,
                           n_splits: int, test_size: int):
    """
    Rolling-origin evaluation: for each fold, train on all data up to a
    cutoff, tune hyperparameters via TimeSeriesSplit *within* the training
    window only (no lookahead), then test on the next `test_size` days.
    Returns a DataFrame of per-fold metrics and the out-of-fold predictions
    (for building an equity curve later).
    """
    n = len(X)
    fold_results = []
    oof_predictions = pd.Series(index=X.index, dtype=float)

    # Build fold boundaries: expanding training window, fixed-size test window
    first_test_start = n - n_splits * test_size
    if first_test_start < 100:
        raise ValueError("Not enough data for the requested number of folds/test size.")

    for fold in range(n_splits):
        test_start = first_test_start + fold * test_size
        test_end = test_start + test_size

        X_train, y_train = X.iloc[:test_start], y.iloc[:test_start]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        # Inner CV for hyperparameter search must also respect time order
        inner_cv = TimeSeriesSplit(n_splits=3)
        search = GridSearchCV(pipeline, param_grid, cv=inner_cv,
                               scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        preds = best_model.predict(X_test)

        oof_predictions.iloc[test_start:test_end] = preds

        fold_results.append({
            "model": model_name,
            "fold": fold + 1,
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
            "best_params": search.best_params_,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

    return pd.DataFrame(fold_results), oof_predictions


# ----------------------------------------------------------------------
# Step 5: Run the full pipeline across tickers and models
# ----------------------------------------------------------------------
def run_pipeline():
    all_fold_results = []
    strategy_curves = {}

    for ticker in TICKERS:
        print(f"\n{'='*70}\n{ticker}\n{'='*70}")

        raw = yf.download(ticker, period=PERIOD, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = build_features(raw)
        y = build_target(df)

        # Drop rows with NaNs from indicator warm-up periods / final NaN target
        valid = df[FEATURE_COLS].notna().all(axis=1) & y.notna()
        X = df.loc[valid, FEATURE_COLS]
        y = y.loc[valid]
        price = df.loc[valid, "Close"]

        if len(X) < 400:
            print(f"Skipping {ticker}: not enough usable rows after feature warm-up.")
            continue

        model_grids = get_model_grids()
        ticker_best_oof = None
        ticker_best_score = -np.inf

        for model_name, (pipeline, grid) in model_grids.items():
            fold_df, oof_preds = walk_forward_evaluate(
                X, y, model_name, pipeline, grid,
                n_splits=N_WALK_FORWARD_SPLITS, test_size=TEST_FOLD_SIZE,
            )
            fold_df["ticker"] = ticker
            all_fold_results.append(fold_df)

            mean_acc = fold_df["accuracy"].mean()
            print(f"  {model_name:12s} walk-forward mean accuracy: {mean_acc:.4f} "
                  f"(mean f1: {fold_df['f1'].mean():.4f})")

            if mean_acc > ticker_best_score:
                ticker_best_score = mean_acc
                ticker_best_oof = oof_preds

        # Build a simple strategy equity curve using the best model's
        # out-of-fold predictions (only defined where we have predictions)
        oof = ticker_best_oof.dropna()
        rets = price.pct_change().reindex(oof.index)
        strategy_return = rets * oof.shift(1).fillna(0)
        strategy_curves[ticker] = {
            "buy_hold": rets.cumsum(),
            "strategy": strategy_return.cumsum(),
        }

    results_df = pd.concat(all_fold_results, ignore_index=True)
    return results_df, strategy_curves


# ----------------------------------------------------------------------
# Step 6: Summarize + plot
# ----------------------------------------------------------------------
def summarize_and_plot(results_df: pd.DataFrame, strategy_curves: dict):
    print(f"\n{'='*70}\nSUMMARY: mean walk-forward accuracy by model, averaged across tickers\n{'='*70}")
    summary = results_df.groupby("model")["accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(summary)

    results_df.to_csv("walk_forward_results.csv", index=False)
    print("\nFull fold-level results saved to walk_forward_results.csv")

    n = len(strategy_curves)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (ticker, curves) in zip(axes, strategy_curves.items()):
        ax.plot(curves["buy_hold"], color="red", label=f"{ticker} Buy & Hold")
        ax.plot(curves["strategy"], color="blue", label=f"{ticker} Best-Model Strategy (walk-forward OOF)")
        ax.set_title(f"{ticker}: Walk-Forward Strategy vs Buy & Hold")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
    plt.tight_layout()
    plt.savefig("walk_forward_strategy_vs_buyhold.png", dpi=150)
    print("Chart saved to walk_forward_strategy_vs_buyhold.png")


if __name__ == "__main__":
    results_df, strategy_curves = run_pipeline()
    summarize_and_plot(results_df, strategy_curves)