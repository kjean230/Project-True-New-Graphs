# ml_model.py
# linear regression models for arthropod abundance vs environmental variables

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _prepare_ml_matrix(
    df: pd.DataFrame,
    response_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Build design matrix X and response y for linear regression:

        response_col ~ z(temp_mean) + z(aqi_mean) + z(year) + season dummies
    """
    required_cols = [response_col, "temp_mean", "aqi_mean", "year", "season_label"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df_used = df.dropna(subset=required_cols).copy()
    if df_used.empty:
        raise ValueError("No data available after dropping rows with missing values.")

    num_cols = ["temp_mean", "aqi_mean", "year"]
    X_num = df_used[num_cols].astype(float).values

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    season_dummies = pd.get_dummies(
        df_used["season_label"],
        prefix="season",
        drop_first=True,
    )

    X = np.hstack([X_num_scaled, season_dummies.values])
    feature_names = [f"z_{col}" for col in num_cols] + list(season_dummies.columns)

    y = df_used[response_col].astype(float).values

    return X, y, df_used, feature_names


def _compute_adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Compute adjusted R^2 given:
        r2: plain R^2
        n: number of observations
        p: number of predictors
    """
    if n <= p + 1:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _plot_residuals_vs_fitted(y_true, y_pred, title: str):
    """
    Residuals vs fitted values plot.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(title + " — Residuals vs Fitted")
    fig.tight_layout()
    plt.show()


def _plot_qq(residuals, title: str):
    """
    QQ plot of residuals using scipy if available.
    """
    if not HAS_SCIPY:
        print("scipy not installed; skipping QQ plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title + " — QQ Plot of Residuals")
    fig.tight_layout()
    plt.show()


def _run_linear_model(
    monthly_df: pd.DataFrame,
    response_col: str,
    taxon_label: str,
    out_prefix: str,
    output_dir: Path,
) -> None:
    """
    Core modeling routine:

    - Prepare X, y.
    - Fit LinearRegression.
    - Print R^2, adjusted R^2, coefficients.
    - Save coefficient table and prediction table to CSV.
    - Show residual diagnostics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, df_used, feature_names = _prepare_ml_matrix(
        monthly_df,
        response_col=response_col,
    )

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    r2 = r2_score(y, y_pred)
    adj_r2 = _compute_adjusted_r2(r2, n=len(y), p=X.shape[1])

    # Coef table
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
        }
    )
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]

    coef_df["response"] = response_col
    coef_df["taxon_label"] = taxon_label
    coef_df["r2"] = r2
    coef_df["adjusted_r2"] = adj_r2

    # Prediction table
    pred_df = df_used.copy()
    pred_df["fitted"] = y_pred
    pred_df["residual"] = residuals

    coef_path = output_dir / f"{out_prefix}_coefficients.csv"
    preds_path = output_dir / f"{out_prefix}_predictions.csv"

    coef_df.to_csv(coef_path, index=False)
    pred_df.to_csv(preds_path, index=False)

    # Console summary
    print(f"\n===== Linear model for {taxon_label} ({response_col}) =====")
    print(f"Number of observations used: {len(y)}")
    print(f"R^2:          {r2:.4f}")
    print(f"Adjusted R^2: {adj_r2:.4f}\n")

    print("Coefficients (standardized numeric predictors):")
    for fname, coef in zip(feature_names, model.coef_):
        print(f"  {fname:20s} -> {coef: .4f}")
    print(f"  {'intercept':20s} -> {model.intercept_: .4f}\n")

    print(f"Coefficient summary saved to: {coef_path}")
    print(f"Predictions (with residuals) saved to: {preds_path}\n")

    # Diagnostics
    title = f"{taxon_label} abundance model"
    _plot_residuals_vs_fitted(y_true=y, y_pred=y_pred, title=title)
    _plot_qq(residuals=residuals, title=title)


def run_spider_model(monthly_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Fit and evaluate a linear regression for spider abundance:

        spider_count ~ z(temp_mean) + z(aqi_mean) + z(year) + season dummies
    """
    if "spider_count" not in monthly_df.columns:
        raise KeyError("'spider_count' not found in monthly_df.")
    _run_linear_model(
        monthly_df=monthly_df,
        response_col="spider_count",
        taxon_label="Spider",
        out_prefix="spider_abundance_model",
        output_dir=output_dir,
    )


def run_fly_model(monthly_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Fit and evaluate a linear regression for fly abundance:

        fly_count ~ z(temp_mean) + z(aqi_mean) + z(year) + season dummies
    """
    if "fly_count" not in monthly_df.columns:
        raise KeyError("'fly_count' not found in monthly_df.")
    _run_linear_model(
        monthly_df=monthly_df,
        response_col="fly_count",
        taxon_label="Fly",
        out_prefix="fly_abundance_model",
        output_dir=output_dir,
    )

def run_temp_aqi_model(monthly_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Fit and evaluate a linear regression modeling the relationship
    between temperature and air quality:

        aqi_mean ~ z(temp_mean) + z(year) + season dummies

    This model examines how temperature predicts air quality levels,
    controlling for year and seasonal effects.
    """
    if "temp_mean" not in monthly_df.columns:
        raise KeyError("'temp_mean' not found in monthly_df.")
    if "aqi_mean" not in monthly_df.columns:
        raise KeyError("'aqi_mean' not found in monthly_df.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    required_cols = ["temp_mean", "aqi_mean", "year", "season_label"]
    df_used = monthly_df.dropna(subset=required_cols).copy()

    if df_used.empty:
        raise ValueError("No data available after dropping rows with missing values.")

    # Standardize predictors
    num_cols = ["temp_mean", "year"]
    X_num = df_used[num_cols].astype(float).values

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Season dummies
    season_dummies = pd.get_dummies(
        df_used["season_label"],
        prefix="season",
        drop_first=True,
    )

    X = np.hstack([X_num_scaled, season_dummies.values])
    feature_names = [f"z_{col}" for col in num_cols] + list(season_dummies.columns)

    y = df_used["aqi_mean"].astype(float).values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    r2 = r2_score(y, y_pred)
    adj_r2 = _compute_adjusted_r2(r2, n=len(y), p=X.shape[1])

    # Coefficient table
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
        }
    )
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]

    coef_df["response"] = "aqi_mean"
    coef_df["model_type"] = "Temperature-AQI relationship"
    coef_df["r2"] = r2
    coef_df["adjusted_r2"] = adj_r2

    # Prediction table
    pred_df = df_used.copy()
    pred_df["fitted_aqi"] = y_pred
    pred_df["residual"] = residuals

    coef_path = output_dir / "temp_aqi_relationship_coefficients.csv"
    preds_path = output_dir / "temp_aqi_relationship_predictions.csv"

    coef_df.to_csv(coef_path, index=False)
    pred_df.to_csv(preds_path, index=False)

    # Console summary
    print("\n===== Temperature-AQI Relationship Model =====")
    print(f"Number of observations used: {len(y)}")
    print(f"R^2:          {r2:.4f}")
    print(f"Adjusted R^2: {adj_r2:.4f}\n")

    print("Coefficients (standardized predictors):")
    for fname, coef in zip(feature_names, model.coef_):
        print(f"  {fname:20s} -> {coef: .4f}")
    print(f"  {'intercept':20s} -> {model.intercept_: .4f}\n")

    print(f"Coefficient summary saved to: {coef_path}")
    print(f"Predictions (with residuals) saved to: {preds_path}\n")

    # Diagnostics
    title = "Temperature-AQI relationship model"
    _plot_residuals_vs_fitted(y_true=y, y_pred=y_pred, title=title)
    _plot_qq(residuals=residuals, title=title)

    # Additional scatter plot: actual temp vs AQI with fitted line
    _plot_temp_aqi_scatter_with_fit(df_used, y_pred)


def _plot_temp_aqi_scatter_with_fit(df: pd.DataFrame, y_pred: np.ndarray):
    """
    Scatter plot of temperature vs AQI with the fitted regression line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color points by season
    season_colors = {
        "Season 1 (Jan-Apr)": "blue",
        "Season 2 (May-Aug)": "red",
        "Season 3 (Sep-Dec)": "green",
    }

    for season, color in season_colors.items():
        mask = df["season_label"] == season
        ax.scatter(
            df.loc[mask, "temp_mean"],
            df.loc[mask, "aqi_mean"],
            c=color,
            label=season,
            alpha=0.6,
            s=50,
        )

    # Sort by temperature for line plot
    df_sorted = df.copy()
    df_sorted["fitted_aqi"] = y_pred
    df_sorted = df_sorted.sort_values("temp_mean")

    # Plot fitted line
    ax.plot(
        df_sorted["temp_mean"],
        df_sorted["fitted_aqi"],
        color="black",
        linewidth=2,
        label="Fitted regression line",
        linestyle="--",
    )

    ax.set_xlabel("Temperature (°F)", fontsize=12)
    ax.set_ylabel("AQI (ppb)", fontsize=12)
    ax.set_title("Temperature vs Air Quality with Fitted Regression Line", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
