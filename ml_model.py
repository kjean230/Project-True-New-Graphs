# ml_model.py
# linear regression models for arthropod abundance vs environmental variables
from pathlib import Path
from typing import Tuple

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
        response_col: str,) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:

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
    season_dummies = pd.get_dummies(df_used["season_label"],
                                    prefix="season",
                                    drop_first=True,)
    X = np.hstack([X_num_scaled, season_dummies.values])
    feature_names = [f"z_{col}" for col in num_cols] + list(season_dummies.columns)

    y = df_used[response_col].astype(float).values

    return X, y, df_used, feature_names

