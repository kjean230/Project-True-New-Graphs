# plotting_arthropods_seaborn.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

sns.set(style="whitegrid")
SEASON_PALETTE = {
    "Season 1 (Jan-Apr)": "blue",
    "Season 2 (May-Aug)": "red",
    "Season 3 (Sep-Dec)": "green",
}

# --- helper functions --- #
def _season_color(season_label: str) -> str:
    return SEASON_PALETTE.get(season_label, "gray")

def _add_season_shading(ax, df: pd.DataFrame, alpha: float = 0.06):
    df = df.sort_values("date_month")
    grouped = df.groupby(["year", "season_label"], sort=True)

    for (_, season_label), g in grouped:
        if g.empty:
            continue
        start = g["date_month"].min()
        end = g["date_month"].max() + pd.offsets.MonthBegin(1)
        color = _season_color(season_label)
        ax.axvspan(start, end, color=color, alpha=alpha)

def _add_lowess_line(ax, x, y, color="black", label="LOWESS"):
    if not HAS_STATSMODELS:
        print("statsmodels is not imported")
        return
    
    import numpy as np
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    smoothed = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=label)

def _prepare_scatter_df(monthly_df: pd.DataFrame, 
                        taxon: str, env: str) -> pd.DataFrame:
    df = monthly_df.copy()
    if taxon == "spider":
        count_col = "spider_count"
    elif taxon == "fly": 
        count_col = "fly_count"
    else:
        raise ValueError("taxon must be 'spider' or 'fly'")
    
    if env == "temp":
        env_col = "temp_mean"
    elif env == "aqi":
        env_col = "aqi_mean"
    else: 
        raise ValueError("env must be 'temp' or 'aqi'")
    
    df = df.dropna(subset=[count_col, env_col])
    df_scatter = df[["date_month", "season_label", count_col, env_col]].copy()
    df_scatter = df_scatter.rename(columns={count_col: "abundance", env_col: "env"})
    return df_scatter

def _scatter_env_vs_abundance_seaborn(df_scatter: pd.DataFrame,
                                      env_label: str,
                                      taxon_label: str):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df_scatter,
                    x="env",
                    y="abundance",
                    hue="season_label",
                    palette=SEASON_PALETTE,
                    alpha=0.7,
                    ax=ax)
    if HAS_STATSMODELS and len(df_scatter) > 4:
        x = df_scatter["env"].to_numpy()
        y = df_scatter["abundance"].to_numpy()
        _add_lowess_line(ax, x, y, color="black", label="LOWESS trend")

        ax.set_xlabel(env_label)
        ax.set_ylabel(f"{taxon_label} abundance (count per month)")
        ax.set_yscale("linear")
        ax.set_title(f"{taxon_label} abundance vs {env_label}")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc="best")
        fig.tight_layout()
        plt.show()

# ---- graph on temp & air quality over time ---- #
def plot_temp_aqi_over_time_seaborn(monthly_df: pd.DataFrame):
    """
    Graph 1 (seaborn):
    - X: date_month
    - Left Y: temp_smooth (3-mo rolling)
    - Right Y: aqi_smooth (3-mo rolling)
    - Season shading.
    """
    df = monthly_df.copy().sort_values("date_month")

    df["temp_smooth"] = (
        df["temp_mean"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    df["aqi_smooth"] = (
        df["aqi_mean"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    _add_season_shading(ax1, df)

    sns.lineplot(
        data=df,
        x="date_month",
        y="temp_smooth",
        ax=ax1,
        color="black",
        label="Temp (3-mo avg, °F)",
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature (°F)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="date_month",
        y="aqi_smooth",
        ax=ax2,
        color="purple",
        label="AQI (3-mo avg, ppb)",
    )
    ax2.set_ylabel("AQI (ppb)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Temperature and AQI over time (2017–2023) – seaborn")
    fig.tight_layout()
    plt.show()

def plot_spider_fly_over_time_seaborn(monthly_df: pd.DataFrame):
    """
    Graph 2 (seaborn):
    - X: date_month
    - Y: smoothed counts (3-mo rolling)
    - Two lines: spiders (black), flies (orange)
    - Season shading.
    """
    df = monthly_df.copy().sort_values("date_month")

    df["spider_smooth"] = (
        df["spider_count"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    df["fly_smooth"] = (
        df["fly_count"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    df.loc[df["spider_count"].isna(), "spider_smooth"] = float("nan")
    df.loc[df["fly_count"].isna(), "fly_smooth"] = float("nan")

    fig, ax = plt.subplots(figsize=(12, 6))

    _add_season_shading(ax, df)

    sns.lineplot(
        data=df,
        x="date_month",
        y="spider_smooth",
        ax=ax,
        color="black",
        label="Spiders (3-mo avg, count)",
    )
    sns.lineplot(
        data=df,
x="date_month",
        y="fly_smooth",
        ax=ax,
        color="orange",
        label="Flies (3-mo avg, count)",
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Abundance (count per month)")
    ax.legend(loc="upper left")

    plt.title("Spider and fly abundance over time (2017–2023) – seaborn")
    fig.tight_layout()
    plt.show()

# scatter plots
def plot_temp_vs_spider_scatter_seaborn(monthly_df: pd.DataFrame):
    """Graph 3A (seaborn): Temp vs spider abundance."""
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="temp")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Spider",
    )

def plot_temp_vs_fly_scatter_seaborn(monthly_df: pd.DataFrame):
    """Graph 3B (seaborn): Temp vs fly abundance."""
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="temp")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Fly",
    )

def plot_aqi_vs_spider_scatter_seaborn(monthly_df: pd.DataFrame):
    """Graph 3C (seaborn): AQI vs spider abundance."""
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="aqi")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Spider",
    )

def plot_aqi_vs_fly_scatter_seaborn(monthly_df: pd.DataFrame):
    """Graph 3D (seaborn): AQI vs fly abundance."""
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="aqi")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Fly",
    )