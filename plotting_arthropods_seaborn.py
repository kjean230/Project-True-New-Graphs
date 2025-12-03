# plotting_arthropods_seaborn.py
# Seaborn-based versions of the graphs, including observation-level time scatters.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set(style="whitegrid")

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

SEASON_PALETTE = {
    "Season 1 (Jan-Apr)": "blue",
    "Season 2 (May-Aug)": "red",
    "Season 3 (Sep-Dec)": "green",
}


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
        return
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    smoothed = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=label)


def _prepare_scatter_df(monthly_df: pd.DataFrame, taxon: str, env: str) -> pd.DataFrame:
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

    df = monthly_df.copy().dropna(subset=[count_col, env_col])
    df_scatter = df[["date_month", "season_label", count_col, env_col]].copy()
    df_scatter = df_scatter.rename(columns={count_col: "abundance", env_col: "env"})
    return df_scatter


def _scatter_env_vs_abundance_seaborn(
    df_scatter: pd.DataFrame, env_label: str, taxon_label: str
):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df_scatter,
        x="env",
        y="abundance",
        hue="season_label",
        palette=SEASON_PALETTE,
        alpha=0.7,
        ax=ax,
    )

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


# ----- Graph 7: temp + AQI over time -----
def plot_temp_aqi_over_time_seaborn(monthly_df: pd.DataFrame):
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


# ----- Graph 8: spider + fly over time -----
def plot_spider_fly_over_time_seaborn(monthly_df: pd.DataFrame):
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


# ----- Graph 9–12: env vs abundance scatters -----
def plot_temp_vs_spider_scatter_seaborn(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="temp")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Spider",
    )


def plot_temp_vs_fly_scatter_seaborn(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="temp")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Fly",
    )


def plot_aqi_vs_spider_scatter_seaborn(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="aqi")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Spider",
    )


def plot_aqi_vs_fly_scatter_seaborn(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="aqi")
    _scatter_env_vs_abundance_seaborn(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Fly",
    )


# ----- Graph 13–16: obs-level time vs env scatters -----
def _scatter_time_env_obs_seaborn(
    obs_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    env_col: str,
    env_label: str,
    taxon_label: str,
):
    """
    Observation-level scatter for graphs 13–16.

    New behavior (Option 1):
      x  = environmental variable from monthly_df (temp_mean or aqi_mean)
      y  = constant (1) per observation, jittered for visibility
      hue= season_label (joined in by month)

    This shows where individual arthropod observations occur along the
    environmental gradient, colored by season.
    """
    if "date_month" not in obs_df.columns:
        raise KeyError(
            "obs_df must contain a 'date_month' column (month-start) "
            "from clean_observation_csv."
        )

    # Look up env + season by month
    env_lookup = monthly_df[["date_month", "season_label", env_col]].dropna(
        subset=[env_col]
    )

    df = obs_df.merge(env_lookup, on="date_month", how="left")
    df = df.dropna(subset=[env_col])

    if df.empty:
        raise ValueError(
            f"No observations could be matched to monthly {env_col} values."
        )

    # Constant y = 1 with small vertical jitter so points don't overlap
    rng = np.random.default_rng(seed=42)
    df["y_jitter"] = 1.0 + rng.uniform(-0.15, 0.15, size=len(df))

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.scatterplot(
        data=df,
        x=env_col,
        y="y_jitter",
        hue="season_label",
        palette=SEASON_PALETTE,
        alpha=0.7,
        ax=ax,
        edgecolor=None,
        s=40,
    )

    ax.set_xlabel(env_label)
    ax.set_ylabel(f"{taxon_label} observations (jittered)")
    ax.set_yticks([])  # y is just a jitter axis, not a numeric quantity
    ax.set_title(f"{taxon_label} observations across {env_label.lower()}")

    # Tighten x-limits slightly around the data
    x_min = df[env_col].min()
    x_max = df[env_col].max()
    pad = (x_max - x_min) * 0.03 if x_max > x_min else 1.0
    ax.set_xlim(x_min - pad, x_max + pad)

    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Season", loc="best")

    fig.tight_layout()
    plt.show()

def plot_spider_temp_time_scatter_seaborn(spider_obs_df: pd.DataFrame,
                                          monthly_df: pd.DataFrame):
    _scatter_time_env_obs_seaborn(
        obs_df=spider_obs_df,
        monthly_df=monthly_df,
        env_col="temp_mean",
        env_label="Temperature (°F)",
        taxon_label="Spider",
    )


def plot_spider_aqi_time_scatter_seaborn(spider_obs_df: pd.DataFrame,
                                         monthly_df: pd.DataFrame):
    _scatter_time_env_obs_seaborn(
        obs_df=spider_obs_df,
        monthly_df=monthly_df,
        env_col="aqi_mean",
        env_label="AQI (ppb)",
        taxon_label="Spider",
    )


def plot_fly_temp_time_scatter_seaborn(fly_obs_df: pd.DataFrame,
                                       monthly_df: pd.DataFrame):
    _scatter_time_env_obs_seaborn(
        obs_df=fly_obs_df,
        monthly_df=monthly_df,
        env_col="temp_mean",
        env_label="Temperature (°F)",
        taxon_label="Fly",
    )


def plot_fly_aqi_time_scatter_seaborn(fly_obs_df: pd.DataFrame,
                                      monthly_df: pd.DataFrame):
    _scatter_time_env_obs_seaborn(
        obs_df=fly_obs_df,
        monthly_df=monthly_df,
        env_col="aqi_mean",
        env_label="AQI (ppb)",
        taxon_label="Fly",
    )