# plotting_graphs.py

import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _season_color(season_label: str) -> str:
    """
    Map 3-band seasons to colors.
    """
    if "Season 1 (Jan–Apr)" in season_label:
        return "blue"
    elif "Season 2 (May–Aug)" in season_label:
        return "red"
    else:
        return "green"


def _add_season_shading(ax, df: pd.DataFrame, alpha: float = 0.06):
    """
    Add vertical bands for each season across years.
    Uses 'season_label' and 'date_month' columns.
    """
    df = df.sort_values("date_month")
    grouped = df.groupby(["year", "season_label"], sort=True)

    for (_, season_label), g in grouped:
        if g.empty:
            continue
        start = g["date_month"].min()
        end = g["date_month"].max()
        # extend to the start of the next month
        end = end + pd.offsets.MonthBegin(1)
        color = _season_color(season_label)
        ax.axvspan(start, end, color=color, alpha=alpha)


def _add_lowess_line(ax, x, y, color="black", label="LOWESS"):
    """
    Add LOWESS smooth if statsmodels is installed.
    """
    if not HAS_STATSMODELS:
        print("statsmodels is not installed; skipping LOWESS line.")
        return

    import numpy as np

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    smoothed = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=label)


# ===== Graph 1: temperature and AQI over time =====
def plot_temp_aqi_over_time(monthly_df: pd.DataFrame):
    """
    Graph 1:
    - X: date_month
    - Left Y: temp_mean (3-month rolling avg)
    - Right Y: aqi_mean (3-month rolling avg)
    - Season shading: 3-band seasons
    """
    df = monthly_df.copy()
    df = df.sort_values("date_month")

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

    ax1.plot(
        df["date_month"],
        df["temp_smooth"],
        label="Temp (3-mo avg, °F)",
        linewidth=2,
        color="black",
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature (°F)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.plot(
        df["date_month"],
        df["aqi_smooth"],
        label="AQI (3-mo avg, ppb)",
        linewidth=2,
        color="purple",
    )
    ax2.set_ylabel("AQI (ppb)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    plt.title("Temperature and AQI over time (2017–2023)")
    plt.show()


# ===== Graph 2: spider and fly abundances over time =====
def plot_spider_fly_over_time(monthly_df: pd.DataFrame):
    """
    Graph 2:
    - X: date_month
    - Y: abundance (counts)
    - 3-month rolling mean for spider_count and fly_count
    - season shading
    """
    df = monthly_df.copy()
    df = df.sort_values("date_month")

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

    # Keep NaN gaps in smoothed series where original counts are NaN
    df.loc[df["spider_count"].isna(), "spider_smooth"] = float("nan")
    df.loc[df["fly_count"].isna(), "fly_smooth"] = float("nan")

    fig, ax = plt.subplots(figsize=(12, 6))

    _add_season_shading(ax, df)

    ax.plot(
        df["date_month"],
        df["spider_smooth"],
        label="Spiders (3-mo avg, count)",
        linewidth=2,
        color="black",
    )
    ax.plot(
        df["date_month"],
        df["fly_smooth"],
        label="Flies (3-mo avg, count)",
        linewidth=2,
        color="orange",
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Abundance (count per month)")
    ax.legend(loc="upper left")

    plt.title("Spider and fly abundance over time (2017–2023)")
    fig.tight_layout()
    plt.show()


# ===== Graph 3: env vs abundance scatterplots =====
def _prepare_scatter_df(
    monthly_df: pd.DataFrame,
    taxon: str,
    env: str,
) -> pd.DataFrame:
    """
    Prepare scatter data for one taxon and one env variable.
    """
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


def _scatter_env_vs_abundance(
    df_scatter: pd.DataFrame,
    env_label: str,
    taxon_label: str,
):
    """
    Core function: scatter + optional LOWESS, colored by season.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for season in df_scatter["season_label"].unique():
        sub = df_scatter[df_scatter["season_label"] == season]
        if sub.empty:
            continue
        color = _season_color(season)
        ax.scatter(
            sub["env"],
            sub["abundance"],
            label=season,
            alpha=0.7,
            color=color,
        )

    if HAS_STATSMODELS and len(df_scatter) > 4:
        x = df_scatter["env"].to_numpy()
        y = df_scatter["abundance"].to_numpy()
        _add_lowess_line(ax, x, y, color="black", label="LOWESS trend")

    ax.set_xlabel(env_label)
    ax.set_ylabel(f"{taxon_label} abundance (count per month)")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.set_title(f"{taxon_label} abundance vs {env_label}")

    fig.tight_layout()
    plt.show()


def plot_temp_vs_spider_scatter(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="temp")
    _scatter_env_vs_abundance(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Spider",
    )


def plot_temp_vs_fly_scatter(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="temp")
    _scatter_env_vs_abundance(
        df_scatter,
        env_label="Temperature (°F)",
        taxon_label="Fly",
    )


def plot_aqi_vs_spider_scatter(monthly_df: pd.DataFrame):
    df_scatter = _prepare_scatter_df(monthly_df, taxon="spider", env="aqi")
    _scatter_env_vs_abundance(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Spider",
    )


def plot_aqi_vs_fly_scatter(monthly_df: pd.DataFrame):
    """
    Graph 3D:
    AQI (ppb) vs fly abundance (monthly counts), colored by season.
    """
    df_scatter = _prepare_scatter_df(monthly_df, taxon="fly", env="aqi")
    _scatter_env_vs_abundance(
        df_scatter,
        env_label="AQI (ppb)",
        taxon_label="Fly",
    )