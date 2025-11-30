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