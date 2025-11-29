# plotting_graphs.py

import pandas as pd
import matplotlib.pyplot as plt

try: 
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def _season_color(season_label: str) -> str:
    if "Season 1 (Jan–Apr)" in season_label:
        return "blue"
    elif "Season 2 (May–Aug)" in season_label:
        return "red"
    else:
        return "green"
    
def _add_season_shading(ax, df: pd.DataFrame, alpha: float = 0.06):
    df = df.sort_values("date_month")
    grouped = df.groupby(["year", "season_label"], sort=True)

    for (_, seasoon_label), g in grouped: 
        if g.empty:
            continue
        start = g["date_month"].min()
        end = g["date_month"].max()
        end = end + pd.offsets.MonthBegin(1)
        color = _season_color(seasoon_label)
        ax.axvspan(start, end, color=color, alpha=alpha)

def _add_lowess_line(ax, x, y, color="black", label="LOWESS"):
    if not HAS_STATSMODELS:
        print("statsmodels is not installed; skipping LOWESS line.")
        return
    
    import numpy as np
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    smoothed = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=label)

# ===== graph one: temperature and AQI over time ===== #