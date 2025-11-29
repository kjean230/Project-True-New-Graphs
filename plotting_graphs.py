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