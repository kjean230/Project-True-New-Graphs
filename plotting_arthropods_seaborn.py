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
