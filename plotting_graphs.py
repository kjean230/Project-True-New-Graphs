# plotting_graphs.py

import pandas as pd
import matplotlib.pyplot as plt

try: 
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

