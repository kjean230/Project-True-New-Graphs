# tree_graphs_liri.py
#
# Tree inventory graphs for paper birch and red maple:
# 1) Condition distribution per species (percentage), grouped by condition group,
#    stacked by original TPCondition categories.
# 2) Count of trees per risk rating, grouped by species.
#
# Intended usage:
#   all_trees_df = load_tree_data("Paper_Birch_Filtered.csv",
#                                 "real_red_maple_tree_data.csv")
#   plot_condition_distribution(all_trees_df)
#   plot_risk_rating_distribution(all_trees_df)

from pathlib import Path
from typing import Union, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _map_species_label(genus_species: str) -> str:
    """
    Map raw GenusSpecies string to a concise species label.

    Rules are case-insensitive and robust to variants like:
      - "Betula papyrifera - paper birch"
      - "Betula papyrifera"
      - "paper birch"
      - "Acer rubrum - red maple"
      - "Acer rubrum"
      - "red maple"
    """
    if pd.isna(genus_species):
        return "Other/Unknown"

    text = str(genus_species).lower()

    if "paper birch" in text or "betula papyrifera" in text:
        return "Paper birch"
    if "red maple" in text or "acer rubrum" in text:
        return "Red maple"

    return "Other/Unknown"


# Condition grouping specification
CONDITION_GROUP_MAP: Dict[str, str] = {
    "Excellent": "Good",
    "Good": "Good",
    "Fair": "Fair",
    "Poor": "Poor",
    "Critical": "Poor",
    "Dead": "Dead",
    "Stump": "Dead",
    "Retired": "Dead",
}

# Order in which to show parent groups on the x-axis
CONDITION_GROUP_ORDER: List[str] = ["Good", "Fair", "Poor", "Dead"]

# Colors for original TPCondition subcategories (stack segments)
CONDITION_COLOR_MAP: Dict[str, str] = {
    "Excellent": "#1b9e77",
    "Good": "#66c2a5",
    "Fair": "#7570b3",
    "Poor": "#e6ab02",
    "Critical": "#e7298a",
    "Dead": "#d95f02",
    "Stump": "#a6761d",
    "Retired": "#666666",
}


def _clean_tree_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw tree-inventory DataFrame:

      - Ensure required columns exist (GenusSpecies, TPCondition, TPStructure,
        RiskRating, DBH, Lat, Long).
      - Map GenusSpecies to a concise species_label.
      - Map TPCondition into a parent condition_group (Good/Fair/Poor/Dead)
        using CONDITION_GROUP_MAP.
    """
    # Make sure we have all expected columns (tolerate missing ones by filling)
    expected_cols = [
        "GenusSpecies",
        "TPCondition",
        "TPStructure",
        "RiskRating",
        "DBH",
        "Lat",
        "Long",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Species label
    df["species_label"] = df["GenusSpecies"].apply(_map_species_label)

    # Normalize TPCondition as string
    df["TPCondition"] = df["TPCondition"].astype(str).str.strip()
    df.loc[df["TPCondition"].isin(["nan", "NaN", "None", ""]), "TPCondition"] = np.nan

    # Map to parent condition group; values not in the map become NaN
    df["condition_group"] = df["TPCondition"].map(CONDITION_GROUP_MAP)

    # Keep only the species of interest (and drop unknown species)
    df = df[df["species_label"].isin(["Paper birch", "Red maple"])].copy()

    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_tree_data(
    birch_csv_path: Union[str, Path],
    maple_csv_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Load paper birch and red maple CSV files and return a unified DataFrame.

    Parameters
    ----------
    birch_csv_path : str or Path
        Path to the paper birch CSV file.
    maple_csv_path : str or Path
        Path to the red maple CSV file.

    Returns
    -------
    all_trees_df : pandas.DataFrame
        Combined DataFrame with at least:
          - species_label ("Paper birch", "Red maple")
          - GenusSpecies
          - TPCondition
          - condition_group ("Good", "Fair", "Poor", "Dead" or NaN)
          - TPStructure
          - RiskRating
          - DBH
          - Lat, Long
    """
    birch_csv_path = Path(birch_csv_path)
    maple_csv_path = Path(maple_csv_path)

    birch_df_raw = pd.read_csv(birch_csv_path, dtype=str)
    maple_df_raw = pd.read_csv(maple_csv_path, dtype=str)

    birch_df = _clean_tree_df(birch_df_raw)
    maple_df = _clean_tree_df(maple_df_raw)

    all_trees_df = pd.concat([birch_df, maple_df], ignore_index=True)

    # Convert DBH to numeric where possible
    all_trees_df["DBH"] = pd.to_numeric(all_trees_df["DBH"], errors="coerce")

    # RiskRating stays as string; we'll coerce in the risk plot function
    return all_trees_df


def plot_condition_distribution(all_trees_df: pd.DataFrame) -> None:
    """
    Plot condition distribution per species (number of trees per condition group).

    - X-axis: condition_group (Good, Fair, Poor, Dead)
    - Hue: species_label (Paper birch, Red maple)
    - Y-axis: count of trees in that condition group for that species.
    """
    df = all_trees_df.copy()

    # Drop rows without a condition_group or species_label
    df = df.dropna(subset=["condition_group", "species_label"])

    # Restrict to the four main groups, in case there are other values
    df = df[df["condition_group"].isin(CONDITION_GROUP_ORDER)]

    if df.empty:
        raise ValueError("No rows with valid condition_group found for plotting.")

    # Count trees per species and condition group (raw counts)
    counts = (
        df.groupby(["species_label", "condition_group"])
        .size()
        .reset_index(name="count")
    )

    # Ensure consistent ordering
    species_order = ["Paper birch", "Red maple"]
    condition_order = CONDITION_GROUP_ORDER

    counts["species_label"] = pd.Categorical(
        counts["species_label"], categories=species_order, ordered=True
    )
    counts["condition_group"] = pd.Categorical(
        counts["condition_group"], categories=condition_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=counts,
        x="condition_group",
        y="count",
        hue="species_label",
        ax=ax,
    )

    ax.set_xlabel("Condition group")
    ax.set_ylabel("Number of trees")
    ax.set_title("Condition distribution per species (counts)")

    ax.legend(title="Species", loc="upper right")

    fig.tight_layout()
    plt.show()



def plot_risk_rating_distribution(all_trees_df: pd.DataFrame) -> None:
    """
    Plot count of trees per risk rating, grouped by species.

    - X-axis: RiskRating (integer categories)
    - Y-axis: count of trees
    - Hue: species_label ("Paper birch", "Red maple")
    - Missing RiskRating values are ignored (dropped).
    """
    df = all_trees_df.copy()

    # Coerce RiskRating to numeric and drop missing
    df["RiskRating"] = pd.to_numeric(df["RiskRating"], errors="coerce")
    df = df.dropna(subset=["RiskRating", "species_label"])

    if df.empty:
        raise ValueError("No rows with valid RiskRating found for plotting.")

    # Make sure RiskRating is treated as a category for x-axis ordering
    df["RiskRating"] = df["RiskRating"].astype(int)

    # Aggregate counts by rating and species
    counts = (
        df.groupby(["RiskRating", "species_label"])
        .size()
        .reset_index(name="count")
    )

    # Ensure risk ratings are sorted
    counts = counts.sort_values("RiskRating")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=counts,
        x="RiskRating",
        y="count",
        hue="species_label",
        ax=ax,
    )

    ax.set_xlabel("Risk rating")
    ax.set_ylabel("Number of trees")
    ax.set_title("Count of trees per risk rating by species")

    ax.legend(title="Species", loc="upper right")

    fig.tight_layout()
    plt.show()