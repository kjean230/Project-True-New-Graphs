
# monthly_dataset.py

from pathlib import Path
from typing import Union
import pandas as pd

from data_loading_cleaning_csv import (
    clean_observation_csv,
    clean_air_quality_monthly,
    clean_weather_monthly,
    build_monthly_grid,
)
def _assign_season_label(month: int) -> str:
    if 1 <= month <= 4:
        return "Season 1 (Jan–Apr)"
    elif 5 <= month <= 8:
        return "Season 2 (May–Aug)"
    else:
        return "Season 3 (Sep–Dec)"


def _assign_season_code(month: int) -> int:
    if 1 <= month <= 4:
        return 1
    elif 5 <= month <= 8:
        return 2
    else:
        return 3

def build_monthly_env_arthropod_df(
    spider_csv: Union[str, Path],
    fly_csv: Union[str, Path],
    aq_csv: Union[str, Path],
    temp_csv: Union[str, Path],
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    station_name: str,
) -> pd.DataFrame:

    base = build_monthly_grid(start, cutoff)

    df_spider = clean_observation_csv(
        spider_csv, start=start, cutoff=cutoff, iconic_taxon="Arachnida"
    )
    spider_monthly = (
        df_spider.groupby(["year", "month", "date_month"])
        .size()
        .reset_index(name="spider_count")
    )

    df_fly = clean_observation_csv(
        fly_csv, start=start, cutoff=cutoff, iconic_taxon="Insecta"
    )
    fly_monthly = (
        df_fly.groupby(["year", "month", "date_month"])
        .size()
        .reset_index(name="fly_count")
    )

    df_aqi = clean_air_quality_monthly(aq_csv, start=start, cutoff=cutoff)
    # df_aqi has: [year, month, date_month, aqi_mean]

    df_temp = clean_weather_monthly(
        temp_csv,
        start=start,
        cutoff=cutoff,
        station_name=station_name,
    )

    df = (
        base
        .merge(spider_monthly, on=["year", "month", "date_month"], how="left")
        .merge(fly_monthly,    on=["year", "month", "date_month"], how="left")
        .merge(df_aqi[["year", "month", "date_month", "aqi_mean"]],
               on=["year", "month", "date_month"], how="left")
        .merge(df_temp[["year", "month", "date_month", "temp_mean"]],
               on=["year", "month", "date_month"], how="left")
    )

    df["spider_count"] = pd.to_numeric(df["spider_count"], errors="coerce")
    df["fly_count"] = pd.to_numeric(df["fly_count"], errors="coerce")

    df["season_label"] = df["month"].apply(_assign_season_label)
    df["season_code_3band"] = df["month"].apply(_assign_season_code)

    df["year_month"] = df["date_month"].dt.to_period("M").astype(str)

    df = df.sort_values("date_month").reset_index(drop=True)

    df["spider_count_3mo"] = (
        df["spider_count"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    df["fly_count_3mo"] = (
        df["fly_count"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df