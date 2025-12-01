# data_loading_cleaning_csv.py
# file to clean csv files and load them into pandas dataframes for future use
# cleaning four csv files

from pathlib import Path
from typing import Union, Optional

import pandas as pd


def clean_observation_csv(
    csv_path: Union[str, Path],
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    iconic_taxon: Optional[str] = None,
) -> pd.DataFrame:
    """
    Clean iNaturalist observation CSV (spiders, flies, etc.).

    Steps:
    - Read CSV as strings.
    - Optionally filter on iconic_taxon_name (e.g., "Arachnida", "Insecta").
    - Parse observed_on to datetime and drop invalid.
    - Restrict to [start, cutoff].
    - Add year, month, date_month (month start).
    - Sort by observed_on.
    """
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, dtype=str)

    if "observed_on" not in df.columns:
        raise KeyError(f"'observed_on' column is missing from {csv_file.name}.")

    if iconic_taxon is not None:
        if "iconic_taxon_name" not in df.columns:
            raise KeyError(
                f"'iconic_taxon_name' column is missing from {csv_file.name} "
                "but iconic_taxon filter was provided."
            )
        df = df[df["iconic_taxon_name"] == iconic_taxon].copy()

    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df.dropna(subset=["observed_on"])

    df = df[
        (df["observed_on"] >= start) & (df["observed_on"] <= cutoff)
    ].copy()

    df["year"] = df["observed_on"].dt.year
    df["month"] = df["observed_on"].dt.month
    df["date_month"] = df["observed_on"].values.astype("datetime64[M]")

    df = df.sort_values("observed_on").reset_index(drop=True)
    return df


def clean_air_quality_monthly(
    csv_path: Union[str, Path],
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Clean air-quality CSV that has rows like:

        time_period: "Annual Average 2017", "Summer 2017", "Winter 2017-18"
        start_date : 2017-01-01, 2017-06-01, 2017-12-01
        data_value : numeric (ppb)

    Expand each row into monthly records:

        - "Winter": Dec(start year), Jan(next year), Feb(next year)
        - "Summer": Jun, Jul, Aug of that year
        - "Annual": Jan–Dec of that year

    Then:
    - filter to [start, cutoff]
    - convert data_value to numeric
    - group by (year, month, date_month) and average to aqi_mean
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    required_cols = {"time_period", "start_date", "data_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path.name}: {missing}")

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"])

    expanded_rows = []

    for _, row in df.iterrows():
        period = row.get("time_period")
        if pd.isnull(period):
            continue

        start_date_row = row["start_date"]
        data_value = row["data_value"]

        # Map time_period to specific months
        if "Winter" in period:
            # Dec of start year, Jan and Feb of next year
            months = [
                start_date_row,
                start_date_row + pd.DateOffset(months=1),
                start_date_row + pd.DateOffset(months=2),
            ]
        elif "Summer" in period:
            # Jun, Jul, Aug of that year
            months = [
                start_date_row,
                start_date_row + pd.DateOffset(months=1),
                start_date_row + pd.DateOffset(months=2),
            ]
        elif "Annual" in period:
            months = [
                pd.Timestamp(f"{start_date_row.year}-{m:02d}-01")
                for m in range(1, 13)
            ]
        else:
            # Ignore other time_period labels, if any
            continue

        for date_month in months:
            expanded_rows.append(
                {
                    "year": date_month.year,
                    "month": date_month.month,
                    "date_month": date_month,
                    "data_value": data_value,
                }
            )

    expanded = pd.DataFrame(expanded_rows)

    expanded = expanded[
        (expanded["date_month"] >= start) & (expanded["date_month"] <= cutoff)
    ].reset_index(drop=True)

    expanded["data_value"] = pd.to_numeric(expanded["data_value"], errors="coerce")
    expanded = expanded.dropna(subset=["data_value"])

    monthly_aqi = (
        expanded.groupby(["year", "month", "date_month"])["data_value"]
        .mean()
        .reset_index()
        .rename(columns={"data_value": "aqi_mean"})
    )

    monthly_aqi = monthly_aqi.sort_values(["year", "month"]).reset_index(drop=True)
    return monthly_aqi


def clean_weather_monthly(
    csv_path: Union[str, Path],
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
    station_name: str,
) -> pd.DataFrame:
    """
    Clean weather CSV to monthly data for a specific station.

    Expected columns:
        station_name, date_month, TAVG (°F)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    required_cols = {"station_name", "date_month", "TAVG"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path.name}: {missing}")

    df["date_month"] = pd.to_datetime(df["date_month"], errors="coerce")
    df = df.dropna(subset=["date_month"])

    df["TAVG"] = pd.to_numeric(df["TAVG"], errors="coerce")
    df = df.dropna(subset=["TAVG"])

    df["station_name"] = df["station_name"].str.strip()
    station_name_clean = station_name.strip()
    df = df[df["station_name"] == station_name_clean].copy()

    df = df[
        (df["date_month"] >= start) & (df["date_month"] <= cutoff)
    ].copy()

    df["year"] = df["date_month"].dt.year
    df["month"] = df["date_month"].dt.month

    monthly_temp = (
        df.groupby(["year", "month", "date_month"])["TAVG"]
        .mean()
        .reset_index()
        .rename(columns={"TAVG": "temp_mean"})
    )

    monthly_temp = monthly_temp.sort_values(["year", "month"]).reset_index(drop=True)
    return monthly_temp


def build_monthly_grid(
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a monthly grid dataframe from start to cutoff (month-start frequency).
    """
    all_months = pd.date_range(start=start, end=cutoff, freq="MS")
    base = pd.DataFrame({"date_month": all_months})
    base["year"] = base["date_month"].dt.year
    base["month"] = base["date_month"].dt.month
    return base