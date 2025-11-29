# data_loading_cleaning_csv.py
# file to clean csv files and load them into pandas dataframes for future use
# cleaning four csv files

from pathlib import Path
import pandas as pd
from typing import Union, Optional

def clean_observation_csv(csv_path: Union[str, Path],
                          start: pd.Timestamp, 
                          cutoff: pd.Timestamp,
                          iconic_taxon: Optional[str] = None,) -> pd.DataFrame:
    # function to clean inaturialist observations in csv file
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, dtype=str)
    if "observed_on" not in df.columns:
        raise KeyError(f"'observed_on' column is missing from {csv_path.name}.")
    df = df[df["iconic_taxon_name"] == iconic_taxon].copy()
    df["observed_on"] = pd.to_datetime(df["observed_on"], errors='coerce')
    df = df.dropna(subset=["observed_on"])

    df = df [
    (df["observed_on"] >= start) & (df["observed_on"] <= cutoff)
    ].copy()

    df['year'] = df['observed_on'].dt.year
    df['month'] = df['observed_on'].dt.month
    df['date_month'] = df['observed_on'].values.astype("datetime64[M]")

    df = df.sort_values("observed_on").reset_index(drop=True)
    return df

def clean_air_quality_monthly(csv_path: Union[str, Path],
                              start: pd.Timestamp,
                              cutoff: pd.Timestamp,) -> pd.DataFrame:
        # function to clean air quality monthly csv file 
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, dtype=str)

        required_cols = {"time_period", "start_date", "data_value"}
        missing = required_cols - set(df.columns)
        if missing:
             raise KeyError(f"Missing columns in {csv_path.name}: {missing}")
        df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
        df = df.dropna(subset=["start_date"])

        expanded_rows = []
        for _, row in df.iterrows():
            period = row.get("time_period")
            if pd.isnull(period):
                 continue
            
            start_date_row = row["start_date"]
            data_value = row["data_value"]

            if "Winter" in period:
                months = [
                    start_date_row,
                    start_date_row + pd.DateOffset(months=1),
                    start_date_row + pd.DateOffset(months=2),
                ]
            elif "Summer" in period:
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
                continue

            for date_month in months:
                expanded_rows.append({
                    "year": date_month.year,
                    "month": date_month.month,
                    "date_month": date_month,
                    "data_value": data_value,
                }) 
            expanded = pd.DataFrame(expanded_rows)

            expanded = expanded[
                (expanded["date_month"] >= start) & (expanded["date_month"] <= cutoff)
            ].reset_index(drop=True)
            
            expanded["data_value"] = pd.to_numeric(expanded["data_value"], errors='coerce')
            expanded = expanded.dropna(subset=["data_value"])

            monthly_aqi = (
        expanded.groupby(["year", "month", "date_month"])["data_value"]
        .mean()
        .reset_index()
        .rename(columns={"data_value": "aqi_mean"})
    )
            monthly_aqi = monthly_aqi.sort_values(["year", "month"]).reset_index(drop=True)
            return monthly_aqi
        
def clean_weather_monthly (csv_path: Union[str, Path],
                           start: pd.Timestamp,
                           cutoff: pd.Timestamp,
                           station_name: str,) -> pd.DataFrame:
    # function to clean weather csv file to monthly data
    # produces a dataframe for final results
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    required_cols = {"stattion_name", "date_month", "TAVG"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path.name}: {missing}")
    df["date_month"] = pd.to_datetime(df["date_month"], errors='coerce')
    df = df.dropna(subset=["date_month"])

    df["TAVG"] = pd.to_numeric(df["TAVG"], errors='coerce')
    df = df.dropna(subset=["TAVG"])

    df["station_name"] = df["station_name"].str.strip()
    station_name_clean = station_name.strip()
    df = df[df["station_name"] == station_name_clean].copy()

    df = df[
         (df["date_month"] >= start) & (df["date_month"] <= cutoff) 
    ].copy()

    df["year"] = df["date_month"].dt.year
    df["month"] = df["date_month"].dt.month

    monthly_temp = monthly_temp.sort_values(["year", "month"]).reset_index(drop=True)
    return monthly_temp

def build_monthly_grid(start: pd.Timestamp,
                       cutoff: pd.Timestamp,) -> pd.DataFrame:
    # function to build a monthly grid dataframe from start to cutoff dates
    all_months = pd.date_range(start=start, end=cutoff, freq='MS')
    base = pd.DataFrame({"date_month": all_months})
    base["year"] = base["date_month"].dt.year
    base["month"] = base["date_month"].dt.month
    return base