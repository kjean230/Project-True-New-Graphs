# data_loading_cleaning_csv.py
# file to clean csv files and load them into pandas dataframes for future use
# cleaning four csv files

from pathlib import Path
import pandas as pd
from typing import Union, Optional

def clean_observation_csv(csv_path: Union[str, Path], 
                          start: pd.Timestamp, 
                          cutoff: pd.Timestamp, 
                          iconic_taxon_name: Optional[str] = None,) -> pd.DataFrame:
    
    # function to clean csv files 
    # created to improve reusability and lessening code duplication in main analysis scripts
    # creates a path object that will read all csv files passed through function
    # csv must be in string format
    # making sure the columns ('taxon_name' and 'observed_on') exists within the dataframe
    # making sure that we restrict the date range to a certain window for graph creation
    # create year, month, and month-start date columns for future merging of dataframes

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    if iconic_taxon_name is not None:
        if "iconic_taxon_name" not in df.columns:
            raise KeyError(f"'iconic_taxon_name' column not found in {csv_path.name} " 
                           "but iconic_taxon_name filter was provided.")
        df = df[df["iconic_taxon_name"] == iconic_taxon_name].copy()

    if "observed_on" not in df.columns:
        raise KeyError(f"'observed_on' column not found in {csv_path.name}.")
    
    df['observed_on'] = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df.dropna(subset=["observed_on"])

    df = df[
        (df["observed_on"] >= start) & (df["observed_on"] <= cutoff)
    ].copy()

    df['year'] = df['observed_on'].dt.year
    df['month'] = df['observed_on'].dt.month
    df['date_month'] = df['observed_on'].values.astype('datetime64[M]')

    # keeps consistent order within dataframe
    df = df.sort_values("observed_on").reset_index(drop=True)
    return df

def load_air_quality_csv():
    ...

def load_weather_temp_csv():
    ...