# clean csv file 
from pathlib import Path
import pandas as pd

# importing a path to CSV files
try:
    # base path files to air quality and temperature csv files
    BASE_PATH_AIRTEMP = Path(__file__).resolve().parent.parent / "Project True Tree Graphs" / "air and temp csvs" 
    BASE_PATH_SPIFLY = Path(__file__).resolve().parent.parent / "Project True Tree Graphs" / "spider and fly csvs"

    # by using the base paths, we can create smaller versions for the csv file
    PATH_AIR_QUALITY = BASE_PATH_AIRTEMP / "file_of_air_quality copy.csv"
    PATH_TEMP = BASE_PATH_AIRTEMP / "file_of_monthly_weather copy.csv"
    PATH_SPIDER = BASE_PATH_SPIFLY / "file_of_spiders - Sheet1 copy.csv"
    PATH_FLY = BASE_PATH_SPIFLY / "files_of_flies - Sheet1 copy.csv"
except: 
    raise FileNotFoundError(f"File for air quality is not found on this path")

try:
    df_air_quality = pd.read_csv(PATH_AIR_QUALITY, dtype=str)
    df_temp = pd.read_csv(PATH_TEMP, dtype=str)
    df_spider = pd.read_csv(PATH_SPIDER, dtype=str)
    df_fly = pd.read_csv(PATH_FLY, dtype=str)
except:
    raise ImportError("the files are not able to be added to dataframes")

print(df_fly)

# print("success!")