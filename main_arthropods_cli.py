# main_arthropods_cli.py
"""
Main CLI entrypoint for arthropod–environment graphs.

Flow:
- Build a combined monthly dataframe for 2017–2023.
- Show a simple text menu.
- Based on user input, call the appropriate plotting function
  (matplotlib or seaborn).
"""

from pathlib import Path
import pandas as pd

from monthly_dataset import build_monthly_env_arthropod_df

# matplotlib-based plots
from plotting_graphs import (
    plot_temp_aqi_over_time,
    plot_spider_fly_over_time,
    plot_temp_vs_spider_scatter,
    plot_temp_vs_fly_scatter,
    plot_aqi_vs_spider_scatter,
    plot_aqi_vs_fly_scatter,
)

# seaborn-based plots
from plotting_arthropods_seaborn import (
    plot_temp_aqi_over_time_seaborn,
    plot_spider_fly_over_time_seaborn,
    plot_temp_vs_spider_scatter_seaborn,
    plot_temp_vs_fly_scatter_seaborn,
    plot_aqi_vs_spider_scatter_seaborn,
    plot_aqi_vs_fly_scatter_seaborn,
)

from user_menu import print_menu


def main():
    # --------- Paths to CSVs (adjust if needed) ---------
    base_dir = Path(__file__).resolve().parent.parent / "Project True Tree Graphs"

    base_path_airtemp = base_dir / "air and temp csvs"
    base_path_spifly = base_dir / "spider and fly csvs"

    air_quality_csv = base_path_airtemp / "file_of_air_quality copy.csv"
    temp_csv = base_path_airtemp / "file_of_monthly_weather copy.csv"
    spider_csv = base_path_spifly / "file_of_spiders - Sheet1 copy.csv"
    fly_csv = base_path_spifly / "files_of_flies - Sheet1 copy.csv"

    # --------- Analysis window and station ---------
    start = pd.Timestamp("2017-01-01")
    cutoff = pd.Timestamp("2023-12-31")
    station_name = "LAGUARDIA AIRPORT, NY US"

    # --------- Build the combined monthly dataframe ---------
    print("Building combined monthly dataframe (this may take a moment)...")
    monthly_df = build_monthly_env_arthropod_df(
        spider_csv=spider_csv,
        fly_csv=fly_csv,
        aq_csv=air_quality_csv,
        temp_csv=temp_csv,
        start=start,
        cutoff=cutoff,
        station_name=station_name,
    )
    print("Done. Rows in monthly_df:", len(monthly_df))

    # --------- Interactive menu loop ---------
    while True:
        print_menu()
        choice = input("Select a graph number (or Q to quit): ").strip()

        if choice.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        # matplotlib graphs
        if choice == "1":
            plot_temp_aqi_over_time(monthly_df)
        elif choice == "2":
            plot_spider_fly_over_time(monthly_df)
        elif choice == "3":
            plot_temp_vs_spider_scatter(monthly_df)
        elif choice == "4":
            plot_temp_vs_fly_scatter(monthly_df)
        elif choice == "5":
            plot_aqi_vs_spider_scatter(monthly_df)
        elif choice == "6":
            plot_aqi_vs_fly_scatter(monthly_df)

        # seaborn graphs
        elif choice == "7":
            plot_temp_aqi_over_time_seaborn(monthly_df)
        elif choice == "8":
            plot_spider_fly_over_time_seaborn(monthly_df)
        elif choice == "9":
            plot_temp_vs_spider_scatter_seaborn(monthly_df)
        elif choice == "10":
            plot_temp_vs_fly_scatter_seaborn(monthly_df)
        elif choice == "11":
            plot_aqi_vs_spider_scatter_seaborn(monthly_df)
        elif choice == "12":
            plot_aqi_vs_fly_scatter_seaborn(monthly_df)

        else:
            print("Unrecognized option. Please try again.")


if __name__ == "__main__":
    main()
