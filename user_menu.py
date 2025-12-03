# user_menu.py
# Text menu for selecting graphs.

GRAPH_MENU = {
    "1": "Graph 1 (matplotlib): Temperature and AQI over time",
    "2": "Graph 2 (matplotlib): Spider and fly abundance over time",
    "3": "Graph 3A (matplotlib): Temperature vs spider abundance (scatter)",
    "4": "Graph 3B (matplotlib): Temperature vs fly abundance (scatter)",
    "5": "Graph 3C (matplotlib): AQI vs spider abundance (scatter)",
    "6": "Graph 3D (matplotlib): AQI vs fly abundance (scatter)",

    "7": "Graph 1 (seaborn): Temperature and AQI over time",
    "8": "Graph 2 (seaborn): Spider and fly abundance over time",
    "9": "Graph 3A (seaborn): Temperature vs spider abundance (scatter)",
    "10": "Graph 3B (seaborn): Temperature vs fly abundance (scatter)",
    "11": "Graph 3C (seaborn): AQI vs spider abundance (scatter)",
    "12": "Graph 3D (seaborn): AQI vs fly abundance (scatter)",

    "13": "Graph 4A (seaborn): Spider obs date vs temperature",
    "14": "Graph 4B (seaborn): Spider obs date vs AQI",
    "15": "Graph 4C (seaborn): Fly obs date vs temperature",
    "16": "Graph 4D (seaborn): Fly obs date vs AQI",
}


def print_menu():
    print("\n=== Arthropods & Environment Graph Menu ===")
    for key in sorted(GRAPH_MENU.keys(), key=int):
        print(f"{key}. {GRAPH_MENU[key]}")
    print("Q. Quit\n")