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

    "13": "Graph 4A (seaborn): Spider observations vs temperature",
    "14": "Graph 4B (seaborn): Spider observations vs AQI",
    "15": "Graph 4C (seaborn): Fly observations vs temperature",
    "16": "Graph 4D (seaborn): Fly observations vs AQI",

    "17": "Tree Graph 1: Condition distribution (Paper Birch vs Red Maple)",
    "18": "Tree Graph 2: High risk proportion",
    "19": "Tree Graph 3: Spatial distribution",
    "20": "Tree Graph 4: DBH distribution",
    "21": "Tree Graph 5: High risk proportion (alternative)",

    "22": "ML Model: Spider abundance regression (with diagnostic plots)",
    "23": "ML Model: Fly abundance regression (with diagnostic plots)",
    "24": "ML Model: Temperature-AQI relationship (with diagnostic plots)",

}


def print_menu():
    print("\n=== Arthropods & Environment Graph Menu ===")
    for key in sorted(GRAPH_MENU.keys(), key=int):
        print(f"{key}. {GRAPH_MENU[key]}")
    print("Q. Quit\n")