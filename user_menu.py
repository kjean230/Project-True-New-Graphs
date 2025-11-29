# user_menu.py

GRAPH_MENU = {
    "1": "Graph 1: Temperature and AQI over time",
    "2": "Graph 2: Spider and fly abundance over time",
    "3": "Graph 3A: Temperature vs spider abundance (scatter)",
    "4": "Graph 3B: Temperature vs fly abundance (scatter)",
    "5": "Graph 3C: AQI vs spider abundance (scatter)",
    "6": "Graph 3D: AQI vs fly abundance (scatter)",
    "7": "ML: Linear regression for spider abundance",
    "8": "ML: Linear regression for fly abundance",
}

def print_menu():
    print("\n=== Arthropods & Environment Graph / Model Menu ===")
    for key in sorted(GRAPH_MENU.keys(), key=int):
        print(f"{key}. {GRAPH_MENU[key]}")
    print("Q. Quit\n")