import sys
import os
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import add_comfort_penalty
from src.plot_utils import plot_energy_vs_penalty_bin

# relative paths
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.csv")
output_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "penalty_bin_bar.png")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Run
df = pd.read_csv(data_path)
df = add_comfort_penalty(df)
plot_energy_vs_penalty_bin(df, save_path=output_path)
