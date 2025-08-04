import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.plot_utils import plot_energy_by_temperature_bin

# File paths
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.csv")
output_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "energy_binned_by_temperature.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Run
df = pd.read_csv(data_path)
plot_energy_by_temperature_bin(df, save_path=output_path)
