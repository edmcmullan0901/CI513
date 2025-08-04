# src/plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_energy_vs_temp(df, save_path=None):
    #air temp vs meter readings (scatterplot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(1000, random_state=42),
                    x='air_temperature',
                    y='meter_reading',
                    alpha=0.5)
    plt.title("Energy Usage vs Temperature")
    plt.xlabel("Air Temperature (°C)")
    plt.ylabel("Energy Usage (kWh)")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_penalty_curve(df, model, poly, save_path=None):
    #plots penalty curve with optimal temp 
    sample = df.sample(frac=0.1, random_state=42)
    temp_range = np.linspace(df['air_temperature'].min(), df['air_temperature'].max(), 200).reshape(-1, 1)
    preds = model.predict(poly.transform(temp_range))
    optimal_temp = temp_range[np.argmin(preds)][0]

    plt.figure(figsize=(10, 6))
    plt.scatter(sample['air_temperature'], sample['penalized_energy'], s=10, alpha=0.2, label="Sampled Data (10%)")
    plt.plot(temp_range, preds, color='red', linewidth=2, label="Regression Curve")
    plt.axvline(optimal_temp, color='green', linestyle='--', label=f"Optimal Temp ≈ {optimal_temp:.2f}°C")
    plt.title("Optimal Temperature Target Based on Comfort Penalty")
    plt.xlabel("Indoor Temperature (°C)")
    plt.ylabel("Penalized Energy Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_energy_vs_penalty_bin(df, save_path=None):
    
    #Avg energy use per comfort penalty bin (bar chart).
    
    bins = [0, 1, 3, 5, 10, 15, 25]
    labels = ['0', '1–2', '3–4', '5–9', '10–14', '15+']
    df['penalty_bin'] = pd.cut(df['temp_deviation'], bins=bins, labels=labels, include_lowest=True)
    bin_means = df.groupby('penalty_bin')['meter_reading'].mean()

    plt.figure(figsize=(10, 6))
    bin_means.plot(kind='bar', color='#4682B4', edgecolor='black')
    plt.title('Average Energy Usage vs Comfort Penalty')
    plt.xlabel('°C away from Comfort Point')
    plt.ylabel('Average Energy Usage (kWh)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_energy_by_temperature_bin(df, save_path=None):
    
    #Plots average energy usage by fixed air temperature bins from -20°C to 40°C.
    
    df = df.copy()
    df = df[df['air_temperature'].notnull()]
    
    # Fixed bin edges from -20 to 40
    bins = np.arange(-20, 45, 5)  # ends at 40
    df['temp_bin'] = pd.cut(df['air_temperature'], bins=bins)

    bin_means = df.groupby('temp_bin')['meter_reading'].mean()

    plt.figure(figsize=(10, 6))
    bin_means.plot(kind='bar', color='teal', edgecolor='black')
    plt.title("Average Energy Usage by Temperature Bin")
    plt.xlabel("Temperature Range (°C)")
    plt.ylabel("Average Energy Usage (kWh)")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_energy_by_building_type(df, save_path=None):
    
    #Bar chart showing average energy usage per building type.
    
    df = df.copy()
    df = df[df['meter_reading'] > 0]

    grouped = df.groupby('primary_use')['meter_reading'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    grouped.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Energy Consumption by Building Use Type")
    plt.xlabel("Building Use Type")
    plt.ylabel("Average Energy Consumption (kWh)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()