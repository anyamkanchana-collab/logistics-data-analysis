"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
Run from the folder containing this script and hypothetical_logistics_dataset.csv:
    python logistics_analysis.py
Requirements:
    pip install pandas numpy matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("hypothetical_logistics_dataset.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate shipment IDs:", df["Shipment_ID"].duplicated().sum())
print("\nDescriptive statistics:\n", df.describe(numeric_only=True).round(2))

numeric_cols = [
    "Distance_km", "Weight_kg", "Shipment_Volume_units",
    "Delivery_Time_days", "Transportation_Cost_USD", "Delay_Days"
]
print("\nCorrelation matrix:\n", df[numeric_cols].corr().round(2))

mode_summary = df.groupby("Transport_Mode").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delivery_Days=("Delivery_Time_days", "mean"),
    Avg_Cost_USD=("Transportation_Cost_USD", "mean"),
    On_Time_Rate=("On_Time", "mean")
).round(2)
print("\nTransport mode summary:\n", mode_summary)

region_summary = df.groupby("Region").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delivery_Days=("Delivery_Time_days", "mean"),
    Avg_Cost_USD=("Transportation_Cost_USD", "mean"),
    On_Time_Rate=("On_Time", "mean")
).round(2)
print("\nRegion summary:\n", region_summary)

# 1. Delivery-time distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Delivery_Time_days"], bins=18, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of shipments")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "delivery_time_distribution.png", dpi=160)
plt.close()

# 2. Average delivery time by transport mode
avg_mode = mode_summary["Avg_Delivery_Days"].sort_values()
plt.figure(figsize=(8, 5))
plt.bar(avg_mode.index, avg_mode.values, edgecolor="black")
plt.title("Average Delivery Time by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("Average delivery time (days)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_delivery_by_mode.png", dpi=160)
plt.close()

# 3. Distance vs transportation cost
plt.figure(figsize=(8, 5))
plt.scatter(df["Distance_km"], df["Transportation_Cost_USD"], alpha=0.65)
plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation cost (USD)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_vs_cost.png", dpi=160)
plt.close()

# 4. Correlation matrix (numeric values annotated)
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
im = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, label="Correlation coefficient")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=70, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160)
plt.close()

# 5. On-time delivery rate by region
region_rates = (df.groupby("Region")["On_Time"].mean() * 100).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(region_rates.index, region_rates.values, edgecolor="black")
plt.axhline(df["On_Time"].mean() * 100, linestyle="--", label="Overall on-time rate")
plt.title("On-Time Delivery Rate by Region")
plt.xlabel("Region")
plt.ylabel("On-time shipments (%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "on_time_by_region.png", dpi=160)
plt.close()

print(f"\nCharts saved to: {OUTPUT_DIR.resolve()}")
