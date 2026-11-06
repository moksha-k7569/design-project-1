import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get project folder
project_folder = Path(__file__).resolve().parent.parent

# Read CSV file
csv_path = project_folder / "results" / "grain_measurements.csv"
df = pd.read_csv(csv_path)

# Plot grain size distribution
plt.figure(figsize=(9, 5))

plt.hist(df["Diameter_mm"], bins=20, edgecolor="black")

plt.title("Estimated Sand Grain-Region Size Distribution")
plt.xlabel("Estimated Diameter (mm)")
plt.ylabel("Number of Regions")

plt.grid(axis="y", alpha=0.3)

# Save graph
output_path = project_folder / "results" / "grain_size_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("Grain size distribution graph created!")
print("Saved at:", output_path)