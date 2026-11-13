from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "krushimitra_all_campaigns.csv"

PREDICTIONS_PATH = PROJECT_DIR / "outputs" / "lai_out_of_fold_predictions.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

predictions = pd.read_csv(PREDICTIONS_PATH)


# ==================================================
# 3. LAI DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

plt.hist(
    pd.to_numeric(df["LAI_mean"], errors="coerce").dropna(), bins=15, edgecolor="black"
)

plt.xlabel("Measured LAI")

plt.ylabel("Observation count")

plt.title("Distribution of Measured Leaf Area Index")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "lai_distribution.png", dpi=300)

plt.close()


# ==================================================
# 4. LAI BY ACQUISITION DATE
# ==================================================

if "S2_Date" in df.columns:
    plt.figure(figsize=(11, 6))

    date_data = df.copy()

    date_data["S2_Date"] = pd.to_datetime(date_data["S2_Date"], errors="coerce")

    date_data.boxplot(column="LAI_mean", by="S2_Date", rot=45)

    plt.xlabel("Sentinel-2 acquisition date")

    plt.ylabel("Measured LAI")

    plt.title("Measured LAI by Acquisition Date")

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "lai_by_acquisition_date.png", dpi=300)

    plt.close()


# ==================================================
# 5. ACTUAL VS PREDICTED
# ==================================================

random_forest = predictions[predictions["Model"] == "Random_Forest"].copy()

experiments = [
    ("NDVI_only", "NDVI-only Random Forest"),
    ("Multispectral_bands", "Multispectral-band Random Forest"),
    ("Bands_and_indices", "Bands-and-indices Random Forest"),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (experiment_name, title) in zip(axes, experiments):
    subset = random_forest[random_forest["Experiment"] == experiment_name]

    if subset.empty:
        ax.set_title(f"{title}\nNo predictions available")

        continue

    ax.scatter(subset["Actual_LAI"], subset["Predicted_LAI"], alpha=0.75)

    minimum = min(subset["Actual_LAI"].min(), subset["Predicted_LAI"].min())

    maximum = max(subset["Actual_LAI"].max(), subset["Predicted_LAI"].max())

    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")

    ax.set_xlabel("Measured LAI")

    ax.set_ylabel("Predicted LAI")

    ax.set_title(title)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "lai_actual_vs_predicted.png", dpi=300)

plt.close()


print("\nPlots generated:")

for filename in [
    "lai_distribution.png",
    "lai_by_acquisition_date.png",
    "lai_actual_vs_predicted.png",
]:
    print(OUTPUT_DIR / filename)
