from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "krushimitra_all_campaigns.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

required = [
    "ESU_ID",
    "Unit_ID",
    "Season",
    "Latitude",
    "Longitude",
    "X_UTM34",
    "Y_UTM34",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ==================================================
# 3. EXTRACT THE UNIT SUFFIX
# ==================================================

df["Unit_Suffix"] = df["Unit_ID"].astype(str).str.extract(r"(U\d+)$", expand=False)

print("\n========== UNIT IDENTIFIERS ==========")

print(
    df.groupby(["Season", "Unit_ID"], dropna=False)
    .agg(
        Observations=("ESU_ID", "size"),
        Unique_ESUs=("ESU_ID", "nunique"),
        Mean_Latitude=("Latitude", "mean"),
        Mean_Longitude=("Longitude", "mean"),
        Min_X=("X_UTM34", "min"),
        Max_X=("X_UTM34", "max"),
        Min_Y=("Y_UTM34", "min"),
        Max_Y=("Y_UTM34", "max"),
    )
    .round(6)
    .to_string()
)


# ==================================================
# 4. COMPARE MATCHING SUFFIXES ACROSS SEASONS
# ==================================================

print("\n========== SUFFIX COMPARISON ==========")

for suffix in sorted(df["Unit_Suffix"].dropna().unique()):
    subset = df[df["Unit_Suffix"] == suffix]

    print(f"\nUnit suffix: {suffix}")

    for season, group in subset.groupby("Season"):
        print(season, "| observations:", len(group), "| coordinate ranges:")

        print("Latitude:", group["Latitude"].min(), "to", group["Latitude"].max())

        print("Longitude:", group["Longitude"].min(), "to", group["Longitude"].max())

        print("X_UTM34:", group["X_UTM34"].min(), "to", group["X_UTM34"].max())

        print("Y_UTM34:", group["Y_UTM34"].min(), "to", group["Y_UTM34"].max())


# ==================================================
# 5. SAVE SUMMARY
# ==================================================

summary = (
    df.groupby(["Season", "Unit_ID"], dropna=False)
    .agg(
        Observations=("ESU_ID", "size"),
        Unique_ESUs=("ESU_ID", "nunique"),
        Mean_Latitude=("Latitude", "mean"),
        Mean_Longitude=("Longitude", "mean"),
        Mean_X_UTM34=("X_UTM34", "mean"),
        Mean_Y_UTM34=("Y_UTM34", "mean"),
    )
    .reset_index()
)

summary.to_csv(OUTPUT_DIR / "physical_unit_coordinate_summary.csv", index=False)

print(
    "\nSaved coordinate summary to:",
    OUTPUT_DIR / "physical_unit_coordinate_summary.csv",
)
