from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "krushimitra_all_campaigns.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Please copy your CSV into the data folder and "
        "rename it to krushimitra_all_campaigns.csv."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Dataset path:", DATA_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")

for column in df.columns:
    print(" -", column)


# ==================================================
# 3. CHECK REQUIRED COLUMNS
# ==================================================

bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

indices = ["NDVI", "NDRE", "EVI", "SAVI", "NDMI", "NDWI"]

targets = ["LAI_mean", "CC_mean"]

required_columns = (
    bands + indices + targets + ["ESU_ID", "Unit_ID", "Season", "S2_Date"]
)

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError("\nMissing required columns:\n" + "\n".join(missing_columns))


# ==================================================
# 4. IDENTIFIERS AND GROUPS
# ==================================================

print("\n========== IDENTIFIERS AND GROUPS ==========")

for column in ["ESU_ID", "Unit_ID", "Season", "S2_Date"]:
    print(f"\n{column}")

    print("Unique values:", df[column].nunique(dropna=True))

    print(df[column].value_counts(dropna=False).sort_index())


print("\nDuplicate ESU_ID rows:")

print(df["ESU_ID"].duplicated().sum())


# ==================================================
# 5. MISSING VALUES
# ==================================================

print("\n========== MISSING VALUES ==========")

missing_values = df[required_columns].isna().sum().sort_values(ascending=False)

print(missing_values)


# ==================================================
# 6. NUMERICAL CONVERSION AND QUALITY CHECKS
# ==================================================

numeric_columns = bands + indices + targets

numeric_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

print("\n========== NUMERICAL SUMMARY ==========")

summary = numeric_df.describe().T.round(4)

print(summary)


print("\n========== NON-FINITE VALUES ==========")

for column in numeric_columns:
    values = numeric_df[column].to_numpy(dtype=float, na_value=np.nan)

    print(f"{column}:", int((~np.isfinite(values)).sum()))


# ==================================================
# 7. SATELLITE DATE CHECKS
# ==================================================

print("\n========== DATE CHECKS ==========")

if "S2_Date" in df.columns:
    df["S2_Date"] = pd.to_datetime(df["S2_Date"], errors="coerce")

    print("\nSentinel-2 acquisition dates:")

    print(df["S2_Date"].value_counts().sort_index())


if "LP80_date" in df.columns:
    df["LP80_date"] = pd.to_datetime(df["LP80_date"], errors="coerce")


if "Date_Difference_Days" in df.columns:
    print("\nGround-to-satellite date differences:")

    print(df["Date_Difference_Days"].value_counts(dropna=False).sort_index())


# ==================================================
# 8. QUALITY-CONTROL FIELDS
# ==================================================

print("\n========== QUALITY CONTROL ==========")

qa_columns = [
    "Has_Usable_Sentinel2",
    "Field_Valid_Pixel_Percentage",
    "S2_Valid_Pixel_Count",
]

for column in qa_columns:
    if column in df.columns:
        print(f"\n{column}")

        print(df[column].describe())

        print(df[column].value_counts(dropna=False).sort_index())


# ==================================================
# 9. TARGET SUMMARIES
# ==================================================

print("\n========== TARGET SUMMARIES ==========")

for target in targets:
    print(f"\n{target}")

    print(pd.to_numeric(df[target], errors="coerce").describe())


# ==================================================
# 10. SAVE AUDIT FILES
# ==================================================

summary.to_csv(OUTPUT_DIR / "spectral_target_summary.csv")

missing_values.to_csv(OUTPUT_DIR / "missing_values.csv")

df.describe(include="all").T.to_csv(OUTPUT_DIR / "dataset_summary.csv")

df.groupby(["Season", "Unit_ID"], dropna=False).size().rename(
    "Observation_Count"
).to_csv(OUTPUT_DIR / "observations_by_season_and_unit.csv")


# ==================================================
# 11. FINISH
# ==================================================

print("\n========== AUDIT COMPLETED ==========")

print("Dataset:", DATA_PATH)

print("Output directory:", OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", file.name)
