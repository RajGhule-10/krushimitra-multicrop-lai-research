from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PHASE 01: CASSAVA DATA PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "cassava" / "raw"
PROCESSED_DIR = ROOT / "data" / "cassava" / "processed"
REPORT_DIR = ROOT / "outputs" / "cassava" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LAI_FILE = RAW_DIR / "Ground-LAI-Clean.csv"
S2_FILE = RAW_DIR / "all_index_zonal_median.csv"


# The 13 vegetation indices identified in the dataset.
VEGETATION_INDICES = [
    "BNDVI",
    "CIG",
    "DVI",
    "EVI",
    "GNDVI",
    "GRVI",
    "NDVI",
    "NDWI",
    "RVI",
    "SAVI",
    "SeLI",
    "TCARI",
    "VIG",
]


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

for file_path in [LAI_FILE, S2_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Place both original CSV files inside data/cassava/raw/"
        )

print("\nPHASE 01: CASSAVA DATA PREPARATION")
print("=" * 60)


# ============================================================
# 2. LOAD DATA
# ============================================================

lai = pd.read_csv(LAI_FILE)
s2 = pd.read_csv(S2_FILE)

print("\nINPUT DATA")
print("LAI shape:", lai.shape)
print("Sentinel-2 shape:", s2.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

lai_required = ["MAP", "year", "month", "day", "Plot", "Sample", "LAI", "Plot_type"]

s2_required = ["Plot", "Value", "date", "MAP", "Index"]

for column in lai_required:
    if column not in lai.columns:
        raise ValueError(f"Missing LAI column: {column}")

for column in s2_required:
    if column not in s2.columns:
        raise ValueError(f"Missing Sentinel-2 column: {column}")


# ============================================================
# 4. NORMALIZE PLOT IDS
# ============================================================


def normalize_plot_id(value):
    """
    Normalize IDs such as E07 -> E7 and E7 -> E7.
    This prevents formatting differences from breaking joins.
    """
    value = str(value).strip().upper()

    match = re.fullmatch(r"([A-Z]+)0*(\d+)", value)

    if match:
        return f"{match.group(1)}{int(match.group(2))}"

    return value


lai["Plot"] = lai["Plot"].astype(str).str.strip()
s2["Plot"] = s2["Plot"].astype(str).str.strip()

lai["Plot_ID"] = lai["Plot"].apply(normalize_plot_id)
s2["Plot_ID"] = s2["Plot"].apply(normalize_plot_id)


# ============================================================
# 5. CLEAN LAI MEASUREMENTS
# ============================================================

lai["LAI"] = pd.to_numeric(lai["LAI"], errors="coerce")
lai["MAP"] = pd.to_numeric(lai["MAP"], errors="coerce")

lai["LAI_Date"] = pd.to_datetime(
    lai[["year", "month", "day"]].rename(
        columns={
            "year": "year",
            "month": "month",
            "day": "day",
        }
    ),
    errors="coerce",
)

lai = lai.replace([np.inf, -np.inf], np.nan)

lai = lai.dropna(subset=["Plot_ID", "MAP", "LAI", "LAI_Date"]).copy()

lai["MAP"] = lai["MAP"].astype(int)


# ============================================================
# 6. AGGREGATE FIELD LAI BY PLOT AND MAP
# ============================================================

# Multiple field samples from the same plot and MAP are
# aggregated into one mean LAI observation.

lai_aggregated = lai.groupby(["Plot_ID", "MAP"], as_index=False).agg(
    Plot=("Plot", "first"),
    LAI=("LAI", "mean"),
    LAI_SD=("LAI", "std"),
    LAI_Sample_Count=("LAI", "count"),
    Plot_type=("Plot_type", "first"),
    LAI_Date_Start=("LAI_Date", "min"),
    LAI_Date_End=("LAI_Date", "max"),
)

lai_aggregated["LAI_Date_Midpoint"] = (
    lai_aggregated["LAI_Date_Start"]
    + (lai_aggregated["LAI_Date_End"] - lai_aggregated["LAI_Date_Start"]) / 2
)

print("\nLAI AGGREGATION")
print("Original LAI measurements:", len(lai))
print("Aggregated plot-MAP records:", len(lai_aggregated))
print("Unique plots:", lai_aggregated["Plot_ID"].nunique())


# ============================================================
# 7. CLEAN SENTINEL-2 DATA
# ============================================================

s2["MAP"] = pd.to_numeric(s2["MAP"], errors="coerce")
s2["Value"] = pd.to_numeric(s2["Value"], errors="coerce")
s2["date"] = pd.to_datetime(s2["date"], errors="coerce")

s2 = s2.replace([np.inf, -np.inf], np.nan)

s2 = s2.dropna(subset=["Plot_ID", "MAP", "Value", "date", "Index"]).copy()

s2["MAP"] = s2["MAP"].astype(int)
s2["Index"] = s2["Index"].astype(str).str.strip()

# Verify there is no duplicate satellite value for the same
# plot, MAP, date and index.
duplicate_count = s2.duplicated(subset=["Plot_ID", "MAP", "date", "Index"]).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate Sentinel-2 records. "
        "Inspect them before continuing."
    )

print("\nSENTINEL-2 DATA")
print("Clean records:", len(s2))
print("Unique plots:", s2["Plot_ID"].nunique())
print("Acquisition dates:", s2["date"].nunique())
print("Unique indices/bands:", s2["Index"].nunique())


# ============================================================
# 8. CONVERT SATELLITE DATA FROM LONG TO WIDE FORMAT
# ============================================================

# Each plot-MAP combination becomes one row.
# Every Index becomes a separate feature column.

s2_wide = s2.pivot(
    index=["Plot_ID", "MAP"], columns="Index", values="Value"
).reset_index()

s2_wide.columns.name = None

# Add the corresponding satellite acquisition date.
date_lookup = s2.groupby(["Plot_ID", "MAP"], as_index=False).agg(
    S2_Date=("date", "first")
)

s2_wide = s2_wide.merge(
    date_lookup, on=["Plot_ID", "MAP"], how="left", validate="one_to_one"
)

print("\nSATELLITE FEATURE TABLE")
print("Rows:", len(s2_wide))
print("Feature columns:", len(s2_wide.columns) - 3)


# ============================================================
# 9. MATCH LAI WITH SENTINEL-2 FEATURES
# ============================================================

matched = lai_aggregated.merge(
    s2_wide, on=["Plot_ID", "MAP"], how="inner", validate="one_to_one"
)

matched["Days_From_LAI"] = (
    matched["S2_Date"] - matched["LAI_Date_Midpoint"]
).dt.total_seconds().abs() / 86400

matched["Days_From_LAI"] = matched["Days_From_LAI"].round(2)

print("\nLAI-SATELLITE MATCHING")
print("Aggregated LAI records:", len(lai_aggregated))
print("Satellite plot-MAP records:", len(s2_wide))
print("Matched records:", len(matched))
print("Unique matched plots:", matched["Plot_ID"].nunique())


# ============================================================
# 10. IDENTIFY UNMATCHED OBSERVATIONS
# ============================================================

lai_keys = lai_aggregated[["Plot_ID", "MAP"]].drop_duplicates()
s2_keys = s2_wide[["Plot_ID", "MAP"]].drop_duplicates()

unmatched_lai = lai_keys.merge(
    s2_keys, on=["Plot_ID", "MAP"], how="left", indicator=True
)

unmatched_lai = unmatched_lai[unmatched_lai["_merge"] == "left_only"].drop(
    columns="_merge"
)

unmatched_s2 = s2_keys.merge(
    lai_keys, on=["Plot_ID", "MAP"], how="left", indicator=True
)

unmatched_s2 = unmatched_s2[unmatched_s2["_merge"] == "left_only"].drop(
    columns="_merge"
)


# ============================================================
# 11. IDENTIFY FEATURE GROUPS
# ============================================================

available_indices = [
    feature for feature in VEGETATION_INDICES if feature in matched.columns
]

missing_indices = [
    feature for feature in VEGETATION_INDICES if feature not in matched.columns
]

# Keep all satellite columns except identifiers and date.
satellite_features = [
    column for column in s2_wide.columns if column not in ["Plot_ID", "MAP", "S2_Date"]
]

quality_columns = [
    "AOT",
    "MSK_CLDPRB",
    "QA10",
    "QA20",
    "QA60",
    "TCI_B",
    "TCI_G",
    "TCI_R",
]

quality_features = [
    feature for feature in satellite_features if feature in quality_columns
]

spectral_features = [
    feature
    for feature in satellite_features
    if feature not in available_indices and feature not in quality_features
]


# ============================================================
# 12. SAVE PROCESSED DATASETS
# ============================================================

lai_output = PROCESSED_DIR / "cassava_lai_plot_map_aggregated.csv"
s2_output = PROCESSED_DIR / "cassava_sentinel2_plot_map_features.csv"
matched_output = PROCESSED_DIR / "cassava_lai_s2_matched.csv"
unmatched_lai_output = PROCESSED_DIR / "cassava_unmatched_lai_records.csv"
unmatched_s2_output = PROCESSED_DIR / "cassava_unmatched_s2_records.csv"

lai_aggregated.to_csv(lai_output, index=False)
s2_wide.to_csv(s2_output, index=False)
matched.to_csv(matched_output, index=False)
unmatched_lai.to_csv(unmatched_lai_output, index=False)
unmatched_s2.to_csv(unmatched_s2_output, index=False)


# ============================================================
# 13. SAVE DATA QUALITY REPORT
# ============================================================

report_path = REPORT_DIR / "phase01_cassava_data_quality.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("CASSAVA DATA PREPARATION REPORT\n")
    report.write("=" * 55 + "\n\n")

    report.write(f"Original LAI measurements: {len(lai)}\n")
    report.write(f"Aggregated LAI plot-MAP records: {len(lai_aggregated)}\n")
    report.write(f"Satellite plot-MAP records: {len(s2_wide)}\n")
    report.write(f"Matched records: {len(matched)}\n")
    report.write(f"Unique matched plots: {matched['Plot_ID'].nunique()}\n")
    report.write(f"Acquisition dates: {s2['date'].nunique()}\n")
    report.write(f"Satellite features: {len(satellite_features)}\n\n")

    report.write("AVAILABLE VEGETATION INDICES\n")
    report.write(", ".join(available_indices) + "\n\n")

    report.write("MISSING EXPECTED INDICES\n")
    report.write(", ".join(missing_indices) if missing_indices else "None")
    report.write("\n\n")

    report.write("SPECTRAL FEATURES\n")
    report.write(", ".join(spectral_features) + "\n\n")

    report.write("QUALITY / AUXILIARY FEATURES\n")
    report.write(", ".join(quality_features) + "\n\n")

    report.write("MATCHED OBSERVATIONS BY MAP\n")
    report.write(matched.groupby("MAP").size().to_string() + "\n\n")

    report.write("UNMATCHED LAI RECORDS BY MAP\n")
    report.write(unmatched_lai.groupby("MAP").size().to_string() + "\n\n")

    report.write("UNMATCHED SATELLITE RECORDS BY MAP\n")
    report.write(unmatched_s2.groupby("MAP").size().to_string() + "\n\n")

    report.write("LAI SUMMARY\n")
    report.write(matched["LAI"].describe().to_string() + "\n\n")

    report.write("SATELLITE DATE DIFFERENCE SUMMARY (DAYS)\n")
    report.write(matched["Days_From_LAI"].describe().to_string())


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\nSAVED FILES")
print(lai_output)
print(s2_output)
print(matched_output)
print(unmatched_lai_output)
print(unmatched_s2_output)
print(report_path)

print("\nVEGETATION INDICES AVAILABLE:", available_indices)
print("MISSING EXPECTED INDICES:", missing_indices)

print("\nMATCHED RECORDS BY MAP")
print(matched.groupby("MAP").size())

print("\nDATE DIFFERENCE SUMMARY")
print(matched["Days_From_LAI"].describe())

print("\nPhase 01 completed successfully.")
