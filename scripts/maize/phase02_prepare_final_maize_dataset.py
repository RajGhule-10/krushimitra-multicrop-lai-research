from pathlib import Path
import re

import pandas as pd
import geopandas as gpd
from shapely.geometry import box


# ============================================================
# PHASE 02: FINAL MAIZE GROUND-TRUTH DATASET PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "maize" / "raw"
PROCESSED_DIR = ROOT / "data" / "maize" / "processed"
REPORT_DIR = ROOT / "outputs" / "maize" / "reports"

INPUT_FILE = PROCESSED_DIR / "maize_ground_truth_audited.csv"
GPKG_FILE = RAW_DIR / "Narok.gpkg"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nMAIZE PHASE 02")
print("=" * 65)


# ============================================================
# 1. LOAD AUDITED GROUND TRUTH
# ============================================================

data = pd.read_csv(INPUT_FILE)

print("\nINPUT DATA")
print("Observations:", len(data))


# ============================================================
# 2. STANDARDIZE FARM IDS AND DATES
# ============================================================


def normalize_farm_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    match = re.fullmatch(r"([A-Z]+)0*(\d+)", value)

    if match:
        return f"{match.group(1)}{int(match.group(2)):03d}"

    return value


data["Farm_ID"] = data["Farm_ID"].apply(normalize_farm_id)

data["LAI_Clean"] = pd.to_numeric(data["LAI_Clean"], errors="coerce")

data["Latitude_Clean"] = pd.to_numeric(data["Latitude_Clean"], errors="coerce")

data["Longitude_Clean"] = pd.to_numeric(data["Longitude_Clean"], errors="coerce")

data["Measurement_Date"] = pd.to_datetime(
    data["Date"], format="%Y/%m/%d", errors="coerce"
)

data["Invalid_LAI_Final"] = data["LAI_Clean"].isna() | (data["LAI_Clean"] < 0)

data["Invalid_GPS_Final"] = (
    data["Latitude_Clean"].isna()
    | data["Longitude_Clean"].isna()
    | (data["Latitude_Clean"] == 0)
    | (data["Longitude_Clean"] == 0)
    | ~data["Latitude_Clean"].between(-90, 90)
    | ~data["Longitude_Clean"].between(-180, 180)
)

data["Invalid_Date_Final"] = data["Measurement_Date"].isna()


# ============================================================
# 3. LOAD MAIZE PARCELS
# ============================================================

parcels = gpd.read_file(GPKG_FILE, layer="Narok - Maize & Wheat")

maize = parcels[
    parcels["Crop Type"].astype(str).str.strip().str.lower() == "maize"
].copy()

maize["Parcel_Farm_ID"] = maize["Parcel Id"].apply(normalize_farm_id)

print("\nPARCEL INFORMATION")
print("Maize polygons:", len(maize))
print("Parcel CRS:", maize.crs)

if maize.crs is None:
    raise ValueError("Maize parcel CRS is missing.")


# ============================================================
# 4. DEFINE A BROAD NAROK STUDY AREA
# ============================================================

# Use UTM Zone 37S for local distance and spatial operations.
# This avoids using Web Mercator for distance calculations.

TARGET_CRS = "EPSG:32737"

maize_utm = maize.to_crs(TARGET_CRS)

xmin, ymin, xmax, ymax = maize_utm.total_bounds

# Expand parcel bounding box by 2 km in every direction.
# This allows nearby GPS points to remain candidates even
# when they do not fall exactly inside a parcel polygon.

study_area = box(xmin - 2000, ymin - 2000, xmax + 2000, ymax + 2000)

print("\nSTUDY AREA")
print("CRS:", TARGET_CRS)
print("Buffer around parcel extent: 2,000 metres")


# ============================================================
# 5. CREATE GPS POINTS
# ============================================================

data["_row_id"] = range(len(data))

valid_gps = data.loc[~data["Invalid_GPS_Final"]].copy()

points = gpd.GeoDataFrame(
    valid_gps,
    geometry=gpd.points_from_xy(
        valid_gps["Longitude_Clean"], valid_gps["Latitude_Clean"]
    ),
    crs="EPSG:4326",
).to_crs(TARGET_CRS)

points["Inside_Study_Area"] = points.geometry.within(study_area)

# Assign broad study-area status back to the master table.
area_lookup = points.set_index("_row_id")["Inside_Study_Area"].to_dict()

data["Inside_Study_Area"] = data["_row_id"].map(area_lookup).fillna(False).astype(bool)


# ============================================================
# 6. VALIDATE GPS AGAINST MAIZE PARCELS
# ============================================================

parcel_geometry = maize_utm[["Parcel_Farm_ID", "geometry"]].copy()

joined = gpd.sjoin(points, parcel_geometry, how="left", predicate="within")

joined["Inside_Any_Maize_Parcel"] = joined["Parcel_Farm_ID"].notna()

joined["Inside_Matched_Parcel"] = joined["Farm_ID"] == joined["Parcel_Farm_ID"]

spatial_summary = joined.groupby("_row_id").agg(
    GPS_Inside_Any_Maize_Parcel=("Inside_Any_Maize_Parcel", "max"),
    GPS_Inside_Matched_Parcel=("Inside_Matched_Parcel", "max"),
    GPS_Containing_Parcel=(
        "Parcel_Farm_ID",
        lambda values: ";".join(sorted(set(values.dropna().astype(str)))),
    ),
)

for column in [
    "GPS_Inside_Any_Maize_Parcel",
    "GPS_Inside_Matched_Parcel",
]:
    data[column] = (
        data["_row_id"].map(spatial_summary[column]).fillna(False).astype(bool)
    )

data["GPS_Containing_Parcel"] = (
    data["_row_id"].map(spatial_summary["GPS_Containing_Parcel"]).fillna("")
)


# ============================================================
# 7. CLASSIFY EACH OBSERVATION
# ============================================================

parcel_ids = set(maize["Parcel_Farm_ID"].dropna())

data["Farm_ID_Exists_In_Parcel_Data"] = data["Farm_ID"].isin(parcel_ids)

# These are known unresolved source identifiers.
data["Unresolved_Farm_ID"] = data["Farm_ID"].isin(["NM", "NW001"])

conditions = [
    data["Invalid_LAI_Final"],
    data["Invalid_Date_Final"],
    data["Invalid_GPS_Final"],
    ~data["Inside_Study_Area"],
    data["Unresolved_Farm_ID"],
    ~data["Farm_ID_Exists_In_Parcel_Data"],
]

choices = [
    "HOLD_INVALID_LAI",
    "HOLD_INVALID_DATE",
    "EXCLUDE_INVALID_GPS",
    "EXCLUDE_OUTSIDE_STUDY_AREA",
    "HOLD_UNRESOLVED_FARM_ID",
    "HOLD_UNMATCHED_FARM_ID",
]

data["Dataset_Status"] = "READY_FOR_S2_EXTRACTION"

import numpy as np

data["Dataset_Status"] = np.select(
    conditions, choices, default="READY_FOR_S2_EXTRACTION"
)

# Preserve records outside their own parcel as candidates,
# but flag them for additional spatial review.
data["Spatial_Review_Required"] = (
    data["Dataset_Status"].eq("READY_FOR_S2_EXTRACTION")
    & ~data["GPS_Inside_Matched_Parcel"]
)


# ============================================================
# 8. CREATE FINAL DATASETS
# ============================================================

candidates = data[data["Dataset_Status"] == "READY_FOR_S2_EXTRACTION"].copy()

held = data[data["Dataset_Status"].str.startswith("HOLD")].copy()

excluded = data[data["Dataset_Status"].str.startswith("EXCLUDE")].copy()

# Remove internal processing index.
for frame in [data, candidates, held, excluded]:
    frame.drop(columns=["_row_id"], inplace=True)

# Output files
master_file = PROCESSED_DIR / "maize_final_master_456.csv"

candidate_file = PROCESSED_DIR / "maize_sentinel2_candidate_observations.csv"

held_file = PROCESSED_DIR / "maize_held_observations.csv"

excluded_file = PROCESSED_DIR / "maize_excluded_observations.csv"

summary_file = REPORT_DIR / "phase02_final_maize_dataset_summary.txt"

data.to_csv(master_file, index=False)
candidates.to_csv(candidate_file, index=False)
held.to_csv(held_file, index=False)
excluded.to_csv(excluded_file, index=False)


# ============================================================
# 9. PRINT FINAL SUMMARY
# ============================================================

print("\nFINAL DATASET CLASSIFICATION")
print("-" * 65)

print(data["Dataset_Status"].value_counts(dropna=False).to_string())

print("\nSPATIAL QUALITY")
print("Inside any maize parcel:", data["GPS_Inside_Any_Maize_Parcel"].sum())

print("Inside matching farm parcel:", data["GPS_Inside_Matched_Parcel"].sum())

print(
    "Candidate observations requiring spatial review:",
    candidates["Spatial_Review_Required"].sum(),
)

print("\nFINAL COUNTS")
print("Original observations:", len(data))
print("Sentinel-2 candidates:", len(candidates))
print("Held observations:", len(held))
print("Excluded observations:", len(excluded))

print("\nCAMPAIGN-WISE CANDIDATE COUNTS")

print(
    candidates.groupby("Campaign")
    .agg(
        Observations=("LAI_Clean", "size"),
        Unique_Farms=("Farm_ID", "nunique"),
        Mean_LAI=("LAI_Clean", "mean"),
        Min_LAI=("LAI_Clean", "min"),
        Max_LAI=("LAI_Clean", "max"),
    )
    .round(3)
    .to_string()
)


# ============================================================
# 10. WRITE REPORT
# ============================================================

with open(summary_file, "w", encoding="utf-8") as report:
    report.write("FINAL MAIZE DATASET PREPARATION REPORT\n")
    report.write("=" * 65 + "\n\n")

    report.write(f"Original observations: {len(data)}\n")
    report.write(f"Sentinel-2 candidates: {len(candidates)}\n")
    report.write(f"Held observations: {len(held)}\n")
    report.write(f"Excluded observations: {len(excluded)}\n\n")

    report.write("DATASET STATUS COUNTS\n")
    report.write(data["Dataset_Status"].value_counts().to_string())

    report.write("\n\nSPATIAL QUALITY\n")
    report.write(
        f"Inside any maize parcel: {data['GPS_Inside_Any_Maize_Parcel'].sum()}\n"
    )
    report.write(
        f"Inside matching farm parcel: {data['GPS_Inside_Matched_Parcel'].sum()}\n"
    )
    report.write(
        f"Candidate records requiring spatial review: "
        f"{candidates['Spatial_Review_Required'].sum()}\n"
    )

    report.write("\n\nCAMPAIGN SUMMARY\n")
    report.write(
        candidates.groupby("Campaign")
        .agg(
            Observations=("LAI_Clean", "size"),
            Unique_Farms=("Farm_ID", "nunique"),
            Mean_LAI=("LAI_Clean", "mean"),
            Min_LAI=("LAI_Clean", "min"),
            Max_LAI=("LAI_Clean", "max"),
        )
        .round(3)
        .to_string()
    )

print("\nOUTPUT FILES")
print(master_file)
print(candidate_file)
print(held_file)
print(excluded_file)
print(summary_file)

print("\nMAIZE PHASE 02 COMPLETED.")
