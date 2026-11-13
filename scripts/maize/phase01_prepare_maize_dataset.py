from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# PHASE 01: MAIZE GROUND-TRUTH AND SPATIAL DATA PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "maize" / "raw"
PROCESSED_DIR = ROOT / "data" / "maize" / "processed"
REPORT_DIR = ROOT / "outputs" / "maize" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_FILES = [
    "Maize_campaign_1_LAI_Data.csv",
    "Maize_campaign_2_LAI_Data.csv",
    "Maize_campaign_3_LAI_Data.csv",
]

GPKG_FILE = RAW_DIR / "Narok.gpkg"

# Known field names from the original maize campaign files.
# The script checks available columns before processing.


# ============================================================
# 1. NORMALIZE FARM IDS
# ============================================================


def normalize_farm_id(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    match = re.fullmatch(r"([A-Z]+)0*(\d+)", value)

    if match:
        return f"{match.group(1)}{int(match.group(2)):03d}"

    return value


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print("\nMAIZE PHASE 01")
print("=" * 65)

for filename in CAMPAIGN_FILES:
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing campaign file: {path}")

if not GPKG_FILE.exists():
    raise FileNotFoundError(f"Missing GeoPackage: {GPKG_FILE}")


# ============================================================
# 3. LOAD CAMPAIGNS
# ============================================================

campaign_frames = []

for number, filename in enumerate(CAMPAIGN_FILES, start=1):
    path = RAW_DIR / filename
    df = pd.read_csv(path)

    print(f"\nCampaign {number}")
    print("File:", filename)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    df["Campaign"] = f"campaign_{number}"

    campaign_frames.append(df)

data = pd.concat(campaign_frames, ignore_index=True)

print("\nCOMBINED DATA")
print("Rows:", len(data))
print("Columns:", data.columns.tolist())


# ============================================================
# 4. IDENTIFY COLUMN NAMES
# ============================================================


def find_column(dataframe, candidates, description):
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify {description}. "
        f"Available columns: {dataframe.columns.tolist()}"
    )


farm_col = find_column(data, ["Farm_ID", "farm_id", "Farm ID"], "farm ID column")

lai_col = find_column(data, ["LAI", "lai", "Leaf Area Index"], "LAI column")

lat_col = find_column(
    data, ["GpsLat", "Latitude", "latitude", "Lat", "lat"], "latitude column"
)

lon_col = find_column(
    data,
    ["GpsLong", "Longitude", "longitude", "Lon", "lon", "Long"],
    "longitude column",
)

date_col = find_column(
    data, ["Date", "date", "Measurement_Date"], "measurement date column"
)

# Store the original farm ID and create normalized IDs.
data["Original_Farm_ID"] = data[farm_col].astype(str)
data["Farm_ID"] = data[farm_col].apply(normalize_farm_id)

data["LAI_Clean"] = pd.to_numeric(data[lai_col], errors="coerce")

data["Latitude_Clean"] = pd.to_numeric(data[lat_col], errors="coerce")

data["Longitude_Clean"] = pd.to_numeric(data[lon_col], errors="coerce")

data["Measurement_Date"] = pd.to_datetime(
    data[date_col], errors="coerce", dayfirst=False
)


# ============================================================
# 5. BASIC DATA QUALITY CHECKS
# ============================================================

data["Invalid_LAI"] = data["LAI_Clean"].isna() | (data["LAI_Clean"] < 0)

data["Invalid_GPS"] = (
    data["Latitude_Clean"].isna()
    | data["Longitude_Clean"].isna()
    | (data["Latitude_Clean"] == 0)
    | (data["Longitude_Clean"] == 0)
    | ~data["Latitude_Clean"].between(-90, 90)
    | ~data["Longitude_Clean"].between(-180, 180)
)

data["Invalid_Date"] = data["Measurement_Date"].isna()

print("\nDATA QUALITY")
print("Total observations:", len(data))
print("Missing/invalid LAI:", data["Invalid_LAI"].sum())
print("Invalid GPS:", data["Invalid_GPS"].sum())
print("Invalid dates:", data["Invalid_Date"].sum())
print("Unique normalized farms:", data["Farm_ID"].nunique())


# ============================================================
# 6. LOAD NAROK GEOPACKAGE
# ============================================================

print("\nGEOPACKAGE LAYERS")

layers = gpd.list_layers(GPKG_FILE)
print(layers.to_string(index=False))

# Select the maize and wheat layer identified in the package.
gdf = gpd.read_file(GPKG_FILE, layer="Narok - Maize & Wheat")

print("\nPARCEL DATA")
print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:", gdf.columns.tolist())


# ============================================================
# 7. FILTER MAIZE PARCELS
# ============================================================

crop_col = find_column(
    gdf, ["Crop Type", "Crop_Type", "Crop"], "parcel crop type column"
)

parcel_id_col = find_column(
    gdf, ["Parcel Id", "Parcel_ID", "Parcel ID"], "parcel ID column"
)

maize_parcels = gdf[gdf[crop_col].astype(str).str.strip().str.lower() == "maize"].copy()

maize_parcels["Farm_ID"] = maize_parcels[parcel_id_col].apply(normalize_farm_id)

print("\nMAIZE PARCELS")
print("Maize polygons:", len(maize_parcels))
print("Unique maize parcel IDs:", maize_parcels["Farm_ID"].nunique())


# ============================================================
# 8. MATCH FIELD FARM IDS WITH PARCEL IDS
# ============================================================

field_farms = set(data["Farm_ID"].dropna().unique())
parcel_farms = set(maize_parcels["Farm_ID"].dropna().unique())

matched_farms = field_farms.intersection(parcel_farms)

unmatched_field_farms = sorted(field_farms - parcel_farms)
unmatched_parcel_farms = sorted(parcel_farms - field_farms)

print("\nFARM-PARCEL MATCHING")
print("Field farms:", len(field_farms))
print("Parcel farms:", len(parcel_farms))
print("Matched farms:", len(matched_farms))
print("Unmatched field farms:", len(unmatched_field_farms))
print("Unmatched parcel farms:", len(unmatched_parcel_farms))


# ============================================================
# 9. GPS POINTS AND PARCEL SPATIAL CHECK
# ============================================================

valid_gps = data.loc[~data["Invalid_GPS"]].copy()

# GPS columns are expected to be longitude/latitude in WGS84.
gps_gdf = gpd.GeoDataFrame(
    valid_gps,
    geometry=gpd.points_from_xy(
        valid_gps["Longitude_Clean"], valid_gps["Latitude_Clean"]
    ),
    crs="EPSG:4326",
)

if maize_parcels.crs is None:
    raise ValueError("Maize parcel CRS is missing.")

gps_projected = gps_gdf.to_crs(maize_parcels.crs)

# Join GPS points to parcel polygons.
spatial_join = gpd.sjoin(
    gps_projected,
    maize_parcels[["Farm_ID", "geometry"]],
    how="left",
    predicate="within",
    lsuffix="field",
    rsuffix="parcel",
)
# Validate that each GPS point falls inside its corresponding farm parcel.

print("\nSPATIAL JOIN COLUMNS:")
print(spatial_join.columns.tolist())

parcel_match_col = "Farm_ID_parcel"
field_match_col = "Farm_ID_field"

if parcel_match_col not in spatial_join.columns:
    raise ValueError("Farm_ID_parcel column is missing.")

if field_match_col not in spatial_join.columns:
    raise ValueError("Farm_ID_field column is missing.")

spatial_join["GPS_Inside_Matched_Parcel"] = (
    spatial_join[field_match_col].eq(spatial_join[parcel_match_col])
    & spatial_join[parcel_match_col].notna()
)

print("\nGPS SPATIAL VALIDATION")
print("Valid GPS observations:", len(spatial_join))
print("Points inside matching parcel:", spatial_join["GPS_Inside_Matched_Parcel"].sum())
print(
    "Points not inside matching parcel:",
    (~spatial_join["GPS_Inside_Matched_Parcel"]).sum(),
)


# ============================================================
# 10. CAMPAIGN SUMMARY
# ============================================================

campaign_summary = (
    data.groupby("Campaign")
    .agg(
        Observations=("LAI_Clean", "size"),
        Unique_Farms=("Farm_ID", "nunique"),
        Mean_LAI=("LAI_Clean", "mean"),
        Min_LAI=("LAI_Clean", "min"),
        Max_LAI=("LAI_Clean", "max"),
        Invalid_LAI=("Invalid_LAI", "sum"),
        Invalid_GPS=("Invalid_GPS", "sum"),
        Invalid_Date=("Invalid_Date", "sum"),
    )
    .reset_index()
)

print("\nCAMPAIGN SUMMARY")
print(campaign_summary.to_string(index=False))


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

# Preserve all original records and add quality flags.
clean_output = PROCESSED_DIR / "maize_ground_truth_audited.csv"

unmatched_output = PROCESSED_DIR / "maize_unmatched_field_farms.csv"

spatial_output = PROCESSED_DIR / "maize_gps_parcel_validation.csv"

summary_output = REPORT_DIR / "phase01_maize_campaign_summary.csv"

data.to_csv(clean_output, index=False)

pd.DataFrame({"Unmatched_Field_Farm_ID": unmatched_field_farms}).to_csv(
    unmatched_output, index=False
)

spatial_join.drop(columns=["geometry"]).to_csv(spatial_output, index=False)

campaign_summary.to_csv(summary_output, index=False)


# ============================================================
# 12. WRITE TEXT REPORT
# ============================================================

report_path = REPORT_DIR / "phase01_maize_data_quality.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("MAIZE DATA QUALITY REPORT\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"Total observations: {len(data)}\n")
    report.write(f"Unique field farms: {len(field_farms)}\n")
    report.write(f"Maize parcel polygons: {len(maize_parcels)}\n")
    report.write(f"Matched farm IDs: {len(matched_farms)}\n")
    report.write(f"Unmatched field farms: {len(unmatched_field_farms)}\n")

    report.write("\nINVALID RECORDS\n")
    report.write(f"Invalid LAI: {data['Invalid_LAI'].sum()}\n")
    report.write(f"Invalid GPS: {data['Invalid_GPS'].sum()}\n")
    report.write(f"Invalid dates: {data['Invalid_Date'].sum()}\n")

    report.write("\nCAMPAIGN SUMMARY\n")
    report.write(campaign_summary.to_string(index=False))

    report.write("\n\nGPS SPATIAL VALIDATION\n")
    report.write(
        spatial_join["GPS_Inside_Matched_Parcel"].value_counts(dropna=False).to_string()
    )

print("\nOUTPUT FILES")
print(clean_output)
print(unmatched_output)
print(spatial_output)
print(summary_output)
print(report_path)

print("\nMAIZE PHASE 01 COMPLETED.")
