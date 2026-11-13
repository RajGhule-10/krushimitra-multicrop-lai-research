from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PHASE 03D: BUILD FINAL MAIZE SATELLITE FEATURE DATASET
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "maize" / "processed"
REPORTS = ROOT / "outputs" / "maize" / "reports"

ORIGINAL_FILE = PROCESSED / "maize_sentinel2_extracted.csv"
PHASE03C_FILE = PROCESSED / "maize_phase03c_comparison.csv"

OUTPUT_FILE = PROCESSED / "maize_final_satellite_features.csv"
SUMMARY_FILE = REPORTS / "phase03d_final_dataset_summary.txt"

REPORTS.mkdir(parents=True, exist_ok=True)

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

INDICES = [
    "NDVI",
    "EVI",
    "GNDVI",
    "SAVI",
    "NDRE",
    "RVI",
    "DVI",
    "NDWI",
    "NDMI",
    "BNDVI",
]

FEATURES = BANDS + INDICES

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------


def first_available(row, names, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def make_key(row):
    """
    Composite observation key.
    Used to prevent duplicate field observations.
    """

    campaign = (
        str(first_available(row, ["campaign_id", "Campaign"], "")).strip().lower()
    )

    farm = (
        str(
            first_available(
                row,
                [
                    "Farm_ID",
                    "farm_id_diagnostic",
                    "original_farm_id",
                    "Original_Farm_ID",
                ],
                "",
            )
        )
        .strip()
        .upper()
    )

    date = pd.to_datetime(
        first_available(row, ["Measurement_Date", "observation_date", "Date"]),
        errors="coerce",
    )

    date_string = date.strftime("%Y-%m-%d") if pd.notna(date) else ""

    time = str(first_available(row, ["Time"], "")).strip()

    lai = pd.to_numeric(
        first_available(row, ["LAI_Clean", "LAI", "original_lai"]), errors="coerce"
    )

    lat = pd.to_numeric(
        first_available(row, ["Latitude_Clean", "GpsLat", "latitude"]), errors="coerce"
    )

    lon = pd.to_numeric(
        first_available(row, ["Longitude_Clean", "GpsLong", "longitude"]),
        errors="coerce",
    )

    return (
        campaign,
        farm,
        date_string,
        time,
        round(lai, 5) if pd.notna(lai) else None,
        round(lat, 7) if pd.notna(lat) else None,
        round(lon, 7) if pd.notna(lon) else None,
    )


def extract_base(row):
    """Standardize field and spatial metadata."""

    return {
        "Campaign": first_available(row, ["Campaign", "campaign_id"]),
        "campaign_id": first_available(row, ["campaign_id", "Campaign"]),
        "Farm_ID": first_available(
            row,
            ["Farm_ID", "farm_id_diagnostic", "original_farm_id", "Original_Farm_ID"],
        ),
        "Original_Farm_ID": first_available(
            row, ["Original_Farm_ID", "original_farm_id", "Farm_ID"]
        ),
        "Measurement_Date": first_available(
            row, ["Measurement_Date", "observation_date", "Date"]
        ),
        "Time": first_available(row, ["Time"]),
        "LAI": first_available(row, ["LAI_Clean", "LAI", "original_lai"]),
        "Latitude": first_available(row, ["Latitude_Clean", "GpsLat", "latitude"]),
        "Longitude": first_available(row, ["Longitude_Clean", "GpsLong", "longitude"]),
        "GPS_Inside_Matched_Parcel": first_available(
            row, ["GPS_Inside_Matched_Parcel"]
        ),
        "GPS_Inside_Any_Maize_Parcel": first_available(
            row, ["GPS_Inside_Any_Maize_Parcel"]
        ),
        "Spatial_Review_Required": first_available(row, ["Spatial_Review_Required"]),
        "Inside_Study_Area": first_available(row, ["Inside_Study_Area"]),
        "Farm_ID_Exists_In_Parcel_Data": first_available(
            row, ["Farm_ID_Exists_In_Parcel_Data"]
        ),
        "Unresolved_Farm_ID": first_available(row, ["Unresolved_Farm_ID"]),
    }


def extract_features(row, prefix=None):
    """Extract a common set of satellite bands and indices."""

    values = {}

    for feature in FEATURES:
        candidates = []

        if prefix:
            candidates.append(f"{prefix}_{feature}")

        candidates.extend([feature, feature.lower(), feature.upper()])

        values[feature] = pd.to_numeric(
            first_available(row, candidates), errors="coerce"
        )

    return values


def make_record(row, source, method, prefix=None):
    record = extract_base(row)

    record.update(extract_features(row, prefix))

    record["Satellite_Source"] = source
    record["Extraction_Method"] = method

    if prefix == "exact":
        record["Satellite_Image_Date"] = first_available(row, ["exact_image_date"])
        record["Satellite_Image_ID"] = first_available(row, ["exact_image_id"])
        record["Image_Date_Difference_Days"] = first_available(
            row, ["exact_day_difference"]
        )
        record["Absolute_Image_Date_Difference_Days"] = first_available(
            row, ["exact_absolute_day_difference"]
        )
        record["Scene_Cloud_Percentage"] = first_available(
            row, ["exact_cloud_percentage"]
        )
        record["Neighbourhood_Valid_Pixel_Count"] = np.nan
        record["Nearest_Pixel_Offset_M"] = 0.0

    elif prefix == "neighbourhood":
        record["Satellite_Image_Date"] = first_available(
            row, ["neighbourhood_image_date"]
        )
        record["Satellite_Image_ID"] = first_available(row, ["neighbourhood_image_id"])
        record["Image_Date_Difference_Days"] = first_available(
            row, ["neighbourhood_day_difference"]
        )
        record["Absolute_Image_Date_Difference_Days"] = first_available(
            row, ["neighbourhood_absolute_day_difference"]
        )
        record["Scene_Cloud_Percentage"] = first_available(
            row, ["neighbourhood_cloud_percentage"]
        )
        record["Neighbourhood_Valid_Pixel_Count"] = first_available(
            row, ["neighbourhood_valid_pixel_count"]
        )
        record["Nearest_Pixel_Offset_M"] = first_available(
            row, ["neighbourhood_nearest_pixel_offset_m"]
        )

    else:
        # Original Phase 03 exact-GPS extraction
        record["Satellite_Image_Date"] = first_available(
            row, ["image_date", "s2_image_date", "selected_image_date"]
        )
        record["Satellite_Image_ID"] = first_available(
            row, ["image_id", "s2_image_id", "selected_image_id"]
        )
        record["Image_Date_Difference_Days"] = first_available(
            row, ["day_difference", "date_difference_days"]
        )
        record["Absolute_Image_Date_Difference_Days"] = (
            abs(record["Image_Date_Difference_Days"])
            if pd.notna(record["Image_Date_Difference_Days"])
            else np.nan
        )
        record["Scene_Cloud_Percentage"] = first_available(
            row, ["cloud_percentage", "CLOUDY_PIXEL_PERCENTAGE"]
        )
        record["Neighbourhood_Valid_Pixel_Count"] = np.nan
        record["Nearest_Pixel_Offset_M"] = 0.0

    return record


# ------------------------------------------------------------
# LOAD INPUTS
# ------------------------------------------------------------

print("=" * 70)
print("PHASE 03D: FINAL MAIZE DATASET CONSTRUCTION")
print("=" * 70)

for file in [ORIGINAL_FILE, PHASE03C_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Required input file not found: {file}")

original = pd.read_csv(ORIGINAL_FILE)
comparison = pd.read_csv(PHASE03C_FILE)

print(f"\nOriginal Phase 03 records: {len(original)}")
print(f"Phase 03C comparison records: {len(comparison)}")

# ------------------------------------------------------------
# IDENTIFY SUCCESSFUL RECORDS
# ------------------------------------------------------------

# Original Phase 03 file contains successful extractions.
original_records = []

for _, row in original.iterrows():
    original_records.append(
        make_record(row, source="Phase 03", method="EXACT_GPS_ORIGINAL")
    )

# Phase 03C: exact GPS successes
exact_success = comparison[
    comparison["exact_status"].astype(str).str.upper() == "SUCCESS"
].copy()

# Phase 03C: neighbourhood-only successes
neighbourhood_only = comparison[
    (comparison["exact_status"].astype(str).str.upper() != "SUCCESS")
    & (comparison["neighbourhood_status"].astype(str).str.upper() == "SUCCESS")
].copy()

print("\nPhase 03C extraction selection:")
print(f"Exact GPS successes: {len(exact_success)}")
print(f"Neighbourhood-only successes: {len(neighbourhood_only)}")

phase03c_records = []

for _, row in exact_success.iterrows():
    phase03c_records.append(
        make_record(row, source="Phase 03C", method="EXACT_GPS", prefix="exact")
    )

for _, row in neighbourhood_only.iterrows():
    phase03c_records.append(
        make_record(
            row, source="Phase 03C", method="NEIGHBOURHOOD_30M", prefix="neighbourhood"
        )
    )

# ------------------------------------------------------------
# COMBINE AND DEDUPLICATE
# ------------------------------------------------------------

final = pd.DataFrame(original_records + phase03c_records)

final["_dedup_key"] = final.apply(make_key, axis=1)

before_dedup = len(final)

duplicate_count = int(final["_dedup_key"].duplicated().sum())

if duplicate_count:
    print(f"\nWARNING: {duplicate_count} duplicate observation keys found.")
    print("Keeping the first record for each observation key.")

    final = final.drop_duplicates(subset="_dedup_key", keep="first").copy()

final.drop(columns=["_dedup_key"], inplace=True)
final.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------
# VALIDATE FEATURE COMPLETENESS
# ------------------------------------------------------------

missing_features = [feature for feature in FEATURES if feature not in final.columns]

if missing_features:
    raise ValueError(f"Missing expected satellite features: {missing_features}")

feature_missingness = final[FEATURES].isna().sum()

complete_feature_rows = int(final[FEATURES].notna().all(axis=1).sum())

print("\nFINAL DATASET VALIDATION")
print("-" * 50)
print(f"Rows before deduplication: {before_dedup}")
print(f"Duplicate keys removed: {duplicate_count}")
print(f"Final unique observations: {len(final)}")
print(f"Rows with all {len(FEATURES)} features: {complete_feature_rows}")

print("\nFeature missingness:")
print(feature_missingness.to_string())

# ------------------------------------------------------------
# EXTRACTION METHOD SUMMARY
# ------------------------------------------------------------

method_summary = (
    final.groupby(["Satellite_Source", "Extraction_Method"], dropna=False)
    .size()
    .rename("observations")
    .reset_index()
)

campaign_summary = (
    final.groupby("campaign_id", dropna=False)
    .agg(
        observations=("LAI", "size"),
        unique_farms=("Farm_ID", "nunique"),
        mean_LAI=("LAI", "mean"),
        min_LAI=("LAI", "min"),
        max_LAI=("LAI", "max"),
    )
    .reset_index()
)

spatial_summary = {
    "inside_matching_parcel": int(
        final["GPS_Inside_Matched_Parcel"].fillna(False).astype(bool).sum()
    ),
    "inside_any_maize_parcel": int(
        final["GPS_Inside_Any_Maize_Parcel"].fillna(False).astype(bool).sum()
    ),
    "spatial_review_required": int(
        final["Spatial_Review_Required"].fillna(False).astype(bool).sum()
    ),
}

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

# Put identifiers and metadata before satellite features.
metadata_columns = [
    "Campaign",
    "campaign_id",
    "Farm_ID",
    "Original_Farm_ID",
    "Measurement_Date",
    "Time",
    "LAI",
    "Latitude",
    "Longitude",
    "Satellite_Source",
    "Extraction_Method",
    "Satellite_Image_Date",
    "Satellite_Image_ID",
    "Image_Date_Difference_Days",
    "Absolute_Image_Date_Difference_Days",
    "Scene_Cloud_Percentage",
    "Neighbourhood_Valid_Pixel_Count",
    "Nearest_Pixel_Offset_M",
    "GPS_Inside_Matched_Parcel",
    "GPS_Inside_Any_Maize_Parcel",
    "Spatial_Review_Required",
    "Inside_Study_Area",
    "Farm_ID_Exists_In_Parcel_Data",
    "Unresolved_Farm_ID",
]

ordered_columns = [c for c in metadata_columns if c in final.columns] + FEATURES

# Retain any additional columns after the standardized columns.
ordered_columns += [c for c in final.columns if c not in ordered_columns]

final = final[ordered_columns]

final.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------
# WRITE REPORT
# ------------------------------------------------------------

with open(SUMMARY_FILE, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 03D: FINAL MAIZE SATELLITE FEATURE DATASET\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Original Phase 03 successful records: {len(original)}\n")
    f.write(f"Phase 03C exact successes: {len(exact_success)}\n")
    f.write(f"Phase 03C neighbourhood-only successes: {len(neighbourhood_only)}\n")
    f.write(f"Rows before deduplication: {before_dedup}\n")
    f.write(f"Duplicate keys removed: {duplicate_count}\n")
    f.write(f"Final unique observations: {len(final)}\n")
    f.write(f"Rows with all {len(FEATURES)} features: {complete_feature_rows}\n\n")

    f.write("EXTRACTION METHOD SUMMARY\n")
    f.write("-" * 50 + "\n")
    f.write(method_summary.to_string(index=False))
    f.write("\n\n")

    f.write("CAMPAIGN SUMMARY\n")
    f.write("-" * 50 + "\n")
    f.write(campaign_summary.to_string(index=False))
    f.write("\n\n")

    f.write("SPATIAL QUALITY SUMMARY\n")
    f.write("-" * 50 + "\n")
    for key, value in spatial_summary.items():
        f.write(f"{key}: {value}\n")

    f.write("\nFEATURE MISSINGNESS\n")
    f.write("-" * 50 + "\n")
    f.write(feature_missingness.to_string())
    f.write("\n\n")

    f.write("OUTPUT FILE\n")
    f.write(str(OUTPUT_FILE))
    f.write("\n")

print("\nEXTRACTION METHOD SUMMARY")
print(method_summary.to_string(index=False))

print("\nCAMPAIGN SUMMARY")
print(campaign_summary.to_string(index=False))

print("\nSPATIAL QUALITY SUMMARY")
for key, value in spatial_summary.items():
    print(f"{key}: {value}")

print("\nOUTPUT FILES")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\n" + "=" * 70)
print("PHASE 03D COMPLETED")
print("=" * 70)
