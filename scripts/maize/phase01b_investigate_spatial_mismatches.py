from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "maize" / "processed"
RAW = ROOT / "data" / "maize" / "raw"
REPORTS = ROOT / "outputs" / "maize" / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

# Load spatial validation results
spatial = pd.read_csv(PROCESSED / "maize_gps_parcel_validation.csv")

# Load original audited ground truth
data = pd.read_csv(PROCESSED / "maize_ground_truth_audited.csv")

print("\nMAIZE PHASE 01B")
print("=" * 60)

print("\nSPATIAL DATA COLUMNS")
print(spatial.columns.tolist())

# Check spatial join results
matched = spatial["Farm_ID_parcel"].notna()

same_farm = spatial["Farm_ID_field"] == spatial["Farm_ID_parcel"]

inside_same = matched & same_farm
inside_different = matched & ~same_farm
outside_all = ~matched

print("\nSPATIAL CLASSIFICATION")
print("Total valid GPS:", len(spatial))
print("Inside matching farm parcel:", inside_same.sum())
print("Inside different maize parcel:", inside_different.sum())
print("Outside all maize parcels:", outside_all.sum())

# Identify farms with spatial mismatches
mismatch = spatial.loc[~spatial["GPS_Inside_Matched_Parcel"]].copy()

farm_summary = (
    mismatch.groupby("Farm_ID_field")
    .agg(
        Mismatch_Observations=("Farm_ID_field", "size"),
        Campaigns=("Campaign", "nunique"),
        Mean_LAI=("LAI_Clean", "mean"),
        Parcel_Containing_Point=("Farm_ID_parcel", "first"),
    )
    .reset_index()
)

print("\nTOP FARM-LEVEL MISMATCHES")
print(
    farm_summary.sort_values("Mismatch_Observations", ascending=False)
    .head(20)
    .to_string(index=False)
)

# Inspect invalid GPS records
invalid_gps = data.loc[data["Invalid_GPS"]].copy()

print("\nINVALID GPS RECORDS")
print(
    invalid_gps[["Campaign", "Original_Farm_ID", "GpsLat", "GpsLong", "LAI"]].to_string(
        index=False
    )
)

# Save diagnostics
mismatch.to_csv(PROCESSED / "maize_spatial_mismatches.csv", index=False)

farm_summary.to_csv(REPORTS / "phase01b_farm_spatial_mismatch_summary.csv", index=False)

invalid_gps.to_csv(PROCESSED / "maize_invalid_gps_records.csv", index=False)

print("\nPHASE 01B OUTPUTS SAVED")
print("maize_spatial_mismatches.csv")
print("phase01b_farm_spatial_mismatch_summary.csv")
print("maize_invalid_gps_records.csv")


# ============================================================
# ADDITIONAL DIAGNOSTIC: FIELD FARM TO PARCEL MAPPING
# ============================================================

print("\n" + "=" * 65)
print("FIELD FARM TO PARCEL MAPPING ANALYSIS")
print("=" * 65)

# Count how often each field farm's GPS points fall in each parcel.
mapping = (
    spatial.dropna(subset=["Farm_ID_parcel"])
    .groupby(["Farm_ID_field", "Farm_ID_parcel"])
    .size()
    .reset_index(name="Observation_Count")
    .sort_values(["Farm_ID_field", "Observation_Count"], ascending=[True, False])
)

print("\nCOMPLETE FIELD-TO-PARCEL MAPPING")
print(mapping.to_string(index=False))

# Determine the most frequent parcel for each field farm.
dominant_mapping = (
    mapping.sort_values(["Farm_ID_field", "Observation_Count"], ascending=[True, False])
    .drop_duplicates("Farm_ID_field")
    .copy()
)

print("\nDOMINANT PARCEL PER FIELD FARM")
print(dominant_mapping.to_string(index=False))

# Show original farm ID values that normalize to unusual IDs.
print("\nORIGINAL FARM ID VALUES")
print(
    data.groupby(["Original_Farm_ID", "Farm_ID"])
    .size()
    .reset_index(name="Observations")
    .sort_values("Original_Farm_ID")
    .to_string(index=False)
)

# Save complete mapping.
mapping.to_csv(REPORTS / "phase01b_complete_farm_parcel_mapping.csv", index=False)

dominant_mapping.to_csv(
    REPORTS / "phase01b_dominant_farm_parcel_mapping.csv", index=False
)

print("\nAdditional mapping reports saved.")
