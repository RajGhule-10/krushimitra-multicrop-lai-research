from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "maize" / "raw"
PROCESSED = ROOT / "data" / "maize" / "processed"
REPORTS = ROOT / "outputs" / "maize" / "reports"

GPKG = RAW / "Narok.gpkg"

print("\nMAIZE PHASE 01C")
print("=" * 60)

# Load original audited observations
data = pd.read_csv(PROCESSED / "maize_ground_truth_audited.csv")

data = data.loc[~data["Invalid_GPS"]].copy()

# Load maize parcel geometries
gdf = gpd.read_file(GPKG, layer="Narok - Maize & Wheat")

maize = gdf[gdf["Crop Type"].astype(str).str.strip().str.lower() == "maize"].copy()

# Use the same farm-ID normalization logic
import re


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    match = re.fullmatch(r"([A-Z]+)0*(\d+)", value)

    if match:
        return f"{match.group(1)}{int(match.group(2)):03d}"

    return value


data["Farm_ID"] = data["Farm_ID"].apply(normalize_id)
maize["Farm_ID"] = maize["Parcel Id"].apply(normalize_id)

# Project GPS points to parcel CRS
points = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data["Longitude_Clean"], data["Latitude_Clean"]),
    crs="EPSG:4326",
).to_crs(maize.crs)

# Calculate distances in metres.
# EPSG:3857 is used by the supplied GeoPackage.
if maize.crs.to_epsg() != 3857:
    print("WARNING: Parcel CRS is not EPSG:3857.")
    print("Distances will be expressed in the parcel CRS units.")

# Match each observation to its own farm parcel
parcel_lookup = maize.set_index("Farm_ID").geometry.to_dict()

distances = []

for _, row in points.iterrows():
    farm_id = row["Farm_ID"]
    point = row.geometry

    own_parcel = parcel_lookup.get(farm_id)

    own_distance = point.distance(own_parcel) if own_parcel is not None else None

    nearest_idx = maize.geometry.distance(point).idxmin()
    nearest_parcel = maize.loc[nearest_idx]

    nearest_distance = point.distance(nearest_parcel.geometry)

    distances.append(
        {
            "Farm_ID": farm_id,
            "Campaign": row["Campaign"],
            "LAI": row["LAI_Clean"],
            "Own_Parcel_Distance": own_distance,
            "Nearest_Parcel_ID": nearest_parcel["Farm_ID"],
            "Nearest_Parcel_Distance": nearest_distance,
        }
    )

result = pd.DataFrame(distances)

result["Own_Parcel_Distance"] = result["Own_Parcel_Distance"].round(2)

result["Nearest_Parcel_Distance"] = result["Nearest_Parcel_Distance"].round(2)

print("\nDISTANCE SUMMARY")
print(result[["Own_Parcel_Distance", "Nearest_Parcel_Distance"]].describe().round(2))

print("\nOBSERVATIONS WITH LARGEST OWN-PARCEL DISTANCES")
print(
    result.sort_values("Own_Parcel_Distance", ascending=False, na_position="last")
    .head(25)
    .to_string(index=False)
)

output = PROCESSED / "maize_parcel_distance_diagnostics.csv"
result.to_csv(output, index=False)

print("\nSaved:", output)
print("\nPHASE 01C COMPLETED")
