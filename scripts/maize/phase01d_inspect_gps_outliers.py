from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "maize" / "processed"

data = pd.read_csv(PROCESSED / "maize_parcel_distance_diagnostics.csv")

ground_truth = pd.read_csv(PROCESSED / "maize_ground_truth_audited.csv")

print("\nMAIZE PHASE 01D")
print("=" * 65)

# Identify observations more than 1 km from their own parcel.
outliers = data[data["Own_Parcel_Distance"] > 1000].copy()

print("\nGPS OUTLIERS ABOVE 1 KM")
print("Number of observations:", len(outliers))

# Attach original GPS coordinates and original IDs.
details = outliers.merge(
    ground_truth[
        ["Farm_ID", "Original_Farm_ID", "Campaign", "Date", "GpsLat", "GpsLong", "LAI"]
    ],
    on=["Farm_ID", "Campaign", "LAI"],
    how="left",
)

print(
    details[
        [
            "Farm_ID",
            "Original_Farm_ID",
            "Campaign",
            "Date",
            "GpsLat",
            "GpsLong",
            "LAI",
            "Own_Parcel_Distance",
            "Nearest_Parcel_ID",
            "Nearest_Parcel_Distance",
        ]
    ].to_string(index=False)
)

print("\nGPS COORDINATE RANGES BY CAMPAIGN")

print(
    ground_truth.groupby("Campaign")
    .agg(
        Latitude_Min=("GpsLat", "min"),
        Latitude_Max=("GpsLat", "max"),
        Longitude_Min=("GpsLong", "min"),
        Longitude_Max=("GpsLong", "max"),
        Observation_Count=("Farm_ID", "size"),
    )
    .to_string()
)

print("\nMISSING OWN-PARCEL DISTANCES")

missing = data[data["Own_Parcel_Distance"].isna()]

print("Count:", len(missing))
print(missing[["Farm_ID", "Campaign", "LAI"]].to_string(index=False))
