from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "maize" / "processed"

data = pd.read_csv(PROCESSED / "maize_ground_truth_audited.csv")

print("\nMAIZE PHASE 01E")
print("=" * 65)

# Six geographically displaced observations
geographic_outliers = (data["Longitude_Clean"] < 35.5) | (
    data["Longitude_Clean"] > 36.5
)

# Unusual farm identifiers
unusual_ids = data["Farm_ID"].isin(["NM", "NW001"])

suspicious = data[geographic_outliers | unusual_ids].copy()

columns = [
    "Campaign",
    "Date",
    "Time",
    "Original_Farm_ID",
    "Farm_ID",
    "GpsLat",
    "GpsLong",
    "LAI",
    "Invalid_GPS",
]

print("\nALL SUSPICIOUS OBSERVATIONS")
print("Count:", len(suspicious))

print(
    suspicious[columns]
    .sort_values(["Campaign", "Original_Farm_ID", "Date"])
    .to_string(index=False)
)

print("\nSUSPICIOUS RECORD COUNTS")
print(suspicious.groupby(["Campaign", "Original_Farm_ID"]).size().to_string())

output = PROCESSED / "maize_suspicious_records_review.csv"

suspicious[columns].to_csv(output, index=False)

print("\nSaved:", output)
print("\nPHASE 01E COMPLETED")
