from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEARCH_DIRS = [
    ROOT / "data" / "rice",
    ROOT / "outputs" / "rice",
]

print("\nAHERO SOURCE FILE INVENTORY")
print("=" * 65)

extensions = {
    ".tif",
    ".tiff",
    ".gpkg",
    ".shp",
    ".kml",
    ".csv",
    ".json",
    ".geojson",
    ".zip",
    ".ipynb",
}

for folder in SEARCH_DIRS:
    print(f"\nDIRECTORY: {folder}")

    if not folder.exists():
        print("Directory not found")
        continue

    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            print(f"{path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")

print("\n" + "=" * 65)
print("MISSING FARM ID INVESTIGATION")
print("=" * 65)

missing_file = (
    ROOT / "outputs" / "rice" / "reports" / "phase10" / "missing_farms_report.csv"
)

if not missing_file.exists():
    raise FileNotFoundError(missing_file)

missing = pd.read_csv(missing_file)

for campaign in sorted(missing["Campaign"].unique()):
    subset = missing[missing["Campaign"] == campaign]

    print(f"\nCampaign {campaign}")
    print("Missing farm IDs:")
    print(", ".join(subset["Farm_ID_norm"].astype(str)))

    print("\nOriginal LAI means:")
    print(
        subset[["Farm_ID_norm", "original_LAI_mean", "original_LAI_count"]].to_string(
            index=False
        )
    )

print("\nInventory completed.")
