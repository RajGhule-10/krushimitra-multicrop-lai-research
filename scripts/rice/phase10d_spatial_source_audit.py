from pathlib import Path
from zipfile import ZipFile
import tempfile
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import fiona

# ============================================================
# PHASE 10D: ORIGINAL AHERO RASTER + GEOMETRY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PACKAGE = Path.home() / "Downloads" / "rice_biophysical_project_complete"

SATELLITE_ZIP = PACKAGE / "satellite_imagery.zip"
PARCEL_ZIP = PACKAGE / "Parcels Geodatabase_updated.zip"

PROCESSED = ROOT / "data" / "rice" / "processed"

OUTPUT = ROOT / "outputs" / "rice" / "reports" / "phase10d"
OUTPUT.mkdir(parents=True, exist_ok=True)

CAMPAIGN_FILES = {
    1: PROCESSED / "final_df_rice_campaign_1.csv",
    2: PROCESSED / "final_df_rice_campaign_2.csv",
    3: PROCESSED / "final_df_rice_campaign_3.csv",
}

IMAGE_DATES = {
    "10-06-2023": "2023-06-10",
    "20-07-2023": "2023-07-20",
    "14-08-2023": "2023-08-14",
}


def normalize_farm_id(value):
    value = str(value).strip().upper()

    if value.startswith("AR") and value[2:].isdigit():
        return "AR" + value[2:].zfill(3)

    return value


def inspect_raster(path):

    print("\n" + "-" * 70)
    print("RASTER:", path.name)
    print("-" * 70)

    with rasterio.open(path) as src:
        print("Dimensions:", src.width, "x", src.height)
        print("Band count:", src.count)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)
        print("Dtypes:", src.dtypes)
        print("Descriptions:", src.descriptions)

        print("\nBand statistics:")

        for index in range(1, src.count + 1):
            data = src.read(index, masked=True)

            values = data.compressed()

            if values.size == 0:
                print(f"Band {index}: NO VALID PIXELS")
                continue

            print(
                f"Band {index}: "
                f"valid={values.size}, "
                f"min={np.min(values):.5f}, "
                f"max={np.max(values):.5f}, "
                f"mean={np.mean(values):.5f}"
            )

        print("\nRaster tags:")
        print(src.tags())

        return {
            "file": path.name,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "nodata": src.nodata,
            "bounds": str(src.bounds),
        }


def inspect_geopackage(path):

    print("\n" + "=" * 70)
    print("AHERO GEOPACKAGE INSPECTION")
    print("=" * 70)

    layers = fiona.listlayers(path)

    print("Layers:", layers)

    results = []

    for layer in layers:
        gdf = gpd.read_file(path, layer=layer)

        print("\nLayer:", layer)
        print("Records:", len(gdf))
        print("Geometry types:", gdf.geom_type.value_counts().to_dict())
        print("CRS:", gdf.crs)
        print("Columns:", list(gdf.columns))

        print("\nFirst records:")
        print(gdf.head(3).to_string())

        for column in gdf.columns:
            if "farm" in column.lower() or "id" in column.lower():
                print(f"\nPotential identifier column: {column}")
                print(gdf[column].head(10).tolist())

        results.append(
            {
                "layer": layer,
                "records": len(gdf),
                "crs": str(gdf.crs),
                "geometry_types": str(gdf.geom_type.value_counts().to_dict()),
                "columns": ",".join(gdf.columns.astype(str)),
            }
        )

    return results


def main():

    if not SATELLITE_ZIP.exists():
        raise FileNotFoundError(SATELLITE_ZIP)

    if not PARCEL_ZIP.exists():
        raise FileNotFoundError(PARCEL_ZIP)

    print("\nPHASE 10D: AHERO SPATIAL SOURCE AUDIT")

    # --------------------------------------------------------
    # Load campaign farm IDs
    # --------------------------------------------------------

    campaign_ids = {}

    for campaign, path in CAMPAIGN_FILES.items():
        df = pd.read_csv(path)

        if "Farm_ID" not in df.columns:
            raise ValueError(f"Farm_ID missing from {path}")

        ids = set(df["Farm_ID"].map(normalize_farm_id))

        campaign_ids[campaign] = ids

        print(f"\nCampaign {campaign}: {len(df)} records, {len(ids)} unique farms")

    all_farms = set.union(*campaign_ids.values())

    print("\nTotal unique campaign farms:", len(all_farms))

    raster_results = []
    gpkg_results = []

    with tempfile.TemporaryDirectory(prefix="ahero_audit_") as temp:
        temp = Path(temp)

        # ----------------------------------------------------
        # Extract original imagery archive
        # ----------------------------------------------------

        satellite_dir = temp / "satellite"

        with ZipFile(SATELLITE_ZIP) as archive:
            for item in archive.namelist():
                if item.startswith("__MACOSX/"):
                    continue

                if item.endswith("/"):
                    continue

                suffix = Path(item).suffix.lower()

                if suffix in {".tif", ".tiff", ".csv"}:
                    archive.extract(item, satellite_dir)

        # ----------------------------------------------------
        # Inspect all three acquisitions
        # ----------------------------------------------------

        for folder in sorted(satellite_dir.rglob("*")):
            if not folder.is_dir():
                continue

            if folder.name.endswith("_sen2res"):
                print("\n\nACQUISITION FOLDER:", folder.name)

                for raster in sorted(folder.glob("*.tif")):
                    result = inspect_raster(raster)
                    result["acquisition_folder"] = folder.name

                    raster_results.append(result)

                for csv in folder.rglob("Ahero*.csv"):
                    print("\nSIDECAR CSV:", csv.relative_to(satellite_dir))

                    try:
                        df = pd.read_csv(csv)

                        print("Shape:", df.shape)
                        print("Columns:", list(df.columns))
                        print(df.head(3).to_string())

                    except Exception as exc:
                        print("CSV read error:", exc)

        # ----------------------------------------------------
        # Extract and inspect Ahero GeoPackage
        # ----------------------------------------------------

        parcel_dir = temp / "parcels"

        with ZipFile(PARCEL_ZIP) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("ahero.gpkg")
                and not name.startswith("__MACOSX/")
            ]

            if not candidates:
                raise FileNotFoundError("Ahero.gpkg not found")

            gpkg_member = candidates[0]

            archive.extract(gpkg_member, parcel_dir)

        gpkg_path = parcel_dir / gpkg_member

        gpkg_results = inspect_geopackage(gpkg_path)

        # ----------------------------------------------------
        # Save metadata reports
        # ----------------------------------------------------

        pd.DataFrame(raster_results).to_csv(OUTPUT / "raster_metadata.csv", index=False)

        pd.DataFrame(gpkg_results).to_csv(
            OUTPUT / "geopackage_metadata.csv", index=False
        )

    print("\n" + "=" * 70)
    print("PHASE 10D COMPLETED")
    print("=" * 70)

    print("Raster metadata:", OUTPUT / "raster_metadata.csv")
    print("GeoPackage metadata:", OUTPUT / "geopackage_metadata.csv")


if __name__ == "__main__":
    main()
