from pathlib import Path
from zipfile import ZipFile
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

# ============================================================
# PHASE 10E: FARM-LEVEL SENTINEL-2 COVERAGE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PACKAGE = Path.home() / "Downloads" / "rice_biophysical_project_complete"

SATELLITE_ZIP = PACKAGE / "satellite_imagery.zip"
PARCEL_ZIP = PACKAGE / "Parcels Geodatabase_updated.zip"

PROCESSED = ROOT / "data" / "rice" / "processed"

OUTPUT = ROOT / "outputs" / "rice" / "reports" / "phase10e"
OUTPUT.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = {
    1: {
        "date": "2023-06-10",
        "folder": "10-06-2023_sen2res",
    },
    2: {
        "date": "2023-07-20",
        "folder": "20-07-2023_sen2res",
    },
    3: {
        "date": "2023-08-14",
        "folder": "14-08-2023_sen2res",
    },
}


def normalize_farm_id(value):
    value = str(value).strip().upper()

    if value.startswith("AR") and value[2:].isdigit():
        return "AR" + value[2:].zfill(3)

    return value


def find_raster(folder, masked):

    candidates = list(folder.glob("*super_resolved*.tif"))

    candidates = [p for p in candidates if "__MACOSX" not in str(p)]

    if masked:
        candidates = [p for p in candidates if "masked" in p.name.lower()]
    else:
        candidates = [p for p in candidates if "masked" not in p.name.lower()]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one raster in {folder}, masked={masked}; found {len(candidates)}"
        )

    return candidates[0]


def inspect_farm_coverage(gdf, raster_path, campaign, date, mode):

    records = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

        if gdf.crs is None:
            raise ValueError("Farm polygons have no CRS")

        projected = gdf.to_crs(raster_crs)

        # Read all bands; a pixel is considered valid only if
        # all 12 bands contain finite values.
        image = src.read()

        valid_pixels = np.all(np.isfinite(image), axis=0)

        print(f"\n{mode.upper()} RASTER")
        print("Raster:", raster_path.name)
        print("CRS:", raster_crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Finite pixels across all bands:", int(valid_pixels.sum()))

        for _, row in projected.iterrows():
            farm_id = normalize_farm_id(row["Parcel Id"])
            geom = row.geometry

            if geom is None or geom.is_empty:
                records.append(
                    {
                        "Campaign": campaign,
                        "Date": date,
                        "Mode": mode,
                        "Farm_ID": farm_id,
                        "Polygon_valid": False,
                        "Polygon_area_m2": np.nan,
                        "Raster_pixels": 0,
                        "Valid_pixels": 0,
                        "Valid_pixel_percentage": 0,
                        "Coverage_status": "INVALID_GEOMETRY",
                    }
                )
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)

            inside = geometry_mask(
                [mapping(geom)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
                all_touched=False,
            )

            total_pixels = int(inside.sum())
            valid_count = int((inside & valid_pixels).sum())

            percentage = 100 * valid_count / total_pixels if total_pixels > 0 else 0

            if total_pixels == 0:
                status = "OUTSIDE_RASTER"
            elif valid_count == 0:
                status = "NO_VALID_PIXELS"
            elif valid_count < 3:
                status = "VERY_LOW_COVERAGE"
            else:
                status = "VALID_PIXELS_AVAILABLE"

            records.append(
                {
                    "Campaign": campaign,
                    "Date": date,
                    "Mode": mode,
                    "Farm_ID": farm_id,
                    "Polygon_valid": geom.is_valid,
                    "Polygon_area_m2": geom.area,
                    "Raster_pixels": total_pixels,
                    "Valid_pixels": valid_count,
                    "Valid_pixel_percentage": round(percentage, 2),
                    "Coverage_status": status,
                }
            )

    return records


def main():

    print("\n" + "=" * 70)
    print("PHASE 10E: FARM-LEVEL RASTER COVERAGE")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="ahero_coverage_") as temp:
        temp = Path(temp)

        # Extract Ahero GeoPackage
        with ZipFile(PARCEL_ZIP) as archive:
            gpkg_member = next(
                name
                for name in archive.namelist()
                if name.lower().endswith("ahero.gpkg")
                and not name.startswith("__MACOSX/")
            )

            archive.extract(gpkg_member, temp / "parcels")

        gpkg_path = temp / "parcels" / gpkg_member

        gdf = gpd.read_file(gpkg_path, layer="Ahero - Rice")

        gdf["Farm_ID_norm"] = gdf["Parcel Id"].apply(normalize_farm_id)

        print("\nParcel records:", len(gdf))
        print("Unique parcel IDs:", gdf["Farm_ID_norm"].nunique())
        print("Parcel CRS:", gdf.crs)

        # Identify the study farms from processed data
        study_farms = set()

        for campaign in (1, 2, 3):
            path = PROCESSED / f"final_df_rice_campaign_{campaign}.csv"
            df = pd.read_csv(path)

            study_farms.update(df["Farm_ID"].apply(normalize_farm_id))

        print("Unique study farms:", len(study_farms))

        selected = gdf[gdf["Farm_ID_norm"].isin(study_farms)].copy()

        print("Matched parcel polygons:", len(selected))

        if selected["Farm_ID_norm"].nunique() != len(study_farms):
            raise ValueError("Not all study farm IDs have matching polygons")

        if selected["Farm_ID_norm"].duplicated().any():
            raise ValueError("Duplicate farm IDs found in selected polygons")

        all_results = []

        # Extract satellite archive
        with ZipFile(SATELLITE_ZIP) as archive:
            for campaign, config in CAMPAIGNS.items():
                folder_prefix = "satellite_imagery/" + config["folder"] + "/"

                folder = temp / config["folder"]
                folder.mkdir(exist_ok=True)

                for name in archive.namelist():
                    if not name.startswith(folder_prefix):
                        continue

                    if name.startswith("__MACOSX/"):
                        continue

                    if name.endswith("/"):
                        continue

                    if Path(name).suffix.lower() not in {".tif", ".tiff"}:
                        continue

                    target = folder / Path(name).name

                    with archive.open(name) as source:
                        target.write_bytes(source.read())

                masked_raster = find_raster(folder, masked=True)
                original_raster = find_raster(folder, masked=False)

                print("\n" + "-" * 70)
                print("CAMPAIGN", campaign, config["date"])
                print("-" * 70)

                for mode, raster in [
                    ("masked", masked_raster),
                    ("unmasked", original_raster),
                ]:
                    results = inspect_farm_coverage(
                        selected,
                        raster,
                        campaign,
                        config["date"],
                        mode,
                    )

                    all_results.extend(results)

        result_df = pd.DataFrame(all_results)

        result_df.to_csv(OUTPUT / "farm_raster_coverage.csv", index=False)

        # Compare masked and unmasked counts
        pivot = result_df.pivot_table(
            index=["Campaign", "Date", "Farm_ID"],
            columns="Mode",
            values=["Raster_pixels", "Valid_pixels"],
            aggfunc="first",
        )

        pivot.columns = [f"{metric}_{mode}" for metric, mode in pivot.columns]

        pivot = pivot.reset_index()

        pivot.to_csv(OUTPUT / "farm_mask_comparison.csv", index=False)

        # Display masked coverage summary
        masked = result_df[result_df["Mode"] == "masked"]

        print("\n" + "=" * 70)
        print("MASKED COVERAGE SUMMARY")
        print("=" * 70)

        print(masked.groupby(["Campaign", "Coverage_status"]).size().to_string())

        print("\nZero valid-pixel farms:")

        print(
            masked[masked["Valid_pixels"] == 0][
                [
                    "Campaign",
                    "Farm_ID",
                    "Raster_pixels",
                    "Valid_pixels",
                    "Coverage_status",
                ]
            ].to_string(index=False)
        )

        print("\nReports saved:")
        print(OUTPUT / "farm_raster_coverage.csv")
        print(OUTPUT / "farm_mask_comparison.csv")

    print("\nPHASE 10E COMPLETED")


if __name__ == "__main__":
    main()
