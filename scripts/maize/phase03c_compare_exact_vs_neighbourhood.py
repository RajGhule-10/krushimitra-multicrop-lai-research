"""
Phase 03C: Compare exact-GPS and neighbourhood Sentinel-2 extraction.

Input:
    data/maize/processed/maize_sentinel2_extraction_failed.csv

Methods:
    A. Exact GPS point extraction.
    B. 30 m neighbourhood mean extraction.

Both methods:
    - Search within +/- 15 days.
    - Use COPERNICUS/S2_SR_HARMONIZED.
    - Accept SCL classes 4 (vegetation) and 5 (bare soil).
    - Select the nearest acquisition independently.
    - Enforce an actual date difference <= 15 days.

Outputs:
    data/maize/processed/maize_phase03c_exact_gps.csv
    data/maize/processed/maize_phase03c_neighbourhood.csv
    data/maize/processed/maize_phase03c_comparison.csv
    outputs/maize/reports/phase03c_comparison_summary.txt

Original Phase 03 and Phase 03B files are not modified.
"""

from pathlib import Path
import time

import ee
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ID = "riparian-eye"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR / "data" / "maize" / "processed" / "maize_sentinel2_extraction_failed.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "maize" / "processed"

REPORT_DIR = BASE_DIR / "outputs" / "maize" / "reports"

EXACT_OUTPUT = OUTPUT_DIR / "maize_phase03c_exact_gps.csv"

NEIGHBOUR_OUTPUT = OUTPUT_DIR / "maize_phase03c_neighbourhood.csv"

COMPARISON_OUTPUT = OUTPUT_DIR / "maize_phase03c_comparison.csv"

SUMMARY_OUTPUT = REPORT_DIR / "phase03c_comparison_summary.txt"

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

DATE_WINDOW_DAYS = 15

NEIGHBOUR_RADIUS_METRES = 30

SAMPLE_SCALE = 20

BATCH_SIZE = 20

VALID_SCL_CLASSES = [4, 5]

REFLECTANCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]

INDEX_BANDS = [
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

FEATURE_BANDS = REFLECTANCE_BANDS + INDEX_BANDS


# ============================================================
# 2. INPUT COLUMN DETECTION
# ============================================================


def find_column(df, possible_names, description):

    lookup = {str(column).strip().lower(): column for column in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    raise ValueError(
        f"Cannot identify {description} column.\nAvailable columns: {list(df.columns)}"
    )


# ============================================================
# 3. PREPARE SENTINEL-2 IMAGE
# ============================================================


def prepare_image(image):

    image = ee.Image(image)

    scl = image.select("SCL")

    # Strict cloud and shadow filtering.
    clear_mask = scl.eq(4).Or(scl.eq(5))

    image = image.updateMask(clear_mask)

    # Convert integer surface reflectance to reflectance.
    reflectance = image.select(REFLECTANCE_BANDS).multiply(0.0001)

    image = image.addBands(reflectance, overwrite=True)

    # Calculate vegetation indices.
    blue = image.select("B2")
    green = image.select("B3")
    red = image.select("B4")
    red_edge_1 = image.select("B5")
    nir = image.select("B8")
    nir_narrow = image.select("B8A")
    swir1 = image.select("B11")

    ndvi = image.expression(
        "(nir - red) / (nir + red)", {"nir": nir, "red": red}
    ).rename("NDVI")

    evi = image.expression(
        "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
        {
            "nir": nir,
            "red": red,
            "blue": blue,
        },
    ).rename("EVI")

    gndvi = image.expression(
        "(nir - green) / (nir + green)", {"nir": nir, "green": green}
    ).rename("GNDVI")

    savi = image.expression(
        "1.5 * (nir - red) / (nir + red + 0.5)", {"nir": nir, "red": red}
    ).rename("SAVI")

    ndre = image.expression(
        "(nir - rededge) / (nir + rededge)", {"nir": nir_narrow, "rededge": red_edge_1}
    ).rename("NDRE")

    rvi = image.expression("nir / red", {"nir": nir, "red": red}).rename("RVI")

    dvi = image.expression("nir - red", {"nir": nir, "red": red}).rename("DVI")

    ndwi = image.expression(
        "(green - nir) / (green + nir)", {"green": green, "nir": nir}
    ).rename("NDWI")

    ndmi = image.expression(
        "(nir - swir) / (nir + swir)", {"nir": nir, "swir": swir1}
    ).rename("NDMI")

    bndvi = image.expression(
        "(nir - blue) / (nir + blue)", {"nir": nir, "blue": blue}
    ).rename("BNDVI")

    image = image.addBands(
        [
            ndvi,
            evi,
            gndvi,
            savi,
            ndre,
            rvi,
            dvi,
            ndwi,
            ndmi,
            bndvi,
        ]
    )

    return image.select(FEATURE_BANDS)


# ============================================================
# 4. DIAGNOSE ONE IMAGE AT ONE FIELD LOCATION
# ============================================================


def inspect_image(image, point, buffer_geometry, target_date):

    image = ee.Image(image)

    prepared = prepare_image(image)

    acquisition_date = image.date()

    absolute_difference = acquisition_date.difference(target_date, "day").abs()

    signed_difference = acquisition_date.difference(target_date, "day")

    # --------------------------------------------------------
    # A. Exact GPS sample
    # --------------------------------------------------------

    exact_sample = prepared.sample(
        region=point, scale=SAMPLE_SCALE, numPixels=1, geometries=False, dropNulls=True
    )

    exact_valid = exact_sample.size().gt(0)

    exact_values = ee.Dictionary(
        ee.Algorithms.If(
            exact_valid,
            ee.Feature(exact_sample.first()).toDictionary(),
            ee.Dictionary({}),
        )
    )

    # --------------------------------------------------------
    # B. Neighbourhood samples
    # --------------------------------------------------------

    neighbourhood_samples = prepared.sample(
        region=buffer_geometry, scale=SAMPLE_SCALE, geometries=True, dropNulls=True
    )

    neighbourhood_count = neighbourhood_samples.size()

    # Calculate mean of every band over valid pixels.
    neighbourhood_mean = prepared.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_geometry,
        scale=SAMPLE_SCALE,
        maxPixels=1000,
        tileScale=2,
    )

    # Find the nearest valid pixel to the original GPS.
    def add_distance(feature):

        feature = ee.Feature(feature)

        distance = point.distance(feature.geometry(), 1)

        return feature.set("distance_from_original_gps_m", distance)

    distance_samples = neighbourhood_samples.map(add_distance)

    nearest_pixel = ee.Feature(
        distance_samples.sort("distance_from_original_gps_m").first()
    )

    nearest_pixel_distance = ee.Algorithms.If(
        neighbourhood_count.gt(0),
        nearest_pixel.get("distance_from_original_gps_m"),
        None,
    )

    # --------------------------------------------------------
    # Return image-level diagnostics.
    # --------------------------------------------------------

    return ee.Feature(
        None,
        {
            "image_date": acquisition_date.format("YYYY-MM-dd"),
            "absolute_day_difference": absolute_difference,
            "signed_day_difference": signed_difference,
            "cloud_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "image_id": image.get("system:index"),
            "exact_valid": ee.Number(exact_valid),
            "exact_values": exact_values,
            "neighbourhood_valid_pixel_count": (neighbourhood_count),
            "neighbourhood_values": neighbourhood_mean,
            "nearest_pixel_offset_m": nearest_pixel_distance,
        },
    )


# ============================================================
# 5. PROCESS ONE FAILED FIELD OBSERVATION
# ============================================================


def extract_observation(feature):

    feature = ee.Feature(feature)

    point = feature.geometry()

    buffer_geometry = point.buffer(NEIGHBOUR_RADIUS_METRES)

    target_date = ee.Date(feature.get("observation_date"))

    search_start = target_date.advance(-DATE_WINDOW_DAYS, "day")

    search_end = target_date.advance(DATE_WINDOW_DAYS + 1, "day")

    # Get imagery intersecting the neighbourhood.
    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(buffer_geometry)
        .filterDate(search_start, search_end)
        .select(REFLECTANCE_BANDS + ["SCL"])
    )

    # Inspect each available image.
    diagnostics = ee.FeatureCollection(
        collection.map(
            lambda image: inspect_image(image, point, buffer_geometry, target_date)
        )
    )

    # Enforce strict actual acquisition difference.
    strict_diagnostics = diagnostics.filter(
        ee.Filter.lte("absolute_day_difference", DATE_WINDOW_DAYS)
    )

    # --------------------------------------------------------
    # Independently select nearest exact-GPS image.
    # --------------------------------------------------------

    exact_candidates = strict_diagnostics.filter(ee.Filter.eq("exact_valid", 1))

    exact_count = exact_candidates.size()

    exact_selected = ee.Feature(
        ee.Algorithms.If(
            exact_count.gt(0),
            exact_candidates.sort("absolute_day_difference").first(),
            ee.Feature(None, {}),
        )
    )

    # --------------------------------------------------------
    # Independently select nearest neighbourhood image.
    # --------------------------------------------------------

    neighbourhood_candidates = strict_diagnostics.filter(
        ee.Filter.gt("neighbourhood_valid_pixel_count", 0)
    )

    neighbourhood_count = neighbourhood_candidates.size()

    neighbourhood_selected = ee.Feature(
        ee.Algorithms.If(
            neighbourhood_count.gt(0),
            neighbourhood_candidates.sort("absolute_day_difference").first(),
            ee.Feature(None, {}),
        )
    )

    # --------------------------------------------------------
    # Extract selected image properties.
    # --------------------------------------------------------

    exact_values = ee.Dictionary(exact_selected.get("exact_values"))

    neighbourhood_values = ee.Dictionary(
        neighbourhood_selected.get("neighbourhood_values")
    )

    # --------------------------------------------------------
    # Prepare output properties.
    # --------------------------------------------------------

    result = feature.set(
        {
            # Exact GPS results.
            "exact_status": ee.Algorithms.If(
                exact_count.gt(0), "SUCCESS", "NO_VALID_EXACT_PIXEL"
            ),
            "exact_image_date": ee.Algorithms.If(
                exact_count.gt(0), exact_selected.get("image_date"), None
            ),
            "exact_day_difference": ee.Algorithms.If(
                exact_count.gt(0), exact_selected.get("signed_day_difference"), None
            ),
            "exact_absolute_day_difference": ee.Algorithms.If(
                exact_count.gt(0), exact_selected.get("absolute_day_difference"), None
            ),
            "exact_cloud_percentage": ee.Algorithms.If(
                exact_count.gt(0), exact_selected.get("cloud_percentage"), None
            ),
            "exact_image_id": ee.Algorithms.If(
                exact_count.gt(0), exact_selected.get("image_id"), None
            ),
            # Neighbourhood results.
            "neighbourhood_status": ee.Algorithms.If(
                neighbourhood_count.gt(0), "SUCCESS", "NO_VALID_NEIGHBOURHOOD_PIXEL"
            ),
            "neighbourhood_image_date": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("image_date"),
                None,
            ),
            "neighbourhood_day_difference": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("signed_day_difference"),
                None,
            ),
            "neighbourhood_absolute_day_difference": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("absolute_day_difference"),
                None,
            ),
            "neighbourhood_cloud_percentage": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("cloud_percentage"),
                None,
            ),
            "neighbourhood_image_id": ee.Algorithms.If(
                neighbourhood_count.gt(0), neighbourhood_selected.get("image_id"), None
            ),
            "neighbourhood_valid_pixel_count": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("neighbourhood_valid_pixel_count"),
                0,
            ),
            "neighbourhood_nearest_pixel_offset_m": ee.Algorithms.If(
                neighbourhood_count.gt(0),
                neighbourhood_selected.get("nearest_pixel_offset_m"),
                None,
            ),
            "exact_neighbourhood_same_image": ee.Algorithms.If(
                exact_count.gt(0),
                ee.Algorithms.If(
                    neighbourhood_count.gt(0),
                    ee.Algorithms.IsEqual(
                        exact_selected.get("image_id"),
                        neighbourhood_selected.get("image_id"),
                    ),
                    False,
                ),
                False,
            ),
            "exact_values": exact_values,
            "neighbourhood_values": neighbourhood_values,
            "calendar_image_count": collection.size(),
            "strict_15day_image_count": (strict_diagnostics.size()),
        }
    )

    return result


# ============================================================
# 6. INPUT PREPARATION
# ============================================================


def prepare_input():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput failed observations: {len(df)}")

    farm_col = find_column(df, ["Farm_ID", "FarmID", "farm_id"], "Farm ID")

    lat_col = find_column(
        df, ["GpsLat", "GPS_Lat", "Latitude_Clean", "Latitude"], "GPS latitude"
    )

    lon_col = find_column(
        df, ["GpsLong", "GPS_Long", "Longitude_Clean", "Longitude"], "GPS longitude"
    )

    date_col = find_column(
        df,
        [
            "Measurement_Date",
            "observation_date",
            "Date",
            "Observation_Date",
        ],
        "measurement date",
    )

    campaign_col = find_column(df, ["Campaign", "campaign_id"], "Campaign")

    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")

    df["_latitude"] = pd.to_numeric(df[lat_col], errors="coerce")

    df["_longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    valid = (
        df["_parsed_date"].notna()
        & df["_latitude"].between(-90, 90)
        & df["_longitude"].between(-180, 180)
        & (df["_latitude"] != 0)
        & (df["_longitude"] != 0)
    )

    invalid = df.loc[~valid].copy()

    df = df.loc[valid].copy().reset_index(drop=True)

    df["_farm"] = df[farm_col].astype(str)

    df["_campaign"] = df[campaign_col].astype(str)

    print(f"Valid observations: {len(df)}")
    print(f"Invalid observations: {len(invalid)}")

    return df, invalid


# ============================================================
# 7. CREATE EARTH ENGINE FEATURES
# ============================================================


def create_features(df):

    features = []

    for index, row in df.iterrows():
        properties = {}

        for column, value in row.items():
            if str(column).startswith("_"):
                continue

            if pd.isna(value):
                continue

            if isinstance(value, pd.Timestamp):
                value = value.strftime("%Y-%m-%d")

            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, (str, int, float, bool)):
                properties[str(column)] = value

        properties.update(
            {
                "diagnostic_id": int(index + 1),
                "observation_date": (row["_parsed_date"].strftime("%Y-%m-%d")),
                "campaign_id": str(row["_campaign"]),
                "farm_id_diagnostic": str(row["_farm"]),
            }
        )

        point = ee.Geometry.Point(
            [
                float(row["_longitude"]),
                float(row["_latitude"]),
            ]
        )

        features.append(ee.Feature(point, properties))

    return features


# ============================================================
# 8. CONVERT EARTH ENGINE RESULTS TO DATAFRAMES
# ============================================================


def flatten_results(result_df):

    exact_rows = []
    neighbourhood_rows = []
    comparison_rows = []

    for _, row in result_df.iterrows():
        exact_values = row.get("exact_values", {})

        neighbourhood_values = row.get("neighbourhood_values", {})

        if not isinstance(exact_values, dict):
            exact_values = {}

        if not isinstance(neighbourhood_values, dict):
            neighbourhood_values = {}

        base = row.drop(
            labels=[
                "exact_values",
                "neighbourhood_values",
            ],
            errors="ignore",
        ).to_dict()

        # ----------------------------------------------------
        # Exact GPS output.
        # ----------------------------------------------------

        exact_row = base.copy()

        exact_row.update({f"exact_{key}": value for key, value in exact_values.items()})

        exact_rows.append(exact_row)

        # ----------------------------------------------------
        # Neighbourhood output.
        # ----------------------------------------------------

        neighbourhood_row = base.copy()

        neighbourhood_row.update(
            {
                f"neighbourhood_{key}": value
                for key, value in neighbourhood_values.items()
            }
        )

        neighbourhood_rows.append(neighbourhood_row)

        # ----------------------------------------------------
        # Comparison output.
        # ----------------------------------------------------

        comparison = base.copy()

        for band in FEATURE_BANDS:
            comparison[f"exact_{band}"] = exact_values.get(band)

            comparison[f"neighbourhood_{band}"] = neighbourhood_values.get(band)

            exact_value = exact_values.get(band)

            neighbourhood_value = neighbourhood_values.get(band)

            if exact_value is not None and neighbourhood_value is not None:
                comparison[f"difference_{band}"] = float(neighbourhood_value) - float(
                    exact_value
                )

            else:
                comparison[f"difference_{band}"] = None

        comparison_rows.append(comparison)

    return (
        pd.DataFrame(exact_rows),
        pd.DataFrame(neighbourhood_rows),
        pd.DataFrame(comparison_rows),
    )


# ============================================================
# 9. GENERATE SUMMARY
# ============================================================


def generate_summary(exact_df, neighbourhood_df, comparison_df, invalid_df, elapsed):

    total = len(comparison_df)

    exact_success = (exact_df["exact_status"] == "SUCCESS").sum()

    neighbourhood_success = (
        neighbourhood_df["neighbourhood_status"] == "SUCCESS"
    ).sum()

    both_success = (
        (comparison_df["exact_status"] == "SUCCESS")
        & (comparison_df["neighbourhood_status"] == "SUCCESS")
    ).sum()

    exact_only = (
        (comparison_df["exact_status"] == "SUCCESS")
        & (comparison_df["neighbourhood_status"] != "SUCCESS")
    ).sum()

    neighbourhood_only = (
        (comparison_df["exact_status"] != "SUCCESS")
        & (comparison_df["neighbourhood_status"] == "SUCCESS")
    ).sum()

    neither = (
        (comparison_df["exact_status"] != "SUCCESS")
        & (comparison_df["neighbourhood_status"] != "SUCCESS")
    ).sum()

    lines = []

    lines.append("=" * 70)
    lines.append("PHASE 03C: EXACT GPS VS NEIGHBOURHOOD COMPARISON")
    lines.append("=" * 70)

    lines.append(f"\nInput observations: {total}")
    lines.append(f"Invalid input observations: {len(invalid_df)}")

    lines.append("\nEXTRACTION SETTINGS")
    lines.append("-" * 50)

    lines.append(f"Date window: +/- {DATE_WINDOW_DAYS} days")
    lines.append(f"Neighbourhood radius: {NEIGHBOUR_RADIUS_METRES} metres")
    lines.append(f"Sampling scale: {SAMPLE_SCALE} metres")
    lines.append(f"Accepted SCL classes: {VALID_SCL_CLASSES}")

    lines.append("\nEXACT GPS RESULTS")
    lines.append("-" * 50)

    lines.append(f"Successful: {exact_success} ({exact_success / total * 100:.2f}%)")

    lines.append(f"Failed: {total - exact_success}")

    lines.append("\nNEIGHBOURHOOD RESULTS")
    lines.append("-" * 50)

    lines.append(
        f"Successful: {neighbourhood_success} "
        f"({neighbourhood_success / total * 100:.2f}%)"
    )

    lines.append(f"Failed: {total - neighbourhood_success}")

    lines.append("\nMETHOD COMPARISON")
    lines.append("-" * 50)

    lines.append(f"Both methods successful: {both_success}")
    lines.append(f"Exact GPS only: {exact_only}")
    lines.append(f"Neighbourhood only: {neighbourhood_only}")
    lines.append(f"Neither method successful: {neither}")

    lines.append("\nCAMPAIGN-WISE RESULTS")
    lines.append("-" * 50)

    campaign_summary = comparison_df.groupby("campaign_id").agg(
        observations=("diagnostic_id", "count"),
        exact_success=("exact_status", lambda values: (values == "SUCCESS").sum()),
        neighbourhood_success=(
            "neighbourhood_status",
            lambda values: (values == "SUCCESS").sum(),
        ),
    )

    lines.append(campaign_summary.to_string())

    lines.append("\nNEIGHBOURHOOD SPATIAL OFFSETS")
    lines.append("-" * 50)

    offsets = pd.to_numeric(
        neighbourhood_df["neighbourhood_nearest_pixel_offset_m"], errors="coerce"
    ).dropna()

    if not offsets.empty:
        lines.append(f"Mean nearest pixel offset: {offsets.mean():.2f} m")

        lines.append(f"Median nearest pixel offset: {offsets.median():.2f} m")

        lines.append(f"Maximum nearest pixel offset: {offsets.max():.2f} m")

    else:
        lines.append("No successful neighbourhood samples.")

    lines.append("\nOUTPUT FILES")
    lines.append("-" * 50)

    lines.append(str(EXACT_OUTPUT))
    lines.append(str(NEIGHBOUR_OUTPUT))
    lines.append(str(COMPARISON_OUTPUT))

    lines.append(f"\nExecution time: {elapsed:.2f} seconds")

    lines.append("\nEND OF REPORT")

    return "\n".join(lines)


# ============================================================
# 10. MAIN
# ============================================================


def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 03C: EXACT GPS VS NEIGHBOURHOOD EXTRACTION")
    print("=" * 70)

    # Load and validate input.
    df, invalid_df = prepare_input()

    if df.empty:
        raise ValueError("No valid observations to process.")

    # Initialize Earth Engine.
    print("\nConnecting to Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    print("Earth Engine connected.")

    # Create point features.
    features = create_features(df)

    print(f"\nCreated {len(features)} field point features.")

    # Batch processing.
    all_results = []

    total_batches = (len(features) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(range(0, len(features), BATCH_SIZE), start=1):
        batch = features[start : start + BATCH_SIZE]

        print(
            f"\nProcessing batch {batch_number}/{total_batches} "
            f"({len(batch)} observations)..."
        )

        collection = ee.FeatureCollection(batch)

        extracted = collection.map(extract_observation)

        result = extracted.getInfo()

        rows = [item["properties"] for item in result["features"]]

        all_results.extend(rows)

        print(f"Completed batch {batch_number}: {len(rows)} observations.")

    result_df = pd.DataFrame(all_results)

    if len(result_df) != len(df):
        raise RuntimeError(f"Expected {len(df)} results, received {len(result_df)}.")

    # Flatten nested feature dictionaries.
    (
        exact_df,
        neighbourhood_df,
        comparison_df,
    ) = flatten_results(result_df)

    # Save separate method outputs.
    exact_df.to_csv(EXACT_OUTPUT, index=False)

    neighbourhood_df.to_csv(NEIGHBOUR_OUTPUT, index=False)

    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)

    # Save invalid input separately if any.
    if not invalid_df.empty:
        invalid_df.to_csv(OUTPUT_DIR / "maize_phase03c_invalid_input.csv", index=False)

    # Generate report.
    elapsed = time.time() - start_time

    report = generate_summary(
        exact_df, neighbourhood_df, comparison_df, invalid_df, elapsed
    )

    SUMMARY_OUTPUT.write_text(report, encoding="utf-8")

    print("\n" + report)

    print("\n" + "=" * 70)
    print("PHASE 03C COMPLETED")
    print("=" * 70)

    print(f"\nExact GPS: {EXACT_OUTPUT}")
    print(f"Neighbourhood: {NEIGHBOUR_OUTPUT}")
    print(f"Comparison: {COMPARISON_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nPHASE 03C FAILED")
        print("-" * 50)
        print(type(error).__name__, ":", error)

        raise
