"""
Phase 03B: Diagnose failed Sentinel-2 matches for Narok maize.

This script investigates the 258 observations that failed Phase 03.

It does NOT:
- Modify Phase 03 outputs.
- Change the original cloud mask.
- Change the temporal matching criteria.
- Train any machine-learning models.

It diagnoses:
1. Whether Sentinel-2 images exist around each field date.
2. Whether images contain a valid SCL pixel at the GPS location.
3. Which SCL classes occur at each location.
4. Whether a valid image exists within a strict +/- 7-day window.
5. The nearest available acquisition date and its cloud percentage.

Outputs:
    data/maize/processed/maize_sentinel2_failure_diagnostics.csv
    outputs/maize/reports/phase03b_campaign_diagnostics.csv
    outputs/maize/reports/phase03b_sentinel2_diagnostic_report.txt
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

FAILED_FILE = (
    BASE_DIR / "data" / "maize" / "processed" / "maize_sentinel2_extraction_failed.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "maize" / "processed"

REPORT_DIR = BASE_DIR / "outputs" / "maize" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "maize_sentinel2_failure_diagnostics.csv"

CAMPAIGN_REPORT = REPORT_DIR / "phase03b_campaign_diagnostics.csv"

REPORT_FILE = REPORT_DIR / "phase03b_sentinel2_diagnostic_report.txt"

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

DATE_WINDOW_DAYS = 7

SAMPLE_SCALE = 20

# SCL classes:
# 0 = No data
# 1 = Saturated/defective
# 2 = Dark area pixels
# 3 = Cloud shadows
# 4 = Vegetation
# 5 = Bare soil
# 6 = Water
# 7 = Unclassified
# 8 = Cloud medium probability
# 9 = Cloud high probability
# 10 = Thin cirrus
# 11 = Snow/ice

VALID_SCL_CLASSES = [4, 5]

# Number of observations retrieved per Earth Engine request.
BATCH_SIZE = 25


# ============================================================
# 2. COLUMN DETECTION
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
# 3. SCL HISTOGRAM EXTRACTION
# ============================================================


def diagnose_observation(feature):

    feature = ee.Feature(feature)

    point = feature.geometry()

    target_date = ee.Date(feature.get("observation_date"))

    # Calendar search window.
    # filterDate's end date is exclusive.
    search_start = target_date.advance(-DATE_WINDOW_DAYS, "day")

    search_end = target_date.advance(DATE_WINDOW_DAYS + 1, "day")

    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(point)
        .filterDate(search_start, search_end)
        .select(["SCL"])
    )

    total_images = collection.size()

    # --------------------------------------------------------
    # Sample SCL at the field GPS point for every image.
    # --------------------------------------------------------

    def sample_image(image):

        image = ee.Image(image)

        samples = image.select("SCL").sample(
            region=point, scale=SAMPLE_SCALE, numPixels=1, geometries=False
        )

        has_sample = samples.size().gt(0)

        scl_value = ee.Number(
            ee.Algorithms.If(has_sample, ee.Feature(samples.first()).get("SCL"), -1)
        )

        acquisition_date = image.date()

        absolute_difference = acquisition_date.difference(target_date, "day").abs()

        signed_difference = acquisition_date.difference(target_date, "day")

        return ee.Feature(
            None,
            {
                "scl_class": scl_value,
                "has_scl_sample": ee.Number(has_sample),
                "image_date": acquisition_date.format("YYYY-MM-dd"),
                "absolute_day_difference": (absolute_difference),
                "signed_day_difference": (signed_difference),
                "cloud_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                "image_id": image.get("system:index"),
            },
        )

    sampled_images = ee.FeatureCollection(collection.map(sample_image))

    # --------------------------------------------------------
    # Restrict to actual acquisitions within +/- 7 days.
    # --------------------------------------------------------

    strict_window = sampled_images.filter(
        ee.Filter.lte("absolute_day_difference", DATE_WINDOW_DAYS)
    )

    # Only pixels classified as vegetation or bare soil.
    clear_images = strict_window.filter(
        ee.Filter.inList("scl_class", VALID_SCL_CLASSES)
    )

    # --------------------------------------------------------
    # Classify image availability.
    # --------------------------------------------------------

    nearest_image = ee.Feature(sampled_images.sort("absolute_day_difference").first())

    nearest_strict_image = ee.Feature(
        strict_window.sort("absolute_day_difference").first()
    )

    nearest_clear_image = ee.Feature(
        clear_images.sort("absolute_day_difference").first()
    )

    # Histogram of sampled SCL classes.
    scl_histogram = sampled_images.aggregate_histogram("scl_class")

    strict_histogram = strict_window.aggregate_histogram("scl_class")

    # Number of images where an SCL pixel was available.
    sampled_count = sampled_images.filter(ee.Filter.eq("has_scl_sample", 1)).size()

    # Number of images with an SCL 4 or 5 pixel.
    clear_count = clear_images.size()

    # --------------------------------------------------------
    # Diagnostic status.
    # --------------------------------------------------------

    diagnostic_status = ee.String(
        ee.Algorithms.If(
            total_images.eq(0),
            "NO_S2_IMAGE_IN_CALENDAR_WINDOW",
            ee.Algorithms.If(
                sampled_count.eq(0),
                "NO_SCL_SAMPLE_AVAILABLE",
                ee.Algorithms.If(
                    strict_window.size().eq(0),
                    "IMAGES_EXIST_BUT_NONE_WITHIN_STRICT_7_DAYS",
                    ee.Algorithms.If(
                        clear_count.eq(0),
                        "NO_VALID_SCL_4_OR_5_PIXEL",
                        "VALID_CLEAR_PIXEL_FOUND",
                    ),
                ),
            ),
        )
    )

    return feature.set(
        {
            "diagnostic_status": diagnostic_status,
            "s2_calendar_image_count": total_images,
            "s2_scl_sampled_image_count": sampled_count,
            "s2_strict_7day_image_count": (strict_window.size()),
            "s2_clear_pixel_image_count": clear_count,
            "s2_scl_histogram_all": scl_histogram,
            "s2_scl_histogram_strict_7day": (strict_histogram),
            "s2_nearest_image_date": ee.Algorithms.If(
                total_images.gt(0), nearest_image.get("image_date"), None
            ),
            "s2_nearest_image_day_difference": ee.Algorithms.If(
                total_images.gt(0), nearest_image.get("absolute_day_difference"), None
            ),
            "s2_nearest_image_cloud_percentage": ee.Algorithms.If(
                total_images.gt(0), nearest_image.get("cloud_percentage"), None
            ),
            "s2_nearest_strict_date": ee.Algorithms.If(
                strict_window.size().gt(0), nearest_strict_image.get("image_date"), None
            ),
            "s2_nearest_strict_day_difference": ee.Algorithms.If(
                strict_window.size().gt(0),
                nearest_strict_image.get("absolute_day_difference"),
                None,
            ),
            "s2_nearest_clear_date": ee.Algorithms.If(
                clear_count.gt(0), nearest_clear_image.get("image_date"), None
            ),
            "s2_nearest_clear_day_difference": ee.Algorithms.If(
                clear_count.gt(0),
                nearest_clear_image.get("absolute_day_difference"),
                None,
            ),
            "s2_nearest_clear_cloud_percentage": ee.Algorithms.If(
                clear_count.gt(0), nearest_clear_image.get("cloud_percentage"), None
            ),
            "s2_nearest_clear_image_id": ee.Algorithms.If(
                clear_count.gt(0), nearest_clear_image.get("image_id"), None
            ),
        }
    )


# ============================================================
# 4. PREPARE INPUT FEATURES
# ============================================================


def create_features(df):

    features = []

    for index, row in df.iterrows():
        latitude = float(row["_latitude"])
        longitude = float(row["_longitude"])

        properties = {}

        # Preserve original observation information.
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

        point = ee.Geometry.Point([longitude, latitude])

        features.append(ee.Feature(point, properties))

    return features


# ============================================================
# 5. MAIN EXECUTION
# ============================================================


def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 03B: SENTINEL-2 FAILURE DIAGNOSTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load failed observations.
    # --------------------------------------------------------

    if not FAILED_FILE.exists():
        raise FileNotFoundError(f"Failed observations file not found: {FAILED_FILE}")

    df = pd.read_csv(FAILED_FILE)

    print(f"\nFailed observations loaded: {len(df)}")

    farm_col = find_column(df, ["Farm_ID", "FarmID", "farm_id"], "Farm ID")

    lat_col = find_column(
        df, ["GpsLat", "GPS_Lat", "Latitude_Clean", "Latitude"], "Latitude"
    )

    lon_col = find_column(
        df, ["GpsLong", "GPS_Long", "Longitude_Clean", "Longitude"], "Longitude"
    )

    date_col = find_column(
        df,
        [
            "Measurement_Date",
            "observation_date",
            "Date",
            "Observation_Date",
        ],
        "Observation date",
    )

    campaign_col = find_column(df, ["Campaign", "campaign_id"], "Campaign")

    # --------------------------------------------------------
    # Validate coordinates and dates.
    # --------------------------------------------------------

    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")

    df["_latitude"] = pd.to_numeric(df[lat_col], errors="coerce")

    df["_longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    df["_campaign"] = df[campaign_col].astype(str)

    df["_farm"] = df[farm_col].astype(str)

    valid = (
        df["_parsed_date"].notna()
        & df["_latitude"].between(-90, 90)
        & df["_longitude"].between(-180, 180)
        & (df["_latitude"] != 0)
        & (df["_longitude"] != 0)
    )

    invalid_df = df.loc[~valid].copy()

    df = df.loc[valid].reset_index(drop=True)

    print(f"Valid diagnostic points: {len(df)}")
    print(f"Invalid diagnostic points: {len(invalid_df)}")

    if len(df) == 0:
        raise ValueError("No valid failed observations to diagnose.")

    print("\nCampaign distribution:")
    print(df["_campaign"].value_counts().sort_index())

    # --------------------------------------------------------
    # Initialize Earth Engine.
    # --------------------------------------------------------

    print("\nConnecting to Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    print("Earth Engine connected.")

    # --------------------------------------------------------
    # Create Earth Engine features.
    # --------------------------------------------------------

    features = create_features(df)

    print(f"\nCreated {len(features)} diagnostic point features.")

    # --------------------------------------------------------
    # Process in batches.
    # --------------------------------------------------------

    all_results = []

    total_batches = (len(features) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(range(0, len(features), BATCH_SIZE), start=1):
        batch = features[start : start + BATCH_SIZE]

        print(
            f"\nProcessing batch {batch_number}/{total_batches} "
            f"({len(batch)} observations)..."
        )

        batch_collection = ee.FeatureCollection(batch)

        diagnosed = batch_collection.map(diagnose_observation)

        result = diagnosed.getInfo()

        batch_rows = [item["properties"] for item in result["features"]]

        all_results.extend(batch_rows)

        print(f"Completed batch {batch_number}: {len(batch_rows)} observations.")

    # --------------------------------------------------------
    # Create diagnostic dataframe.
    # --------------------------------------------------------

    result_df = pd.DataFrame(all_results)

    if len(result_df) != len(df):
        raise RuntimeError(
            "Diagnostic result count does not match input count. "
            f"Expected {len(df)}, got {len(result_df)}."
        )

    result_df.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Campaign summary.
    # --------------------------------------------------------

    campaign_summary = (
        result_df.groupby(["campaign_id", "diagnostic_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    campaign_summary.to_csv(CAMPAIGN_REPORT, index=False)

    # --------------------------------------------------------
    # Status counts.
    # --------------------------------------------------------

    status_counts = result_df["diagnostic_status"].value_counts().to_dict()

    campaign_image_summary = (
        result_df.groupby("campaign_id")
        .agg(
            observations=("diagnostic_id", "count"),
            mean_calendar_images=("s2_calendar_image_count", "mean"),
            median_calendar_images=("s2_calendar_image_count", "median"),
            mean_sampled_images=("s2_scl_sampled_image_count", "mean"),
            mean_strict_images=("s2_strict_7day_image_count", "mean"),
            mean_clear_images=("s2_clear_pixel_image_count", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Generate report.
    # --------------------------------------------------------

    lines = []

    lines.append("=" * 70)
    lines.append("PHASE 03B: SENTINEL-2 FAILURE DIAGNOSTIC REPORT")
    lines.append("=" * 70)

    lines.append(f"\nGoogle Cloud project: {PROJECT_ID}")
    lines.append(f"Collection: {COLLECTION_ID}")
    lines.append(f"Input observations: {len(df)}")
    lines.append(f"Invalid observations: {len(invalid_df)}")

    lines.append("\nDIAGNOSTIC SETTINGS")
    lines.append("-" * 50)

    lines.append(f"Calendar search window: +/- {DATE_WINDOW_DAYS} days")

    lines.append(f"Strict actual acquisition difference: <= {DATE_WINDOW_DAYS} days")

    lines.append(f"Point sampling scale: {SAMPLE_SCALE} metres")

    lines.append(f"Accepted SCL classes: {VALID_SCL_CLASSES}")

    lines.append("\nDIAGNOSTIC STATUS COUNTS")
    lines.append("-" * 50)

    for status, count in status_counts.items():
        percentage = count / len(result_df) * 100

        lines.append(f"{status}: {count} ({percentage:.2f}%)")

    lines.append("\nCAMPAIGN IMAGE AVAILABILITY")
    lines.append("-" * 50)

    lines.append(campaign_image_summary.to_string(index=False))

    lines.append("\nCAMPAIGN DIAGNOSTIC SUMMARY")
    lines.append("-" * 50)

    lines.append(campaign_summary.to_string(index=False))

    lines.append("\nOUTPUT FILES")
    lines.append("-" * 50)

    lines.append(str(OUTPUT_FILE))
    lines.append(str(CAMPAIGN_REPORT))

    elapsed = time.time() - start_time

    lines.append(f"\nExecution time: {elapsed:.2f} seconds")

    lines.append("\nEND OF REPORT")

    report_text = "\n".join(lines)

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)

    print("\n" + "=" * 70)
    print("PHASE 03B COMPLETED")
    print("=" * 70)

    print(f"\nDiagnostic CSV: {OUTPUT_FILE}")
    print(f"Campaign CSV: {CAMPAIGN_REPORT}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nPHASE 03B FAILED")
        print("-" * 50)

        print(type(error).__name__, ":", error)

        raise
