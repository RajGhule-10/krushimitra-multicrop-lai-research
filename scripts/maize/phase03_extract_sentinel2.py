"""
Phase 03: Sentinel-2 extraction for Narok maize LAI research.

Input:
    data/maize/processed/maize_sentinel2_candidate_observations.csv

Outputs:
    data/maize/processed/maize_sentinel2_extracted.csv
    data/maize/processed/maize_sentinel2_extraction_failed.csv
    outputs/maize/reports/phase03_sentinel2_extraction_summary.txt

Method:
    - Sentinel-2 Level-2A Harmonized surface reflectance
    - Nearest valid image within +/- 7 days
    - SCL clear-pixel filtering
    - GPS point extraction at 20 m
    - Reflectance scaling and vegetation indices
"""

from pathlib import Path
import sys
import time

import ee
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ID = "riparian-eye"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "maize"
    / "processed"
    / "maize_sentinel2_candidate_observations.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "maize" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "maize" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "maize_sentinel2_extracted.csv"
FAILED_FILE = OUTPUT_DIR / "maize_sentinel2_extraction_failed.csv"
REPORT_FILE = REPORT_DIR / "phase03_sentinel2_extraction_summary.txt"

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

DATE_WINDOW_DAYS = 7
SAMPLE_SCALE = 20

# Sentinel-2 reflectance bands.
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

# Only accept clearly classified vegetation or bare-soil pixels.
# SCL 4 = vegetation
# SCL 5 = bare soil
VALID_SCL_CLASSES = [4, 5]


# ============================================================
# 2. COLUMN DETECTION
# ============================================================


def find_column(df, possible_names, description):

    lookup = {str(column).strip().lower(): column for column in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def clean_property(value):

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ============================================================
# 3. VEGETATION INDEX CALCULATION
# ============================================================


def add_vegetation_indices(image):

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

    return image.addBands(
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


# ============================================================
# 4. PREPARE SATELLITE IMAGE
# ============================================================


def prepare_image(image):

    scl = image.select("SCL")

    # Retain only SCL classes 4 and 5.
    clear_mask = scl.eq(4).Or(scl.eq(5))

    # Apply cloud/classification mask.
    image = image.updateMask(clear_mask)

    # Convert Sentinel-2 integer reflectance to surface
    # reflectance values using the 0.0001 scale factor.
    reflectance = image.select(REFLECTANCE_BANDS).multiply(0.0001)

    image = image.addBands(reflectance, overwrite=True)

    image = add_vegetation_indices(image)

    return image


# ============================================================
# 5. SELECT NEAREST CLEAR IMAGE FOR EACH OBSERVATION
# ============================================================


def extract_observation(feature):

    feature = ee.Feature(feature)
    point = feature.geometry()

    target_date = ee.Date(feature.get("observation_date"))

    start_date = target_date.advance(-DATE_WINDOW_DAYS, "day")

    end_date = target_date.advance(DATE_WINDOW_DAYS + 1, "day")

    candidates = (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .select(REFLECTANCE_BANDS + ["SCL"])
    )

    # Evaluate SCL at the exact field GPS point.
    def check_point_quality(image):

        image = ee.Image(image)

        scl_sample = image.select("SCL").sample(
            region=point, scale=20, numPixels=1, geometries=False
        )

        has_scl = scl_sample.size().gt(0)

        scl_value = ee.Number(
            ee.Algorithms.If(has_scl, ee.Feature(scl_sample.first()).get("SCL"), -1)
        )

        is_valid = scl_value.eq(4).Or(scl_value.eq(5))

        date_difference = image.date().difference(target_date, "day").abs()

        return image.set(
            {
                "valid_point_pixel": ee.Number(is_valid),
                "date_difference_days": date_difference,
                "point_scl_class": scl_value,
            }
        )

    valid_images = (
        candidates.map(check_point_quality)
        .filter(ee.Filter.eq("valid_point_pixel", 1))
        .sort("date_difference_days")
    )

    def extract_from_image():

        selected_image = ee.Image(valid_images.first())

        selected_image = prepare_image(selected_image)

        sample = selected_image.sample(
            region=point, scale=SAMPLE_SCALE, numPixels=1, geometries=False
        )

        def successful_sample():

            sampled_values = ee.Feature(sample.first()).toDictionary()

            return feature.set(sampled_values).set(
                {
                    "s2_status": "SUCCESS",
                    "s2_image_date": selected_image.date().format("YYYY-MM-dd"),
                    "s2_day_difference": (
                        selected_image.date().difference(target_date, "day")
                    ),
                    "s2_absolute_day_difference": (
                        selected_image.date().difference(target_date, "day").abs()
                    ),
                    "s2_cloud_percentage": (
                        selected_image.get("CLOUDY_PIXEL_PERCENTAGE")
                    ),
                    "s2_point_scl_class": (selected_image.get("point_scl_class")),
                    "s2_image_id": selected_image.get("system:index"),
                }
            )

        def unsuccessful_sample():

            return feature.set(
                {
                    "s2_status": "NO_VALID_SAMPLE",
                    "s2_image_date": None,
                    "s2_day_difference": None,
                    "s2_absolute_day_difference": None,
                    "s2_cloud_percentage": None,
                    "s2_point_scl_class": None,
                    "s2_image_id": None,
                }
            )

        return ee.Feature(
            ee.Algorithms.If(
                sample.size().gt(0), successful_sample(), unsuccessful_sample()
            )
        )

    def no_clear_image():

        return feature.set(
            {
                "s2_status": "NO_CLEAR_IMAGE",
                "s2_image_date": None,
                "s2_day_difference": None,
                "s2_absolute_day_difference": None,
                "s2_cloud_percentage": None,
                "s2_point_scl_class": None,
                "s2_image_id": None,
            }
        )

    return ee.Feature(
        ee.Algorithms.If(
            valid_images.size().gt(0), extract_from_image(), no_clear_image()
        )
    )


# ============================================================
# 6. MAIN EXECUTION
# ============================================================


def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PHASE 03: NAROK MAIZE SENTINEL-2 EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput observations: {len(df)}")
    print(f"Input columns: {list(df.columns)}")

    if len(df) == 0:
        raise ValueError("Input dataset is empty.")

    # Identify required columns.
    farm_col = find_column(df, ["Farm_ID", "FarmID", "farm_id"], "Farm ID")

    lai_col = find_column(df, ["LAI", "lai", "LAI_Value"], "LAI")

    lat_col = find_column(
        df, ["GpsLat", "GPS_Lat", "Latitude", "latitude", "Lat"], "GPS latitude"
    )

    lon_col = find_column(
        df,
        ["GpsLong", "GPS_Long", "Longitude", "longitude", "Lon", "Long"],
        "GPS longitude",
    )

    date_col = find_column(
        df,
        [
            "Date",
            "Measurement_Date",
            "Observation_Date",
            "LAI_Date",
            "date",
        ],
        "measurement date",
    )

    campaign_col = find_column(df, ["Campaign", "Campaign_ID", "campaign"], "Campaign")

    # --------------------------------------------------------
    # Parse and validate observations
    # --------------------------------------------------------

    df["_parsed_date"] = pd.to_datetime(
        df[date_col], format="%Y/%m/%d", errors="coerce"
    )

    # Fallback for other date representations.
    missing_dates = df["_parsed_date"].isna()

    if missing_dates.any():
        df.loc[missing_dates, "_parsed_date"] = pd.to_datetime(
            df.loc[missing_dates, date_col], errors="coerce"
        )

    df["_latitude"] = pd.to_numeric(df[lat_col], errors="coerce")

    df["_longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    df["_lai"] = pd.to_numeric(df[lai_col], errors="coerce")

    valid_mask = (
        df["_parsed_date"].notna()
        & df["_latitude"].between(-90, 90)
        & df["_longitude"].between(-180, 180)
        & df["_lai"].notna()
        & (df["_latitude"] != 0)
        & (df["_longitude"] != 0)
    )

    invalid_df = df.loc[~valid_mask].copy()
    df = df.loc[valid_mask].copy()

    if len(df) == 0:
        raise ValueError("No valid observations remain after input validation.")

    df = df.reset_index(drop=True)

    print(f"Valid observations: {len(df)}")
    print(f"Invalid observations: {len(invalid_df)}")

    print("\nCampaign distribution:")
    print(df[campaign_col].value_counts().sort_index())

    # --------------------------------------------------------
    # Earth Engine initialization
    # --------------------------------------------------------

    print("\nConnecting to Google Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    print("Earth Engine connection verified.")

    # --------------------------------------------------------
    # Create Earth Engine feature collection
    # --------------------------------------------------------

    features = []

    for index, row in df.iterrows():
        latitude = float(row["_latitude"])
        longitude = float(row["_longitude"])

        observation_date = row["_parsed_date"].strftime("%Y-%m-%d")

        properties = {}

        # Preserve the original observation attributes,
        # including the spatial quality flags from Phase 02.
        for column, value in row.items():
            if str(column).startswith("_"):
                continue

            cleaned = clean_property(value)

            if cleaned is not None:
                properties[str(column)] = cleaned

        properties.update(
            {
                "observation_id": int(index + 1),
                "observation_date": observation_date,
                "original_lai": float(row["_lai"]),
                "original_farm_id": str(row[farm_col]),
                "campaign_id": str(row[campaign_col]),
            }
        )

        point = ee.Geometry.Point([longitude, latitude])

        features.append(ee.Feature(point, properties))

    print(f"\nCreated {len(features)} Earth Engine point features.")

    feature_collection = ee.FeatureCollection(features)

    # --------------------------------------------------------
    # Run satellite extraction
    # --------------------------------------------------------

    print("\nSearching for Sentinel-2 images...")
    print(f"Date matching window: +/- {DATE_WINDOW_DAYS} days")
    print(f"Point sampling scale: {SAMPLE_SCALE} metres")
    print("SCL accepted classes: vegetation (4), bare soil (5)")

    extracted_collection = feature_collection.map(extract_observation)

    print("\nRetrieving extracted satellite values...")

    result = extracted_collection.getInfo()

    output_rows = [feature["properties"] for feature in result["features"]]

    output_df = pd.DataFrame(output_rows)

    # --------------------------------------------------------
    # Separate successful and failed observations
    # --------------------------------------------------------

    successful_df = output_df[output_df["s2_status"] == "SUCCESS"].copy()

    failed_df = output_df[output_df["s2_status"] != "SUCCESS"].copy()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    successful_df.to_csv(OUTPUT_FILE, index=False)

    failed_df.to_csv(FAILED_FILE, index=False)

    # Also preserve observations invalid before EE processing.
    if not invalid_df.empty:
        invalid_df.to_csv(
            FAILED_FILE.with_name("maize_sentinel2_invalid_input.csv"), index=False
        )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    total = len(df)
    success_count = len(successful_df)
    failed_count = len(failed_df)

    if success_count > 0:
        mean_day_difference = (
            successful_df["s2_absolute_day_difference"].astype(float).mean()
        )

        median_day_difference = (
            successful_df["s2_absolute_day_difference"].astype(float).median()
        )

        campaign_success = successful_df.groupby("campaign_id").size().to_dict()

        date_difference_summary = (
            successful_df["s2_absolute_day_difference"].astype(float).describe()
        )

    else:
        mean_day_difference = None
        median_day_difference = None
        campaign_success = {}
        date_difference_summary = None

    status_counts = output_df["s2_status"].value_counts().to_dict()

    elapsed = time.time() - start_time

    report = []

    report.append("=" * 65)
    report.append("PHASE 03: MAIZE SENTINEL-2 EXTRACTION REPORT")
    report.append("=" * 65)

    report.append(f"\nInput file: {INPUT_FILE}")
    report.append(f"Satellite collection: {COLLECTION_ID}")
    report.append(f"Google Cloud project: {PROJECT_ID}")

    report.append("\nDATASET SUMMARY")
    report.append("-" * 45)
    report.append(f"Valid input observations: {total}")
    report.append(f"Invalid input observations: {len(invalid_df)}")
    report.append(f"Successful satellite matches: {success_count}")
    report.append(f"Failed satellite matches: {failed_count}")

    report.append(f"Satellite match rate: {success_count / total * 100:.2f}%")

    report.append("\nEXTRACTION SETTINGS")
    report.append("-" * 45)
    report.append(f"Matching window: +/- {DATE_WINDOW_DAYS} days")
    report.append(f"Sampling scale: {SAMPLE_SCALE} metres")
    report.append(f"Accepted SCL classes: {VALID_SCL_CLASSES}")
    report.append("Image selection: nearest valid point pixel")

    report.append("\nEXTRACTION STATUS")
    report.append("-" * 45)

    for status, count in status_counts.items():
        report.append(f"{status}: {count}")

    report.append("\nCAMPAIGN SUCCESS COUNTS")
    report.append("-" * 45)

    for campaign, count in sorted(campaign_success.items()):
        report.append(f"{campaign}: {count}")

    report.append("\nDATE MATCHING QUALITY")
    report.append("-" * 45)

    report.append(f"Mean absolute date difference: {mean_day_difference}")

    report.append(f"Median absolute date difference: {median_day_difference}")

    if date_difference_summary is not None:
        report.append(str(date_difference_summary))

    report.append("\nOUTPUT FILES")
    report.append("-" * 45)
    report.append(str(OUTPUT_FILE))
    report.append(str(FAILED_FILE))

    report.append(f"\nExecution time: {elapsed:.2f} seconds")
    report.append("\nEND OF REPORT")

    report_text = "\n".join(report)

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)

    print("\n" + "=" * 65)
    print("PHASE 03 COMPLETED")
    print("=" * 65)

    print(f"\nSuccessful output: {OUTPUT_FILE}")
    print(f"Failed output: {FAILED_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nPHASE 03 FAILED")
        print("-" * 45)
        print(type(error).__name__, ":", error)

        raise
