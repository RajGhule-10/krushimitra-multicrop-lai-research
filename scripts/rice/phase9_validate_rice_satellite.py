"""
KRUSHIMITRA — PHASE 9A
RANCHI RICE SATELLITE DATA VALIDATION AND INTEGRATION

Inputs:
1. Phase 9 pixel-level Sentinel-2 features
2. Phase 9 GEE site-level temporal statistics
3. Phase 9 image quality inventory

Outputs:
- Validated pixel dataset
- Validated site-summary dataset
- Validated image inventory
- Acquisition-level quality report
- Pixel-sample temporal statistics
- Pixel-sample spatial variability
- Spectral outlier flags
- Correlation matrix
- Temporal and spatial plots
- Research validation report

IMPORTANT:
- No observations are automatically deleted.
- Low-coverage dates are retained and flagged.
- GEE reduceRegion statistics and pixel-sample statistics
  are kept separate.
- Pixel samples are site-buffer observations, NOT
  experimental rice plot observations.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "rice" / "raw"
PROCESSED = ROOT / "data" / "rice" / "processed"

REPORTS = ROOT / "outputs" / "rice" / "reports" / "phase9"
PLOTS = ROOT / "outputs" / "rice" / "plots" / "phase9"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

PIXEL_FILE = RAW / "ranchi_rice_phase9_pixel_features_2019.csv"
SITE_FILE = RAW / "ranchi_rice_phase9_site_summary_2019.csv"
INVENTORY_FILE = RAW / "ranchi_rice_phase9_image_inventory_2019.csv"

# Project-specific screening threshold.
# This is not a universal scientific quality threshold.
LOW_COVERAGE_THRESHOLD = 70

INDICES = ["NDVI", "EVI", "SAVI", "GNDVI", "NDRE", "NDMI", "MNDWI"]

SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

FEATURES = SPECTRAL_BANDS + INDICES

print("=" * 65)
print("PHASE 9A — RICE SATELLITE VALIDATION")
print("=" * 65)


# ============================================================
# 2. LOAD INPUT DATA
# ============================================================

for path in [PIXEL_FILE, SITE_FILE, INVENTORY_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

pixels = pd.read_csv(PIXEL_FILE)
site = pd.read_csv(SITE_FILE)
inventory = pd.read_csv(INVENTORY_FILE)

print("\nINPUT DATA SHAPES")
print("Pixel records:", pixels.shape)
print("Site summary:", site.shape)
print("Image inventory:", inventory.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

pixel_required = FEATURES + [
    "acquisition_date",
    "image_id",
    "scene_cloud_percentage",
    "longitude",
    "latitude",
]

site_required = [
    "acquisition_date",
    "image_id",
    "scene_cloud_percentage",
    "valid_pixel_count",
    "study_area",
]

for feature in FEATURES:
    for statistic in ["mean", "median", "stdDev"]:
        site_required.append(f"{feature}_{statistic}")

inventory_required = [
    "acquisition_date",
    "image_id",
    "scene_cloud_percentage",
    "valid_pixel_count",
    "study_area",
]


def check_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    print(f"{name}: required columns verified")


check_columns(pixels, pixel_required, "Pixel dataset")
check_columns(site, site_required, "Site summary")
check_columns(inventory, inventory_required, "Inventory")


# ============================================================
# 4. NORMALIZE DATA TYPES
# ============================================================

for df in [pixels, site, inventory]:
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce")

    df["image_id"] = df["image_id"].astype(str)

for column in FEATURES + ["scene_cloud_percentage", "longitude", "latitude"]:
    pixels[column] = pd.to_numeric(pixels[column], errors="coerce")

for feature in FEATURES:
    for statistic in ["mean", "median", "stdDev"]:
        column = f"{feature}_{statistic}"

        site[column] = pd.to_numeric(site[column], errors="coerce")

for df in [site, inventory]:
    df["valid_pixel_count"] = pd.to_numeric(df["valid_pixel_count"], errors="coerce")

if pixels["acquisition_date"].isna().any():
    raise ValueError("Invalid dates found in pixel dataset")

if site["acquisition_date"].isna().any():
    raise ValueError("Invalid dates found in site summary")

if inventory["acquisition_date"].isna().any():
    raise ValueError("Invalid dates found in inventory")


# ============================================================
# 5. CHECK DUPLICATES AND MISSING VALUES
# ============================================================

duplicate_mask = pixels.duplicated(
    subset=["acquisition_date", "longitude", "latitude"], keep=False
)

duplicate_pixels = pixels[duplicate_mask].copy()

missing_report = pd.DataFrame(
    {
        "column": pixels.columns,
        "missing_count": [pixels[column].isna().sum() for column in pixels.columns],
        "missing_percentage": [
            pixels[column].isna().mean() * 100 for column in pixels.columns
        ],
    }
)

duplicate_pixels.to_csv(REPORTS / "duplicate_pixel_records.csv", index=False)

missing_report.to_csv(REPORTS / "pixel_missing_value_report.csv", index=False)

print("\nDUPLICATE CHECK")
print("Duplicate coordinate-date records:", len(duplicate_pixels))

print("Pixel feature missing values:", pixels[FEATURES].isna().sum().sum())


# ============================================================
# 6. VALIDATE PIXEL COUNTS AGAINST INVENTORY
# ============================================================

pixel_counts = (
    pixels.groupby("acquisition_date").size().rename("pixel_csv_count").reset_index()
)

quality = inventory.merge(pixel_counts, on="acquisition_date", how="left")

quality["pixel_csv_count"] = quality["pixel_csv_count"].fillna(0).astype(int)

quality["valid_pixel_count"] = quality["valid_pixel_count"].fillna(0).astype(int)

quality["count_difference"] = quality["valid_pixel_count"] - quality["pixel_csv_count"]

quality["count_check"] = np.where(quality["count_difference"] == 0, "PASS", "FAIL")

if not (quality["count_check"] == "PASS").all():
    raise ValueError(
        "Pixel counts do not match the image inventory. "
        "Inspect acquisition-level count differences."
    )

print("\nPIXEL COUNT VALIDATION")
print("Inventory total:", quality["valid_pixel_count"].sum())

print("Pixel CSV total:", len(pixels))

print("Count consistency:", (quality["count_check"] == "PASS").all())


# ============================================================
# 7. ASSIGN ACQUISITION QUALITY FLAGS
# ============================================================


def classify_coverage(count):

    if count == 0:
        return "NO_VALID_PIXELS"

    if count < LOW_COVERAGE_THRESHOLD:
        return "LOW_COVERAGE"

    return "ACCEPTABLE_COVERAGE"


quality["coverage_status"] = quality["valid_pixel_count"].apply(classify_coverage)

pixels = pixels.merge(
    quality[["acquisition_date", "coverage_status", "valid_pixel_count"]],
    on="acquisition_date",
    how="left",
    validate="many_to_one",
)

site = site.merge(
    quality[["acquisition_date", "coverage_status"]],
    on="acquisition_date",
    how="left",
    validate="one_to_one",
)

print("\nCOVERAGE CLASSIFICATION")
print(quality["coverage_status"].value_counts())


# ============================================================
# 8. VALIDATE SITE SUMMARY AGAINST INVENTORY
# ============================================================

site_inventory_check = site.merge(
    inventory[["acquisition_date", "valid_pixel_count"]],
    on="acquisition_date",
    how="outer",
    suffixes=("_site", "_inventory"),
    indicator=True,
)

site_inventory_check["count_match"] = (
    site_inventory_check["valid_pixel_count_site"]
    == site_inventory_check["valid_pixel_count_inventory"]
)

if not site_inventory_check["_merge"].eq("both").all():
    raise ValueError("Site summary and inventory acquisition dates differ.")

if not site_inventory_check["count_match"].all():
    raise ValueError("Site summary and inventory pixel counts differ.")

print("Site-summary/inventory consistency: PASS")


# ============================================================
# 9. PIXEL-SAMPLE TEMPORAL STATISTICS
# ============================================================

# These statistics are calculated from exported pixel rows.
# They are NOT the same as GEE reduceRegion statistics.

temporal_statistics = pixels.groupby("acquisition_date")[FEATURES].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

temporal_statistics.columns = [
    f"{feature}_{stat}" for feature, stat in temporal_statistics.columns
]

temporal_statistics = temporal_statistics.reset_index()

temporal_statistics = temporal_statistics.merge(
    quality[
        [
            "acquisition_date",
            "valid_pixel_count",
            "coverage_status",
            "scene_cloud_percentage",
        ]
    ],
    on="acquisition_date",
    how="left",
)

temporal_statistics.to_csv(
    REPORTS / "pixel_sample_temporal_statistics.csv", index=False
)


# ============================================================
# 10. SPATIAL VARIABILITY STATISTICS
# ============================================================

spatial_rows = []

for date, group in pixels.groupby("acquisition_date"):
    for index in INDICES:
        values = group[index].dropna()

        if len(values) == 0:
            continue

        mean_value = values.mean()
        std_value = values.std()

        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)

        spatial_rows.append(
            {
                "acquisition_date": date,
                "index": index,
                "pixel_count": len(values),
                "mean": mean_value,
                "std": std_value,
                "variance": values.var(),
                "minimum": values.min(),
                "maximum": values.max(),
                "range": values.max() - values.min(),
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "coefficient_of_variation": (
                    std_value / abs(mean_value) if mean_value != 0 else np.nan
                ),
            }
        )

spatial_variability = pd.DataFrame(spatial_rows)

spatial_variability = spatial_variability.merge(
    quality[["acquisition_date", "coverage_status"]], on="acquisition_date", how="left"
)

spatial_variability.to_csv(REPORTS / "pixel_spatial_variability.csv", index=False)


# ============================================================
# 11. FLAG SPECTRAL INDEX OUTLIERS
# ============================================================

# IQR-based flags are diagnostic only.
# No pixel is deleted or automatically treated as invalid.

outlier_rows = []

for date, group in pixels.groupby("acquisition_date"):
    for index in INDICES:
        values = group[index]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flags = (values < lower) | (values > upper)

        for row_index in group.index[flags.fillna(False)]:
            outlier_rows.append(
                {
                    "acquisition_date": date,
                    "pixel_row_index": row_index,
                    "index": index,
                    "observed_value": pixels.loc[row_index, index],
                    "lower_iqr_limit": lower,
                    "upper_iqr_limit": upper,
                    "flag": "IQR_OUTLIER",
                }
            )

outlier_report = pd.DataFrame(outlier_rows)

outlier_report.to_csv(REPORTS / "spectral_index_outlier_flags.csv", index=False)


# ============================================================
# 12. INDEX CORRELATION ANALYSIS
# ============================================================

# Pooled correlations can be influenced by date-to-date changes.
# They are descriptive, not causal.

correlation_matrix = pixels[INDICES].corr(method="pearson")

correlation_matrix.to_csv(REPORTS / "pooled_index_correlation_matrix.csv")

date_correlations = []

for date, group in pixels.groupby("acquisition_date"):
    if len(group) < 3:
        continue

    matrix = group[INDICES].corr(method="pearson")

    for index_a in INDICES:
        for index_b in INDICES:
            date_correlations.append(
                {
                    "acquisition_date": date,
                    "index_a": index_a,
                    "index_b": index_b,
                    "pearson_correlation": matrix.loc[index_a, index_b],
                    "pixel_count": len(group),
                }
            )

pd.DataFrame(date_correlations).to_csv(
    REPORTS / "date_wise_index_correlations.csv", index=False
)


# ============================================================
# 13. COMPARE GEE AND PIXEL-SAMPLE MEANS
# ============================================================

# Keep both aggregation methods visible.
# Small differences are possible due to sampling and
# reduction methodology.

pixel_means = pixels.groupby("acquisition_date")[INDICES].mean().reset_index()

pixel_means = pixel_means.rename(
    columns={index: f"{index}_pixel_sample_mean" for index in INDICES}
)

gee_means = site[["acquisition_date"] + [f"{index}_mean" for index in INDICES]].copy()

comparison = gee_means.merge(pixel_means, on="acquisition_date", how="left")

for index in INDICES:
    comparison[f"{index}_mean_difference"] = (
        comparison[f"{index}_mean"] - comparison[f"{index}_pixel_sample_mean"]
    )

comparison.to_csv(REPORTS / "gee_vs_pixel_sample_means.csv", index=False)


# ============================================================
# 14. PLOT TEMPORAL INDEX TRENDS
# ============================================================

for index in ["NDVI", "EVI", "NDRE", "NDMI"]:
    fig, ax = plt.subplots(figsize=(12, 6))

    gee_column = f"{index}_mean"
    pixel_column = f"{index}_pixel_sample_mean"

    ax.plot(
        comparison["acquisition_date"],
        comparison[gee_column],
        marker="o",
        label="GEE reduceRegion mean",
    )

    ax.plot(
        comparison["acquisition_date"],
        comparison[pixel_column],
        marker="s",
        linestyle="--",
        label="Pixel-sample mean",
    )

    low_dates = quality.loc[
        quality["coverage_status"] == "LOW_COVERAGE", "acquisition_date"
    ]

    for date in low_dates:
        ax.axvline(date, linestyle=":", alpha=0.35)

    ax.set_title(f"Ranchi 2019 — {index} Temporal Comparison")

    ax.set_xlabel("Acquisition date")
    ax.set_ylabel(index)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(PLOTS / f"phase9_{index.lower()}_temporal_comparison.png", dpi=300)

    plt.close(fig)


# ============================================================
# 15. PLOT PIXEL DISTRIBUTIONS
# ============================================================

for index in ["NDVI", "EVI", "NDRE"]:
    dates = sorted(pixels["acquisition_date"].unique())

    data = [
        pixels.loc[pixels["acquisition_date"] == date, index].dropna().values
        for date in dates
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.boxplot(
        data,
        tick_labels=[pd.Timestamp(date).strftime("%d %b") for date in dates],
        showfliers=True,
    )

    ax.set_title(f"Ranchi 2019 — Pixel Distribution of {index}")

    ax.set_xlabel("Acquisition date")
    ax.set_ylabel(index)
    ax.grid(axis="y", alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(PLOTS / f"phase9_{index.lower()}_pixel_distribution.png", dpi=300)

    plt.close(fig)


# ============================================================
# 16. PLOT POOLED CORRELATION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(9, 8))

image = ax.imshow(correlation_matrix.values, vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(INDICES)))
ax.set_yticks(range(len(INDICES)))

ax.set_xticklabels(INDICES, rotation=45, ha="right")
ax.set_yticklabels(INDICES)

for i in range(len(INDICES)):
    for j in range(len(INDICES)):
        ax.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

ax.set_title("Pooled Sentinel-2 Index Correlation")

fig.colorbar(image, ax=ax, label="Pearson correlation")
fig.tight_layout()

fig.savefig(PLOTS / "phase9_index_correlation_heatmap.png", dpi=300)

plt.close(fig)


# ============================================================
# 17. EXPORT VALIDATED DATASETS
# ============================================================

pixels.to_csv(PROCESSED / "ranchi_phase9_validated_pixel_dataset.csv", index=False)

site.to_csv(PROCESSED / "ranchi_phase9_validated_site_summary.csv", index=False)

quality.to_csv(PROCESSED / "ranchi_phase9_validated_image_inventory.csv", index=False)


# ============================================================
# 18. GENERATE RESEARCH VALIDATION REPORT
# ============================================================

total_dates = len(quality)

valid_dates = (quality["valid_pixel_count"] > 0).sum()

zero_dates = (quality["valid_pixel_count"] == 0).sum()

low_dates = (quality["coverage_status"] == "LOW_COVERAGE").sum()

acceptable_dates = (quality["coverage_status"] == "ACCEPTABLE_COVERAGE").sum()

monthly_counts = (
    quality.assign(month=quality["acquisition_date"].dt.strftime("%B"))
    .groupby("month")
    .agg(
        candidate_scenes=("image_id", "count"),
        dates_with_pixels=("valid_pixel_count", lambda x: (x > 0).sum()),
    )
)

report = []

report.append("KRUSHIMITRA — PHASE 9A VALIDATION REPORT")
report.append("=" * 65)

report.append(f"Total candidate scenes: {total_dates}")
report.append(f"Dates with valid pixels: {valid_dates}")
report.append(f"Dates with zero valid pixels: {zero_dates}")
report.append(f"Low-coverage dates: {low_dates}")
report.append(f"Acceptable-coverage dates: {acceptable_dates}")

report.append(f"Total exported pixel observations: {len(pixels)}")

report.append(f"Sum of inventory pixel counts: {quality['valid_pixel_count'].sum()}")

report.append(f"Duplicate coordinate-date records: {len(duplicate_pixels)}")

report.append(f"Spectral IQR flags: {len(outlier_report)}")

report.append("")
report.append("MONTHLY COVERAGE")
report.append(monthly_counts.to_string())

report.append("")
report.append("ACQUISITION QUALITY")
report.append(
    quality[
        [
            "acquisition_date",
            "scene_cloud_percentage",
            "valid_pixel_count",
            "coverage_status",
            "count_check",
        ]
    ].to_string(index=False)
)

report.append("")
report.append("SCIENTIFIC LIMITATIONS")
report.append(
    "1. Study area represents a 100 m site buffer, not experimental plot boundaries."
)

report.append(
    "2. Pixel samples cannot be assigned to individual "
    "rice treatments or LAI observations."
)

report.append(
    "3. July and August have no valid pixels under the "
    "current masking and sampling procedure."
)

report.append(
    "4. Low-coverage dates are retained but should be interpreted cautiously."
)

report.append("5. Spectral indices are not direct LAI measurements.")

report.append(
    "6. Pooled correlations are descriptive and may be "
    "influenced by acquisition-date effects."
)

report.append(
    "7. This dataset does not independently validate a rice LAI prediction model."
)

report.append("")
report.append("MONTHLY COVERAGE INTERPRETATION")
report.append(
    "The available pixel observations begin in September. "
    "The exported data therefore do not provide usable "
    "July-August spectral coverage."
)

report.append("")
report.append("FINAL STATUS")
report.append(
    "Phase 9A validation and integration completed. "
    "No observations were automatically deleted."
)

report_path = REPORTS / "phase9_validation_report.txt"

report_path.write_text("\n".join(report), encoding="utf-8")


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PHASE 9A COMPLETED")
print("=" * 65)

print("Candidate scenes:", total_dates)
print("Dates with valid pixels:", valid_dates)
print("Zero-pixel dates:", zero_dates)
print("Low-coverage dates:", low_dates)
print("Acceptable-coverage dates:", acceptable_dates)

print("Pixel records:", len(pixels))
print("Duplicate coordinate-date records:", len(duplicate_pixels))

print("\nProcessed datasets:", PROCESSED)
print("Reports:", REPORTS)
print("Plots:", PLOTS)

print("\nNo observations were automatically deleted.")
print("Rice LAI model training remains on hold.")
