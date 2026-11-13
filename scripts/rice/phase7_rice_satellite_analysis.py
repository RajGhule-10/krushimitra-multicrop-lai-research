from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# KRUSHIMITRA — PHASE 7
# RICE SATELLITE TEMPORAL AND SPATIAL ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "rice" / "raw"
OUTPUT = ROOT / "outputs" / "rice"
PLOTS = OUTPUT / "plots"
REPORTS = OUTPUT / "reports"

INPUT_FILE = RAW / "ranchi_rice_2019_pixel_features.csv"

PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Sentinel-2 spectral bands available in the CSV.
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Indices available in the CSV.
INDICES = ["NDVI", "EVI", "SAVI", "GNDVI", "NDRE", "NDMI", "MNDWI"]

# Columns required for the analysis.
REQUIRED_COLUMNS = (
    BANDS
    + INDICES
    + [
        "acquisition_date",
        "image_id",
        "latitude",
        "longitude",
        "scene_cloud_percentage",
    ]
)


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Pixel dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Parse acquisition dates.
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce")

    # Convert numerical fields safely.
    numeric_columns = (
        BANDS
        + INDICES
        + [
            "latitude",
            "longitude",
            "scene_cloud_percentage",
        ]
    )

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve original pixel rows.
    df["pixel_record_id"] = np.arange(1, len(df) + 1)

    if df["acquisition_date"].isna().any():
        raise ValueError(
            "Invalid acquisition dates found. Inspect the source CSV before continuing."
        )

    if df["latitude"].isna().any() or df["longitude"].isna().any():
        raise ValueError(
            "Missing pixel coordinates detected. "
            "Inspect the source CSV before continuing."
        )

    if df.empty:
        raise ValueError("Pixel dataset is empty.")

    return df


# ============================================================
# 2. PIXEL DATA QUALITY ASSESSMENT
# ============================================================


def create_quality_report(df):

    rows = []

    # Overall quality.
    rows.append(
        {
            "acquisition_date": "ALL_DATES",
            "pixel_count": len(df),
            "missing_band_values": int(df[BANDS].isna().sum().sum()),
            "missing_index_values": int(df[INDICES].isna().sum().sum()),
            "missing_coordinates": int(
                df[["latitude", "longitude"]].isna().sum().sum()
            ),
            "duplicate_coordinate_date_records": int(
                df.duplicated(
                    subset=[
                        "acquisition_date",
                        "latitude",
                        "longitude",
                    ],
                    keep="first",
                ).sum()
            ),
            "unique_images": df["image_id"].nunique(),
            "mean_scene_cloud_percentage": (df["scene_cloud_percentage"].mean()),
        }
    )

    # Date-wise quality.
    for date, group in df.groupby("acquisition_date"):
        rows.append(
            {
                "acquisition_date": date.strftime("%Y-%m-%d"),
                "pixel_count": len(group),
                "missing_band_values": int(group[BANDS].isna().sum().sum()),
                "missing_index_values": int(group[INDICES].isna().sum().sum()),
                "missing_coordinates": int(
                    group[["latitude", "longitude"]].isna().sum().sum()
                ),
                "duplicate_coordinate_date_records": int(
                    group.duplicated(
                        subset=["latitude", "longitude"],
                        keep="first",
                    ).sum()
                ),
                "unique_images": group["image_id"].nunique(),
                "mean_scene_cloud_percentage": (group["scene_cloud_percentage"].mean()),
            }
        )

    quality = pd.DataFrame(rows)

    # Reflectance sanity check.
    # Sentinel-2 reflectance values in this dataset are scaled
    # to reflectance units using the GEE export.
    band_values = df[BANDS]

    reflectance_outside_range = (band_values < 0) | (band_values > 1.5)

    reflectance_report = pd.DataFrame(
        [
            {
                "assessment": "Total band values",
                "value": int(band_values.size),
            },
            {
                "assessment": "Missing band values",
                "value": int(band_values.isna().sum().sum()),
            },
            {
                "assessment": "Negative reflectance values",
                "value": int((band_values < 0).sum().sum()),
            },
            {
                "assessment": "Reflectance values above 1.5",
                "value": int(reflectance_outside_range.sum().sum()),
            },
        ]
    )

    return quality, reflectance_report


# ============================================================
# 3. DATE-WISE INDEX STATISTICS
# ============================================================


def calculate_temporal_statistics(df):

    rows = []

    for date, group in df.groupby("acquisition_date"):
        for index in INDICES:
            values = group[index].dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "acquisition_date": date,
                    "index": index,
                    "pixel_count": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "minimum": values.min(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "maximum": values.max(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 4. SPATIAL VARIABILITY STATISTICS
# ============================================================


def calculate_spatial_statistics(df):

    rows = []

    for date, group in df.groupby("acquisition_date"):
        for index in INDICES:
            values = group[index].dropna()

            if values.empty:
                continue

            mean_value = values.mean()
            std_value = values.std()

            cv = (
                std_value / abs(mean_value)
                if mean_value != 0 and not np.isnan(std_value)
                else np.nan
            )

            rows.append(
                {
                    "acquisition_date": date,
                    "index": index,
                    "pixel_count": len(values),
                    "mean": mean_value,
                    "std": std_value,
                    "variance": values.var(),
                    "range": values.max() - values.min(),
                    "iqr": (values.quantile(0.75) - values.quantile(0.25)),
                    "coefficient_of_variation": cv,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 5. IQR-BASED PIXEL OUTLIER FLAGS
# ============================================================


def detect_outliers(df):

    flagged = df.copy()

    summary_rows = []

    # Calculate outliers separately for each acquisition date.
    # This avoids directly comparing different dates' distributions.
    for index in INDICES:
        flag_column = f"{index}_outlier_flag"

        flagged[flag_column] = False

        for date, group in flagged.groupby("acquisition_date"):
            values = group[index].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (flagged["acquisition_date"] == date) & (
                (flagged[index] < lower_bound) | (flagged[index] > upper_bound)
            )

            flagged.loc[mask, flag_column] = True

            summary_rows.append(
                {
                    "acquisition_date": date,
                    "index": index,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": int(mask.sum()),
                    "pixel_count": len(values),
                }
            )

    flag_columns = [f"{index}_outlier_flag" for index in INDICES]

    flagged["any_index_outlier"] = flagged[flag_columns].any(axis=1)

    summary = pd.DataFrame(summary_rows)

    return flagged, summary


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================


def calculate_correlations(df):

    # Pooled correlation is descriptive and may be influenced
    # by differences between acquisition dates.
    pooled = df[INDICES].corr(method="pearson")

    pooled.to_csv(REPORTS / "index_correlation_matrix.csv")

    # Also calculate correlation separately for each date.
    date_matrices = []

    for date, group in df.groupby("acquisition_date"):
        matrix = group[INDICES].corr(method="pearson")

        matrix.to_csv(
            REPORTS / ("index_correlation_" + date.strftime("%Y%m%d") + ".csv")
        )

        for index_a in INDICES:
            for index_b in INDICES:
                date_matrices.append(
                    {
                        "acquisition_date": date,
                        "index_1": index_a,
                        "index_2": index_b,
                        "pearson_correlation": matrix.loc[index_a, index_b],
                    }
                )

    return pooled, pd.DataFrame(date_matrices)


# ============================================================
# 7. PLOT — INDEX DISTRIBUTIONS
# ============================================================


def plot_index_distributions(df):

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))

    axes = axes.flatten()

    for i, index in enumerate(INDICES):
        dates = sorted(df["acquisition_date"].unique())

        data = [
            df.loc[df["acquisition_date"] == date, index].dropna().values
            for date in dates
        ]

        axes[i].boxplot(
            data,
            tick_labels=[pd.Timestamp(date).strftime("%d %b") for date in dates],
            showfliers=True,
        )

        axes[i].set_title(f"{index} Distribution")
        axes[i].set_xlabel("Acquisition date")
        axes[i].set_ylabel(index)
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].grid(axis="y", alpha=0.25)

    axes[-1].axis("off")

    fig.suptitle("Ranchi Rice Site — Pixel-Level Index Distributions", fontsize=15)

    fig.tight_layout()

    fig.savefig(PLOTS / "pixel_index_distributions.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 8. PLOT — TEMPORAL INDEX TRENDS
# ============================================================


def plot_temporal_trends(temporal):

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))

    axes = axes.flatten()

    for i, index in enumerate(INDICES):
        subset = temporal[temporal["index"] == index].sort_values("acquisition_date")

        axes[i].plot(
            subset["acquisition_date"], subset["mean"], marker="o", label="Mean"
        )

        axes[i].plot(
            subset["acquisition_date"],
            subset["median"],
            marker="s",
            linestyle="--",
            label="Median",
        )

        axes[i].fill_between(
            subset["acquisition_date"],
            subset["q25"],
            subset["q75"],
            alpha=0.2,
            label="IQR",
        )

        axes[i].set_title(f"{index} Temporal Trend")
        axes[i].set_ylabel(index)
        axes[i].tick_params(axis="x", rotation=35)
        axes[i].grid(alpha=0.25)
        axes[i].legend(fontsize=8)

    axes[-1].axis("off")

    fig.suptitle("Temporal Changes in Rice-Site Spectral Indices", fontsize=15)

    fig.tight_layout()

    fig.savefig(PLOTS / "temporal_index_trends.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 9. PLOT — SPATIAL NDVI AND EVI DISTRIBUTION
# ============================================================


def plot_spatial_distribution(df, index):

    dates = sorted(df["acquisition_date"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    axes = axes.flatten()

    scatter = None

    for i, date in enumerate(dates):
        group = df[df["acquisition_date"] == date]

        scatter = axes[i].scatter(
            group["longitude"],
            group["latitude"],
            c=group[index],
            cmap="RdYlGn",
            s=35,
            edgecolors="black",
            linewidths=0.15,
        )

        axes[i].set_title(f"{index} — {date.strftime('%d %b %Y')}")

        axes[i].set_xlabel("Longitude")
        axes[i].set_ylabel("Latitude")
        axes[i].grid(alpha=0.2)

        axes[i].set_aspect("equal", adjustable="datalim")

    fig.colorbar(scatter, ax=axes.tolist(), label=index, shrink=0.8)

    fig.suptitle(
        f"Spatial Distribution of {index} Across Sentinel-2 Dates", fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS / f"spatial_{index.lower()}_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 10. PLOT — INDEX CORRELATION HEATMAP
# ============================================================


def plot_correlation_heatmap(correlation):

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(correlation.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(INDICES)))

    ax.set_yticks(range(len(INDICES)))

    ax.set_xticklabels(INDICES, rotation=45, ha="right")

    ax.set_yticklabels(INDICES)

    for i in range(len(INDICES)):
        for j in range(len(INDICES)):
            value = correlation.iloc[i, j]

            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Pearson correlation")

    ax.set_title("Pooled Pixel-Level Index Correlation")

    fig.tight_layout()

    fig.savefig(PLOTS / "index_correlation_heatmap.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 11. PLOT — OUTLIER COUNTS
# ============================================================


def plot_outliers(outlier_summary):

    totals = (
        outlier_summary.groupby("index")["outlier_count"]
        .sum()
        .reindex(INDICES, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(totals.index, totals.values)

    ax.set_title("IQR-Flagged Pixel Values by Spectral Index")

    ax.set_xlabel("Spectral index")
    ax.set_ylabel("Flagged index observations")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(PLOTS / "spectral_outlier_analysis.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 12. GENERATE RESEARCH REPORT
# ============================================================


def generate_report(
    df, quality, reflectance, temporal, spatial, outlier_summary, correlation
):

    report = []

    dates = sorted(df["acquisition_date"].unique())

    report.extend(
        [
            "KRUSHIMITRA — PHASE 7",
            "RICE SATELLITE TEMPORAL AND SPATIAL ANALYSIS",
            "=" * 65,
            "",
            "1. DATASET OVERVIEW",
            "",
            f"Total pixel records: {len(df)}",
            f"Unique acquisition dates: {len(dates)}",
            f"Unique satellite images: {df['image_id'].nunique()}",
            f"Spectral bands: {len(BANDS)}",
            f"Spectral indices: {len(INDICES)}",
            "",
            "Acquisition dates:",
        ]
    )

    for date in dates:
        count = (df["acquisition_date"] == date).sum()

        report.append(f"  {date.strftime('%Y-%m-%d')}: {count} pixel records")

    report.extend(
        [
            "",
            "2. DATA QUALITY",
            "",
        ]
    )

    for _, row in quality.iterrows():
        report.append(
            f"{row['acquisition_date']}: "
            f"{row['pixel_count']} pixels, "
            f"{row['missing_band_values']} missing band values, "
            f"{row['missing_index_values']} missing index values, "
            f"{row['duplicate_coordinate_date_records']} "
            "duplicate coordinate/date records."
        )

    report.extend(
        [
            "",
            "Reflectance checks:",
        ]
    )

    for _, row in reflectance.iterrows():
        report.append(f"  {row['assessment']}: {row['value']}")

    report.extend(
        [
            "",
            "3. TEMPORAL INDEX SUMMARY",
            "",
        ]
    )

    for index in INDICES:
        subset = temporal[temporal["index"] == index].sort_values("acquisition_date")

        if subset.empty:
            continue

        report.append(f"{index}:")

        for _, row in subset.iterrows():
            report.append(
                f"  {row['acquisition_date'].strftime('%Y-%m-%d')}: "
                f"mean={row['mean']:.4f}, "
                f"median={row['median']:.4f}, "
                f"std={row['std']:.4f}"
            )

        report.append("")

    report.extend(
        [
            "4. SPATIAL VARIABILITY",
            "",
            (
                "Spatial variability statistics are calculated "
                "within each acquisition date."
            ),
            (
                "Standard deviation, variance, range, IQR and "
                "coefficient of variation are descriptive statistics."
            ),
            "",
            "5. OUTLIER ANALYSIS",
            "",
            (
                "Outliers are flagged using the 1.5 × IQR rule "
                "separately for each index and acquisition date."
            ),
            (
                "Outlier flags do not automatically indicate "
                "measurement errors or invalid pixels."
            ),
            ("No observations have been deleted because of outlier status."),
            "",
        ]
    )

    for index in INDICES:
        subset = outlier_summary[outlier_summary["index"] == index]

        total = int(subset["outlier_count"].sum())

        report.append(f"{index}: {total} flagged index observations")

    report.extend(
        [
            "",
            "6. INDEX CORRELATION",
            "",
            (
                "Pooled Pearson correlations describe relationships "
                "among pixel-level indices across all acquisition dates."
            ),
            (
                "These correlations may be influenced by date-related "
                "differences and should not be interpreted as "
                "independent causal relationships."
            ),
            "",
        ]
    )

    for index_a in INDICES:
        for index_b in INDICES:
            if index_a >= index_b:
                continue

            value = correlation.loc[index_a, index_b]

            report.append(f"{index_a} vs {index_b}: {value:.4f}")

    report.extend(
        [
            "",
            "7. SCIENTIFIC LIMITATIONS",
            "",
            (
                "1. The study area is represented by a temporary "
                "100-metre buffer around a site-level coordinate."
            ),
            ("2. Experimental plot boundaries are unavailable."),
            (
                "3. Pixel samples cannot be assumed to represent "
                "individual experimental rice plots."
            ),
            ("4. Four acquisition dates provide limited temporal coverage."),
            (
                "5. Scene cloud percentage is scene-level metadata "
                "and does not represent cloud contamination "
                "for each sampled pixel."
            ),
            ("6. Spectral indices are not direct measurements of leaf area index."),
            "",
            "8. FINAL INTERPRETATION",
            "",
            (
                "The dataset supports exploratory analysis of "
                "site-level Sentinel-2 spectral behaviour and "
                "spatial variability across four acquisition dates."
            ),
            (
                "The analysis does not establish plot-level rice "
                "monitoring accuracy or satellite-based LAI prediction."
            ),
            "",
            "=" * 65,
            "END OF PHASE 7 REPORT",
        ]
    )

    output_file = REPORTS / "phase7_rice_satellite_report.txt"

    output_file.write_text("\n".join(report), encoding="utf-8")


# ============================================================
# 13. MAIN EXECUTION
# ============================================================


def main():

    print("\n" + "=" * 55)
    print("KRUSHIMITRA — PHASE 7")
    print("RICE SATELLITE TEMPORAL AND SPATIAL ANALYSIS")
    print("=" * 55)

    df = load_data()

    print("\nDataset loaded:", df.shape)

    print("\nAcquisition date distribution:")
    print(df.groupby("acquisition_date").size().to_string())

    print("\nCalculating data quality...")

    quality, reflectance = create_quality_report(df)

    print("\nCalculating temporal statistics...")

    temporal = calculate_temporal_statistics(df)

    print("\nCalculating spatial variability...")

    spatial = calculate_spatial_statistics(df)

    print("\nDetecting spectral outliers...")

    flagged, outlier_summary = detect_outliers(df)

    print("\nCalculating index correlations...")

    correlation, date_correlations = calculate_correlations(df)

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    quality.to_csv(REPORTS / "pixel_quality_report.csv", index=False)

    reflectance.to_csv(REPORTS / "reflectance_quality_report.csv", index=False)

    temporal.to_csv(REPORTS / "temporal_index_statistics.csv", index=False)

    spatial.to_csv(REPORTS / "spatial_variability_report.csv", index=False)

    outlier_summary.to_csv(REPORTS / "spectral_outlier_summary.csv", index=False)

    date_correlations.to_csv(REPORTS / "date_wise_index_correlations.csv", index=False)

    flagged.to_csv(REPORTS / "pixel_index_outlier_flags.csv", index=False)

    # --------------------------------------------------------
    # GENERATE PLOTS
    # --------------------------------------------------------

    print("\nGenerating plots...")

    plot_index_distributions(df)

    plot_temporal_trends(temporal)

    plot_spatial_distribution(df, "NDVI")

    plot_spatial_distribution(df, "EVI")

    plot_correlation_heatmap(correlation)

    plot_outliers(outlier_summary)

    # --------------------------------------------------------
    # GENERATE FINAL REPORT
    # --------------------------------------------------------

    generate_report(
        df, quality, reflectance, temporal, spatial, outlier_summary, correlation
    )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\nPixel quality summary:")
    print(quality.to_string(index=False))

    print("\nTemporal index statistics:")
    print(temporal[["acquisition_date", "index", "mean", "std"]].to_string(index=False))

    print("\nOutlier summary:")
    print(
        outlier_summary[["acquisition_date", "index", "outlier_count"]].to_string(
            index=False
        )
    )

    print("\n" + "=" * 55)
    print("PHASE 7 COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("\nPlots saved in:", PLOTS)
    print("Reports saved in:", REPORTS)


if __name__ == "__main__":
    main()
