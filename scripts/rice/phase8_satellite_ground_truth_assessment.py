"""
KRUSHIMITRA — PHASE 8
SATELLITE AND GROUND-TRUTH INTEGRATION ASSESSMENT

Purpose:
1. Assess temporal compatibility between Ranchi ground-truth LAI
   observations and Sentinel-2 acquisition dates.
2. Quantify date offsets and matching eligibility.
3. Evaluate the available temporal coverage.
4. Assess whether exploratory satellite-LAI comparisons are justified.
5. Generate auditable CSV reports and visualizations.

Scientific safeguards:
- Satellite pixels are site-buffer samples, not experimental plots.
- No pixel is assigned to a treatment or LAI observation.
- No satellite-to-LAI prediction model is trained.
- Date matching does not establish spatial or experimental compatibility.
- Existing suspicious observations remain identifiable.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "rice" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "rice" / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "rice" / "reports"
PLOT_DIR = PROJECT_ROOT / "outputs" / "rice" / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_FILE = PROCESSED_DIR / "rice_ground_truth_all_records.csv"
CANDIDATES_FILE = PROCESSED_DIR / "rice_ground_truth_model_candidates.csv"
MATCHING_FILE = PROCESSED_DIR / "ranchi_satellite_date_matching.csv"

PIXEL_FILE = RAW_DIR / "ranchi_rice_2019_pixel_features.csv"
SITE_SUMMARY_FILE = RAW_DIR / "ranchi_rice_2019_site_summary.csv"

MAX_MATCH_OFFSET_DAYS = 10

SITE_NAME = "Ranchi"

SATELLITE_INDEX_COLUMNS = [
    "NDVI",
    "EVI",
    "SAVI",
    "GNDVI",
    "NDRE",
    "NDMI",
    "MNDWI",
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def load_csv(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")

    df = pd.read_csv(path)

    print(f"{description}: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def normalize_name(value):
    """Normalize column names for flexible matching."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, aliases, description, required=True):
    """
    Find a column using known aliases.

    Matching is case-insensitive and ignores spaces/hyphens.
    """

    normalized = {normalize_name(column): column for column in df.columns}

    for alias in aliases:
        key = normalize_name(alias)

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not identify {description}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected one of: {aliases}"
        )

    return None


def parse_dates(series):
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def save_report(df, filename):
    path = REPORT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path.name}")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# 3. LOAD INPUT DATASETS
# ============================================================


def load_inputs():

    print_section("PHASE 8 — LOADING DATASETS")

    ground_truth = load_csv(GROUND_TRUTH_FILE, "Ground-truth all-records dataset")

    candidates = load_csv(CANDIDATES_FILE, "Ground-truth model-candidates dataset")

    matching = load_csv(MATCHING_FILE, "Phase 4 satellite date-matching dataset")

    pixels = load_csv(PIXEL_FILE, "Ranchi Sentinel-2 pixel dataset")

    site_summary = load_csv(SITE_SUMMARY_FILE, "Ranchi Sentinel-2 site summary")

    return (
        ground_truth,
        candidates,
        matching,
        pixels,
        site_summary,
    )


# ============================================================
# 4. IDENTIFY RANCHI GROUND-TRUTH RECORDS
# ============================================================


def prepare_ground_truth(ground_truth, candidates):

    print_section("PREPARING RANCHI GROUND TRUTH")

    location_col = find_column(
        ground_truth,
        ["location", "site", "study_location"],
        "ground-truth location column",
    )

    lai_col = find_column(
        ground_truth, ["lai", "LAI", "lai_value"], "ground-truth LAI column"
    )

    date_col = find_column(
        ground_truth,
        [
            "observation_date",
            "date",
            "measurement_date",
            "sampling_date",
            "lai_date",
        ],
        "ground-truth observation date column",
    )

    source_id_col = find_column(
        ground_truth,
        [
            "source_row_id",
            "record_id",
            "observation_id",
            "id",
        ],
        "ground-truth source row identifier",
        required=False,
    )

    df = ground_truth.copy()

    df["_location_normalized"] = df[location_col].astype(str).str.strip().str.lower()

    ranchi = df[df["_location_normalized"].str.contains("ranchi", na=False)].copy()

    if ranchi.empty:
        raise ValueError("No Ranchi records were found in the ground-truth dataset.")

    ranchi["observation_date_parsed"] = parse_dates(ranchi[date_col])

    ranchi["lai_numeric"] = safe_numeric(ranchi[lai_col])

    if source_id_col:
        ranchi["source_record_id"] = ranchi[source_id_col].astype(str)
    else:
        ranchi["source_record_id"] = ranchi.index.astype(str)

    ranchi["date_available"] = ranchi["observation_date_parsed"].notna()

    ranchi["lai_available"] = ranchi["lai_numeric"].notna()

    # Identify candidate membership using stable source IDs
    candidate_id_col = find_column(
        candidates,
        [
            "source_row_id",
            "record_id",
            "observation_id",
            "id",
        ],
        "candidate source row identifier",
        required=False,
    )

    candidate_lai_col = find_column(
        candidates, ["lai", "LAI", "lai_value"], "candidate LAI column", required=False
    )

    if source_id_col and candidate_id_col:
        candidate_ids = set(candidates[candidate_id_col].astype(str))

        ranchi["is_model_candidate"] = ranchi["source_record_id"].isin(candidate_ids)

    elif candidate_lai_col:
        # Do not match candidates using LAI values alone.
        # Source identity is necessary to establish membership.
        ranchi["is_model_candidate"] = np.nan

        warnings.warn(
            "Candidate membership could not be linked reliably "
            "because a common source identifier was unavailable."
        )

    else:
        ranchi["is_model_candidate"] = np.nan

    print(f"Ranchi ground-truth records: {len(ranchi)}")
    print("Records with valid dates:", ranchi["date_available"].sum())
    print("Records with numeric LAI:", ranchi["lai_available"].sum())

    if ranchi["is_model_candidate"].notna().any():
        print("Ranchi candidate records:", int(ranchi["is_model_candidate"].sum()))

    return ranchi


# ============================================================
# 5. EXTRACT SATELLITE ACQUISITION DATES
# ============================================================


def prepare_satellite_data(pixels, site_summary):

    print_section("PREPARING SATELLITE ACQUISITIONS")

    date_col = find_column(
        pixels, ["acquisition_date", "date"], "satellite acquisition date"
    )

    pixel_dates = parse_dates(pixels[date_col])

    if pixel_dates.isna().any():
        raise ValueError("Satellite pixel dataset contains invalid acquisition dates.")

    pixels = pixels.copy()
    pixels["acquisition_date_parsed"] = pixel_dates

    acquisition_summary = (
        pixels.groupby("acquisition_date_parsed")
        .agg(
            pixel_count=("acquisition_date_parsed", "size"),
            unique_coordinates=(
                find_column(pixels, ["latitude", "lat"], "pixel latitude"),
                "count",
            ),
            unique_images=(
                find_column(
                    pixels,
                    ["image_id", "system:index"],
                    "satellite image identifier",
                    required=False,
                )
                or date_col,
                "nunique",
            ),
        )
        .reset_index()
        .rename(columns={"acquisition_date_parsed": "satellite_date"})
    )

    # Get scene cloud percentage if available.
    cloud_col = find_column(
        pixels,
        [
            "scene_cloud_percentage",
            "cloud_percentage",
            "cloud_cover",
        ],
        "scene cloud percentage",
        required=False,
    )

    if cloud_col:
        cloud_summary = (
            pixels.groupby("acquisition_date_parsed")[cloud_col]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "acquisition_date_parsed": "satellite_date",
                    cloud_col: "mean_scene_cloud_percentage",
                }
            )
        )

        acquisition_summary = acquisition_summary.merge(
            cloud_summary, on="satellite_date", how="left"
        )

    print("Satellite acquisition dates:", len(acquisition_summary))

    print(acquisition_summary.to_string(index=False))

    # Summarize available site-level index data.
    site_date_col = find_column(
        site_summary, ["acquisition_date", "date"], "site-summary acquisition date"
    )

    site_summary = site_summary.copy()

    site_summary["satellite_date"] = parse_dates(site_summary[site_date_col])

    return (
        pixels,
        acquisition_summary,
        site_summary,
    )


# ============================================================
# 6. TEMPORAL MATCHING ASSESSMENT
# ============================================================


def calculate_temporal_matches(ranchi, acquisition_summary):

    print_section("TEMPORAL MATCHING ASSESSMENT")

    satellite_dates = sorted(acquisition_summary["satellite_date"].dropna().unique())

    if not satellite_dates:
        raise ValueError("No valid satellite dates available.")

    records = []

    for _, row in ranchi.iterrows():
        observation_date = row["observation_date_parsed"]

        result = {
            "source_record_id": row["source_record_id"],
            "observation_date": observation_date,
            "lai": row["lai_numeric"],
            "date_available": pd.notna(observation_date),
            "lai_available": pd.notna(row["lai_numeric"]),
            "is_model_candidate": row["is_model_candidate"],
        }

        if pd.isna(observation_date):
            result.update(
                {
                    "nearest_satellite_date": pd.NaT,
                    "date_offset_days": np.nan,
                    "absolute_date_offset_days": np.nan,
                    "temporal_match_status": "missing_ground_truth_date",
                    "within_10_days": False,
                }
            )

        else:
            differences = [
                (sat_date - observation_date).days for sat_date in satellite_dates
            ]

            nearest_index = int(np.argmin(np.abs(differences)))

            nearest_date = satellite_dates[nearest_index]
            signed_offset = differences[nearest_index]
            absolute_offset = abs(signed_offset)

            status = (
                "within_10_days"
                if absolute_offset <= MAX_MATCH_OFFSET_DAYS
                else "no_scene_within_10_days"
            )

            result.update(
                {
                    "nearest_satellite_date": nearest_date,
                    "date_offset_days": signed_offset,
                    "absolute_date_offset_days": absolute_offset,
                    "temporal_match_status": status,
                    "within_10_days": (absolute_offset <= MAX_MATCH_OFFSET_DAYS),
                }
            )

        records.append(result)

    matches = pd.DataFrame(records)

    matches["nearest_satellite_date"] = pd.to_datetime(
        matches["nearest_satellite_date"], errors="coerce"
    )

    matches["observation_date"] = pd.to_datetime(
        matches["observation_date"], errors="coerce"
    )

    return matches


# ============================================================
# 7. COMPARE WITH PHASE 4 MATCHING RESULTS
# ============================================================


def assess_phase4_consistency(matches, phase4_matching):

    print_section("PHASE 4 CONSISTENCY CHECK")

    if phase4_matching.empty:
        print("Phase 4 matching dataset is empty.")
        return pd.DataFrame(
            [
                {
                    "check": "phase4_matching_available",
                    "result": False,
                    "details": "No Phase 4 matching records.",
                }
            ]
        )

    phase4_date_col = find_column(
        phase4_matching,
        [
            "observation_date",
            "ground_truth_date",
            "date",
            "lai_date",
        ],
        "Phase 4 ground-truth observation date",
        required=False,
    )

    phase4_status_col = find_column(
        phase4_matching,
        [
            "match_status",
            "matching_status",
            "satellite_match_status",
            "status",
        ],
        "Phase 4 match status",
        required=False,
    )

    checks = []

    checks.append(
        {
            "check": "phase4_matching_available",
            "result": True,
            "details": (f"{len(phase4_matching)} Phase 4 records loaded."),
        }
    )

    if phase4_date_col:
        phase4_dates = parse_dates(phase4_matching[phase4_date_col])

        checks.append(
            {
                "check": "phase4_dates_parseable",
                "result": bool(phase4_dates.notna().all()),
                "details": (
                    f"{phase4_dates.notna().sum()} valid dates "
                    f"out of {len(phase4_dates)}."
                ),
            }
        )

    if phase4_status_col:
        status_counts = (
            phase4_matching[phase4_status_col].astype(str).value_counts().to_dict()
        )

        checks.append(
            {
                "check": "phase4_status_available",
                "result": True,
                "details": str(status_counts),
            }
        )

    else:
        checks.append(
            {
                "check": "phase4_status_available",
                "result": False,
                "details": (
                    "Could not identify a Phase 4 status column. "
                    "Phase 8 independently calculated date offsets."
                ),
            }
        )

    return pd.DataFrame(checks)


# ============================================================
# 8. DATE-LEVEL SUMMARY
# ============================================================


def create_date_level_summary(matches):

    valid = matches[matches["observation_date"].notna()].copy()

    summary = (
        valid.groupby("observation_date")
        .agg(
            ground_truth_record_count=("source_record_id", "count"),
            valid_lai_count=("lai", "count"),
            mean_lai=("lai", "mean"),
            median_lai=("lai", "median"),
            nearest_satellite_date=("nearest_satellite_date", "first"),
            mean_absolute_offset_days=("absolute_date_offset_days", "mean"),
            matched_records=("within_10_days", "sum"),
        )
        .reset_index()
    )

    summary["ground_truth_date"] = summary["observation_date"]

    summary["match_percentage"] = (
        summary["matched_records"] / summary["ground_truth_record_count"] * 100
    )

    return summary


# ============================================================
# 9. SATELLITE INDEX DATE SUMMARY
# ============================================================


def create_satellite_index_summary(site_summary):

    print_section("SATELLITE INDEX SUMMARY")

    available_indices = [
        column
        for column in SATELLITE_INDEX_COLUMNS
        if f"{column}_mean" in site_summary.columns
    ]

    if not available_indices:
        raise ValueError(
            "No recognized *_mean satellite index columns found. "
            f"Available columns: {list(site_summary.columns)}"
        )

    result = pd.DataFrame()

    date_col = find_column(
        site_summary, ["acquisition_date", "date"], "site-summary acquisition date"
    )

    result["satellite_date"] = parse_dates(site_summary[date_col])

    for index_name in available_indices:
        mean_col = f"{index_name}_mean"
        median_col = f"{index_name}_median"
        std_col = f"{index_name}_stdDev"

        result[index_name] = safe_numeric(site_summary[mean_col])

        if median_col in site_summary.columns:
            result[f"{index_name}_median"] = safe_numeric(site_summary[median_col])

        if std_col in site_summary.columns:
            result[f"{index_name}_stdDev"] = safe_numeric(site_summary[std_col])

    result = result.sort_values("satellite_date").reset_index(drop=True)

    print("Recognized satellite indices:", available_indices)
    print("\nSite-level index statistics:")
    print(result.to_string(index=False))

    return result


# ============================================================
# 10. RESEARCH ELIGIBILITY ASSESSMENT
# ============================================================


def create_eligibility_report(
    ranchi, matches, acquisition_summary, satellite_index_summary
):

    print_section("RESEARCH ELIGIBILITY ASSESSMENT")

    total_records = len(matches)

    # All Ranchi records
    all_matched = int(matches["within_10_days"].sum())
    all_unmatched = total_records - all_matched

    # Candidate records only
    candidates = matches[matches["is_model_candidate"] == True].copy()

    candidate_total = len(candidates)

    candidate_matched = int(candidates["within_10_days"].sum())

    candidate_unmatched = candidate_total - candidate_matched

    # Excluded observations
    excluded = matches[matches["is_model_candidate"] == False].copy()

    excluded_total = len(excluded)

    all_match_percentage = all_matched / total_records * 100 if total_records else 0

    candidate_match_percentage = (
        candidate_matched / candidate_total * 100 if candidate_total else 0
    )

    dated_records = int(matches["date_available"].sum())

    valid_lai_records = int(matches["lai_available"].sum())

    eligible_dates = candidates[candidates["within_10_days"]][
        "observation_date"
    ].nunique()

    unique_ground_truth_dates = matches["observation_date"].nunique()

    satellite_dates = len(acquisition_summary)

    assessment = [
        {
            "assessment": "All Ranchi ground-truth records",
            "result": f"{total_records}",
            "interpretation": (
                "Includes all records, including the suspicious "
                "observation excluded from model candidates."
            ),
            "eligibility": "available",
        },
        {
            "assessment": "Model candidate records",
            "result": f"{candidate_total}",
            "interpretation": ("Records retained by Phase 4 for potential modelling."),
            "eligibility": "available",
        },
        {
            "assessment": "Excluded Ranchi observations",
            "result": f"{excluded_total}",
            "interpretation": ("Records excluded from model candidates by Phase 4."),
            "eligibility": "excluded_from_candidates",
        },
        {
            "assessment": "Valid observation dates",
            "result": f"{dated_records}/{total_records}",
            "interpretation": ("Ground-truth records with valid observation dates."),
            "eligibility": "available",
        },
        {
            "assessment": "Numeric LAI observations",
            "result": f"{valid_lai_records}/{total_records}",
            "interpretation": ("Ground-truth records containing numeric LAI."),
            "eligibility": "available",
        },
        {
            "assessment": "Satellite acquisition coverage",
            "result": f"{satellite_dates} dates",
            "interpretation": ("Four available Sentinel-2 acquisitions."),
            "eligibility": "limited",
        },
        {
            "assessment": "All-record temporal matching",
            "result": (f"{all_matched}/{total_records} ({all_match_percentage:.2f}%)"),
            "interpretation": (
                f"{all_matched} of all Ranchi records have a "
                f"nearest acquisition within {MAX_MATCH_OFFSET_DAYS} days."
            ),
            "eligibility": "temporal_only",
        },
        {
            "assessment": "All-record unmatched observations",
            "result": f"{all_unmatched}",
            "interpretation": (
                "All Ranchi records without a satellite acquisition "
                "within the matching threshold."
            ),
            "eligibility": "limited",
        },
        {
            "assessment": "Candidate temporal matching",
            "result": (
                f"{candidate_matched}/{candidate_total} "
                f"({candidate_match_percentage:.2f}%)"
            ),
            "interpretation": (
                "Matching result for Phase 4 retained model candidates."
            ),
            "eligibility": "temporal_only",
        },
        {
            "assessment": "Candidate unmatched observations",
            "result": f"{candidate_unmatched}",
            "interpretation": (
                "Retained candidate observations without a match "
                "within the matching threshold."
            ),
            "eligibility": "limited",
        },
        {
            "assessment": "Distinct candidate matched dates",
            "result": f"{eligible_dates}",
            "interpretation": (
                "Unique ground-truth dates represented among "
                "temporally matched candidates."
            ),
            "eligibility": "temporal_only",
        },
        {
            "assessment": "Distinct ground-truth dates",
            "result": f"{unique_ground_truth_dates}",
            "interpretation": ("Unique observation dates in all Ranchi records."),
            "eligibility": "available",
        },
        {
            "assessment": "Plot-level spatial correspondence",
            "result": "Not established",
            "interpretation": (
                "Satellite pixels represent a temporary site buffer, "
                "not georeferenced experimental plots."
            ),
            "eligibility": "not_eligible",
        },
        {
            "assessment": "Treatment-level satellite correspondence",
            "result": "Not established",
            "interpretation": (
                "No verified mapping between satellite pixels "
                "and individual experimental treatments."
            ),
            "eligibility": "not_eligible",
        },
        {
            "assessment": "Supervised satellite-to-LAI modelling",
            "result": "Not justified",
            "interpretation": (
                "Temporal matches alone do not establish spatially "
                "compatible training observations or independent validation."
            ),
            "eligibility": "not_eligible",
        },
    ]

    report = pd.DataFrame(assessment)

    print(report.to_string(index=False))

    # Explicit Phase 4 regression check
    print("\nPHASE 4 CANDIDATE MATCHING REGRESSION CHECK")

    print(f"Candidate records: {candidate_total}")
    print(f"Candidate matched: {candidate_matched}")
    print(f"Candidate unmatched: {candidate_unmatched}")

    if candidate_total == 40 and candidate_matched == 24 and candidate_unmatched == 16:
        print("PASS: Phase 8 reproduces Phase 4 matching counts.")
    else:
        print(
            "WARNING: Candidate matching counts differ from "
            "the previously established Phase 4 results."
        )

    return report


# ============================================================
# 11. VISUALIZATIONS
# ============================================================


def plot_temporal_matching(matches):

    print_section("GENERATING TEMPORAL MATCHING PLOTS")

    data = matches[matches["observation_date"].notna()].copy()

    if data.empty:
        print("No valid dates available for temporal plots.")
        return

    # Plot 1: Observation dates and nearest satellite dates
    fig, ax = plt.subplots(figsize=(12, 6))

    for index, row in data.reset_index(drop=True).iterrows():
        if pd.isna(row["nearest_satellite_date"]):
            continue

        ax.plot(
            [row["observation_date"], row["nearest_satellite_date"]],
            [index, index],
            marker="o",
            alpha=0.65,
        )

    ax.set_title("Ranchi Ground-Truth Dates and Nearest Sentinel-2 Dates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ground-truth record")
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "phase8_temporal_matching_timeline.png", dpi=300, bbox_inches="tight"
    )

    plt.close(fig)

    # Plot 2: Absolute date offsets
    fig, ax = plt.subplots(figsize=(10, 5))

    offset_data = data[data["absolute_date_offset_days"].notna()]

    ax.hist(
        offset_data["absolute_date_offset_days"],
        bins=np.arange(
            0, max(12, offset_data["absolute_date_offset_days"].max() + 2), 1
        ),
        edgecolor="black",
    )

    ax.axvline(
        MAX_MATCH_OFFSET_DAYS,
        linestyle="--",
        label=f"{MAX_MATCH_OFFSET_DAYS}-day threshold",
    )

    ax.set_title("Distribution of Absolute Ground-Truth / Satellite Date Offsets")
    ax.set_xlabel("Absolute date difference (days)")
    ax.set_ylabel("Number of ground-truth records")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "phase8_date_offset_distribution.png", dpi=300, bbox_inches="tight"
    )

    plt.close(fig)

    # Plot 3: Ground-truth LAI by date
    fig, ax = plt.subplots(figsize=(11, 5))

    valid_lai = data.dropna(subset=["lai", "observation_date"])

    if not valid_lai.empty:
        grouped = (
            valid_lai.groupby("observation_date")["lai"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        grouped["std"] = grouped["std"].fillna(0)

        ax.errorbar(
            grouped["observation_date"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=4,
        )

        ax.set_title("Ranchi Ground-Truth LAI by Observation Date")
        ax.set_xlabel("Ground-truth observation date")
        ax.set_ylabel("LAI (mean ± standard deviation)")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR / "phase8_ground_truth_lai_by_date.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_satellite_indices(index_summary):

    if index_summary.empty:
        return

    indices = [col for col in SATELLITE_INDEX_COLUMNS if col in index_summary.columns]

    fig, axes = plt.subplots(
        len(indices), 1, figsize=(11, max(3, len(indices) * 2.6)), squeeze=False
    )

    for i, column in enumerate(indices):
        ax = axes[i, 0]

        ax.plot(index_summary["satellite_date"], index_summary[column], marker="o")

        ax.set_title(f"{column} — Site-Level Mean")
        ax.set_ylabel(column)
        ax.grid(alpha=0.3)

        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Ranchi Site-Level Sentinel-2 Vegetation Index Trends", fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(
        PLOT_DIR / "phase8_satellite_index_temporal_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. GENERATE FINAL RESEARCH REPORT
# ============================================================


def generate_text_report(
    ranchi, matches, acquisition_summary, eligibility_report, consistency_report
):

    path = REPORT_DIR / "phase8_satellite_ground_truth_report.txt"

    total = len(matches)

    matched = int(matches["within_10_days"].sum())

    unmatched = total - matched

    valid_dates = int(matches["observation_date"].notna().sum())

    valid_lai = int(matches["lai"].notna().sum())

    with open(path, "w", encoding="utf-8") as file:
        file.write("=" * 70 + "\n")
        file.write("KRUSHIMITRA — PHASE 8\n")
        file.write("SATELLITE AND GROUND-TRUTH INTEGRATION ASSESSMENT\n")
        file.write("=" * 70 + "\n\n")

        file.write("1. DATASET OVERVIEW\n")
        file.write("-" * 70 + "\n")

        file.write(f"Ranchi ground-truth records: {total}\n")

        file.write(f"Records with valid observation dates: {valid_dates}\n")

        file.write(f"Records with numeric LAI: {valid_lai}\n")

        file.write(f"Sentinel-2 acquisition dates: {len(acquisition_summary)}\n\n")

        file.write("2. TEMPORAL MATCHING\n")
        file.write("-" * 70 + "\n")

        file.write(f"Matching threshold: {MAX_MATCH_OFFSET_DAYS} days\n")

        file.write(f"Records matched within threshold: {matched}\n")

        file.write(f"Records without a match within threshold: {unmatched}\n")

        file.write(
            f"Matching percentage: {matched / total * 100 if total else 0:.2f}%\n\n"
        )

        file.write("3. PHASE 4 CONSISTENCY CHECKS\n")
        file.write("-" * 70 + "\n")

        file.write(consistency_report.to_string(index=False))

        file.write("\n\n4. RESEARCH ELIGIBILITY\n")
        file.write("-" * 70 + "\n")

        file.write(eligibility_report.to_string(index=False))

        file.write("\n\n5. SCIENTIFIC LIMITATIONS\n")
        file.write("-" * 70 + "\n")

        limitations = [
            "Satellite observations are sampled from a temporary 100 m site buffer.",
            "Experimental plot boundaries and plot-level coordinates "
            "are not available.",
            "A temporal date match does not establish spatial correspondence.",
            "Satellite vegetation indices are not direct LAI measurements.",
            "Four satellite acquisition dates do not provide "
            "continuous crop-season coverage.",
            "Ground-truth records have different experimental "
            "structures and treatment groupings.",
            "The nearest-date matching procedure does not establish "
            "a causal or predictive relationship.",
            "The suspicious Ranchi LAI observation remains excluded "
            "from model candidates according to Phase 4 processing.",
            "No satellite-to-LAI supervised model was trained.",
        ]

        for item in limitations:
            file.write(f"- {item}\n")

        file.write("\n6. FINAL CONCLUSION\n")
        file.write("-" * 70 + "\n")

        file.write(
            "The available Ranchi data support exploratory temporal "
            "assessment of site-level Sentinel-2 indices and "
            "ground-truth observation dates.\n\n"
        )

        file.write(
            "Temporal matching alone is insufficient to justify "
            "plot-level satellite-LAI integration or supervised "
            "LAI prediction. Georeferenced experimental plot "
            "boundaries, compatible field measurements, and "
            "independent validation are required.\n"
        )

    print(f"\nFinal report saved: {path}")


# ============================================================
# 13. MAIN PIPELINE
# ============================================================


def main():

    print("\n" + "=" * 65)
    print("KRUSHIMITRA — PHASE 8")
    print("SATELLITE AND GROUND-TRUTH INTEGRATION ASSESSMENT")
    print("=" * 65)

    (
        ground_truth,
        candidates,
        phase4_matching,
        pixels,
        site_summary,
    ) = load_inputs()

    ranchi = prepare_ground_truth(ground_truth, candidates)

    (
        pixels,
        acquisition_summary,
        site_summary,
    ) = prepare_satellite_data(pixels, site_summary)

    matches = calculate_temporal_matches(ranchi, acquisition_summary)

    consistency_report = assess_phase4_consistency(matches, phase4_matching)

    date_summary = create_date_level_summary(matches)

    index_summary = create_satellite_index_summary(site_summary)

    eligibility_report = create_eligibility_report(
        ranchi, matches, acquisition_summary, index_summary
    )

    # Save detailed outputs
    save_report(matches, "phase8_ground_truth_satellite_temporal_matches.csv")

    save_report(date_summary, "phase8_ground_truth_date_summary.csv")

    save_report(acquisition_summary, "phase8_satellite_acquisition_summary.csv")

    save_report(index_summary, "phase8_satellite_index_summary.csv")

    save_report(eligibility_report, "phase8_research_eligibility_report.csv")

    save_report(consistency_report, "phase8_phase4_consistency_checks.csv")

    # Generate plots
    plot_temporal_matching(matches)

    plot_satellite_indices(index_summary)

    # Generate final report
    generate_text_report(
        ranchi, matches, acquisition_summary, eligibility_report, consistency_report
    )

    print_section("PHASE 8 COMPLETED SUCCESSFULLY")

    print(f"Ranchi ground-truth records: {len(ranchi)}")

    print("Temporal matches within threshold:", int(matches["within_10_days"].sum()))

    print("Records without temporal match:", int((~matches["within_10_days"]).sum()))

    print("\nReports saved in:", REPORT_DIR)

    print("Plots saved in:", PLOT_DIR)


if __name__ == "__main__":
    main()
