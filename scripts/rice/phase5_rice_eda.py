"""
KRUSHIMITRA
PHASE 5 — RICE EXPLORATORY DATA ANALYSIS

Analyses:
1. Ground-truth LAI distributions
2. LAI by location and observation stage
3. Variety-wise LAI analysis
4. Treatment-wise LAI analysis
5. Missing values and duplicate records
6. Ranchi satellite vegetation-index trends
7. Satellite date matching analysis
8. Modelling feasibility report

Important:
- Does not train any model.
- Does not treat pixels as independent LAI labels.
- Preserves source-level experimental differences.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "rice" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "rice"

PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_FILE = DATA_DIR / "rice_ground_truth_model_candidates.csv"

ALL_RECORDS_FILE = DATA_DIR / "rice_ground_truth_all_records.csv"

MATCH_FILE = DATA_DIR / "ranchi_satellite_date_matching.csv"

SUMMARY_FILE = ROOT / "data" / "rice" / "raw" / "ranchi_rice_2019_site_summary.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n======================================")
print("KRUSHIMITRA — PHASE 5 RICE EDA")
print("======================================")

ground = pd.read_csv(GROUND_FILE)
all_records = pd.read_csv(ALL_RECORDS_FILE)
matching = pd.read_csv(MATCH_FILE)

ground["observation_date"] = pd.to_datetime(ground["observation_date"], errors="coerce")

matching["ground_truth_date"] = pd.to_datetime(
    matching["ground_truth_date"], errors="coerce"
)

matching["satellite_date"] = pd.to_datetime(matching["satellite_date"], errors="coerce")

ground["lai"] = pd.to_numeric(ground["lai"], errors="coerce")

matching["lai"] = pd.to_numeric(matching["lai"], errors="coerce")

if SUMMARY_FILE.exists():
    satellite = pd.read_csv(SUMMARY_FILE)
    satellite["acquisition_date"] = pd.to_datetime(
        satellite["acquisition_date"], errors="coerce"
    )
else:
    satellite = pd.DataFrame()

print("\nGround-truth shape:", ground.shape)
print("All-records shape:", all_records.shape)
print("Matching shape:", matching.shape)


# ============================================================
# 3. BASIC DATASET SUMMARY
# ============================================================

dataset_summary = (
    ground.groupby("location")
    .agg(
        observation_count=("lai", "count"),
        mean_lai=("lai", "mean"),
        median_lai=("lai", "median"),
        std_lai=("lai", "std"),
        minimum_lai=("lai", "min"),
        maximum_lai=("lai", "max"),
        unique_dates=("observation_date", "nunique"),
    )
    .reset_index()
)

dataset_summary.to_csv(REPORT_DIR / "rice_dataset_summary.csv", index=False)

print("\nDataset summary:")
print(dataset_summary.to_string(index=False))


# ============================================================
# 4. OBSERVATION STAGE SUMMARY
# ============================================================

stage_summary = (
    ground.groupby(["location", "observation_stage"], dropna=False)
    .agg(
        count=("lai", "count"),
        mean_lai=("lai", "mean"),
        median_lai=("lai", "median"),
        std_lai=("lai", "std"),
        min_lai=("lai", "min"),
        max_lai=("lai", "max"),
    )
    .reset_index()
)

stage_summary.to_csv(REPORT_DIR / "lai_stage_summary.csv", index=False)


# ============================================================
# 5. LAI DISTRIBUTION BY LOCATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(data=ground, x="location", y="lai")

plt.title("Rice LAI Distribution by Location")
plt.xlabel("Research Location")
plt.ylabel("Leaf Area Index (LAI)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(PLOT_DIR / "lai_distribution_by_location.png", dpi=300)

plt.close()


# ============================================================
# 6. LAI BY OBSERVATION STAGE
# ============================================================

plt.figure(figsize=(12, 6))

sns.boxplot(data=ground, x="observation_stage", y="lai", hue="location")

plt.title("Rice LAI by Observation Stage and Location")
plt.xlabel("Observation Stage")
plt.ylabel("Leaf Area Index")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(PLOT_DIR / "lai_by_observation_stage.png", dpi=300)

plt.close()


# ============================================================
# 7. VARIETY-WISE ANALYSIS
# ============================================================

variety_summary = (
    ground.groupby(["location", "variety"], dropna=False)
    .agg(
        count=("lai", "count"),
        mean_lai=("lai", "mean"),
        median_lai=("lai", "median"),
        std_lai=("lai", "std"),
    )
    .reset_index()
)

variety_summary.to_csv(REPORT_DIR / "variety_lai_summary.csv", index=False)

plt.figure(figsize=(12, 6))

sns.barplot(data=variety_summary, x="variety", y="mean_lai", hue="location")

plt.title("Mean LAI by Rice Variety")
plt.xlabel("Variety")
plt.ylabel("Mean LAI")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(PLOT_DIR / "variety_wise_lai.png", dpi=300)

plt.close()


# ============================================================
# 8. TREATMENT-WISE ANALYSIS
# ============================================================

treatment_summary = (
    ground.groupby(["location", "treatment"], dropna=False)
    .agg(count=("lai", "count"), mean_lai=("lai", "mean"), median_lai=("lai", "median"))
    .reset_index()
)

treatment_summary.to_csv(REPORT_DIR / "treatment_lai_summary.csv", index=False)


# ============================================================
# 9. DATE-WISE LAI TREND
# ============================================================

date_summary = (
    ground.groupby(["location", "observation_date"], dropna=False)
    .agg(mean_lai=("lai", "mean"), count=("lai", "count"))
    .reset_index()
)

date_summary.to_csv(REPORT_DIR / "date_wise_lai_summary.csv", index=False)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=date_summary, x="observation_date", y="mean_lai", hue="location", marker="o"
)

plt.title("Rice LAI Trends Across Observation Dates")
plt.xlabel("Observation Date")
plt.ylabel("Mean LAI")
plt.xticks(rotation=35)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(PLOT_DIR / "lai_temporal_trends.png", dpi=300)

plt.close()


# ============================================================
# 10. MISSING VALUE ANALYSIS
# ============================================================

missing_summary = all_records.isnull().sum().reset_index()

missing_summary.columns = ["column", "missing_count"]

missing_summary["missing_percentage"] = (
    missing_summary["missing_count"] / len(all_records) * 100
)

missing_summary.to_csv(REPORT_DIR / "missing_value_report.csv", index=False)


# ============================================================
# 11. DUPLICATE ANALYSIS
# ============================================================

duplicate_count = int(ground.duplicated().sum())

duplicate_report = pd.DataFrame(
    [{"total_records": len(ground), "exact_duplicate_rows": duplicate_count}]
)

duplicate_report.to_csv(REPORT_DIR / "duplicate_report.csv", index=False)


# ============================================================
# 12. RANCHI SATELLITE VEGETATION INDEX ANALYSIS
# ============================================================

index_columns = [
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "GNDVI_mean",
    "NDRE_mean",
    "NDMI_mean",
    "MNDWI_mean",
]

if not satellite.empty:
    available_indices = [col for col in index_columns if col in satellite.columns]

    satellite_trends = satellite[["acquisition_date"] + available_indices].copy()

    satellite_trends.to_csv(
        REPORT_DIR / "ranchi_satellite_index_trends.csv", index=False
    )

    plt.figure(figsize=(12, 6))

    for col in available_indices:
        plt.plot(
            satellite["acquisition_date"],
            satellite[col],
            marker="o",
            label=col.replace("_mean", ""),
        )

    plt.title("Ranchi Sentinel-2 Vegetation Index Trends")
    plt.xlabel("Satellite Acquisition Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=35)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "ranchi_satellite_indices.png", dpi=300)

    plt.close()


# ============================================================
# 13. SATELLITE MATCHING ANALYSIS
# ============================================================

matching_summary = (
    matching.groupby("match_status").size().reset_index(name="observation_count")
)

matching_summary.to_csv(REPORT_DIR / "satellite_matching_summary.csv", index=False)

plt.figure(figsize=(8, 5))

sns.countplot(data=matching, x="match_status")

plt.title("Ranchi Satellite Matching Status")
plt.xlabel("Matching Status")
plt.ylabel("Observation Count")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(PLOT_DIR / "satellite_matching_status.png", dpi=300)

plt.close()


# ============================================================
# 14. DATE OFFSET ANALYSIS
# ============================================================

if "date_offset_days" in matching.columns:
    plt.figure(figsize=(9, 5))

    sns.histplot(data=matching, x="date_offset_days", bins=10, kde=False)

    plt.title("Satellite Acquisition Date Offset")
    plt.xlabel("Days Between Satellite and Ground Observation")
    plt.ylabel("Number of Observations")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "satellite_date_offsets.png", dpi=300)

    plt.close()


# ============================================================
# 15. MODELLING FEASIBILITY REPORT
# ============================================================

retained_count = len(ground)

ranchi_count = len(ground[ground["location"] == "Ranchi"])

jabalpur_count = len(ground[ground["location"] == "Jabalpur"])

kuthulia_count = len(ground[ground["location"] == "Kuthulia"])

matched_count = int(matching["satellite_date"].notna().sum())

unmatched_count = int(matching["satellite_date"].isna().sum())

report = f"""
KRUSHIMITRA — PHASE 5
RICE DATASET EDA AND MODELLING FEASIBILITY REPORT

==================================================

1. GROUND-TRUTH DATA

Total retained LAI records: {retained_count}

Jabalpur records: {jabalpur_count}
Kuthulia records: {kuthulia_count}
Ranchi records: {ranchi_count}

2. DATA QUALITY

Exact duplicate rows: {duplicate_count}

Suspicious Ranchi value 42.26:
Preserved in all-records source with exclusion flag.

3. SATELLITE MATCHING

Temporal matches: {matched_count}
Unmatched observations: {unmatched_count}

Matching is site-level temporal matching only.
No plot-level satellite correspondence is established.

4. MODELLING FEASIBILITY

The three datasets have different experimental designs,
observation stages and measurement structures.

Jabalpur:
Variety-level LAI observations.

Kuthulia:
Treatment-level harvest LAI means.

Ranchi:
Treatment-level LAI observations.

The Ranchi satellite features represent a general
100-metre site buffer, not individual experimental plots.

Therefore, a combined supervised LAI model should not
be trained from these records without further validation
of observation units, spatial correspondence and
experimental independence.

5. RECOMMENDATION

Continue collecting rice datasets with:
- Field-level LAI observations
- Exact observation dates
- Georeferenced experimental plots
- Satellite imagery matching the observation period
- Clear measurement and replication methodology

==================================================
END OF REPORT
"""

with open(
    REPORT_DIR / "rice_modeling_feasibility_report.txt", "w", encoding="utf-8"
) as file:
    file.write(report)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n======================================")
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGround-truth records:", retained_count)
print("Jabalpur:", jabalpur_count)
print("Kuthulia:", kuthulia_count)
print("Ranchi:", ranchi_count)

print("\nDuplicate rows:", duplicate_count)

print("\nSatellite matches:", matched_count)
print("Unmatched:", unmatched_count)

print("\nPlots saved in:", PLOT_DIR)
print("Reports saved in:", REPORT_DIR)

print("\nPHASE 5 EXECUTION COMPLETE")
