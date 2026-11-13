from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# KRUSHIMITRA — PHASE 4
# Rice Ground-Truth Preparation and Satellite Date Matching
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "rice" / "raw"
OUT = ROOT / "data" / "rice" / "processed"

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Jabalpur": RAW / "IND_RI_JAB_2019_Timeseries.csv",
    "Kuthulia": RAW / "IND_RI_KUT_2019_Timeseries.csv",
    "Ranchi": RAW / "IND_RI_RAN_2019_Timeseries.csv",
}

SUMMARY_FILE = RAW / "ranchi_rice_2019_site_summary.csv"


# ============================================================
# COMMON HELPERS
# ============================================================


def find_column(df, phrase):
    matches = [c for c in df.columns if phrase.lower() in c.lower()]

    if not matches:
        raise ValueError(f"Missing source column: {phrase}")

    return matches[0]


def parse_dates(values):
    return pd.to_datetime(values, format="mixed", dayfirst=True, errors="coerce")


def base_metadata(df, location):

    date_col = find_column(df, "DATE/TIME")
    stage_col = find_column(df, "Serial number")

    species_matches = [c for c in df.columns if "Species" in c]

    if species_matches:
        variety_values = df[species_matches[0]].astype(str)

    else:
        treatment_matches = [c for c in df.columns if "Treatment type" in c]

        if treatment_matches:
            variety_values = df[treatment_matches[0]].astype(str)
        else:
            variety_values = pd.Series(["Unknown"] * len(df), index=df.index)

    result = pd.DataFrame(
        {
            # Original CSV row identity (1-based).
            "source_row_id": [
                f"{location}_row_{idx + 1:03d}" for idx in range(len(df))
            ],
            "location": location,
            "observation_date": parse_dates(df[date_col]),
            "observation_stage": (df[stage_col].astype(str).str.strip()),
            "variety_or_species": variety_values,
            "latitude": pd.to_numeric(df["1600"], errors="coerce"),
            "longitude": pd.to_numeric(df["1601"], errors="coerce"),
        },
        index=df.index,
    )

    return result


# ============================================================
# JABALPUR — WIDE TO LONG
# ============================================================


def process_jabalpur():

    df = pd.read_csv(FILES["Jabalpur"])
    meta = base_metadata(df, "Jabalpur")

    lai_cols = [c for c in df.columns if "Leaf area index" in c]

    records = []

    for col in lai_cols:
        variety = col.split("Treatment:")[-1].strip()

        for idx, value in df[col].items():
            if pd.isna(value):
                continue

            row = meta.loc[idx].to_dict()

            row.update(
                {
                    "variety": variety,
                    "treatment": variety,
                    "lai": float(value),
                    "measurement_type": ("variety_level_source_value"),
                    "source_dataset": "Jabalpur_2019",
                    "quality_status": "retained",
                }
            )

            records.append(row)

    return pd.DataFrame(records)


# ============================================================
# KUTHULIA — TREATMENT COLUMNS TO LONG
# PRESERVE SOWING-DATE GROUP AND SOURCE ROW ID
# ============================================================


def process_kuthulia():

    df = pd.read_csv(FILES["Kuthulia"])
    meta = base_metadata(df, "Kuthulia")

    # Identify the beginning of each experimental schedule.
    # Each schedule starts with a 0 DAS record.
    stage = meta["observation_stage"].astype(str).str.strip().str.upper()

    starts_group = stage.str.fullmatch(r"0\s*DAS")

    if not starts_group.iloc[0]:
        raise ValueError(
            "Kuthulia source does not start with a 0 DAS record. "
            "Cannot safely infer sowing-date groups."
        )

    group_numbers = starts_group.cumsum()

    if group_numbers.nunique() != 3:
        raise ValueError(
            "Expected 3 Kuthulia sowing-date groups, "
            f"but found {group_numbers.nunique()}."
        )

    # Derive each sowing date from its 0 DAS observation.
    group_metadata = {}

    for group_number in sorted(group_numbers.unique()):
        group_rows = meta.loc[group_numbers == group_number]

        zero_rows = group_rows[
            group_rows["observation_stage"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.fullmatch(r"0\s*DAS")
        ]

        if len(zero_rows) != 1:
            raise ValueError(
                f"Kuthulia group {group_number} must contain exactly one 0 DAS row."
            )

        sowing_date = zero_rows.iloc[0]["observation_date"]

        group_metadata[group_number] = {
            "sowing_date_group": f"KUT_S{group_number}",
            "sowing_date": sowing_date,
        }

    # Add experimental schedule information to every source row.
    meta["sowing_date_group"] = [
        group_metadata[group_number]["sowing_date_group"]
        for group_number in group_numbers
    ]

    meta["sowing_date"] = [
        group_metadata[group_number]["sowing_date"] for group_number in group_numbers
    ]

    lai_cols = [c for c in df.columns if "Leaf area index" in c]

    records = []

    for col in lai_cols:
        treatment = col.split("Treatment")[-1].strip()
        treatment = treatment.replace("T", "").strip()

        for idx, value in df[col].items():
            if pd.isna(value):
                continue

            row = meta.loc[idx].to_dict()

            row.update(
                {
                    "variety": "MTU-1010",
                    "treatment": treatment,
                    "lai": float(value),
                    "measurement_type": ("treatment_level_mean"),
                    "source_dataset": "Kuthulia_2019",
                    "quality_status": "retained",
                }
            )

            records.append(row)

    result = pd.DataFrame(records)

    # Verify all original LAI values were preserved.
    expected_count = int(df[lai_cols].notna().sum().sum())

    if len(result) != expected_count:
        raise ValueError(
            "Kuthulia LAI count mismatch: "
            f"expected {expected_count}, generated {len(result)}."
        )

    return result


# ============================================================
# RANCHI — LONG-FORMAT GROUND TRUTH
# ============================================================


def process_ranchi():

    df = pd.read_csv(FILES["Ranchi"])
    meta = base_metadata(df, "Ranchi")

    lai_col = find_column(df, "Leaf area index")

    records = []

    for idx, value in df[lai_col].items():
        if pd.isna(value):
            continue

        raw_species = str(meta.loc[idx, "variety_or_species"])

        treatment = raw_species.split("_T")[-1] if "_T" in raw_species else None

        variety = raw_species.split("_T")[0].replace("_", " ").strip()

        lai = float(value)

        suspicious = lai == 42.26

        row = meta.loc[idx].to_dict()

        row.update(
            {
                "variety": variety,
                "treatment": treatment,
                "lai": lai,
                "measurement_type": ("treatment_level_source_value"),
                "source_dataset": "Ranchi_2019",
                "quality_status": (
                    "excluded_suspicious_value" if suspicious else "retained"
                ),
            }
        )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# PHASE 4.1 — COMBINE GROUND-TRUTH RECORDS
# ============================================================

ground_truth = pd.concat(
    [
        process_jabalpur(),
        process_kuthulia(),
        process_ranchi(),
    ],
    ignore_index=True,
)

ground_truth["observation_stage"] = (
    ground_truth["observation_stage"].astype(str).str.strip()
)

ground_truth["observation_date"] = pd.to_datetime(
    ground_truth["observation_date"], errors="coerce"
)

ground_truth["lai"] = pd.to_numeric(ground_truth["lai"], errors="coerce")

ground_truth["observation_year"] = ground_truth["observation_date"].dt.year

ground_truth["observation_month"] = ground_truth["observation_date"].dt.month

# Save all source records, including flagged observations.
ground_truth.to_csv(OUT / "rice_ground_truth_all_records.csv", index=False)

# Exclude only explicitly flagged observations.
model_candidates = ground_truth[ground_truth["quality_status"] == "retained"].copy()

model_candidates.to_csv(OUT / "rice_ground_truth_model_candidates.csv", index=False)


# ============================================================
# PHASE 4.2 — RANCHI SATELLITE DATE MATCHING
# ============================================================

ranchi_gt = model_candidates[model_candidates["location"] == "Ranchi"].copy()

satellite = pd.read_csv(SUMMARY_FILE)

satellite["acquisition_date"] = pd.to_datetime(
    satellite["acquisition_date"], errors="coerce"
)

matches = []

for _, ground_row in ranchi_gt.iterrows():
    ground_date = ground_row["observation_date"]

    candidates = satellite.dropna(subset=["acquisition_date"]).copy()

    candidates["date_offset_days"] = (
        candidates["acquisition_date"] - ground_date
    ).dt.days

    candidates["absolute_offset_days"] = candidates["date_offset_days"].abs()

    candidates = candidates[candidates["absolute_offset_days"] <= 10]

    match_record = {
        "source_row_id": ground_row["source_row_id"],
        "ground_truth_date": ground_date,
        "treatment": ground_row["treatment"],
        "variety": ground_row["variety"],
        "lai": ground_row["lai"],
        "satellite_date": pd.NaT,
        "date_offset_days": np.nan,
        "match_status": "no_scene_within_10_days",
        "satellite_image_id": None,
    }

    if not candidates.empty:
        # Nearest scene; ties prefer the earlier date.
        candidates = candidates.sort_values(
            ["absolute_offset_days", "acquisition_date"]
        )

        selected = candidates.iloc[0]

        match_record.update(
            {
                "satellite_date": selected["acquisition_date"],
                "date_offset_days": selected["date_offset_days"],
                "match_status": ("site_level_temporal_match_only"),
                "satellite_image_id": selected["image_id"],
            }
        )

    matches.append(match_record)

matching_df = pd.DataFrame(matches)

matching_df.to_csv(OUT / "ranchi_satellite_date_matching.csv", index=False)


# ============================================================
# PHASE 4.3 — DATA QUALITY REPORT
# ============================================================

quality_report = (
    ground_truth.groupby(["location", "quality_status"])
    .size()
    .reset_index(name="record_count")
)

quality_report.to_csv(OUT / "rice_ground_truth_quality_report.csv", index=False)


# ============================================================
# PHASE 4.4 — SOURCE IDENTITY VALIDATION
# ============================================================

if ground_truth["source_row_id"].isna().any():
    raise ValueError("Missing source row identifiers detected.")

if ground_truth.duplicated(
    subset=["source_dataset", "source_row_id", "treatment"]
).any():
    raise ValueError("Unexpected duplicate source-row/treatment combinations.")

kuthulia_check = ground_truth[ground_truth["location"] == "Kuthulia"].copy()

if len(kuthulia_check) != 15:
    raise ValueError(f"Expected 15 Kuthulia LAI values, found {len(kuthulia_check)}.")

if kuthulia_check["source_row_id"].nunique() != 3:
    raise ValueError("Expected 3 Kuthulia harvest source rows.")

print("\n======================================")
print("KRUSHIMITRA PHASE 4 COMPLETED")
print("======================================")

print("\nGround-truth records:")
print(ground_truth.groupby("location").size())

print("\nQuality status:")
print(ground_truth["quality_status"].value_counts())

print("\nKuthulia sowing-date groups:")
print(
    kuthulia_check[["sowing_date_group", "sowing_date"]]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nKuthulia harvest records:")
print(
    kuthulia_check[kuthulia_check["observation_stage"] == "At harvest"][
        [
            "source_row_id",
            "sowing_date_group",
            "sowing_date",
            "observation_date",
            "treatment",
            "lai",
        ]
    ].to_string(index=False)
)

print("\nSatellite match status:")
print(matching_df["match_status"].value_counts())

print("\nOutput directory:", OUT)
