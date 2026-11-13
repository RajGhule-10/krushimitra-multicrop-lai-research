from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# KRUSHIMITRA — PHASE 6
# RICE RESEARCH READINESS ASSESSMENT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "rice" / "raw"
PROCESSED = ROOT / "data" / "rice" / "processed"
OUTPUT = ROOT / "outputs" / "rice" / "reports"

OUTPUT.mkdir(parents=True, exist_ok=True)

GROUND_FILE = PROCESSED / "rice_ground_truth_all_records.csv"
CANDIDATE_FILE = PROCESSED / "rice_ground_truth_model_candidates.csv"
MATCH_FILE = PROCESSED / "ranchi_satellite_date_matching.csv"

# Satellite summary is stored in the raw data directory.
SATELLITE_FILE = RAW / "ranchi_rice_2019_site_summary.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================


def load_data():

    required_files = [
        GROUND_FILE,
        CANDIDATE_FILE,
        MATCH_FILE,
        SATELLITE_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required Phase 6 input not found: {file}")

    all_records = pd.read_csv(GROUND_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    matching = pd.read_csv(MATCH_FILE)
    satellite = pd.read_csv(SATELLITE_FILE)

    all_records["observation_date"] = pd.to_datetime(
        all_records["observation_date"], errors="coerce"
    )

    candidates["observation_date"] = pd.to_datetime(
        candidates["observation_date"], errors="coerce"
    )

    matching["ground_truth_date"] = pd.to_datetime(
        matching["ground_truth_date"], errors="coerce"
    )

    matching["satellite_date"] = pd.to_datetime(
        matching["satellite_date"], errors="coerce"
    )

    satellite["acquisition_date"] = pd.to_datetime(
        satellite["acquisition_date"], errors="coerce"
    )

    return all_records, candidates, matching, satellite


# ============================================================
# 2. DATASET COMPATIBILITY ASSESSMENT
# ============================================================


def assess_dataset_compatibility(candidates):

    definitions = {
        "Jabalpur": {
            "measurement_structure": "Variety-level source observations",
            "observation_design": "Multiple growth stages and varieties",
            "spatial_resolution": "Site-level coordinates",
            "temporal_coverage": "Multiple observation dates",
            "limitations": (
                "Experimental plot boundaries and plot-level "
                "satellite correspondence are unavailable."
            ),
        },
        "Kuthulia": {
            "measurement_structure": "Treatment-level means",
            "observation_design": (
                "Three sowing-date groups and five establishment-method treatments"
            ),
            "spatial_resolution": "Site-level coordinates",
            "temporal_coverage": "Harvest observations",
            "limitations": (
                "Treatment means are not independent plot-level "
                "measurements. Only one variety is represented."
            ),
        },
        "Ranchi": {
            "measurement_structure": "Treatment-level source observations",
            "observation_design": ("Multiple treatments, varieties and growth stages"),
            "spatial_resolution": "Site-level coordinates",
            "temporal_coverage": "Multiple observation dates",
            "limitations": (
                "Plot boundaries are unavailable. Satellite features "
                "represent a general 100-metre site buffer."
            ),
        },
    }

    rows = []

    for location, details in definitions.items():
        subset = candidates[candidates["location"] == location]

        rows.append(
            {
                "location": location,
                "record_count": len(subset),
                "measurement_structure": details["measurement_structure"],
                "observation_design": details["observation_design"],
                "spatial_resolution": details["spatial_resolution"],
                "temporal_coverage": details["temporal_coverage"],
                "lai_minimum": subset["lai"].min(),
                "lai_maximum": subset["lai"].max(),
                "unique_observation_dates": (subset["observation_date"].nunique()),
                "combined_supervised_model_eligible": "No",
                "limitations": details["limitations"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 3. EXPERIMENTAL STRUCTURE ASSESSMENT
# ============================================================


def assess_experimental_structure(candidates):

    rows = []

    for location, subset in candidates.groupby("location"):
        rows.append(
            {
                "location": location,
                "retained_records": len(subset),
                "unique_source_rows": (
                    subset["source_row_id"].nunique()
                    if "source_row_id" in subset.columns
                    else np.nan
                ),
                "unique_varieties": (
                    subset["variety"].nunique()
                    if "variety" in subset.columns
                    else np.nan
                ),
                "unique_treatments": (
                    subset["treatment"].nunique()
                    if "treatment" in subset.columns
                    else np.nan
                ),
                "unique_observation_stages": (subset["observation_stage"].nunique()),
                "unique_observation_dates": (subset["observation_date"].nunique()),
                "measurement_types": (
                    ", ".join(subset["measurement_type"].dropna().astype(str).unique())
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. SATELLITE DATA ELIGIBILITY
# ============================================================


def assess_satellite_eligibility(matching, satellite):

    matched = matching[
        matching["match_status"] == "site_level_temporal_match_only"
    ].copy()

    unmatched = matching[matching["match_status"] == "no_scene_within_10_days"].copy()

    total = len(matching)

    match_rate = len(matched) / total * 100 if total > 0 else 0

    rows = [
        {
            "assessment": "Total Ranchi ground-truth records",
            "value": total,
            "interpretation": "Retained Ranchi observations assessed.",
        },
        {
            "assessment": "Temporal matches",
            "value": len(matched),
            "interpretation": (
                "Ground-truth dates with a satellite acquisition "
                "within the selected 10-day window."
            ),
        },
        {
            "assessment": "Unmatched records",
            "value": len(unmatched),
            "interpretation": (
                "No selected satellite acquisition within the 10-day matching window."
            ),
        },
        {
            "assessment": "Temporal match rate percentage",
            "value": round(match_rate, 2),
            "interpretation": (
                "Temporal availability only; not model accuracy "
                "or plot-level matching quality."
            ),
        },
        {
            "assessment": "Satellite acquisition dates",
            "value": satellite["acquisition_date"].nunique(),
            "interpretation": (
                "Number of unique satellite acquisition dates "
                "in the site-summary dataset."
            ),
        },
        {
            "assessment": "Satellite spatial correspondence",
            "value": "Site-level 100-metre buffer",
            "interpretation": (
                "No experimental plot boundaries or plot-specific "
                "satellite samples are available."
            ),
        },
        {
            "assessment": "Plot-level LAI validation",
            "value": "Not established",
            "interpretation": (
                "Cannot associate site-buffer satellite features "
                "with individual experimental treatments."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 5. MODELLING ELIGIBILITY
# ============================================================


def assess_modelling_eligibility(candidates, matching):

    ranchi_matches = matching[
        matching["match_status"] == "site_level_temporal_match_only"
    ]

    ranchi_unmatched = matching[matching["match_status"] == "no_scene_within_10_days"]

    definitions = [
        {
            "research_objective": "Rice LAI descriptive analysis",
            "current_status": "Eligible with limitations",
            "reason": (
                "Source LAI measurements are available, but "
                "experimental designs must be analysed separately."
            ),
        },
        {
            "research_objective": "Rice variety and treatment comparisons",
            "current_status": "Eligible with limitations",
            "reason": (
                "Comparisons can be performed within compatible "
                "experimental designs and observation stages."
            ),
        },
        {
            "research_objective": "Ranchi satellite vegetation-index analysis",
            "current_status": "Eligible for exploratory analysis",
            "reason": (
                "Four site-summary acquisition dates and pixel-level "
                "spectral features are available."
            ),
        },
        {
            "research_objective": "Satellite-based rice LAI prediction",
            "current_status": "Not currently justified",
            "reason": (
                "Plot-level satellite correspondence and "
                "independent ground validation are unavailable."
            ),
        },
        {
            "research_objective": "Combined three-location LAI model",
            "current_status": "Not currently justified",
            "reason": (
                "Measurement structures, experimental designs, "
                "stages and locations differ."
            ),
        },
        {
            "research_objective": "Independent model validation",
            "current_status": "Insufficient evidence",
            "reason": (
                "Independent plot-level validation samples have not been established."
            ),
        },
    ]

    result = pd.DataFrame(definitions)

    result["ranchi_temporal_matches"] = len(ranchi_matches)
    result["ranchi_temporal_unmatched"] = len(ranchi_unmatched)

    return result


# ============================================================
# 6. DATA INTEGRITY CHECKS
# ============================================================


def run_integrity_checks(all_records, candidates, matching, satellite):

    checks = []

    def add_check(name, passed, details):
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "REVIEW",
                "details": details,
            }
        )

    add_check(
        "All-records dataset loaded",
        len(all_records) > 0,
        f"{len(all_records)} records",
    )

    add_check(
        "Model-candidate dataset loaded",
        len(candidates) > 0,
        f"{len(candidates)} retained records",
    )

    add_check(
        "Suspicious Ranchi observation preserved",
        (
            (
                (all_records["location"] == "Ranchi")
                & (all_records["lai"] == 42.26)
                & (all_records["quality_status"] == "excluded_suspicious_value")
            ).any()
        ),
        "42.26 should remain in all-records with exclusion flag.",
    )

    add_check(
        "Suspicious value excluded from candidates",
        not (
            ((candidates["location"] == "Ranchi") & (candidates["lai"] == 42.26)).any()
        ),
        "Suspicious value must not appear in model candidates.",
    )

    add_check(
        "Source-row identity available",
        (
            "source_row_id" in all_records.columns
            and all_records["source_row_id"].notna().all()
        ),
        "Source-row identifiers should be populated.",
    )

    add_check(
        "Kuthulia sowing groups preserved",
        (
            "sowing_date_group" in all_records.columns
            and all_records.loc[
                all_records["location"] == "Kuthulia", "sowing_date_group"
            ].nunique()
            == 3
        ),
        "Three distinct Kuthulia sowing-date groups expected.",
    )

    add_check(
        "Satellite acquisition dates available",
        satellite["acquisition_date"].notna().any(),
        "Satellite dates should be available.",
    )

    add_check(
        "Ranchi matching covers retained Ranchi records",
        (len(matching) == len(candidates[candidates["location"] == "Ranchi"])),
        (
            f"{len(matching)} matching rows for "
            f"{len(candidates[candidates['location'] == 'Ranchi'])} "
            "retained Ranchi records."
        ),
    )

    return pd.DataFrame(checks)


# ============================================================
# 7. GENERATE RESEARCH READINESS REPORT
# ============================================================


def generate_text_report(
    all_records,
    candidates,
    compatibility,
    experimental,
    satellite_eligibility,
    modelling,
    integrity,
):

    report = []

    report.extend(
        [
            "KRUSHIMITRA — PHASE 6",
            "RICE RESEARCH READINESS ASSESSMENT",
            "=" * 60,
            "",
            "1. DATASET OVERVIEW",
            "",
            f"All source records: {len(all_records)}",
            f"Retained candidates: {len(candidates)}",
            (f"Excluded suspicious records: {len(all_records) - len(candidates)}"),
            "",
            "2. LOCATION-WISE DATASET",
            "",
        ]
    )

    for _, row in compatibility.iterrows():
        report.extend(
            [
                f"{row['location']}:",
                f"  Records: {row['record_count']}",
                (f"  Measurement structure: {row['measurement_structure']}"),
                (f"  Observation stages: {row['observation_design']}"),
                (f"  Spatial information: {row['spatial_resolution']}"),
                f"  Limitations: {row['limitations']}",
                "",
            ]
        )

    report.extend(
        [
            "3. SATELLITE DATA ELIGIBILITY",
            "",
        ]
    )

    for _, row in satellite_eligibility.iterrows():
        report.append(f"{row['assessment']}: {row['value']}")
        report.append(f"  {row['interpretation']}")
        report.append("")

    report.extend(
        [
            "4. RESEARCH OBJECTIVE ELIGIBILITY",
            "",
        ]
    )

    for _, row in modelling.iterrows():
        report.append(f"{row['research_objective']}: {row['current_status']}")
        report.append(f"  Reason: {row['reason']}")
        report.append("")

    report.extend(
        [
            "5. DATA INTEGRITY CHECKS",
            "",
        ]
    )

    for _, row in integrity.iterrows():
        report.append(f"{row['check']}: {row['status']}")
        report.append(f"  {row['details']}")
        report.append("")

    report.extend(
        [
            "=" * 60,
            "6. FINAL RESEARCH RECOMMENDATION",
            "",
            (
                "The available datasets support descriptive LAI "
                "analysis within compatible experimental designs."
            ),
            (
                "Ranchi Sentinel-2 observations support exploratory "
                "site-level vegetation-index analysis."
            ),
            (
                "A combined supervised satellite-based LAI prediction "
                "model is not currently justified."
            ),
            "",
            (
                "Further work requires georeferenced experimental "
                "plots, exact field observation dates, compatible "
                "LAI measurement methodology and independent "
                "field-level validation."
            ),
            "",
            "END OF PHASE 6 REPORT",
        ]
    )

    output_file = OUTPUT / "rice_research_readiness_report.txt"

    output_file.write_text("\n".join(report), encoding="utf-8")


# ============================================================
# 8. MAIN EXECUTION
# ============================================================


def main():

    print("\n" + "=" * 50)
    print("KRUSHIMITRA — PHASE 6")
    print("RICE RESEARCH READINESS ASSESSMENT")
    print("=" * 50)

    (
        all_records,
        candidates,
        matching,
        satellite,
    ) = load_data()

    print("\nLoaded datasets:")
    print("All records:", all_records.shape)
    print("Candidates:", candidates.shape)
    print("Satellite matching:", matching.shape)
    print("Satellite summaries:", satellite.shape)

    compatibility = assess_dataset_compatibility(candidates)

    experimental = assess_experimental_structure(candidates)

    satellite_eligibility = assess_satellite_eligibility(matching, satellite)

    modelling = assess_modelling_eligibility(candidates, matching)

    integrity = run_integrity_checks(all_records, candidates, matching, satellite)

    # Save individual reports.
    compatibility.to_csv(OUTPUT / "dataset_compatibility.csv", index=False)

    experimental.to_csv(OUTPUT / "experimental_structure.csv", index=False)

    satellite_eligibility.to_csv(OUTPUT / "satellite_data_eligibility.csv", index=False)

    modelling.to_csv(OUTPUT / "modelling_eligibility.csv", index=False)

    integrity.to_csv(OUTPUT / "phase6_integrity_checks.csv", index=False)

    generate_text_report(
        all_records,
        candidates,
        compatibility,
        experimental,
        satellite_eligibility,
        modelling,
        integrity,
    )

    print("\nDataset compatibility:")
    print(compatibility.to_string(index=False))

    print("\nExperimental structure:")
    print(experimental.to_string(index=False))

    print("\nModelling eligibility:")
    print(modelling[["research_objective", "current_status"]].to_string(index=False))

    print("\nIntegrity checks:")
    print(integrity.to_string(index=False))

    print("\n" + "=" * 50)
    print("PHASE 6 COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print("\nReports saved in:", OUTPUT)


if __name__ == "__main__":
    main()
