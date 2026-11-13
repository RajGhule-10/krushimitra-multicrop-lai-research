from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PHASE 10: AHERO RICE DATASET INTEGRITY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "rice" / "raw"
PROCESSED = ROOT / "data" / "rice" / "processed"

REPORT_DIR = ROOT / "outputs" / "rice" / "reports" / "phase10"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = {
    1: {
        "processed": "final_df_rice_campaign_1.csv",
        "ground_truth": "Rice_campaign_1_LAI_Data.csv",
        "date": "2023-06-10",
    },
    2: {
        "processed": "final_df_rice_campaign_2.csv",
        "ground_truth": "Rice_campaign_2_LAI_Data.csv",
        "date": "2023-07-20",
    },
    3: {
        "processed": "final_df_rice_campaign_3.csv",
        "ground_truth": "Rice_campaign_3_LAI_Data.csv",
        "date": "2023-08-14",
    },
}

FEATURES = [
    "NDVI",
    "GNDVI",
    "NDVIre",
    "NDREI",
    "CIRE",
    "VREI",
    "EVI",
    "mND₇₀₅",
    "TCI",
    "mSR740",
    "AFRI₁₅₀₀",
    "AFRI₂₁₀₀",
    "NDII8",
    "SAVI",
    "SVHI",
    "NPCI",
    "CI783-B8",
    "CI783-B8A",
    "NBIB8",
    "NBIB8A",
    "MSI8A",
    "LSWI8",
    "NMDI",
    "REDRIG2",
]

TARGET = "LAI"


def normalize_farm_id(value):
    """Normalize AR01, AR001 and AR1 to AR001."""
    value = str(value).strip().upper()

    if not value.startswith("AR"):
        raise ValueError(f"Unexpected farm ID: {value}")

    number = value[2:]

    if not number.isdigit():
        raise ValueError(f"Invalid farm ID: {value}")

    return "AR" + number.zfill(3)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\nCheck the filename and data/rice folder."
        )
    return pd.read_csv(path)


def main():

    print("\n" + "=" * 65)
    print("AHERO RICE DATASET INTEGRITY AUDIT")
    print("=" * 65)

    all_campaigns = []
    campaign_reports = []
    missing_farms = []

    for campaign, files in CAMPAIGNS.items():
        print(f"\n--- CAMPAIGN {campaign} ---")

        processed = load_csv(PROCESSED / files["processed"])
        ground_truth = load_csv(RAW / files["ground_truth"])

        required_processed = ["Farm_ID", TARGET, "Growth_stage"] + FEATURES

        required_ground_truth = ["Farm_ID", TARGET, "Date"]

        for col in required_processed:
            if col not in processed.columns:
                raise ValueError(f"Campaign {campaign}: missing processed column {col}")

        for col in required_ground_truth:
            if col not in ground_truth.columns:
                raise ValueError(
                    f"Campaign {campaign}: missing ground-truth column {col}"
                )

        processed = processed.copy()
        ground_truth = ground_truth.copy()

        processed["Farm_ID_norm"] = processed["Farm_ID"].apply(normalize_farm_id)

        ground_truth["Farm_ID_norm"] = ground_truth["Farm_ID"].apply(normalize_farm_id)

        processed[TARGET] = pd.to_numeric(processed[TARGET], errors="coerce")

        ground_truth[TARGET] = pd.to_numeric(ground_truth[TARGET], errors="coerce")

        # Reconstruct ground-truth mean LAI per farm
        ground_truth_mean = (
            ground_truth.groupby("Farm_ID_norm")
            .agg(
                original_LAI_mean=(TARGET, "mean"),
                original_LAI_count=(TARGET, "count"),
                original_first_date=("Date", "min"),
                original_last_date=("Date", "max"),
            )
            .reset_index()
        )

        # Verify farm IDs and LAI values
        comparison = processed.merge(
            ground_truth_mean,
            on="Farm_ID_norm",
            how="outer",
            indicator=True,
            validate="one_to_one",
        )

        comparison["LAI_difference"] = (
            comparison[TARGET] - comparison["original_LAI_mean"]
        )

        comparison["LAI_match"] = np.isclose(
            comparison[TARGET],
            comparison["original_LAI_mean"],
            atol=1e-4,
            equal_nan=False,
        )

        comparison["Campaign"] = campaign

        # Farm IDs missing from processed satellite dataset
        missing = comparison[comparison["_merge"] == "right_only"].copy()

        if not missing.empty:
            missing_farms.append(missing)

        # Processed farms without original ground truth
        unexpected = comparison[comparison["_merge"] == "left_only"]

        # Satellite feature quality
        numeric_features = processed[FEATURES].apply(pd.to_numeric, errors="coerce")

        missing_features = int(numeric_features.isna().sum().sum())
        infinite_features = int(np.isinf(numeric_features.to_numpy()).sum())

        duplicate_farms = int(processed["Farm_ID_norm"].duplicated().sum())

        lai_mismatches = int(
            ((comparison["_merge"] == "both") & (~comparison["LAI_match"])).sum()
        )

        result = {
            "Campaign": campaign,
            "Processed_records": len(processed),
            "Processed_unique_farms": processed["Farm_ID_norm"].nunique(),
            "Original_LAI_measurements": len(ground_truth),
            "Original_unique_farms": ground_truth["Farm_ID_norm"].nunique(),
            "Missing_processed_farms": len(missing),
            "Unexpected_processed_farms": len(unexpected),
            "LAI_mismatches": lai_mismatches,
            "Duplicate_processed_farms": duplicate_farms,
            "Missing_feature_values": missing_features,
            "Infinite_feature_values": infinite_features,
        }

        campaign_reports.append(result)

        print(f"Processed records: {len(processed)}")
        print(f"Processed farms: {processed['Farm_ID_norm'].nunique()}")
        print(f"Original LAI measurements: {len(ground_truth)}")
        print(f"Original LAI farms: {ground_truth['Farm_ID_norm'].nunique()}")
        print(f"Farms absent from processed data: {len(missing)}")
        print(f"LAI mismatches: {lai_mismatches}")
        print(f"Duplicate processed farm IDs: {duplicate_farms}")
        print(f"Missing satellite feature values: {missing_features}")
        print(f"Infinite satellite feature values: {infinite_features}")

        # Keep campaign identity: the same farm can occur in multiple campaigns.
        processed["Campaign"] = campaign
        processed["Campaign_Date"] = files["date"]
        processed["Farm_ID_norm"] = processed["Farm_ID_norm"]

        all_campaigns.append(processed)

        comparison.to_csv(
            REPORT_DIR / f"campaign_{campaign}_lai_comparison.csv", index=False
        )

    # --------------------------------------------------------
    # Combined dataset checks
    # --------------------------------------------------------

    combined = pd.concat(all_campaigns, ignore_index=True)

    combined["Farm_ID_norm"] = combined["Farm_ID"].apply(normalize_farm_id)

    repeated_farms = (
        combined.groupby("Farm_ID_norm")
        .agg(
            Campaign_count=("Campaign", "nunique"),
            Record_count=("Campaign", "size"),
            Campaigns=("Campaign", lambda x: ",".join(map(str, sorted(x.unique())))),
        )
        .reset_index()
    )

    combined_duplicate_keys = combined.duplicated(
        subset=["Farm_ID_norm", "Campaign"]
    ).sum()

    report = pd.DataFrame(campaign_reports)

    report.to_csv(REPORT_DIR / "campaign_integrity_summary.csv", index=False)

    repeated_farms.to_csv(REPORT_DIR / "farm_campaign_history.csv", index=False)

    if missing_farms:
        pd.concat(missing_farms, ignore_index=True).to_csv(
            REPORT_DIR / "missing_farms_report.csv", index=False
        )

    combined.to_csv(REPORT_DIR / "audited_rice_dataset.csv", index=False)

    print("\n" + "=" * 65)
    print("COMBINED DATASET SUMMARY")
    print("=" * 65)

    print(f"Total records: {len(combined)}")
    print(f"Unique farms: {combined['Farm_ID_norm'].nunique()}")
    print(f"Campaigns: {combined['Campaign'].nunique()}")
    print(f"Duplicate farm-campaign records: {combined_duplicate_keys}")
    print(
        "Farms appearing in multiple campaigns:",
        int((repeated_farms["Campaign_count"] > 1).sum()),
    )

    print("\nCampaign summary:")
    print(report.to_string(index=False))

    print("\nReports saved to:")
    print(REPORT_DIR)

    print("\nPHASE 10 AUDIT COMPLETED")


if __name__ == "__main__":
    main()
