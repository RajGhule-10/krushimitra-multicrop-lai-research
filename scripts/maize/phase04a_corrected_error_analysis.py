from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 04A: CORRECTED MAIZE ERROR ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "maize" / "processed" / "maize_final_satellite_features.csv"

REPORT_DIR = ROOT / "outputs" / "maize" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["NDVI", "EVI", "GNDVI", "SAVI", "NDRE", "RVI", "DVI", "NDMI", "BNDVI"]

TARGET = "LAI"
GROUP = "Farm_ID"

N_SPLITS = 5
RANDOM_STATE = 42

PERMUTATION_REPEATS = 10

# ============================================================
# OUTPUT FILES
# ============================================================

SUMMARY_FILE = REPORT_DIR / "phase04a_model_comparison.csv"
FOLD_FILE = REPORT_DIR / "phase04a_fold_metrics.csv"
PREDICTION_FILE = REPORT_DIR / "phase04a_oof_predictions.csv"

IMPORTANCE_FOLD_FILE = REPORT_DIR / "phase04a_permutation_importance_by_fold.csv"

IMPORTANCE_SUMMARY_FILE = REPORT_DIR / "phase04a_permutation_importance_summary.csv"

CAMPAIGN_ERROR_FILE = REPORT_DIR / "phase04a_campaign_error_analysis.csv"

LAI_ERROR_FILE = REPORT_DIR / "phase04a_lai_range_error_analysis.csv"

METHOD_ERROR_FILE = REPORT_DIR / "phase04a_extraction_method_error_analysis.csv"

REPORT_FILE = REPORT_DIR / "phase04a_error_analysis_report.txt"


# ============================================================
# HELPERS
# ============================================================


def parse_boolean(series):
    """Safely interpret CSV boolean values."""

    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def build_models():
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Dummy Mean": DummyRegressor(strategy="mean"),
    }


def prepare_variants(data):
    variants = {"Full_316": data.copy()}

    reliable = data[parse_boolean(data["GPS_Inside_Matched_Parcel"])].copy()

    variants["Spatial_Reliable"] = reliable

    original = data[
        data["Satellite_Source"].astype(str).str.strip() == "Phase 03"
    ].copy()

    variants["Original_Phase03"] = original

    return variants


def clean_dataset(data):
    data = data.copy()

    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=[TARGET, GROUP] + FEATURES).copy()

    data[GROUP] = data[GROUP].astype(str)

    return data


# ============================================================
# DATASET EVALUATION
# ============================================================


def evaluate_variant(dataset_name, raw_data):

    data = clean_dataset(raw_data)

    X = data[FEATURES]
    y = data[TARGET]
    groups = data[GROUP]

    n_farms = groups.nunique()
    n_splits = min(N_SPLITS, n_farms)

    if n_splits < 2:
        raise ValueError(f"Not enough farms in {dataset_name}")

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print(f"Observations: {len(data)}")
    print(f"Unique farms: {n_farms}")
    print(f"CV folds: {n_splits}")

    cv = GroupKFold(n_splits=n_splits)

    models = build_models()

    summary_records = []
    fold_records = []
    prediction_records = []
    importance_records = []

    # One OOF prediction array per model.
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        oof_predictions = np.full(len(data), np.nan)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_farms = set(groups.iloc[train_idx])
            test_farms = set(groups.iloc[test_idx])

            if train_farms.intersection(test_farms):
                raise RuntimeError("Farm leakage detected.")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            oof_predictions[test_idx] = predictions

            metrics = calculate_metrics(y_test.to_numpy(), predictions)

            fold_records.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold,
                    "Train_Observations": len(train_idx),
                    "Test_Observations": len(test_idx),
                    "Train_Farms": len(train_farms),
                    "Test_Farms": len(test_farms),
                    **metrics,
                }
            )

            # ------------------------------------------------
            # HELD-OUT PERMUTATION IMPORTANCE
            # Random Forest only
            # ------------------------------------------------

            if model_name == "Random Forest":
                permutation = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    scoring="neg_mean_absolute_error",
                    n_repeats=PERMUTATION_REPEATS,
                    random_state=RANDOM_STATE + fold,
                    n_jobs=-1,
                )

                for feature_idx, feature in enumerate(FEATURES):
                    importance_records.append(
                        {
                            "Dataset": dataset_name,
                            "Fold": fold,
                            "Feature": feature,
                            "Importance_Mean": (
                                permutation.importances_mean[feature_idx]
                            ),
                            "Importance_Std": (
                                permutation.importances_std[feature_idx]
                            ),
                            "Test_Observations": len(test_idx),
                        }
                    )

        if np.isnan(oof_predictions).any():
            raise RuntimeError(
                f"Incomplete OOF predictions: {dataset_name}, {model_name}"
            )

        overall_metrics = calculate_metrics(y.to_numpy(), oof_predictions)

        summary_records.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Observations": len(data),
                "Unique_Farms": n_farms,
                "CV_Folds": n_splits,
                **overall_metrics,
            }
        )

        # ----------------------------------------------------
        # SAVE EXACTLY ONE PREDICTION PER OBSERVATION/MODEL
        # ----------------------------------------------------

        prediction_frame = pd.DataFrame(
            {
                "Observation_ID": data["Observation_ID"].to_numpy(),
                "Dataset": dataset_name,
                "Model": model_name,
                "Farm_ID": groups.to_numpy(),
                "Campaign": data["campaign_id"].to_numpy(),
                "Actual_LAI": y.to_numpy(),
                "Predicted_LAI": oof_predictions,
                "Prediction_Error": (oof_predictions - y.to_numpy()),
                "Absolute_Error": np.abs(oof_predictions - y.to_numpy()),
                "Extraction_Method": (data["Extraction_Method"].to_numpy()),
                "Satellite_Source": (data["Satellite_Source"].to_numpy()),
                "Satellite_Image_Date": (data["Satellite_Image_Date"].to_numpy()),
                "Image_Date_Difference_Days": (
                    data["Absolute_Image_Date_Difference_Days"].to_numpy()
                ),
            }
        )

        prediction_records.append(prediction_frame)

        print(
            f"MAE:  {overall_metrics['MAE']:.6f}\n"
            f"RMSE: {overall_metrics['RMSE']:.6f}\n"
            f"R2:   {overall_metrics['R2']:.6f}"
        )

    return (summary_records, fold_records, prediction_records, importance_records)


# ============================================================
# ERROR ANALYSIS
# ============================================================


def grouped_error_analysis(predictions, group_columns, output_name):

    results = []

    for (dataset, model), subset in predictions.groupby(["Dataset", "Model"]):
        for group_values, group_data in subset.groupby(group_columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            actual = group_data["Actual_LAI"]
            predicted = group_data["Predicted_LAI"]

            record = {
                "Dataset": dataset,
                "Model": model,
                "Observations": len(group_data),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "Mean_Bias": (predicted - actual).mean(),
            }

            for column, value in zip(group_columns, group_values):
                record[column] = value

            results.append(record)

    result_df = pd.DataFrame(results)

    result_df.to_csv(REPORT_DIR / output_name, index=False)

    return result_df


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("PHASE 04A: CORRECTED MAIZE ERROR ANALYSIS")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    data = pd.read_csv(DATA_FILE)

    required = FEATURES + [
        TARGET,
        GROUP,
        "campaign_id",
        "Satellite_Source",
        "Extraction_Method",
        "Satellite_Image_Date",
        "Absolute_Image_Date_Difference_Days",
        "GPS_Inside_Matched_Parcel",
    ]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Stable source-row identifier.
    data = data.reset_index(drop=True)
    data["Observation_ID"] = [f"MAIZE_{i + 1:04d}" for i in range(len(data))]

    variants = prepare_variants(data)

    all_summaries = []
    all_folds = []
    all_predictions = []
    all_importance = []

    for dataset_name, dataset in variants.items():
        if dataset.empty:
            print(f"Skipping empty dataset: {dataset_name}")
            continue

        results = evaluate_variant(dataset_name, dataset)

        summaries, folds, predictions, importance = results

        all_summaries.extend(summaries)
        all_folds.extend(folds)
        all_predictions.extend(predictions)
        all_importance.extend(importance)

    summary_df = pd.DataFrame(all_summaries)
    fold_df = pd.DataFrame(all_folds)

    prediction_df = pd.concat(all_predictions, ignore_index=True)

    importance_df = pd.DataFrame(all_importance)

    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    expected_prediction_rows = sum(
        len(clean_dataset(dataset)) * 3
        for dataset in variants.values()
        if not dataset.empty
    )

    if len(prediction_df) != expected_prediction_rows:
        raise RuntimeError("Prediction row count does not match expected count.")

    duplicate_predictions = prediction_df.duplicated(
        subset=["Dataset", "Model", "Observation_ID"]
    ).sum()

    if duplicate_predictions:
        raise RuntimeError(f"Duplicate OOF predictions found: {duplicate_predictions}")

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE SUMMARY
    # --------------------------------------------------------

    importance_summary = (
        importance_df.groupby("Feature")
        .agg(
            Mean_Importance=("Importance_Mean", "mean"),
            Std_Importance=("Importance_Mean", "std"),
            Dataset_Count=("Dataset", "nunique"),
        )
        .reset_index()
    )

    # Save dataset-specific importance separately.
    importance_by_dataset = (
        importance_df.groupby(["Dataset", "Feature"])
        .agg(
            Mean_Importance=("Importance_Mean", "mean"),
            Std_Importance=("Importance_Mean", "std"),
        )
        .reset_index()
        .sort_values(["Dataset", "Mean_Importance"], ascending=[True, False])
    )

    # --------------------------------------------------------
    # LAI RANGE ERROR ANALYSIS
    # --------------------------------------------------------

    prediction_df["LAI_Range"] = pd.cut(
        prediction_df["Actual_LAI"],
        bins=[-np.inf, 1, 2, 3, 4, np.inf],
        labels=["<=1", "1-2", "2-3", "3-4", ">4"],
        right=True,
    )

    # --------------------------------------------------------
    # GROUPED ERROR ANALYSES
    # --------------------------------------------------------

    campaign_errors = grouped_error_analysis(
        prediction_df, ["Campaign"], "phase04a_campaign_error_analysis.csv"
    )

    lai_errors = grouped_error_analysis(
        prediction_df, ["LAI_Range"], "phase04a_lai_range_error_analysis.csv"
    )

    method_errors = grouped_error_analysis(
        prediction_df,
        ["Extraction_Method"],
        "phase04a_extraction_method_error_analysis.csv",
    )

    # --------------------------------------------------------
    # SAVE ALL MAIN RESULTS
    # --------------------------------------------------------

    summary_df.to_csv(SUMMARY_FILE, index=False)
    fold_df.to_csv(FOLD_FILE, index=False)
    prediction_df.to_csv(PREDICTION_FILE, index=False)

    importance_df.to_csv(IMPORTANCE_FOLD_FILE, index=False)

    importance_by_dataset.to_csv(IMPORTANCE_SUMMARY_FILE, index=False)

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    with open(REPORT_FILE, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 04A: CORRECTED MAIZE ERROR ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("FOLD METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n\n")

        f.write("CAMPAIGN ERROR ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(campaign_errors.to_string(index=False))
        f.write("\n\n")

        f.write("LAI RANGE ERROR ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(lai_errors.to_string(index=False))
        f.write("\n\n")

        f.write("EXTRACTION METHOD ERROR ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(method_errors.to_string(index=False))
        f.write("\n\n")

        f.write("DATASET-SPECIFIC PERMUTATION IMPORTANCE\n")
        f.write("-" * 50 + "\n")
        f.write(importance_by_dataset.to_string(index=False))
        f.write("\n\n")

        f.write("OUTPUT VALIDATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Expected prediction rows: {expected_prediction_rows}\n")
        f.write(f"Actual prediction rows: {len(prediction_df)}\n")
        f.write(f"Duplicate predictions: {duplicate_predictions}\n")
        f.write("Validation: GroupKFold by Farm_ID\n")
        f.write("Permutation importance: held-out test folds\n")
        f.write(
            "Prediction files contain one OOF prediction per "
            "observation, model and dataset variant.\n"
        )

    # --------------------------------------------------------
    # FINAL TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRECTED MODEL COMPARISON")
    print("=" * 70)

    print(summary_df.sort_values(["Dataset", "MAE"]).to_string(index=False))

    print("\nDATASET-SPECIFIC PERMUTATION IMPORTANCE")
    print(importance_by_dataset.to_string(index=False))

    print("\nOUTPUT VALIDATION")
    print(f"Expected prediction rows: {expected_prediction_rows}")
    print(f"Actual prediction rows: {len(prediction_df)}")
    print(f"Duplicate predictions: {duplicate_predictions}")

    print("\nOUTPUT FILES")
    for path in [
        SUMMARY_FILE,
        FOLD_FILE,
        PREDICTION_FILE,
        IMPORTANCE_FOLD_FILE,
        IMPORTANCE_SUMMARY_FILE,
        REPORT_DIR / "phase04a_campaign_error_analysis.csv",
        REPORT_DIR / "phase04a_lai_range_error_analysis.csv",
        REPORT_DIR / "phase04a_extraction_method_error_analysis.csv",
        REPORT_FILE,
    ]:
        print(path)

    print("\n" + "=" * 70)
    print("PHASE 04A COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
