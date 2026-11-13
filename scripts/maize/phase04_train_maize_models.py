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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 04: MAIZE BASELINE MODEL TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "maize" / "processed" / "maize_final_satellite_features.csv"

MODEL_DIR = ROOT / "outputs" / "maize" / "models"
REPORT_DIR = ROOT / "outputs" / "maize" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Nine non-duplicate vegetation indices.
FEATURES = ["NDVI", "EVI", "GNDVI", "SAVI", "NDRE", "RVI", "DVI", "NDMI", "BNDVI"]

TARGET = "LAI"
GROUP = "Farm_ID"

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# METRICS
# ============================================================


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
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


# ============================================================
# DATASET VARIANTS
# ============================================================


def prepare_dataset_variants(data):
    variants = {}

    # 1. Full dataset
    variants["Full_316"] = data.copy()

    # 2. Spatially reliable observations:
    # GPS must fall inside its matching farm parcel.
    reliable = data[data["GPS_Inside_Matched_Parcel"] == True].copy()

    variants["Spatial_Reliable"] = reliable

    # 3. Original Phase 03 extraction window only.
    original = data[data["Satellite_Source"].astype(str) == "Phase 03"].copy()

    variants["Original_Phase03"] = original

    return variants


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================


def evaluate_dataset(dataset_name, data):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    data = data.copy()

    # Keep only rows with valid target, group and features.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=[TARGET, GROUP] + FEATURES).copy()

    X = data[FEATURES]
    y = data[TARGET]
    groups = data[GROUP].astype(str)

    unique_groups = groups.nunique()

    print(f"Observations: {len(data)}")
    print(f"Unique farms: {unique_groups}")
    print(f"Features: {len(FEATURES)}")

    if unique_groups < 2:
        raise ValueError(f"{dataset_name} has fewer than 2 farms.")

    if len(data) < 5:
        raise ValueError(f"{dataset_name} has insufficient observations.")

    n_splits = min(N_SPLITS, unique_groups)

    cv = GroupKFold(n_splits=n_splits)

    models = build_models()

    all_predictions = []
    fold_results = []
    feature_importance_results = []

    # --------------------------------------------------------
    # TRAIN EACH MODEL
    # --------------------------------------------------------

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        oof_predictions = np.full(len(data), np.nan)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            # Explicit leakage check.
            overlap = train_groups.intersection(test_groups)

            if overlap:
                raise RuntimeError(f"Farm leakage detected: {overlap}")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            oof_predictions[test_idx] = predictions

            fold_metrics = calculate_metrics(y_test, predictions)

            fold_results.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold,
                    "Train_Observations": len(train_idx),
                    "Test_Observations": len(test_idx),
                    "Train_Farms": len(train_groups),
                    "Test_Farms": len(test_groups),
                    **fold_metrics,
                }
            )

            # Collect Random Forest feature importance.
            if model_name == "Random Forest":
                feature_importance_results.append(
                    pd.DataFrame(
                        {
                            "Dataset": dataset_name,
                            "Fold": fold,
                            "Feature": FEATURES,
                            "Importance": model.feature_importances_,
                        }
                    )
                )

        if np.isnan(oof_predictions).any():
            raise RuntimeError(f"Missing out-of-fold predictions for {model_name}")

        # Overall metrics from all out-of-fold predictions.
        overall_metrics = calculate_metrics(y.to_numpy(), oof_predictions)

        print(
            f"MAE:  {overall_metrics['MAE']:.6f}\n"
            f"RMSE: {overall_metrics['RMSE']:.6f}\n"
            f"R2:   {overall_metrics['R2']:.6f}"
        )

        summary = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Observations": len(data),
            "Unique_Farms": unique_groups,
            "CV_Folds": n_splits,
            **overall_metrics,
        }

        all_predictions.append(
            pd.DataFrame(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Farm_ID": groups.to_numpy(),
                    "Campaign": data["campaign_id"].to_numpy(),
                    "Actual_LAI": y.to_numpy(),
                    "Predicted_LAI": oof_predictions,
                    "Extraction_Method": (data["Extraction_Method"].to_numpy()),
                    "Satellite_Source": (data["Satellite_Source"].to_numpy()),
                }
            )
        )

        yield summary, fold_results, all_predictions, feature_importance_results


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("PHASE 04: MAIZE BASELINE MODEL TRAINING")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    print(f"\nLoaded observations: {len(data)}")

    required_columns = FEATURES + [
        TARGET,
        GROUP,
        "Satellite_Source",
        "GPS_Inside_Matched_Parcel",
        "campaign_id",
        "Extraction_Method",
    ]

    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Unique farms in complete dataset: {data[GROUP].nunique()}")

    variants = prepare_dataset_variants(data)

    summary_results = []
    all_fold_results = []
    all_prediction_results = []
    all_importance_results = []

    for dataset_name, dataset in variants.items():
        if dataset.empty:
            print(f"\nSkipping {dataset_name}: no observations.")
            continue

        for summary, fold_results, predictions, importance in evaluate_dataset(
            dataset_name, dataset
        ):
            summary_results.append(summary)

            # Keep only the folds generated in this run.
            all_fold_results.extend(
                r for r in fold_results if r["Dataset"] == dataset_name
            )

            all_prediction_results.extend(predictions)

            all_importance_results.extend(importance)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_results)
    folds_df = pd.DataFrame(all_fold_results)

    predictions_df = pd.concat(all_prediction_results, ignore_index=True)

    importance_df = (
        pd.concat(all_importance_results, ignore_index=True)
        if all_importance_results
        else pd.DataFrame()
    )

    summary_file = REPORT_DIR / "phase04_maize_model_comparison.csv"

    folds_file = REPORT_DIR / "phase04_maize_fold_metrics.csv"

    predictions_file = REPORT_DIR / "phase04_maize_oof_predictions.csv"

    importance_file = REPORT_DIR / "phase04_maize_rf_feature_importance.csv"

    text_report = REPORT_DIR / "phase04_maize_model_report.txt"

    summary_df.to_csv(summary_file, index=False)
    folds_df.to_csv(folds_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    if not importance_df.empty:
        importance_df.to_csv(importance_file, index=False)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE SUMMARY
    # --------------------------------------------------------

    importance_summary = pd.DataFrame()

    if not importance_df.empty:
        importance_summary = (
            importance_df.groupby("Feature")["Importance"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )

        importance_summary.to_csv(
            REPORT_DIR / "phase04_maize_rf_feature_importance_summary.csv", index=False
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    with open(text_report, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 04: MAIZE BASELINE MODEL REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("VALIDATION STRATEGY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Cross-validation: GroupKFold, maximum {N_SPLITS} folds\n")
        f.write(f"Grouping variable: {GROUP}\n")
        f.write("Models: Ridge, Random Forest, Dummy Mean\n")
        f.write(f"Features: {FEATURES}\n\n")

        f.write("OVERALL MODEL COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("FOLD METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(folds_df.to_string(index=False))
        f.write("\n\n")

        if not importance_summary.empty:
            f.write("RANDOM FOREST FEATURE IMPORTANCE\n")
            f.write("-" * 50 + "\n")
            f.write(importance_summary.to_string(index=False))
            f.write("\n\n")

        f.write("IMPORTANT NOTES\n")
        f.write("-" * 50 + "\n")
        f.write("Metrics are based on pooled out-of-fold predictions.\n")
        f.write("GroupKFold keeps observations from the same farm in one fold.\n")
        f.write(
            "NDWI was excluded because it is exactly the negative "
            "of GNDVI in this dataset.\n"
        )
        f.write(
            "The full, spatially reliable, and original-window "
            "datasets are evaluated separately.\n"
        )
        f.write(
            "Results measure within-dataset farm-held-out "
            "performance, not external geographic or seasonal "
            "generalization.\n"
        )
        f.write(
            "Random Forest feature importance is predictive "
            "importance, not causal importance.\n"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 04 MODEL COMPARISON")
    print("=" * 70)

    print(summary_df.sort_values(["Dataset", "MAE"]).to_string(index=False))

    print("\nRandom Forest feature importance:")
    if not importance_summary.empty:
        print(importance_summary.to_string(index=False))

    print("\nOUTPUT FILES")
    print(summary_file)
    print(folds_file)
    print(predictions_file)
    print(importance_file)
    print(text_report)

    print("\n" + "=" * 70)
    print("PHASE 04 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
