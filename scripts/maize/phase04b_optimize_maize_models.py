from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 04B: NESTED CV AND TARGETED MODEL OPTIMIZATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "maize" / "processed" / "maize_final_satellite_features.csv"

REPORT_DIR = ROOT / "outputs" / "maize" / "reports"
MODEL_DIR = ROOT / "outputs" / "maize" / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "LAI"
GROUP = "Farm_ID"

FULL_FEATURES = ["NDVI", "EVI", "GNDVI", "SAVI", "NDRE", "RVI", "DVI", "NDMI", "BNDVI"]

# Selected from the earlier Phase 04A exploratory analysis.
REDUCED_FEATURES = ["NDMI", "BNDVI", "EVI"]

OUTER_SPLITS = 5
INNER_SPLITS = 4

RANDOM_STATE = 42

# ------------------------------------------------------------
# HYPERPARAMETER GRIDS
# ------------------------------------------------------------

RIDGE_GRID = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

RF_GRID = {
    "max_depth": [None, 8],
    "min_samples_leaf": [2, 4],
    "max_features": [0.7, 1.0],
}

N_ESTIMATORS = 400

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

SUMMARY_FILE = REPORT_DIR / "phase04b_nested_model_comparison.csv"

FOLD_FILE = REPORT_DIR / "phase04b_outer_fold_metrics.csv"

PREDICTION_FILE = REPORT_DIR / "phase04b_oof_predictions.csv"

PARAMETER_FILE = REPORT_DIR / "phase04b_best_parameters_by_fold.csv"

REPORT_FILE = REPORT_DIR / "phase04b_optimization_report.txt"


# ============================================================
# HELPERS
# ============================================================


def parse_boolean(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def make_rf():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1
    )


def make_ridge():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge())])


# ============================================================
# LOAD AND FILTER DATA
# ============================================================


def load_spatial_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    data = pd.read_csv(DATA_FILE)

    required = FULL_FEATURES + [
        TARGET,
        GROUP,
        "GPS_Inside_Matched_Parcel",
        "campaign_id",
        "Satellite_Source",
        "Extraction_Method",
        "Satellite_Image_Date",
        "Absolute_Image_Date_Difference_Days",
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use only observations inside their matching parcels.
    data = data[parse_boolean(data["GPS_Inside_Matched_Parcel"])].copy()

    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    for feature in FULL_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=[TARGET, GROUP] + FULL_FEATURES).copy()

    data = data.reset_index(drop=True)

    # Stable identifier for tracking predictions.
    data["Observation_ID"] = [f"MAIZE_SPATIAL_{i + 1:04d}" for i in range(len(data))]

    return data


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================


def main():

    print("=" * 70)
    print("PHASE 04B: NESTED MAIZE MODEL OPTIMIZATION")
    print("=" * 70)

    data = load_spatial_dataset()

    print(f"\nSpatially reliable observations: {len(data)}")
    print(f"Unique farms: {data[GROUP].nunique()}")
    print(f"Full feature count: {len(FULL_FEATURES)}")
    print(f"Reduced feature count: {len(REDUCED_FEATURES)}")

    X_full = data[FULL_FEATURES]
    X_reduced = data[REDUCED_FEATURES]

    y = data[TARGET]
    groups = data[GROUP].astype(str)

    if groups.nunique() < OUTER_SPLITS:
        raise ValueError("Not enough unique farms for outer GroupKFold.")

    outer_cv = GroupKFold(n_splits=OUTER_SPLITS)

    model_names = [
        "Ridge_Full",
        "RandomForest_Full",
        "RandomForest_Reduced",
        "Dummy_Mean",
    ]

    prediction_arrays = {name: np.full(len(data), np.nan) for name in model_names}

    fold_records = []
    parameter_records = []

    # ========================================================
    # OUTER CV
    # ========================================================

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_full, y, groups), start=1
    ):
        print("\n" + "=" * 70)
        print(f"OUTER FOLD {outer_fold}/{OUTER_SPLITS}")
        print("=" * 70)

        X_train_full = X_full.iloc[train_idx]
        X_test_full = X_full.iloc[test_idx]

        X_train_reduced = X_reduced.iloc[train_idx]
        X_test_reduced = X_reduced.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = groups.iloc[train_idx]
        test_groups = groups.iloc[test_idx]

        if set(train_groups).intersection(set(test_groups)):
            raise RuntimeError("Farm leakage detected in outer fold.")

        print(f"Training observations: {len(train_idx)}")
        print(f"Testing observations: {len(test_idx)}")
        print(f"Training farms: {train_groups.nunique()}")
        print(f"Testing farms: {test_groups.nunique()}")

        inner_splits = min(INNER_SPLITS, train_groups.nunique())

        inner_cv = GroupKFold(n_splits=inner_splits)

        # ----------------------------------------------------
        # 1. RIDGE - FULL FEATURES
        # ----------------------------------------------------

        print("\nTuning Ridge Regression...")

        ridge_search = GridSearchCV(
            estimator=make_ridge(),
            param_grid=RIDGE_GRID,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        ridge_search.fit(X_train_full, y_train, groups=train_groups)

        ridge_prediction = ridge_search.predict(X_test_full)

        prediction_arrays["Ridge_Full"][test_idx] = ridge_prediction

        print(f"Best Ridge parameters: {ridge_search.best_params_}")

        parameter_records.append(
            {
                "Outer_Fold": outer_fold,
                "Model": "Ridge_Full",
                "Best_Parameters": json.dumps(ridge_search.best_params_),
                "Inner_CV_MAE": -ridge_search.best_score_,
            }
        )

        # ----------------------------------------------------
        # 2. RANDOM FOREST - FULL FEATURES
        # ----------------------------------------------------

        print("\nTuning Random Forest with all features...")

        rf_full_search = GridSearchCV(
            estimator=make_rf(),
            param_grid=RF_GRID,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        rf_full_search.fit(X_train_full, y_train, groups=train_groups)

        rf_full_prediction = rf_full_search.predict(X_test_full)

        prediction_arrays["RandomForest_Full"][test_idx] = rf_full_prediction

        print(f"Best full RF parameters: {rf_full_search.best_params_}")

        parameter_records.append(
            {
                "Outer_Fold": outer_fold,
                "Model": "RandomForest_Full",
                "Best_Parameters": json.dumps(rf_full_search.best_params_),
                "Inner_CV_MAE": -rf_full_search.best_score_,
            }
        )

        # ----------------------------------------------------
        # 3. RANDOM FOREST - REDUCED FEATURES
        # ----------------------------------------------------

        print("\nTuning Random Forest with three features...")

        rf_reduced_search = GridSearchCV(
            estimator=make_rf(),
            param_grid=RF_GRID,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        rf_reduced_search.fit(X_train_reduced, y_train, groups=train_groups)

        rf_reduced_prediction = rf_reduced_search.predict(X_test_reduced)

        prediction_arrays["RandomForest_Reduced"][test_idx] = rf_reduced_prediction

        print(f"Best reduced RF parameters: {rf_reduced_search.best_params_}")

        parameter_records.append(
            {
                "Outer_Fold": outer_fold,
                "Model": "RandomForest_Reduced",
                "Best_Parameters": json.dumps(rf_reduced_search.best_params_),
                "Inner_CV_MAE": -rf_reduced_search.best_score_,
            }
        )

        # ----------------------------------------------------
        # 4. DUMMY MEAN BASELINE
        # ----------------------------------------------------

        dummy = DummyRegressor(strategy="mean")

        dummy.fit(X_train_full, y_train)

        dummy_prediction = dummy.predict(X_test_full)

        prediction_arrays["Dummy_Mean"][test_idx] = dummy_prediction

        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        fold_predictions = {
            "Ridge_Full": ridge_prediction,
            "RandomForest_Full": rf_full_prediction,
            "RandomForest_Reduced": rf_reduced_prediction,
            "Dummy_Mean": dummy_prediction,
        }

        for model_name, predictions in fold_predictions.items():
            metrics = calculate_metrics(y_test.to_numpy(), predictions)

            fold_records.append(
                {
                    "Outer_Fold": outer_fold,
                    "Model": model_name,
                    "Train_Observations": len(train_idx),
                    "Test_Observations": len(test_idx),
                    "Train_Farms": train_groups.nunique(),
                    "Test_Farms": test_groups.nunique(),
                    **metrics,
                }
            )

            print(
                f"{model_name}: "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}"
            )

    # ========================================================
    # AGGREGATE OUTER OOF RESULTS
    # ========================================================

    prediction_records = []
    summary_records = []

    for model_name, predictions in prediction_arrays.items():
        if np.isnan(predictions).any():
            raise RuntimeError(f"Incomplete predictions for {model_name}")

        metrics = calculate_metrics(y.to_numpy(), predictions)

        summary_records.append(
            {
                "Model": model_name,
                "Observations": len(data),
                "Unique_Farms": groups.nunique(),
                "Outer_Folds": OUTER_SPLITS,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

        prediction_records.append(
            pd.DataFrame(
                {
                    "Observation_ID": data["Observation_ID"],
                    "Model": model_name,
                    "Farm_ID": groups,
                    "Campaign": data["campaign_id"],
                    "Actual_LAI": y,
                    "Predicted_LAI": predictions,
                    "Prediction_Error": predictions - y.to_numpy(),
                    "Absolute_Error": np.abs(predictions - y.to_numpy()),
                    "Extraction_Method": data["Extraction_Method"],
                    "Satellite_Source": data["Satellite_Source"],
                    "Satellite_Image_Date": data["Satellite_Image_Date"],
                    "Image_Date_Difference_Days": (
                        data["Absolute_Image_Date_Difference_Days"]
                    ),
                }
            )
        )

    summary_df = pd.DataFrame(summary_records)
    fold_df = pd.DataFrame(fold_records)
    parameter_df = pd.DataFrame(parameter_records)

    prediction_df = pd.concat(prediction_records, ignore_index=True)

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    expected_rows = len(data) * len(model_names)

    duplicates = prediction_df.duplicated(subset=["Observation_ID", "Model"]).sum()

    if len(prediction_df) != expected_rows:
        raise RuntimeError("Prediction row count validation failed.")

    if duplicates:
        raise RuntimeError(f"Duplicate predictions detected: {duplicates}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary_df.to_csv(SUMMARY_FILE, index=False)

    fold_df.to_csv(FOLD_FILE, index=False)

    prediction_df.to_csv(PREDICTION_FILE, index=False)

    parameter_df.to_csv(PARAMETER_FILE, index=False)

    with open(REPORT_FILE, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 04B: NESTED MAIZE MODEL OPTIMIZATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET\n")
        f.write("-" * 50 + "\n")
        f.write(f"Observations: {len(data)}\n")
        f.write(f"Unique farms: {groups.nunique()}\n\n")

        f.write("VALIDATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Outer GroupKFold: {OUTER_SPLITS}\n")
        f.write(f"Inner GroupKFold: maximum {INNER_SPLITS}\n")
        f.write("Grouping: Farm_ID\n")
        f.write("Hyperparameter selection: inner CV MAE\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("OUTER FOLD METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n\n")

        f.write("BEST PARAMETERS BY OUTER FOLD\n")
        f.write("-" * 50 + "\n")
        f.write(parameter_df.to_string(index=False))
        f.write("\n\n")

        f.write("IMPORTANT METHODOLOGICAL NOTES\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Outer test farms are excluded from all inner hyperparameter selection.\n"
        )
        f.write("All reported predictions are outer-fold out-of-fold predictions.\n")
        f.write(
            "Reduced features were selected based on earlier "
            "exploratory Phase 04A results. Therefore, this "
            "comparison is exploratory, not an independent "
            "confirmation of feature selection.\n"
        )
        f.write(
            "Results apply to the spatially reliable dataset "
            "and do not establish external geographic or "
            "seasonal generalization.\n"
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 04B: FINAL NESTED CV COMPARISON")
    print("=" * 70)

    print(summary_df.sort_values("MAE").to_string(index=False))

    print("\nOUTPUT VALIDATION")
    print(f"Expected prediction rows: {expected_rows}")
    print(f"Actual prediction rows: {len(prediction_df)}")
    print(f"Duplicate predictions: {duplicates}")

    print("\nOUTPUT FILES")
    print(SUMMARY_FILE)
    print(FOLD_FILE)
    print(PREDICTION_FILE)
    print(PARAMETER_FILE)
    print(REPORT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 04B COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
