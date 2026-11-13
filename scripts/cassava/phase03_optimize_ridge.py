from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PHASE 03: CASSAVA RIDGE OPTIMIZATION + FEATURE IMPORTANCE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "cassava" / "processed" / "cassava_lai_s2_matched.csv"

OUTPUT_DIR = ROOT / "outputs" / "cassava" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "LAI"
GROUP = "Plot_ID"

FEATURES = [
    "BNDVI",
    "CIG",
    "DVI",
    "EVI",
    "GNDVI",
    "GRVI",
    "NDVI",
    "NDWI",
    "RVI",
    "SAVI",
    "SeLI",
    "TCARI",
    "VIG",
]

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_STATE = 42
PERMUTATION_REPEATS = 20


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET, GROUP, "MAP"]

missing = [column for column in required_columns if column not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for column in FEATURES + [TARGET]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna(subset=FEATURES + [TARGET, GROUP]).reset_index(drop=True)

X = data[FEATURES]
y = data[TARGET]
groups = data[GROUP]

if groups.nunique() < OUTER_SPLITS:
    raise ValueError("Not enough independent plots for outer CV.")

print("\nCASSAVA PHASE 03")
print("=" * 60)
print("Rows:", len(data))
print("Unique plots:", groups.nunique())
print("Features:", len(FEATURES))
print("Alpha candidates:", ALPHAS)


# ============================================================
# 2. DEFINE VALIDATION AND MODEL
# ============================================================

outer_cv = GroupKFold(n_splits=OUTER_SPLITS)
inner_cv = GroupKFold(n_splits=INNER_SPLITS)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ]
)

parameter_grid = {"ridge__alpha": ALPHAS}


# ============================================================
# 3. PREPARE RESULT STORAGE
# ============================================================

fixed_predictions = np.full(len(data), np.nan)
tuned_predictions = np.full(len(data), np.nan)

selected_alphas = []
importance_results = []


# ============================================================
# 4. NESTED GROUP CROSS-VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    print(f"\nOUTER FOLD {fold}")
    print("-" * 45)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    # --------------------------------------------------------
    # A. FIXED RIDGE: alpha = 1.0
    # --------------------------------------------------------

    fixed_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    fixed_model.fit(X_train, y_train)

    fixed_predictions[test_idx] = fixed_model.predict(X_test)

    # --------------------------------------------------------
    # B. TUNED RIDGE: INNER GROUP CROSS-VALIDATION
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    best_model = search.best_estimator_

    tuned_predictions[test_idx] = best_model.predict(X_test)

    best_alpha = search.best_params_["ridge__alpha"]
    selected_alphas.append(best_alpha)

    print("Training plots:", groups_train.nunique())
    print("Testing plots:", groups.iloc[test_idx].nunique())
    print("Selected alpha:", best_alpha)

    # --------------------------------------------------------
    # C. PERMUTATION IMPORTANCE ON HELD-OUT PLOTS
    # --------------------------------------------------------

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    for index, feature in enumerate(FEATURES):
        importance_results.append(
            {
                "Outer_Fold": fold,
                "Feature": feature,
                "Importance_Mean": importance.importances_mean[index],
                "Importance_SD": importance.importances_std[index],
            }
        )


# ============================================================
# 5. CALCULATE MODEL METRICS
# ============================================================


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


fixed_metrics = calculate_metrics(y, fixed_predictions)
tuned_metrics = calculate_metrics(y, tuned_predictions)

comparison = pd.DataFrame(
    [
        {
            "Model": "Ridge alpha=1.0 (fixed)",
            **fixed_metrics,
        },
        {
            "Model": "Ridge (nested CV tuned)",
            **tuned_metrics,
        },
    ]
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))


# ============================================================
# 6. SELECTED ALPHA SUMMARY
# ============================================================

alpha_table = pd.DataFrame(
    {
        "Outer_Fold": range(1, OUTER_SPLITS + 1),
        "Selected_Alpha": selected_alphas,
    }
)

print("\nSELECTED ALPHA VALUES")
print(alpha_table.to_string(index=False))

print("\nALPHA FREQUENCY")
print(alpha_table["Selected_Alpha"].value_counts().sort_index().to_string())


# ============================================================
# 7. AGGREGATE PERMUTATION IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(importance_results)

importance_summary = (
    importance_df.groupby("Feature")
    .agg(
        Mean_Importance=("Importance_Mean", "mean"),
        Mean_Fold_SD=("Importance_SD", "mean"),
        Fold_Stability=("Importance_Mean", "std"),
    )
    .reset_index()
    .sort_values("Mean_Importance", ascending=False)
    .reset_index(drop=True)
)

print("\nPERMUTATION FEATURE IMPORTANCE")
print(importance_summary.to_string(index=False))


# ============================================================
# 8. SAVE PREDICTIONS AND RESULTS
# ============================================================

predictions = data[[GROUP, "MAP", TARGET]].copy()

predictions["Fixed_Ridge_Predicted_LAI"] = fixed_predictions
predictions["Tuned_Ridge_Predicted_LAI"] = tuned_predictions

comparison_path = OUTPUT_DIR / "phase03_cassava_ridge_comparison.csv"

predictions_path = OUTPUT_DIR / "phase03_cassava_ridge_predictions.csv"

alpha_path = OUTPUT_DIR / "phase03_cassava_selected_alphas.csv"

importance_path = OUTPUT_DIR / "phase03_cassava_feature_importance.csv"

fold_importance_path = OUTPUT_DIR / "phase03_cassava_feature_importance_by_fold.csv"

comparison.to_csv(comparison_path, index=False)
predictions.to_csv(predictions_path, index=False)
alpha_table.to_csv(alpha_path, index=False)
importance_summary.to_csv(importance_path, index=False)
importance_df.to_csv(fold_importance_path, index=False)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\nFILES SAVED")
print(comparison_path)
print(predictions_path)
print(alpha_path)
print(importance_path)
print(fold_importance_path)

print("\nPHASE 03 COMPLETED SUCCESSFULLY.")
