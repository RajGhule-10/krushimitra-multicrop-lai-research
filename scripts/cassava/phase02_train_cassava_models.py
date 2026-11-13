from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PHASE 02: CASSAVA BASELINE MODEL TRAINING
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

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET, GROUP, "MAP", "Plot_type"]

missing_columns = [column for column in required_columns if column not in data.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

for column in FEATURES + [TARGET]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna(subset=FEATURES + [TARGET, GROUP, "MAP", "Plot_type"]).reset_index(
    drop=True
)

X = data[FEATURES]
y = data[TARGET]
groups = data[GROUP]

if groups.nunique() < N_SPLITS:
    raise ValueError(
        f"Only {groups.nunique()} plots available for {N_SPLITS}-fold GroupKFold."
    )

print("\nCASSAVA BASELINE MODEL TRAINING")
print("=" * 60)

print("\nDATASET")
print("Rows:", len(data))
print("Unique plots:", groups.nunique())
print("MAP periods:", sorted(data["MAP"].unique().tolist()))
print("Features:", len(FEATURES))
print("LAI mean:", round(y.mean(), 4))
print("LAI standard deviation:", round(y.std(), 4))

print("\nPLOT TYPES")
print(data["Plot_type"].value_counts().to_string())


# ============================================================
# 2. DEFINE CROSS-VALIDATION
# ============================================================

cv = GroupKFold(n_splits=N_SPLITS)

# Show the number of held-out plots in each fold.
print("\nGROUPED VALIDATION FOLDS")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    train_plots = groups.iloc[train_idx].nunique()
    test_plots = groups.iloc[test_idx].nunique()

    print(f"Fold {fold}: train plots={train_plots}, test plots={test_plots}")


# ============================================================
# 3. DEFINE MODELS
# ============================================================

models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# Add XGBoost only if it is installed in the active environment.
if importlib.util.find_spec("xgboost") is not None:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
else:
    print(
        "\nXGBoost is not installed. It will be skipped; other models will still run."
    )


# ============================================================
# 4. TRAIN AND GENERATE OUT-OF-FOLD PREDICTIONS
# ============================================================

predictions = data[[GROUP, "MAP", "Plot_type", TARGET]].copy()

metrics_rows = []

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    # Every prediction is generated for a plot excluded
    # entirely from that fold's training data.
    predicted = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=1,
        method="predict",
    )

    predictions[model_name.replace(" ", "_") + "_Predicted_LAI"] = predicted

    mae = mean_absolute_error(y, predicted)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    r2 = r2_score(y, predicted)

    metrics_rows.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")


# ============================================================
# 5. SAVE MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(metrics_rows)
comparison = comparison.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

comparison_path = OUTPUT_DIR / "phase02_cassava_model_comparison.csv"

comparison.to_csv(comparison_path, index=False)


# ============================================================
# 6. SAVE OUT-OF-FOLD PREDICTIONS
# ============================================================

predictions_path = OUTPUT_DIR / "phase02_cassava_oof_predictions.csv"

predictions.to_csv(predictions_path, index=False)


# ============================================================
# 7. CAMPAIGN / GROWTH-PERIOD ERROR ANALYSIS
# ============================================================

best_model_name = comparison.iloc[0]["Model"]

best_prediction_column = best_model_name.replace(" ", "_") + "_Predicted_LAI"

predictions["Best_Model_Absolute_Error"] = (
    predictions[TARGET] - predictions[best_prediction_column]
).abs()

map_analysis = (
    predictions.groupby("MAP")
    .agg(
        Observations=(TARGET, "count"),
        Mean_Actual_LAI=(TARGET, "mean"),
        MAE=("Best_Model_Absolute_Error", "mean"),
    )
    .reset_index()
)

map_path = OUTPUT_DIR / "phase02_cassava_map_error_analysis.csv"
map_analysis.to_csv(map_path, index=False)

print("\nERROR BY MAP")
print(map_analysis.to_string(index=False))


# ============================================================
# 8. MEASUREMENT-TYPE ERROR ANALYSIS
# ============================================================

type_analysis = (
    predictions.groupby("Plot_type")
    .agg(
        Observations=(TARGET, "count"),
        Mean_Actual_LAI=(TARGET, "mean"),
        MAE=("Best_Model_Absolute_Error", "mean"),
    )
    .reset_index()
)

type_path = OUTPUT_DIR / "phase02_cassava_plot_type_error_analysis.csv"

type_analysis.to_csv(type_path, index=False)

print("\nERROR BY PLOT TYPE")
print(type_analysis.to_string(index=False))


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\nFILES SAVED")
print(comparison_path)
print(predictions_path)
print(map_path)
print(type_path)

print("\nBEST MODEL BY RMSE:", best_model_name)

print("\nPhase 02 completed successfully.")
