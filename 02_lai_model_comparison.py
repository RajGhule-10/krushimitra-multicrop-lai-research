from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "krushimitra_all_campaigns.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. CONFIGURATION
# ==================================================

TARGET = "LAI_mean"

GROUP_COLUMN = "Unit_ID"

RANDOM_STATE = 42

N_SPLITS = 5


# ==================================================
# 3. FEATURE SETS
# ==================================================

bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

indices = ["NDVI", "NDRE", "EVI", "SAVI", "NDMI", "NDWI"]

experiments = {
    "NDVI_only": ["NDVI"],
    "Multispectral_bands": bands,
    "Bands_and_indices": bands + indices,
}


# ==================================================
# 4. MODEL FACTORIES
# ==================================================


def make_random_forest():

    return RandomForestRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_ridge_regression():

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


models = {
    "Random_Forest": make_random_forest,
    "Ridge_Regression": make_ridge_regression,
}


# ==================================================
# 5. LOAD DATA
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\n========== LAI MODEL COMPARISON ==========")

print("Input dataset:", DATA_PATH)

print("Original rows:", len(df))


# ==================================================
# 6. VALIDATE COLUMNS
# ==================================================

all_features = sorted(
    set(feature for feature_list in experiments.values() for feature in feature_list)
)

required_columns = all_features + [TARGET, GROUP_COLUMN, "ESU_ID", "Season", "S2_Date"]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# ==================================================
# 7. CLEAN DATA
# ==================================================

for column in all_features + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# Keep one consistent set of observations across
# all experiments to make the comparison fair.

df = df.dropna(subset=required_columns).copy()

if df.empty:
    raise ValueError("No usable observations remain after cleaning.")

if df["ESU_ID"].duplicated().any():
    raise ValueError(
        "Duplicate ESU_ID rows detected. Investigate duplicates before modelling."
    )

n_groups = df[GROUP_COLUMN].nunique()

if n_groups < N_SPLITS:
    raise ValueError(
        f"Only {n_groups} groups are available, but {N_SPLITS} folds were requested."
    )


# ==================================================
# 8. PREPARE CROSS-VALIDATION
# ==================================================

X_groups = df[GROUP_COLUMN]

y = df[TARGET]

cv = GroupKFold(n_splits=N_SPLITS)

print("\nRows retained:", len(df))

print("Unique ESUs:", df["ESU_ID"].nunique())

print("Unique groups:", n_groups)

print("Cross-validation folds:", N_SPLITS)

print("\nGroup sizes:")

print(df[GROUP_COLUMN].value_counts().sort_index())


# ==================================================
# 9. RUN EXPERIMENTS
# ==================================================

results = []

prediction_tables = []

for experiment_name, feature_columns in experiments.items():
    print("\n" + "=" * 60)

    print("Feature experiment:", experiment_name)

    print("Number of features:", len(feature_columns))

    X = df[feature_columns]

    for model_name, model_factory in models.items():
        model = model_factory()

        predictions = cross_val_predict(model, X, y, groups=X_groups, cv=cv, n_jobs=1)

        r2 = r2_score(y, predictions)

        rmse = np.sqrt(mean_squared_error(y, predictions))

        mae = mean_absolute_error(y, predictions)

        results.append(
            {
                "Target": TARGET,
                "Experiment": experiment_name,
                "Model": model_name,
                "N": len(df),
                "Groups": n_groups,
                "Folds": N_SPLITS,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        prediction_tables.append(
            pd.DataFrame(
                {
                    "ESU_ID": df["ESU_ID"].to_numpy(),
                    "Unit_ID": df[GROUP_COLUMN].to_numpy(),
                    "Season": df["Season"].to_numpy(),
                    "S2_Date": df["S2_Date"].to_numpy(),
                    "Actual_LAI": y.to_numpy(),
                    "Predicted_LAI": predictions,
                    "Residual": y.to_numpy() - predictions,
                    "Experiment": experiment_name,
                    "Model": model_name,
                }
            )
        )

        print(f"\nModel: {model_name}")

        print(f"R²:   {r2:.4f}")

        print(f"RMSE: {rmse:.4f}")

        print(f"MAE:  {mae:.4f}")


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by=["RMSE", "MAE"], ascending=[True, True])

results_path = OUTPUT_DIR / "lai_model_comparison.csv"

results_df.to_csv(results_path, index=False)

predictions_df = pd.concat(prediction_tables, ignore_index=True)

predictions_path = OUTPUT_DIR / "lai_out_of_fold_predictions.csv"

predictions_df.to_csv(predictions_path, index=False)


# ==================================================
# 11. PRINT COMPARISON
# ==================================================

print("\n========== FINAL MODEL COMPARISON ==========")

print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

print("\nResults saved to:")

print(results_path)

print(predictions_path)

print("\nExperiment completed.")
