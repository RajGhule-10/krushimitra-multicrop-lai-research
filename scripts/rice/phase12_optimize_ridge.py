from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "rice" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "rice" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMPAIGN_FILES = [
    "final_df_rice_campaign_1.csv",
    "final_df_rice_campaign_2.csv",
    "final_df_rice_campaign_3.csv",
]

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
GROUP = "Farm_ID"

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

frames = []

for filename in CAMPAIGN_FILES:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    df["Campaign"] = filename.replace("final_df_rice_", "").replace(".csv", "")
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

required_columns = FEATURES + [TARGET, GROUP, "Campaign"]
missing_columns = [c for c in required_columns if c not in data.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

data = data[required_columns].copy()

for column in FEATURES + [TARGET]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=FEATURES + [TARGET, GROUP]).reset_index(drop=True)

X = data[FEATURES]
y = data[TARGET]
groups = data[GROUP]

print("\nDATASET")
print("Rows:", len(data))
print("Unique farms:", groups.nunique())
print("Campaigns:", data["Campaign"].nunique())
print("Features:", len(FEATURES))


# --------------------------------------------------
# 3. DEFINE MODEL AND VALIDATION
# --------------------------------------------------

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=4)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ]
)

parameter_grid = {"ridge__alpha": ALPHAS}

# Store predictions from both fixed and tuned Ridge models.
fixed_predictions = np.full(len(data), np.nan)
tuned_predictions = np.full(len(data), np.nan)

selected_alphas = []


# --------------------------------------------------
# 4. NESTED GROUP CROSS-VALIDATION
# --------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    # Fixed Ridge: alpha = 1.0, matching Phase 11.
    fixed_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    fixed_model.fit(X_train, y_train)
    fixed_predictions[test_idx] = fixed_model.predict(X_test)

    # Tune alpha using only the outer training farms.
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train, groups=groups_train)

    tuned_predictions[test_idx] = search.predict(X_test)
    selected_alphas.append(search.best_params_["ridge__alpha"])

    print(f"Fold {fold}: best alpha = {search.best_params_['ridge__alpha']}")


# --------------------------------------------------
# 5. EVALUATE OUTER-FOLD PREDICTIONS
# --------------------------------------------------


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

print("\nPHASE 12 MODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nSELECTED ALPHA VALUES")
print(pd.Series(selected_alphas).value_counts().sort_index())


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

predictions = data[[GROUP, "Campaign", TARGET]].copy()
predictions["Fixed_Ridge_Predicted_LAI"] = fixed_predictions
predictions["Tuned_Ridge_Predicted_LAI"] = tuned_predictions

comparison_path = OUTPUT_DIR / "phase12_ridge_comparison.csv"
predictions_path = OUTPUT_DIR / "phase12_ridge_predictions.csv"
alpha_path = OUTPUT_DIR / "phase12_selected_alphas.csv"

comparison.to_csv(comparison_path, index=False)
predictions.to_csv(predictions_path, index=False)

pd.DataFrame(
    {
        "Outer_Fold": range(1, len(selected_alphas) + 1),
        "Selected_Alpha": selected_alphas,
    }
).to_csv(alpha_path, index=False)

print("\nFILES SAVED")
print(comparison_path)
print(predictions_path)
print(alpha_path)

print("\nPhase 12 completed.")
