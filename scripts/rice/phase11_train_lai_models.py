from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PHASE 11: RICE LAI MODEL TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "rice" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "rice" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

FILES = [
    ("campaign_1", "final_df_rice_campaign_1.csv"),
    ("campaign_2", "final_df_rice_campaign_2.csv"),
    ("campaign_3", "final_df_rice_campaign_3.csv"),
]


def load_dataset():

    frames = []

    for campaign, filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df["Campaign"] = campaign

        frames.append(df)

        print(f"{campaign}: {len(df)} records loaded")

    data = pd.concat(frames, ignore_index=True)

    required = ["Farm_ID", TARGET] + FEATURES

    missing_columns = [col for col in required if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[required + ["Campaign"]].copy()

    for col in [TARGET] + FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    before = len(data)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    print("\nDATASET CHECK")
    print("Rows before cleaning:", before)
    print("Rows used:", len(data))
    print("Unique farms:", data["Farm_ID"].nunique())
    print("Campaigns:", data["Campaign"].nunique())

    if data["Farm_ID"].nunique() < 5:
        raise ValueError("Insufficient unique farms for 5-fold CV.")

    return data


def evaluate_model(name, model, X, y, groups, cv):

    print(f"\n{'=' * 65}")
    print("MODEL:", name)
    print("=" * 65)

    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return predictions, {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():

    data = load_dataset()

    X = data[FEATURES]
    y = data[TARGET]
    groups = data["Farm_ID"]

    cv = GroupKFold(n_splits=5)

    print("\nVALIDATION")
    print("Method: 5-fold GroupKFold")
    print("Grouping: Farm_ID")
    print("Features:", len(FEATURES))

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge Regression": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ),
    }

    results = []
    prediction_data = data[["Farm_ID", "Campaign", TARGET]].copy()

    for name, model in models.items():
        predictions, metrics = evaluate_model(name, model, X, y, groups, cv)

        results.append(metrics)

        prediction_data[f"{name.replace(' ', '_')}_Predicted_LAI"] = predictions

    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "phase11_model_comparison.csv"
    predictions_path = OUTPUT_DIR / "phase11_oof_predictions.csv"

    results_df.to_csv(results_path, index=False)
    prediction_data.to_csv(predictions_path, index=False)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(results_path)
    print(predictions_path)

    print("\nPHASE 11 COMPLETED")


if __name__ == "__main__":
    main()
