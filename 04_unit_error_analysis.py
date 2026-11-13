from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = PROJECT_DIR / "outputs" / "lai_out_of_fold_predictions.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs"


# ==================================================
# 2. LOAD PREDICTIONS
# ==================================================

df = pd.read_csv(PREDICTIONS_PATH)


# ==================================================
# 3. FILTER MULTISPECTRAL RANDOM FOREST
# ==================================================

df = df[
    (df["Experiment"] == "Bands_and_indices") & (df["Model"] == "Random_Forest")
].copy()


if df.empty:
    raise ValueError(
        "No multispectral Random Forest predictions found. "
        "Run 02_lai_model_comparison.py first."
    )


# ==================================================
# 4. CALCULATE ERRORS
# ==================================================

df["Residual"] = df["Actual_LAI"] - df["Predicted_LAI"]

df["Absolute_Error"] = df["Residual"].abs()

df["Squared_Error"] = df["Residual"] ** 2


# ==================================================
# 5. SUMMARIZE BY UNIT
# ==================================================

summary = (
    df.groupby("Unit_ID")
    .agg(
        Observations=("ESU_ID", "count"),
        Mean_Actual_LAI=("Actual_LAI", "mean"),
        Mean_Predicted_LAI=("Predicted_LAI", "mean"),
        MAE=("Absolute_Error", "mean"),
        Mean_Squared_Error=("Squared_Error", "mean"),
    )
    .reset_index()
)

summary["RMSE"] = np.sqrt(summary["Mean_Squared_Error"])

summary = summary.drop(columns=["Mean_Squared_Error"])


# ==================================================
# 6. PRINT AND SAVE
# ==================================================

print("\n========== ERROR BY FIELD UNIT ==========")

print(summary.round(4).to_string(index=False))

output_path = OUTPUT_DIR / "lai_error_by_unit.csv"

summary.to_csv(output_path, index=False)

print("\nSaved:", output_path)
