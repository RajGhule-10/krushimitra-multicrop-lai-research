"""Crop-aware result adapters. Source files remain untouched."""

from __future__ import annotations

import pandas as pd

from config.crops import CropConfig, get_crop
from services.crop_data_service import read_csv


def model_results(crop_id: str) -> pd.DataFrame:
    config = get_crop(crop_id)
    rows: list[pd.DataFrame] = []
    for experiment, path, protocol in config.model_files:
        frame = read_csv(path)
        if frame.empty:
            continue
        result = frame.copy()
        rename = {"Observations": "N", "Unique_Farms": "Groups", "Outer_Folds": "Folds"}
        result = result.rename(columns=rename)
        for metric in ("R2", "RMSE", "MAE", "N", "Groups", "Folds"):
            if metric in result:
                result[metric] = pd.to_numeric(result[metric], errors="coerce")
        if "Experiment" in result:
            result["Feature_Configuration"] = result["Experiment"]
        result["Experiment"] = experiment
        result["Protocol"] = protocol
        if "N" not in result:
            result["N"] = config.final_observations
        if "Groups" not in result:
            result["Groups"] = config.final_groups
        if "Folds" not in result:
            result["Folds"] = 5
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


WHEAT_CONFIGURATION_LABELS = {
    "NDVI_only": "NDVI Only",
    "Multispectral_bands": "Spectral Bands Only",
    "Bands_and_indices": "Spectral Bands + Vegetation Indices",
}
WHEAT_MODEL_LABELS = {
    "Random_Forest": "Random Forest",
    "Ridge_Regression": "Ridge Regression",
}


def wheat_feature_configuration_results() -> pd.DataFrame:
    """Return only the documented wheat feature experiment in presentation form."""
    results = model_results("wheat")
    required = {"Feature_Configuration", "Model", "R2", "RMSE", "MAE", "N", "Folds"}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame()
    result = results[results["Feature_Configuration"].isin(WHEAT_CONFIGURATION_LABELS)].copy()
    result["Feature configuration"] = result["Feature_Configuration"].map(WHEAT_CONFIGURATION_LABELS)
    result["Model display"] = result["Model"].map(WHEAT_MODEL_LABELS).fillna(result["Model"])
    order = list(WHEAT_CONFIGURATION_LABELS.values())
    result["Feature configuration"] = pd.Categorical(result["Feature configuration"], categories=order, ordered=True)
    return result.sort_values(["Feature configuration", "Model display"]).reset_index(drop=True)


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_Predicted_LAI")] + (["Predicted_LAI"] if "Predicted_LAI" in frame else [])


def normalise_predictions(crop_id: str) -> pd.DataFrame:
    """Convert each crop's documented OOF schema to one in-memory chart schema."""
    config = get_crop(crop_id)
    normalized: list[pd.DataFrame] = []
    for experiment, path in config.prediction_files:
        raw = read_csv(path)
        if raw.empty:
            continue
        actual = "Actual_LAI" if "Actual_LAI" in raw else config.target
        if actual not in raw:
            continue
        group = config.group if config.group in raw else None
        campaign = config.campaign if config.campaign and config.campaign in raw else None
        pred_columns = _prediction_columns(raw)
        if not pred_columns:
            continue
        for pred_column in pred_columns:
            use = pd.DataFrame({
                "crop_id": crop_id,
                "observation_id": raw.get("Observation_ID", raw.index.astype(str)),
                "group_id": raw[group] if group else pd.NA,
                "campaign_or_season": raw[campaign] if campaign else pd.NA,
                "actual_lai": pd.to_numeric(raw[actual], errors="coerce"),
                "predicted_lai": pd.to_numeric(raw[pred_column], errors="coerce"),
                "model": raw["Model"] if pred_column == "Predicted_LAI" and "Model" in raw else pred_column.removesuffix("_Predicted_LAI").replace("_", " "),
                "experiment": experiment,
            })
            use["residual"] = use["actual_lai"] - use["predicted_lai"]
            use["absolute_error"] = use["residual"].abs()
            normalized.append(use.dropna(subset=["actual_lai", "predicted_lai"]))
    return pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame()


def feature_importance(crop_id: str) -> pd.DataFrame:
    config = get_crop(crop_id)
    if not config.feature_importance_files:
        return pd.DataFrame()
    return read_csv(config.feature_importance_files[0])


def best_result(crop_id: str) -> pd.Series | None:
    results = model_results(crop_id)
    if results.empty or "R2" not in results:
        return None
    return results.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
