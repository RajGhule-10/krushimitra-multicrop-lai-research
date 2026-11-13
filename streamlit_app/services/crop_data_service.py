"""Read-only loaders for crop-specific source data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.crops import CropConfig, DatasetSpec, get_crop
from config.paths import project_path


DATE_HINTS = ("Date", "date", "S2_Date", "Satellite_Image_Date", "LAI_Date")


def read_csv(relative_path: str) -> pd.DataFrame:
    """Return an empty frame for an unavailable/unreadable CSV, never altering it."""
    path = project_path(relative_path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()
    for column in frame.columns:
        if any(hint in column for hint in DATE_HINTS):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                frame[column] = parsed
    return frame


def _rice_observations(config: CropConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(config.primary_paths, start=1):
        frame = read_csv(path)
        if not frame.empty:
            frame = frame.copy()
            frame["Campaign"] = f"campaign_{index}"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _maize_evaluation_observations(config: CropConfig) -> pd.DataFrame:
    frame = read_csv(config.primary_paths[0])
    if frame.empty:
        return frame
    key = "Observation_ID" if "Observation_ID" in frame else None
    result = frame.drop_duplicates(key).copy() if key else frame.copy()
    result["_source_note"] = "Unique records from the 201-observation nested-CV evaluation sample."
    return result


def load_observations(crop_id: str) -> pd.DataFrame:
    """Load only the selected crop's primary observation/evaluation data."""
    config = get_crop(crop_id)
    if crop_id == "rice":
        return _rice_observations(config)
    if crop_id == "maize":
        return _maize_evaluation_observations(config)
    return read_csv(config.primary_paths[0])


def load_feature_source(crop_id: str) -> pd.DataFrame:
    config = get_crop(crop_id)
    if config.feature_source_path:
        return read_csv(config.feature_source_path)
    return load_observations(crop_id)


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    return read_csv(spec.path)


def available_datasets(crop_id: str) -> tuple[DatasetSpec, ...]:
    return get_crop(crop_id).datasets


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """Discover numeric Sentinel-like bands and vegetation indices without IDs/targets."""
    excluded = {target, "LAI", "LAI_mean", "Actual_LAI", "Latitude", "Longitude", "MAP", "LCC", "AOT"}
    candidates: list[str] = []
    for column in frame.select_dtypes(include="number").columns:
        upper = column.upper()
        satellite_like = upper.startswith("B") or any(token in upper for token in ("ND", "VI", "SAVI", "RVI", "CIG", "TCI", "MSI", "DVI", "AFRI", "CIRE", "VREI"))
        if satellite_like and column not in excluded:
            candidates.append(column)
    return candidates


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Column", "Missing", "Percent missing"])
    missing = frame.isna().sum()
    return pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Percent missing": (100 * missing / len(frame)).round(1)}).sort_values("Missing", ascending=False)


def data_path_exists(relative_path: str) -> bool:
    return Path(project_path(relative_path)).is_file()
