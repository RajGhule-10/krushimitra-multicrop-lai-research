from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SOURCES: Final = {
    "observations": "data/krushimitra_all_campaigns.csv",
    "comparison": "outputs/lai_model_comparison.csv",
    "predictions": "outputs/lai_out_of_fold_predictions.csv",
    "unit_errors": "outputs/lai_error_by_unit.csv",
    "season_units": "outputs/observations_by_season_and_unit.csv",
    "spectral_summary": "outputs/spectral_target_summary.csv",
    "missing_values": "outputs/missing_values.csv",
    "coordinates": "outputs/physical_unit_coordinate_summary.csv",
    "dataset_summary": "outputs/dataset_summary.csv",
}
DATE_COLUMNS: Final = {"S2_Date", "Primary_Field_Date", "LP80_date", "Phenology_Date", "Soil_Date", "AGB_N_date"}


@st.cache_data(show_spinner=False)
def read_csv(relative_path: str) -> pd.DataFrame:
    """Load a project CSV defensively, returning an empty frame for unavailable data."""
    path = PROJECT_ROOT / relative_path
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    frame = frame.copy()
    for column in DATE_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_sources() -> dict[str, pd.DataFrame]:
    return {name: read_csv(path) for name, path in SOURCES.items()}


def source_path(name: str) -> str:
    return SOURCES[name]


def source_status(data: dict[str, pd.DataFrame]) -> list[str]:
    return [source_path(name) for name, frame in data.items() if frame.empty]


def resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return a column only for an exact, unambiguous semantic candidate."""
    matches = [candidate for candidate in candidates if candidate in frame.columns]
    return matches[0] if len(matches) == 1 else None


def numeric_copy(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def display_name(value: object) -> str:
    labels = {
        "NDVI_only": "NDVI Only",
        "Multispectral_bands": "Multispectral Bands",
        "Bands_and_indices": "Bands + Vegetation Indices",
        "Random_Forest": "Random Forest",
        "Ridge_Regression": "Ridge Regression",
        "LAI_mean": "Measured LAI",
    }
    return labels.get(str(value), str(value).replace("_", " "))
