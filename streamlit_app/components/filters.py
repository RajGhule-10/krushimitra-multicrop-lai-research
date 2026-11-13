from __future__ import annotations

import pandas as pd
import streamlit as st

from config.crops import CropConfig


def apply_observation_filters(frame: pd.DataFrame, config: CropConfig) -> pd.DataFrame:
    """Render only applicable filters and return the chosen subset."""
    if frame.empty:
        return frame
    filtered = frame.copy()
    with st.expander("Filter observations", expanded=True):
        columns = st.columns(3)
        candidates = [(config.campaign, "Campaign / season"), (config.group, "Farm / plot / field unit")]
        for index, (field, label) in enumerate(candidates):
            if field and field in filtered:
                options = sorted(filtered[field].dropna().astype(str).unique().tolist())
                selected = columns[index].multiselect(label, options, default=options)
                filtered = filtered[filtered[field].astype(str).isin(selected)]
        if config.target in filtered:
            values = pd.to_numeric(filtered[config.target], errors="coerce").dropna()
            if not values.empty:
                low, high = float(values.min()), float(values.max())
                chosen = columns[2].slider("Measured LAI range", low, high, (low, high))
                filtered = filtered[pd.to_numeric(filtered[config.target], errors="coerce").between(*chosen)]
    return filtered
