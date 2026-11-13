from __future__ import annotations

import streamlit as st

from config.crops import get_crop


METHODS = {
    "wheat": "Ground observations were temporally paired with Sentinel-2 Surface Reflectance Harmonized bands and indices. Random Forest and Ridge configurations were evaluated with Unit_ID GroupKFold.",
    "rice": "The primary dashboard combines only the three Ahero 2023 campaign datasets. Sentinel-2-derived indices were evaluated using Farm_ID GroupKFold, with a separate nested Ridge study.",
    "maize": "Ground observations underwent cleaning, parcel/spatial validation, Sentinel-2 extraction, and exact/neighbourhood diagnostics. The final nested evaluation uses 201 spatially reliable observations, grouped by Farm_ID.",
    "cassava": "Ground LAI was aggregated and matched to plot-MAP satellite features. Ridge models were assessed with nested Plot_ID-grouped cross-validation on 232 matched records.",
}


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    st.title(f"Methodology — {config.name}")
    st.markdown("#### Crop-specific workflow")
    st.write(METHODS[crop_id])
    st.markdown("#### Validation")
    st.write(config.validation)
    st.markdown("#### Research limitations")
    st.write(config.caveat)
    st.write("LAI model performance can vary with growth stage, sample coverage, temporal matching, soil/background effects, atmospheric conditions, and data quality. Grouped validation is useful but does not prove performance in new regions or seasons.")
    st.markdown("#### Appropriate use")
    st.warning("This is an academic analytics dashboard. It is not a validated agronomic advisory, live monitoring service, or basis for fertilizer, irrigation, pesticide, or disease recommendations.")
