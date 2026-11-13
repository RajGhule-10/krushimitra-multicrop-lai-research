"""KrushiMitra's read-only multi-crop Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from components.navigation import sidebar
from components.ui_helpers import apply_theme
from views import crop_explorer, data_explorer, feature_analysis, methodology, model_performance, overview, prediction_analysis


st.set_page_config(page_title="KrushiMitra Research", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")
apply_theme()


def main() -> None:
    page, crop_id = sidebar()
    if page == "Research Overview":
        overview.render()
    elif page == "Crop Explorer":
        crop_explorer.render(crop_id)
    elif page == "Model Performance":
        model_performance.render(crop_id)
    elif page == "Prediction Analysis":
        prediction_analysis.render(crop_id)
    elif page == "Feature Analysis":
        feature_analysis.render(crop_id)
    elif page == "Data Explorer":
        data_explorer.render(crop_id)
    else:
        methodology.render(crop_id)


if __name__ == "__main__":
    main()
