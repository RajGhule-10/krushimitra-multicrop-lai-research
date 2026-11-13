from __future__ import annotations

import streamlit as st

from config.crops import CROPS


PAGES = ["Research Overview", "Crop Explorer", "Model Performance", "Prediction Analysis", "Feature Analysis", "Data Explorer", "Methodology"]


def sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## KrushiMitra")
        st.caption("Multi-Crop LAI Research Analytics")
        page = st.radio("Navigate", PAGES)
        st.divider()
        crop_id = st.selectbox("Active crop study", list(CROPS), format_func=lambda key: CROPS[key].name)
        crop = CROPS[crop_id]
        st.caption(crop.location)
        st.divider()
        st.caption("Read-only local research data")
        st.caption("No live satellite retrieval · no agronomic advice")
    return page, crop_id
