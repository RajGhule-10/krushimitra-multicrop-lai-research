from __future__ import annotations

import streamlit as st

from config.crops import get_crop
from services.crop_data_service import available_datasets, load_dataset, missing_summary


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    specs = available_datasets(crop_id)
    st.title(f"Data Explorer — {config.name}")
    spec = st.selectbox("Dataset", specs, format_func=lambda item: f"{item.label} · {item.kind}")
    frame = load_dataset(spec)
    st.caption(f"{spec.description} Source: `{spec.path}`")
    if frame.empty:
        st.warning("This source could not be read. The dashboard did not modify it.")
        return
    left, right = st.columns(2)
    left.metric("Rows", f"{len(frame):,}")
    right.metric("Columns", f"{len(frame.columns):,}")
    with st.expander("Schema and missing values", expanded=True):
        schema = frame.dtypes.astype(str).rename_axis("Column").reset_index(name="Data type")
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.dataframe(missing_summary(frame), use_container_width=True, hide_index=True)
    with st.expander("Descriptive statistics"):
        numeric = frame.describe(include="number").transpose().reset_index(names="Column")
        st.dataframe(numeric, use_container_width=True, hide_index=True)
    st.download_button("Download this dataset as CSV", frame.to_csv(index=False).encode("utf-8"), file_name=f"krushimitra_{crop_id}_{spec.label.lower().replace(' ', '_')}.csv", mime="text/csv")
    st.markdown("#### Preview")
    st.dataframe(frame, use_container_width=True, hide_index=True, height=420)
