from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import generic_scatter, lai_histogram, time_series
from components.filters import apply_observation_filters
from config.crops import get_crop
from services.crop_data_service import feature_columns, load_observations, missing_summary


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    observations = load_observations(crop_id)
    st.title(f"Crop Explorer — {config.name}")
    st.caption(config.location)
    st.info(config.caveat)
    if observations.empty:
        st.warning(f"The primary source for this study could not be loaded: {', '.join(config.primary_paths)}")
        return
    filtered = apply_observation_filters(observations, config)
    groups = filtered[config.group].nunique() if config.group in filtered else None
    dates = pd.to_datetime(filtered[config.date], errors="coerce") if config.date and config.date in filtered else pd.Series(dtype="datetime64[ns]")
    metrics = st.columns(4)
    metrics[0].metric("Showing observations", f"{len(filtered):,} / {len(observations):,}")
    metrics[1].metric("Groups", f"{groups:,}" if groups is not None else "—")
    metrics[2].metric("LAI range", f"{filtered[config.target].min():.2f}–{filtered[config.target].max():.2f}" if config.target in filtered and not filtered.empty else "—")
    metrics[3].metric("Satellite features", len(feature_columns(observations, config.target)))
    if dates.notna().any():
        st.caption(f"Observation/satellite date range: {dates.min():%d %b %Y} – {dates.max():%d %b %Y}")
    st.markdown("#### LAI and observation patterns")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(lai_histogram(filtered, config.target), use_container_width=True)
    with right:
        if config.date and config.date in filtered and dates.notna().any():
            display = filtered.copy()
            display[config.date] = dates
            if config.campaign and config.campaign in display and config.campaign != "Season":
                display["Season"] = display[config.campaign].astype(str)
            st.plotly_chart(time_series(display.dropna(subset=[config.date]), config.date, config.target), use_container_width=True)
        else:
            st.info("No documented observation date is available for this selected crop dataset.")
    features = feature_columns(filtered, config.target)
    if features:
        selected = st.selectbox("Feature versus measured LAI", features)
        plot = filtered.dropna(subset=[selected, config.target])
        color = config.campaign if config.campaign and config.campaign in plot else None
        st.plotly_chart(generic_scatter(plot, selected, config.target, color, f"{selected} versus measured LAI"), use_container_width=True)
    with st.expander("Available fields and missing values"):
        st.dataframe(missing_summary(filtered), use_container_width=True, hide_index=True)
    st.markdown("#### Filtered observations")
    priority = [column for column in [config.group, config.campaign, config.date, config.target] if column and column in filtered]
    remaining = [column for column in filtered.columns if column not in priority]
    st.download_button("Download filtered observations as CSV", filtered.to_csv(index=False).encode("utf-8"), file_name=f"krushimitra_{crop_id}_filtered_observations.csv", mime="text/csv")
    st.dataframe(filtered[priority + remaining], use_container_width=True, hide_index=True, height=390)
