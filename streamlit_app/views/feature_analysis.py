from __future__ import annotations

import streamlit as st

from components.charts import correlation_heatmap, generic_metric_bar, generic_scatter
from config.crops import get_crop
from services.crop_data_service import feature_columns, load_feature_source
from services.model_results_service import feature_importance


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    frame = load_feature_source(crop_id)
    st.title(f"Feature Analysis — {config.name}")
    if frame.empty:
        st.warning("The selected feature source could not be loaded.")
        return
    target = config.target if config.target in frame else "LAI" if "LAI" in frame else None
    features = feature_columns(frame, target or "")
    if crop_id == "maize":
        st.info("Feature exploration uses the 316-row satellite-feature source. It is distinct from the 201-observation nested-CV evaluation sample used on Model Performance.")
    if not target or not features:
        st.warning("No compatible target or numeric spectral features were found.")
        return
    selected = st.selectbox("Feature versus measured LAI", features)
    st.plotly_chart(generic_scatter(frame.dropna(subset=[selected, target]), selected, target, config.campaign, f"{selected} versus measured LAI"), use_container_width=True)
    limited = features[:25]
    if len(limited) >= 2:
        st.plotly_chart(correlation_heatmap(frame, [target] + limited), use_container_width=True)
    importance = feature_importance(crop_id)
    st.markdown("#### Documented feature importance")
    if importance.empty:
        st.info("No crop-specific feature-importance output is available. Correlation and scatter views are descriptive only; correlation does not establish causation.")
    elif {"Feature", "Mean_Importance"}.issubset(importance.columns):
        st.plotly_chart(generic_metric_bar(importance.rename(columns={"Feature": "Model", "Mean_Importance": "Importance"}), "Importance"), use_container_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)
        st.caption("Importance is shown only from the documented model output; it is not a causal ranking.")
