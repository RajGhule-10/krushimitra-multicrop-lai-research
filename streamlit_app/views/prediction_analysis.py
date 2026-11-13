from __future__ import annotations

import streamlit as st

from components.charts import error_by_group, normalized_actual_predicted, residual_histogram
from config.crops import get_crop
from services.model_results_service import normalise_predictions


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    predictions = normalise_predictions(crop_id)
    st.title(f"Prediction Analysis — {config.name}")
    st.caption("Historical validation outputs only — not new-field predictions.")
    if predictions.empty:
        st.warning("No compatible out-of-fold prediction schema could be loaded.")
        return
    experiments = predictions["experiment"].drop_duplicates().tolist()
    experiment = st.selectbox("Prediction experiment", experiments)
    subset = predictions[predictions["experiment"] == experiment]
    model = st.selectbox("Model", subset["model"].drop_duplicates().tolist())
    subset = subset[subset["model"] == model]
    negative = int((subset["predicted_lai"] < 0).sum())
    if negative:
        st.warning(f"{negative} raw prediction(s) are negative. They are deliberately not clipped: negative LAI is physically invalid and requires research investigation.")
    a, b, c = st.columns(3)
    a.metric("Predictions", f"{len(subset):,}")
    b.metric("Mean absolute error", f"{subset['absolute_error'].mean():.3f} LAI")
    c.metric("Mean residual", f"{subset['residual'].mean():.3f} LAI")
    left, right = st.columns(2)
    with left: st.plotly_chart(normalized_actual_predicted(subset), use_container_width=True)
    with right: st.plotly_chart(residual_histogram(subset.rename(columns={"residual": "Residual"}), "Residual"), use_container_width=True)
    st.plotly_chart(error_by_group(subset), use_container_width=True)
    if subset["campaign_or_season"].notna().any():
        campaign = subset.groupby("campaign_or_season", dropna=False).agg(Observations=("actual_lai", "size"), MAE=("absolute_error", "mean"), RMSE=("residual", lambda values: (values.pow(2).mean()) ** .5)).reset_index()
        st.markdown("#### Campaign / season error")
        st.dataframe(campaign, use_container_width=True, hide_index=True)
    st.caption("Residual = measured LAI − predicted LAI. Error variation across groups or campaigns is descriptive and does not establish causation.")
