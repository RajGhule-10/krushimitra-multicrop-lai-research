from __future__ import annotations

import streamlit as st

from config.crops import CROPS
from services.crop_data_service import load_observations
from services.model_results_service import best_result


def render() -> None:
    st.title("KrushiMitra — Multi-Crop LAI Research")
    st.caption("Sentinel-2, ground-measured LAI, and crop-specific validation analytics")
    st.info("The four studies are independent experiments. Their datasets, feature configurations and validation protocols differ, so their metrics are not a controlled crop-to-crop ranking.")
    st.markdown("#### Research objective")
    st.write("Explore documented, historical Leaf Area Index estimation experiments using local research data and existing model outputs. This dashboard neither retrieves live imagery nor gives crop-management recommendations.")
    cards = st.columns(2)
    for column, config in zip(cards * 2, CROPS.values()):
        observation = load_observations(config.id)
        best = best_result(config.id)
        with column:
            st.markdown(f"### {config.name}")
            st.caption(config.location)
            one, two = st.columns(2)
            one.metric("Final modelling observations", f"{config.final_observations or len(observation):,}")
            groups = config.final_groups or (observation[config.group].nunique() if config.group in observation else None)
            two.metric("Verified groups", f"{groups:,}" if groups is not None else "See source")
            if best is not None:
                st.write(f"Best documented result: **{best['Model']}** · R² {best['R2']:.3f} · RMSE {best['RMSE']:.3f} · MAE {best['MAE']:.3f}")
            else:
                st.warning("Model result file unavailable.")
            st.caption(config.validation)
    st.markdown("#### Overall workflow")
    st.markdown("Ground measurements → crop-specific matching and feature preparation → model evaluation with grouped validation → out-of-fold error analysis → cautious research interpretation")
    st.warning("Historical out-of-fold predictions demonstrate past validation performance only. They are not an inference service for new fields.")
