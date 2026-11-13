from __future__ import annotations

import streamlit as st

from components.charts import feature_configuration_metric, generic_metric_bar
from config.crops import get_crop
from services.model_results_service import model_results, wheat_feature_configuration_results


def _render_wheat_feature_experiment() -> None:
    results = wheat_feature_configuration_results()
    st.markdown("#### Feature Configuration Experiment")
    st.write("This grouped comparison holds the evaluated wheat dataset and GroupKFold protocol constant while changing the satellite feature representation. NDVI-only is a single-index baseline; spectral bands use Sentinel-2 reflectance; the combined configuration adds derived vegetation indices to those bands.")
    if results.empty:
        st.warning("The documented winter-wheat feature-configuration result file is unavailable or has an unexpected schema.")
        return
    best = results.sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True]).iloc[0]
    cards = st.columns(4)
    cards[0].metric("Best documented configuration", f"{best['Model display']} · {best['Feature configuration']}")
    cards[1].metric("R²", f"{best['R2']:.3f}")
    cards[2].metric("RMSE", f"{best['RMSE']:.3f} LAI")
    cards[3].metric("MAE", f"{best['MAE']:.3f} LAI")
    st.caption(f"All documented configurations use N = {int(results['N'].iloc[0])} observations and five-fold GroupKFold grouped by Unit_ID.")
    for metric in ("R2", "RMSE", "MAE"):
        st.plotly_chart(feature_configuration_metric(results, metric), use_container_width=True)
    display = results[["Feature configuration", "Model display", "N", "Groups", "Folds", "R2", "RMSE", "MAE"]]
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={"R2": st.column_config.NumberColumn("R²", format="%.3f"), "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"), "MAE": st.column_config.NumberColumn("MAE", format="%.3f")})
    st.info("A separate all-vegetation-indices-only experiment is not present in `02_lai_model_comparison.py`, `outputs/lai_model_comparison.csv`, or the matching out-of-fold results. It is intentionally shown as unavailable; no metrics were inferred and no model was retrained.")
    st.caption("Interpret the documented pattern by model: Random Forest performs best with bands plus indices, while Ridge performs best with spectral bands alone. This does not establish that adding indices always improves LAI estimation.")


def render(crop_id: str) -> None:
    config = get_crop(crop_id)
    results = model_results(crop_id)
    st.title(f"Model Performance — {config.name}")
    st.info(config.validation)
    if results.empty:
        st.warning("No valid model-comparison file could be loaded for this crop.")
        return
    if crop_id == "wheat":
        _render_wheat_feature_experiment()
        return
    show = [column for column in ["Experiment", "Protocol", "Model", "N", "Groups", "Folds", "R2", "RMSE", "MAE"] if column in results]
    st.dataframe(results[show], use_container_width=True, hide_index=True, column_config={"R2": st.column_config.NumberColumn("R²", format="%.3f"), "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"), "MAE": st.column_config.NumberColumn("MAE", format="%.3f")})
    st.caption("Compare values only within their documented experiment and validation protocol. R² can be negative; RMSE and MAE are LAI units.")
    choice = st.selectbox("Show experiment", results["Experiment"].unique().tolist())
    subset = results[results["Experiment"] == choice]
    a, b, c = st.columns(3)
    with a: st.plotly_chart(generic_metric_bar(subset, "R2"), use_container_width=True)
    with b: st.plotly_chart(generic_metric_bar(subset, "RMSE"), use_container_width=True)
    with c: st.plotly_chart(generic_metric_bar(subset, "MAE"), use_container_width=True)
    if crop_id == "maize":
        st.warning("The primary nested-CV maize results are modest. Negative R² values indicate a model can underperform a mean-prediction baseline under this evaluation.")
