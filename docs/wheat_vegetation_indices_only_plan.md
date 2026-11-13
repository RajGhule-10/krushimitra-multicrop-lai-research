# Optional Winter-Wheat Vegetation-Indices-Only Experiment

## Status

Not implemented. The documented wheat modelling script and outputs contain only `NDVI_only`, `Multispectral_bands`, and `Bands_and_indices`; they do not contain a full vegetation-indices-only evaluation.

## Proposed, approval-gated work

1. Define and review the exact feature set: `NDVI`, `NDRE`, `EVI`, `SAVI`, `NDMI`, and `NDWI` from `02_lai_model_comparison.py`.
2. Confirm it uses the identical cleaned 142-row sample, `LAI_mean` target, and five-fold `Unit_ID` GroupKFold protocol.
3. Run the existing Random Forest and standardized Ridge model factories with no other changes.
4. Save a separately labelled result and out-of-fold prediction output; do not overwrite existing outputs.
5. Independently review the metrics before adding this fourth configuration to the dashboard.

No model is trained by the dashboard or by this navigation/model-intelligence enhancement.
