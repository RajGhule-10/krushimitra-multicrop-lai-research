# KrushiMitra Crop Biophysical Estimation Research

## Research Objective

Investigate whether multispectral Sentinel-2 bands and vegetation
indices can estimate measured winter-wheat Leaf Area Index (LAI)
more accurately than an NDVI-only baseline.

## Dataset

The project uses matched Sentinel-2 spectral observations and
in-situ winter-wheat measurements.

Primary target:

- LAI_mean

Secondary exploratory target:

- CC_mean

## Models

- Random Forest Regression
- Ridge Regression

## Feature Experiments

1. NDVI only
2. Ten Sentinel-2 spectral bands
3. Spectral bands combined with vegetation indices

## Validation

Five-fold GroupKFold cross-validation grouped by Unit_ID.

## Evaluation Metrics

- R-squared
- Root Mean Squared Error
- Mean Absolute Error

## Scripts

- 01_data_audit.py: inspect dataset quality
- 02_lai_model_comparison.py: train and compare models
- 03_plot_lai_predictions.py: generate visualizations
- 04_unit_error_analysis.py: summarize errors by field unit

## Run Order

1. python 01_data_audit.py
2. python 02_lai_model_comparison.py
3. python 03_plot_lai_predictions.py
4. python 04_unit_error_analysis.py

## Limitations

The dataset contains a limited number of field-unit groups and
satellite acquisition dates. Performance should not be assumed
to generalize to unseen seasons, regions, or crops.

Satellite-to-ground matching and valid-pixel diagnostics require
verification before final research conclusions are drawn.

## Multi-Crop Research Dashboard

The repository also includes a read-only Streamlit dashboard for the four independent LAI experiments: Bulgarian winter wheat, Ahero (Kenya) rice, Kenya maize, and Thailand cassava.

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app/app.py
```

See [streamlit_app/README.md](streamlit_app/README.md) for dashboard safeguards, including Ahero-only rice views, explicit distinction between the maize 201-observation evaluation sample and 316-row satellite feature source, and the limitation that historical out-of-fold outputs are not live field predictions.

The winter-wheat dashboard shows the three documented feature configurations. A full vegetation-indices-only comparison is not in the research outputs; its approval-gated implementation outline is in [docs/wheat_vegetation_indices_only_plan.md](docs/wheat_vegetation_indices_only_plan.md).
