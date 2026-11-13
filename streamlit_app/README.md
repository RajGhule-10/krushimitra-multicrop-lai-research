# KrushiMitra Multi-Crop LAI Research Dashboard

This Streamlit application is a read-only research analytics dashboard for four independent LAI experiments: Bulgarian winter wheat, Ahero (Kenya) rice, Kenya maize, and Thailand cassava.

It reads existing repository CSVs only. It does not retrain models, write data, contact Google Earth Engine, create new-field predictions, or provide agronomic advice.

## Run

From the repository root:

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app/app.py
```

## Included research safeguards

- Ahero 2023 rice is kept separate from Ranchi/Kuthulia material.
- Maize's 201-observation nested-CV evaluation sample is clearly separated from the 316-row satellite-feature source.
- Charts normalize crop-specific output schemas only in memory; source CSVs are never modified.
- Historical out-of-fold predictions are labelled as validation outputs, not live prediction capability.
- Missing files produce warnings without blocking unrelated crop pages.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
