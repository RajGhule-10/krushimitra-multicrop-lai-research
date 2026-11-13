"""Regression checks for the read-only dashboard adapters."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from config.crops import CROPS  # noqa: E402
from services.crop_data_service import load_observations  # noqa: E402
from services.model_results_service import model_results, normalise_predictions, wheat_feature_configuration_results  # noqa: E402


class DashboardDataTests(unittest.TestCase):
    def test_primary_observation_counts_and_targets(self) -> None:
        expected = {"wheat": 142, "rice": 177, "maize": 201, "cassava": 232}
        for crop_id, count in expected.items():
            with self.subTest(crop=crop_id):
                frame = load_observations(crop_id)
                self.assertEqual(len(frame), count)
                config = CROPS[crop_id]
                self.assertIn(config.target, frame.columns)
                self.assertIn(config.group, frame.columns)

    def test_maize_evaluation_is_not_satellite_source_row_count(self) -> None:
        frame = load_observations("maize")
        self.assertEqual(frame["Observation_ID"].nunique(), 201)
        self.assertNotEqual(len(frame), 316)

    def test_model_files_provide_real_metrics(self) -> None:
        for crop_id in CROPS:
            with self.subTest(crop=crop_id):
                results = model_results(crop_id)
                self.assertFalse(results.empty)
                self.assertTrue({"Model", "R2", "RMSE", "MAE"}.issubset(results.columns))
        wheat = model_results("wheat")
        best = wheat.sort_values("R2", ascending=False).iloc[0]
        self.assertAlmostEqual(float(best["R2"]), 0.715990, places=6)

    def test_prediction_normalization_uses_crop_specific_columns(self) -> None:
        expected_counts = {"wheat": 852, "rice": 708, "maize": 804, "cassava": 464}
        for crop_id, count in expected_counts.items():
            with self.subTest(crop=crop_id):
                frame = normalise_predictions(crop_id)
                self.assertEqual(len(frame), count)
                self.assertTrue({"actual_lai", "predicted_lai", "group_id", "model", "residual", "absolute_error"}.issubset(frame.columns))

    def test_wheat_feature_configuration_mapping_uses_documented_results(self) -> None:
        results = wheat_feature_configuration_results()
        self.assertEqual(len(results), 6)
        self.assertEqual(set(results["Feature configuration"]), {"NDVI Only", "Spectral Bands Only", "Spectral Bands + Vegetation Indices"})
        random_forest = results[results["Model display"] == "Random Forest"]
        best = random_forest.sort_values("R2", ascending=False).iloc[0]
        self.assertEqual(best["Feature configuration"], "Spectral Bands + Vegetation Indices")
        self.assertAlmostEqual(float(best["R2"]), 0.7159902780988887, places=12)
        ridge = results[results["Model display"] == "Ridge Regression"].sort_values("R2", ascending=False).iloc[0]
        self.assertEqual(ridge["Feature configuration"], "Spectral Bands Only")


if __name__ == "__main__":
    unittest.main()
