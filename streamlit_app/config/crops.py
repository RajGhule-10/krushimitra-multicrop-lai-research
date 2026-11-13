"""Explicit, research-safe configuration for the independent crop studies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: str
    kind: str
    description: str


@dataclass(frozen=True)
class CropConfig:
    id: str
    name: str
    location: str
    target: str
    group: str
    date: str | None
    campaign: str | None
    primary_label: str
    primary_paths: tuple[str, ...]
    model_files: tuple[tuple[str, str, str], ...]
    prediction_files: tuple[tuple[str, str], ...]
    feature_importance_files: tuple[str, ...] = ()
    datasets: tuple[DatasetSpec, ...] = ()
    validation: str = ""
    caveat: str = ""
    final_observations: int | None = None
    final_groups: int | None = None
    feature_source_path: str | None = None


CROPS: dict[str, CropConfig] = {
    "wheat": CropConfig(
        id="wheat", name="Bulgarian Winter Wheat", location="Zlatia, Bulgaria",
        target="LAI_mean", group="Unit_ID", date="S2_Date", campaign="Season",
        primary_label="Matched modelling observations",
        primary_paths=("data/krushimitra_all_campaigns.csv",),
        model_files=(("Grouped model comparison", "outputs/lai_model_comparison.csv", "grouped"),),
        prediction_files=(("Grouped out-of-fold predictions", "outputs/lai_out_of_fold_predictions.csv"),),
        validation="Five-fold GroupKFold grouped by Unit_ID (verified in 02_lai_model_comparison.py).",
        caveat="This experiment has limited field-unit and seasonal coverage; it does not establish independent geographic generalization.",
        final_observations=142, final_groups=12,
        datasets=(
            DatasetSpec("Matched modelling observations", "data/krushimitra_all_campaigns.csv", "processed/modelling", "142 matched Sentinel-2 and winter-wheat observations."),
            DatasetSpec("Model comparison", "outputs/lai_model_comparison.csv", "model output", "Grouped evaluation metrics."),
            DatasetSpec("Out-of-fold predictions", "outputs/lai_out_of_fold_predictions.csv", "model output", "Historical cross-validation predictions."),
        ),
    ),
    "rice": CropConfig(
        id="rice", name="Kenya Rice", location="Ahero, Kenya · 2023",
        target="LAI", group="Farm_ID", date=None, campaign="Campaign",
        primary_label="Ahero campaign modelling observations",
        primary_paths=(
            "data/rice/processed/final_df_rice_campaign_1.csv",
            "data/rice/processed/final_df_rice_campaign_2.csv",
            "data/rice/processed/final_df_rice_campaign_3.csv",
        ),
        model_files=(
            ("Phase 11 · 5-fold grouped comparison", "outputs/rice/models/phase11_model_comparison.csv", "grouped"),
            ("Phase 12 · nested Ridge comparison", "outputs/rice/models/phase12_ridge_comparison.csv", "nested"),
        ),
        prediction_files=(
            ("Phase 11 · grouped out-of-fold predictions", "outputs/rice/models/phase11_oof_predictions.csv"),
            ("Phase 12 · nested Ridge predictions", "outputs/rice/models/phase12_ridge_predictions.csv"),
        ),
        validation="Phase 11 uses five-fold Farm_ID GroupKFold. Phase 12 evaluates fixed and tuned Ridge with five outer folds and four inner folds.",
        caveat="The primary rice views use Ahero 2023 only. Ranchi/Kuthulia files remain separate and are not combined with this experiment.",
        final_observations=177,
        datasets=(
            DatasetSpec("Ahero campaign 1", "data/rice/processed/final_df_rice_campaign_1.csv", "processed/modelling", "43 Ahero observations."),
            DatasetSpec("Ahero campaign 2", "data/rice/processed/final_df_rice_campaign_2.csv", "processed/modelling", "60 Ahero observations."),
            DatasetSpec("Ahero campaign 3", "data/rice/processed/final_df_rice_campaign_3.csv", "processed/modelling", "74 Ahero observations."),
            DatasetSpec("Phase 11 predictions", "outputs/rice/models/phase11_oof_predictions.csv", "model output", "Farm-grouped historical predictions."),
            DatasetSpec("Phase 12 predictions", "outputs/rice/models/phase12_ridge_predictions.csv", "model output", "Nested Ridge historical predictions."),
        ),
    ),
    "maize": CropConfig(
        id="maize", name="Kenya Maize", location="Kenya",
        target="Actual_LAI", group="Farm_ID", date="Satellite_Image_Date", campaign="Campaign",
        primary_label="Final nested-CV evaluation observations",
        primary_paths=("outputs/maize/reports/phase04b_oof_predictions.csv",),
        model_files=(("Phase 04b · grouped nested comparison", "outputs/maize/reports/phase04b_nested_model_comparison.csv", "nested"),),
        prediction_files=(("Phase 04b · nested out-of-fold predictions", "outputs/maize/reports/phase04b_oof_predictions.csv"),),
        validation="Five outer Farm_ID-grouped folds; up to four inner folds for tuning. Phase 04b is the primary evaluation.",
        caveat="The 201-observation nested-model evaluation sample is distinct from the 316-row satellite-feature source. Weak and negative R² values are retained.",
        final_observations=201, final_groups=50,
        feature_source_path="data/maize/processed/maize_final_satellite_features.csv",
        datasets=(
            DatasetSpec("Final nested-CV evaluation sample", "outputs/maize/reports/phase04b_oof_predictions.csv", "model output", "201 unique observations repeated across evaluated models."),
            DatasetSpec("Satellite-feature source", "data/maize/processed/maize_final_satellite_features.csv", "processed/source", "316 satellite-feature records; not the final nested evaluation sample."),
            DatasetSpec("Audited master observations", "data/maize/processed/maize_final_master_456.csv", "processed/source", "456 audited source observations."),
        ),
    ),
    "cassava": CropConfig(
        id="cassava", name="Thailand Cassava", location="Thailand",
        target="LAI", group="Plot_ID", date="S2_Date", campaign="MAP",
        primary_label="Matched LAI and Sentinel-2 observations",
        primary_paths=("data/cassava/processed/cassava_lai_s2_matched.csv",),
        model_files=(("Phase 03 · nested Ridge comparison", "outputs/cassava/models/phase03_cassava_ridge_comparison.csv", "nested"),),
        prediction_files=(("Phase 03 · fixed and tuned Ridge predictions", "outputs/cassava/models/phase03_cassava_ridge_predictions.csv"),),
        feature_importance_files=("outputs/cassava/models/phase03_cassava_feature_importance.csv", "outputs/cassava/models/phase03_cassava_feature_importance_by_fold.csv"),
        validation="Five outer Plot_ID-grouped folds and four inner folds for Ridge tuning.",
        caveat="The 232 matched records are temporally matched plot-MAP observations; source ground and satellite records are larger, separate inputs.",
        final_observations=232, final_groups=47,
        datasets=(
            DatasetSpec("Matched modelling observations", "data/cassava/processed/cassava_lai_s2_matched.csv", "processed/modelling", "232 LAI / Sentinel-2 matched observations."),
            DatasetSpec("Original ground LAI", "data/cassava/raw/Ground-LAI-Clean.csv", "original", "Original ground measurements."),
            DatasetSpec("Original satellite indices", "data/cassava/raw/all_index_zonal_median.csv", "original", "Original satellite-index zonal medians."),
            DatasetSpec("Ridge predictions", "outputs/cassava/models/phase03_cassava_ridge_predictions.csv", "model output", "Historical fixed/tuned Ridge predictions."),
        ),
    ),
}


def crop_options() -> list[str]:
    return list(CROPS)


def get_crop(crop_id: str) -> CropConfig:
    return CROPS[crop_id]
