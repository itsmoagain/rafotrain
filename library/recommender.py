"""Recommendation utilities built on top of the correlation model."""
from __future__ import annotations

from typing import Dict, List, Optional

from statistics import mean

from .correlation_model import PracticeCorrelationModel
from .loader import load_practice_metadata, load_training_dataset
from .schema import REPO_ROOT, encode_crop, encode_practice, encode_soil, encode_stage

MODEL_PATH = REPO_ROOT / "models" / "correlation_model.pkl"

_training_dataset_cache: Optional[List[Dict[str, object]]] = None
_model_cache: Optional[PracticeCorrelationModel] = None


def _get_training_dataset() -> List[Dict[str, object]]:
    global _training_dataset_cache
    if _training_dataset_cache is None:
        _training_dataset_cache = load_training_dataset()
    return _training_dataset_cache


def _load_model() -> PracticeCorrelationModel:
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Correlation model not found. Run 'python scripts/train_practice_correlations.py' to "
                f"generate {MODEL_PATH.name}."
            )
        _model_cache = PracticeCorrelationModel.load(MODEL_PATH)
    return _model_cache


def _default_numeric_value(column: str) -> float:
    dataset = _get_training_dataset()
    values = [float(record[column]) for record in dataset if isinstance(record.get(column), (int, float))]
    if not values:
        return 0.0
    return mean(values)


def _prepare_feature_rows(
    crop: str,
    anomaly_type: str,
    spi: float,
    ndvi_anomaly: float,
    soil_type: str,
    stage: str,
    soil_moisture: Optional[float],
    temp_mean: Optional[float],
) -> List[Dict[str, object]]:
    practices = load_practice_metadata()
    candidates = [p for p in practices.values() if p.crop == crop and p.anomaly_type == anomaly_type]

    if not candidates:
        return []

    soil_moisture_val = soil_moisture if soil_moisture is not None else _default_numeric_value("soil_moisture")
    temp_mean_val = temp_mean if temp_mean is not None else _default_numeric_value("temp_mean")

    rows: List[Dict[str, object]] = []
    for practice in candidates:
        rows.append(
            {
                "practice": practice,
                "practice_id": practice.id,
                "spi": float(spi),
                "ndvi_anomaly": float(ndvi_anomaly),
                "soil_moisture": float(soil_moisture_val),
                "temp_mean": float(temp_mean_val),
                "phenology_stage_encoded": encode_stage(stage),
                "soil_type_encoded": encode_soil(soil_type),
                "practice_id_encoded": encode_practice(practice.id),
                "crop_encoded": encode_crop(crop),
            }
        )
    return rows


def rank_practices(crop: str, anomaly_type: str, scores: List[float]) -> List[Dict[str, object]]:
    practices = load_practice_metadata()
    candidates = [p for p in practices.values() if p.crop == crop and p.anomaly_type == anomaly_type]
    ranked: List[Dict[str, object]] = []
    for practice, score in zip(candidates, scores):
        ranked.append(
            {
                "practice_id": practice.id,
                "score": float(score),
                "description": practice.description,
                "expected_outcomes": practice.expected_outcomes,
                "tags": practice.tags,
                "cost_level": practice.cost_level,
                "labor_intensity": practice.labor_intensity,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def get_recommendations(
    crop: str,
    anomaly_type: str,
    spi: float,
    ndvi_anomaly: float,
    soil_type: str,
    stage: str,
    soil_moisture: Optional[float] = None,
    temp_mean: Optional[float] = None,
) -> List[Dict[str, object]]:
    rows = _prepare_feature_rows(
        crop=crop,
        anomaly_type=anomaly_type,
        spi=spi,
        ndvi_anomaly=ndvi_anomaly,
        soil_type=soil_type,
        stage=stage,
        soil_moisture=soil_moisture,
        temp_mean=temp_mean,
    )
    if not rows:
        return []

    model = _load_model()
    scores = model.predict(rows)
    return rank_practices(crop, anomaly_type, scores)
