"""Train the RandomForest correlation model on the prepared dataset."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statistics import mean
from typing import Dict, List

from library.correlation_model import PracticeCorrelationModel
from library.loader import load_training_dataset
from library.schema import REPO_ROOT

MODEL_PATH = REPO_ROOT / "models" / "correlation_model.pkl"

NUMERIC_COLUMNS = ["spi", "ndvi_anomaly", "soil_moisture", "temp_mean", "effectiveness_score"]
ENCODED_COLUMNS = ["phenology_stage_encoded", "soil_type_encoded", "practice_id_encoded", "crop_encoded"]


def _column_means(records: List[Dict[str, object]], columns: List[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for column in columns:
        values = [float(record[column]) for record in records if isinstance(record.get(column), (int, float))]
        stats[column] = mean(values) if values else 0.0
    return stats


def _prepare_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_means = _column_means(records, NUMERIC_COLUMNS)
    prepared: List[Dict[str, object]] = []
    for record in records:
        item = dict(record)
        for column in NUMERIC_COLUMNS:
            value = item.get(column)
            try:
                item[column] = float(value)
            except (TypeError, ValueError):
                item[column] = numeric_means[column]
        for column in ENCODED_COLUMNS:
            value = item.get(column)
            try:
                item[column] = int(value)
            except (TypeError, ValueError):
                item[column] = -1
        prepared.append(item)
    return prepared


def main() -> None:
    dataset = load_training_dataset()
    prepared = _prepare_records(dataset)

    model = PracticeCorrelationModel.create_default()
    model.fit(prepared)
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
