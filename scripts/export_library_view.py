"""Export a human-readable practice library ranked by model confidence."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statistics import mean
from typing import Dict, List

import json

from library.correlation_model import PracticeCorrelationModel
from library.loader import load_practice_metadata, load_training_dataset
from library.schema import REPO_ROOT
from scripts.train_practice_correlations import _prepare_records

MODEL_PATH = REPO_ROOT / "models" / "correlation_model.pkl"
OUTPUT_PATH = REPO_ROOT / "library" / "practice_library.yml"


def _to_yaml(entries: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    for entry in entries:
        lines.append("-")
        for key, value in entry.items():
            if isinstance(value, list):
                list_items = ", ".join(json.dumps(item) for item in value)
                lines.append(f"  {key}: [{list_items}]")
            else:
                lines.append(f"  {key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    dataset = load_training_dataset()
    prepared = _prepare_records(dataset)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Correlation model not found. Run 'python scripts/train_practice_correlations.py' before "
            "exporting the library view."
        )
    model = PracticeCorrelationModel.load(MODEL_PATH)
    predictions = model.predict(prepared)

    aggregated: Dict[str, Dict[str, object]] = {}
    for record, prediction in zip(prepared, predictions):
        practice_id = str(record.get("practice_id"))
        stats = aggregated.setdefault(
            practice_id,
            {"observed": [], "predicted": [], "sample_size": 0},
        )
        stats["observed"].append(float(record.get("effectiveness_score", 0.0)))
        stats["predicted"].append(float(prediction))
        stats["sample_size"] += 1

    practices = load_practice_metadata()
    entries: List[Dict[str, object]] = []
    for practice_id, practice in practices.items():
        stats = aggregated.get(practice_id, {"observed": [], "predicted": [], "sample_size": 0})
        entry: Dict[str, object] = {
            "practice_id": practice_id,
            "crop": practice.crop,
            "anomaly_type": practice.anomaly_type,
            "description": practice.description,
            "expected_outcomes": practice.expected_outcomes,
            "tags": practice.tags,
            "evidence": practice.evidence,
            "cost_level": practice.cost_level or "",
            "labor_intensity": practice.labor_intensity or "",
        }
        if stats["observed"]:
            entry["observed_effectiveness"] = round(mean(stats["observed"]), 3)
        if stats["predicted"]:
            entry["predicted_effectiveness"] = round(mean(stats["predicted"]), 3)
        entry["sample_size"] = stats["sample_size"]
        entries.append(entry)

    entries.sort(key=lambda item: item.get("predicted_effectiveness", 0.0), reverse=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(_to_yaml(entries), encoding="utf-8")
    print(f"Practice library exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
