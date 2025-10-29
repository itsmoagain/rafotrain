"""Construct the model-ready training dataset by merging metadata and outcomes."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
from typing import Dict, List

from library.loader import (
    encode_records,
    load_outcome_metrics,
    load_practice_metadata,
    load_region_metadata,
    load_training_examples,
)
from library.schema import ENCODINGS_PATH, data_file

ENCODING_KEYS = {
    "phenology_stage": "phenology_stage",
    "soil_type": "soil_type",
    "crop": "crop",
    "practice_id": "practice_id",
}


def build_encodings(records: List[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    encodings: Dict[str, Dict[str, int]] = {}
    for column, key in ENCODING_KEYS.items():
        values = sorted({str(record[column]) for record in records if record.get(column) not in {None, ""}})
        encodings[key] = {value: index for index, value in enumerate(values)}
    ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ENCODINGS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(encodings, fp, indent=2)
    return encodings


def merge_metadata() -> List[Dict[str, object]]:
    examples = load_training_examples()
    practices = load_practice_metadata()
    regions = load_region_metadata()
    outcomes = load_outcome_metrics()

    outcome_lookup: Dict[tuple, Dict[str, object]] = {}
    for outcome in outcomes:
        key = (
            outcome.get("region"),
            outcome.get("crop"),
            outcome.get("anomaly_type"),
            outcome.get("metric"),
        )
        outcome_lookup[key] = outcome

    records: List[Dict[str, object]] = []
    for example in examples:
        record = dict(example)
        practice = practices.get(str(example.get("practice_id")))
        if practice:
            record.update(
                {
                    "practice_crop": practice.crop,
                    "practice_anomaly_type": practice.anomaly_type,
                    "practice_description": practice.description,
                    "practice_tags": ",".join(practice.tags),
                    "practice_expected_outcomes": ",".join(practice.expected_outcomes),
                    "practice_evidence": ",".join(practice.evidence),
                    "practice_cost_level": practice.cost_level or "",
                    "practice_labor_intensity": practice.labor_intensity or "",
                }
            )
        region = regions.get(str(example.get("region")))
        if region:
            record.update(
                {
                    "region_country": region.country,
                    "region_climate": region.climate,
                    "region_primary_crops": ",".join(region.primary_crops),
                    "region_elevation_m": region.elevation_m if region.elevation_m is not None else "",
                    "region_soil_types": ",".join(region.soil_types),
                    "region_notes": region.notes or "",
                }
            )
        outcome_key = (
            example.get("region"),
            example.get("crop"),
            example.get("anomaly_type"),
            example.get("outcome_metric"),
        )
        outcome = outcome_lookup.get(outcome_key)
        if outcome:
            record.update(
                {
                    "outcome_before_value": outcome.get("before_value", ""),
                    "outcome_after_value": outcome.get("after_value", ""),
                    "outcome_delta": outcome.get("delta", ""),
                    "outcome_days_since_practice": outcome.get("days_since_practice", ""),
                }
            )
        records.append(record)
    return records


def write_dataset(records: List[Dict[str, object]]) -> None:
    output_path = data_file("training_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"Training dataset written to {output_path}")


def main() -> None:
    records = merge_metadata()
    build_encodings(records)
    encoded = encode_records(records)
    write_dataset(encoded)


if __name__ == "__main__":
    main()
