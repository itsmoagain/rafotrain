"""Utility functions for loading data and metadata for the practice library."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import Practice, Region, data_file, encode_crop, encode_practice, encode_soil, encode_stage


class SimpleYAMLParser:
    """A very small subset YAML parser that supports the files in this repository."""

    def parse(self, text: str) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        current_key: Optional[str] = None
        current_map: Optional[Dict[str, object]] = None

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.strip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                key = line[:-1].strip()
                current_key = key
                current_map = {}
                result[key] = current_map
                continue
            if current_map is None:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            field, value = stripped.split(":", 1)
            field = field.strip()
            value = value.strip()
            parsed_value = self._parse_value(value)
            current_map[field] = parsed_value
        return result

    @staticmethod
    def _parse_value(value: str) -> object:
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [part.strip() for part in inner.split(",")]
            return [SimpleYAMLParser._strip_quotes(part) for part in parts if part]
        return SimpleYAMLParser._coerce_scalar(SimpleYAMLParser._strip_quotes(value))

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        return value

    @staticmethod
    def _coerce_scalar(value: str) -> object:
        if value in {"null", "None", ""}:
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def _load_yaml(path: Path) -> Dict[str, Dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    parser = SimpleYAMLParser()
    return parser.parse(text)


def load_practice_metadata() -> Dict[str, Practice]:
    raw = _load_yaml(data_file("practice_metadata.yml"))
    practices: Dict[str, Practice] = {}
    for practice_id, attrs in raw.items():
        practices[practice_id] = Practice(id=practice_id, **attrs)
    return practices


def load_region_metadata() -> Dict[str, Region]:
    raw = _load_yaml(data_file("region_metadata.yml"))
    regions: Dict[str, Region] = {}
    for region_id, attrs in raw.items():
        regions[region_id] = Region(id=region_id, **attrs)
    return regions


def load_training_examples() -> List[Dict[str, object]]:
    return _read_csv_records(data_file("training_examples.csv"))


def load_outcome_metrics() -> List[Dict[str, object]]:
    path = data_file("outcome_metrics.csv")
    if not path.exists():
        return []
    return _read_csv_records(path)


def load_training_dataset() -> List[Dict[str, object]]:
    path = data_file("training_dataset.csv")
    if not path.exists():
        raise FileNotFoundError(
            "training_dataset.csv not found. Run scripts/build_training_dataset.py first."
        )
    return _read_csv_records(path)


def load_encodings() -> Dict[str, Dict[str, int]]:
    from .schema import ENCODINGS_PATH

    if not ENCODINGS_PATH.exists():
        raise FileNotFoundError(
            "encodings.json not found. Run scripts/build_training_dataset.py first."
        )
    with ENCODINGS_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def encode_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    encoded: List[Dict[str, object]] = []
    for record in records:
        item = dict(record)
        if "phenology_stage" in record:
            item["phenology_stage_encoded"] = encode_stage(
                record.get("phenology_stage") or None
            )
        if "soil_type" in record:
            item["soil_type_encoded"] = encode_soil(record.get("soil_type") or None)
        if "crop" in record:
            item["crop_encoded"] = encode_crop(record.get("crop") or None)
        if "practice_id" in record:
            item["practice_id_encoded"] = encode_practice(record.get("practice_id") or None)
        encoded.append(item)
    return encoded


def _read_csv_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        records: List[Dict[str, object]] = []
        for row in reader:
            records.append({key: _coerce_csv_value(value) for key, value in row.items()})
        return records


def _coerce_csv_value(value: str) -> object:
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"na", "null", "none"}:
        return ""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
