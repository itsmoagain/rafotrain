"""Model helpers for training and scoring practice effectiveness correlations."""
from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FEATURE_COLUMNS = [
    "spi",
    "ndvi_anomaly",
    "soil_moisture",
    "temp_mean",
    "phenology_stage_encoded",
    "soil_type_encoded",
    "practice_id_encoded",
    "crop_encoded",
]
TARGET_COLUMN = "effectiveness_score"


def _mse(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


class _DecisionTree:
    def __init__(self, max_depth: int, min_samples_split: int = 2, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random = random.Random(random_state)
        self.tree: Dict[str, object] = {}
        self.feature_importances: List[float] = [0.0 for _ in FEATURE_COLUMNS]

    def fit(self, rows: List[List[float]], targets: List[float]) -> None:
        self.tree = self._build_tree(rows, targets, depth=0)

    def _build_tree(
        self, rows: List[List[float]], targets: List[float], depth: int
    ) -> Dict[str, object]:
        node = {
            "type": "leaf",
            "value": sum(targets) / len(targets) if targets else 0.0,
        }

        if depth >= self.max_depth or len(set(targets)) <= 1 or len(rows) < self.min_samples_split:
            return node

        best_feature = None
        best_threshold = None
        best_score = math.inf
        best_split: Optional[Tuple[List[List[float]], List[float], List[List[float]], List[float]]] = None

        feature_indices = list(range(len(FEATURE_COLUMNS)))
        self.random.shuffle(feature_indices)
        max_features = max(1, int(math.sqrt(len(FEATURE_COLUMNS))))
        candidate_features = feature_indices[:max_features]

        for feature_index in candidate_features:
            values = sorted({row[feature_index] for row in rows})
            if len(values) <= 1:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left_rows: List[List[float]] = []
                left_targets: List[float] = []
                right_rows: List[List[float]] = []
                right_targets: List[float] = []
                for row, target in zip(rows, targets):
                    if row[feature_index] <= threshold:
                        left_rows.append(row)
                        left_targets.append(target)
                    else:
                        right_rows.append(row)
                        right_targets.append(target)
                if not left_rows or not right_rows:
                    continue
                left_mse = _mse(left_targets)
                right_mse = _mse(right_targets)
                score = (len(left_rows) / len(rows)) * left_mse + (
                    len(right_rows) / len(rows)
                ) * right_mse
                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold
                    best_split = (left_rows, left_targets, right_rows, right_targets)

        if best_split is None or best_feature is None or best_threshold is None:
            return node

        left_rows, left_targets, right_rows, right_targets = best_split
        parent_mse = _mse(targets)
        improvement = parent_mse - best_score
        self.feature_importances[best_feature] += max(improvement, 0.0)

        node = {
            "type": "split",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_rows, left_targets, depth + 1),
            "right": self._build_tree(right_rows, right_targets, depth + 1),
        }
        return node

    def predict_row(self, row: List[float]) -> float:
        node = self.tree
        while node.get("type") == "split":
            feature = node["feature"]
            threshold = node["threshold"]
            if row[feature] <= threshold:
                node = node["left"]  # type: ignore[assignment]
            else:
                node = node["right"]  # type: ignore[assignment]
        return node.get("value", 0.0)


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random = random.Random(random_state)
        self.trees: List[_DecisionTree] = []
        self.feature_importances_: List[float] = [0.0 for _ in FEATURE_COLUMNS]

    def fit(self, rows: List[List[float]], targets: List[float]) -> None:
        self.trees = []
        n_samples = len(rows)
        for index in range(self.n_estimators):
            tree_seed = self.random.randint(0, 10_000_000)
            tree = _DecisionTree(self.max_depth, random_state=tree_seed)
            sampled_rows: List[List[float]] = []
            sampled_targets: List[float] = []
            for _ in range(n_samples):
                choice = self.random.randrange(n_samples)
                sampled_rows.append(rows[choice])
                sampled_targets.append(targets[choice])
            tree.fit(sampled_rows, sampled_targets)
            self.trees.append(tree)

        total_importances = [0.0 for _ in FEATURE_COLUMNS]
        for tree in self.trees:
            for idx, value in enumerate(tree.feature_importances):
                total_importances[idx] += value
        total = sum(total_importances)
        if total > 0:
            self.feature_importances_ = [value / total for value in total_importances]
        else:
            self.feature_importances_ = total_importances

    def predict(self, rows: List[List[float]]) -> List[float]:
        predictions: List[float] = []
        for row in rows:
            tree_values = [tree.predict_row(row) for tree in self.trees]
            predictions.append(sum(tree_values) / len(tree_values) if tree_values else 0.0)
        return predictions

    def save(self, path: Path) -> None:
        with path.open("wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path: Path) -> "RandomForestRegressor":
        with path.open("rb") as fp:
            model = pickle.load(fp)
        if not isinstance(model, RandomForestRegressor):
            raise TypeError("Expected a RandomForestRegressor in the saved model file.")
        return model


@dataclass
class PracticeCorrelationModel:
    """Wrapper around the custom RandomForestRegressor for practice effectiveness."""

    model: RandomForestRegressor

    @classmethod
    def create_default(cls) -> "PracticeCorrelationModel":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        return cls(model=model)

    def fit(self, records: List[Dict[str, object]]) -> None:
        rows, targets = self._prepare(records)
        self.model.fit(rows, targets)

    def predict(self, records: List[Dict[str, object]]) -> List[float]:
        rows, _ = self._prepare(records, include_target=False)
        return self.model.predict(rows)

    def feature_importances(self) -> List[float]:
        return list(self.model.feature_importances_)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: Path) -> "PracticeCorrelationModel":
        model = RandomForestRegressor.load(path)
        return cls(model=model)

    def _prepare(
        self, records: List[Dict[str, object]], include_target: bool = True
    ) -> Tuple[List[List[float]], List[float]]:
        rows: List[List[float]] = []
        targets: List[float] = []
        for record in records:
            row: List[float] = []
            for feature in FEATURE_COLUMNS:
                value = record.get(feature)
                if isinstance(value, (int, float)):
                    row.append(float(value))
                elif value == "":
                    row.append(0.0)
                else:
                    row.append(float(value) if value is not None else 0.0)
            rows.append(row)
            if include_target:
                target_value = record.get(TARGET_COLUMN, 0.0)
                targets.append(float(target_value) if target_value is not None else 0.0)
        return rows, targets
