"""Evaluate the trained correlation model and report key metrics."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statistics import mean
from typing import List

from library.correlation_model import FEATURE_COLUMNS, PracticeCorrelationModel
from library.loader import load_training_dataset
from library.schema import REPO_ROOT
from scripts.train_practice_correlations import _prepare_records

MODEL_PATH = REPO_ROOT / "models" / "correlation_model.pkl"


def r2_score(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    actual_mean = mean(actual)
    ss_tot = sum((value - actual_mean) ** 2 for value in actual)
    ss_res = sum((a - b) ** 2 for a, b in zip(actual, predicted))
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def main() -> None:
    dataset = load_training_dataset()
    prepared = _prepare_records(dataset)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Correlation model not found. Run 'python scripts/train_practice_correlations.py' before "
            "evaluating."
        )
    model = PracticeCorrelationModel.load(MODEL_PATH)
    predictions = model.predict(prepared)
    actual = [float(record["effectiveness_score"]) for record in prepared]

    score = r2_score(actual, predictions)
    print("R^2 on training dataset: {:.3f}".format(score))
    print("Feature importances:")
    for name, importance in zip(FEATURE_COLUMNS, model.feature_importances()):
        print(f"  {name}: {importance:.3f}")


if __name__ == "__main__":
    main()
