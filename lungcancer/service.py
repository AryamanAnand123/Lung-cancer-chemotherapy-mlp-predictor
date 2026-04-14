"""Prediction service used by the Flask app.

This module serves tree-stack models only (LightGBM/XGBoost/TabNet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .schema import FEATURE_ORDER, feature_defaults

try:
    from .tree_stack import TreePredictor

    TREE_STACK_AVAILABLE = True
    TREE_STACK_IMPORT_ERROR = None
except Exception:
    TreePredictor = None  # type: ignore[assignment]
    TREE_STACK_AVAILABLE = False
    TREE_STACK_IMPORT_ERROR = "tree_stack import failed"


RESPONSE_DECISION_THRESHOLD = 0.50
MORTALITY_DECISION_THRESHOLD = 0.50
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
ALGORITHMS = ("lightgbm", "xgboost", "tabnet")


def _report_path(task: str, algorithm: str) -> Path:
    return REPORTS_DIR / f"{task}_{algorithm}_nested_cv.json"


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class PredictionService:
    """Provide tree-stack predictions and model status for the Flask app."""

    def __init__(self) -> None:
        self._load_errors: Dict[str, List[str]] = {"response": [], "mortality": []}
        self._response_tree_predictor, self._response_algorithm = self._load_tree_predictor(task="response")
        self._mortality_tree_predictor, self._mortality_algorithm = self._load_tree_predictor(task="mortality")
        self._response_threshold = self._load_task_threshold(
            task="response",
            algorithm=self._response_algorithm,
            default=RESPONSE_DECISION_THRESHOLD,
        )
        self._mortality_threshold = self._load_task_threshold(
            task="mortality",
            algorithm=self._mortality_algorithm,
            default=MORTALITY_DECISION_THRESHOLD,
        )
        self._response_report = self._task_report_summary("response", self._response_algorithm)
        self._mortality_report = self._task_report_summary("mortality", self._mortality_algorithm)

    def _task_report_summary(self, task: str, algorithm: str | None) -> Dict[str, Any] | None:
        if algorithm is None:
            return None
        payload = _safe_read_json(_report_path(task=task, algorithm=algorithm))
        if payload is None:
            return None
        notes = payload.get("notes", [])
        dropped: List[str] = []
        for note in notes:
            if isinstance(note, str) and note.startswith("dropped_features="):
                dropped = [part for part in note.split("=", 1)[1].split(",") if part]
        used_features = [name for name in FEATURE_ORDER if name not in set(dropped)]
        return {
            "task": task,
            "algorithm": algorithm,
            "auc": payload.get("outer_mean", {}).get("auc"),
            "brier": payload.get("outer_mean", {}).get("brier"),
            "threshold": payload.get("best_threshold", {}).get("threshold"),
            "used_features": used_features,
            "dropped_features": dropped,
            "report": str(_report_path(task=task, algorithm=algorithm)),
        }

    def _load_tree_predictor(self, task: str) -> tuple[TreePredictor | None, str | None]:
        if not TREE_STACK_AVAILABLE:
            if TREE_STACK_IMPORT_ERROR:
                self._load_errors[task].append(TREE_STACK_IMPORT_ERROR)
            return None, None

        ranked_algorithms: list[tuple[float, float, str]] = []
        for algorithm in ALGORITHMS:
            payload = _safe_read_json(_report_path(task=task, algorithm=algorithm))
            auc = float("-inf")
            brier = float("inf")
            if payload is not None:
                auc = float(payload.get("outer_mean", {}).get("auc", float("-inf")))
                brier = float(payload.get("outer_mean", {}).get("brier", float("inf")))
            ranked_algorithms.append((auc, -brier, algorithm))

        for _, _, algorithm in sorted(ranked_algorithms, reverse=True):
            try:
                predictor = TreePredictor.load(task=task, algorithm=algorithm, strict=True)
            except Exception as exc:
                self._load_errors[task].append(f"{algorithm}: {type(exc).__name__}: {exc}")
                predictor = None
            if predictor is not None:
                return predictor, algorithm
            self._load_errors[task].append(f"{algorithm}: model unavailable")
        return None, None

    def _load_task_threshold(self, task: str, algorithm: str | None, default: float) -> float:
        if algorithm is None:
            return default
        payload = _safe_read_json(_report_path(task=task, algorithm=algorithm))
        if payload is None:
            return default
        threshold = float(payload.get("best_threshold", {}).get("threshold", default))
        if 0.0 <= threshold <= 1.0:
            return threshold
        return default

    @property
    def response_ready(self) -> bool:
        return self._response_tree_predictor is not None

    @property
    def mortality_ready(self) -> bool:
        return self._mortality_tree_predictor is not None

    def status(self) -> Dict[str, Any]:
        mode = "unavailable"
        if self._response_tree_predictor is not None and self._mortality_tree_predictor is not None:
            mode = "tree-stack split models (lightgbm/xgboost/tabnet + conformal)"
        return {
            "response_model_ready": self.response_ready,
            "mortality_model_ready": self.mortality_ready,
            "feature_count": len(FEATURE_ORDER),
            "feature_names": FEATURE_ORDER,
            "response_threshold": self._response_threshold,
            "mortality_threshold": self._mortality_threshold,
            "response_algorithm": self._response_algorithm,
            "mortality_algorithm": self._mortality_algorithm,
            "response_report": self._response_report,
            "mortality_report": self._mortality_report,
            "load_errors": self._load_errors,
            "mode": mode,
        }

    def predict_response(self, features: Dict[str, Any], drug_name: str = "Cisplatin") -> Dict[str, Any]:
        """Predict both treatment response and mortality."""

        if not self.response_ready or not self.mortality_ready:
            response_err = "; ".join(self._load_errors.get("response", [])[:2])
            mortality_err = "; ".join(self._load_errors.get("mortality", [])[:2])
            return {
                "probability": None,
                "label": "Model not trained",
                "confidence": "0.0%",
                "confidence_level": "Low",
                "mortality_probability": None,
                "mortality_label": "Model not trained",
                "drug_name": drug_name,
                "model_type": "tree-stack-unavailable",
                "mortality_message": f"Model unavailable. Response load: {response_err or 'none'}. Mortality load: {mortality_err or 'none'}.",
                "drug_info": None,
            }

        response_prob: float | None = None
        mortality_prob: float | None = None
        response_interval: Dict[str, float] | None = None
        mortality_interval: Dict[str, float] | None = None

        if self._response_tree_predictor is not None:
            response_prob = float(self._response_tree_predictor.predict_probability(features))
            response_interval = self._response_tree_predictor.predict_interval(features)

        if self._mortality_tree_predictor is not None:
            mortality_prob = float(self._mortality_tree_predictor.predict_probability(features))
            mortality_interval = self._mortality_tree_predictor.predict_interval(features)
        drug_info = None

        if response_prob is None:
            response_label = "Model not trained"
            confidence = 0.0
            confidence_level = "Low"
        else:
            response_label = (
                "Positive response expected"
                if response_prob >= self._response_threshold
                else "Negative response expected"
            )
            confidence = max(response_prob, 1.0 - response_prob)
            if confidence >= 0.8:
                confidence_level = "High"
            elif confidence >= 0.7:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

        if mortality_prob is None:
            mortality_label = "Model not trained"
        else:
            mortality_label = (
                "Higher mortality risk"
                if mortality_prob >= self._mortality_threshold
                else "Lower mortality risk"
            )

        model_type = (
            f"Tree stack split models (response={self._response_algorithm}, mortality={self._mortality_algorithm})"
        )

        return {
            "probability": response_prob,
            "label": response_label,
            "confidence": f"{confidence * 100:.1f}%",
            "confidence_level": confidence_level,
            "mortality_probability": mortality_prob,
            "mortality_label": mortality_label,
            "drug_name": drug_name,
            "model_type": model_type,
            "response_algorithm": self._response_algorithm,
            "mortality_algorithm": self._mortality_algorithm,
            "mortality_message": "Response and mortality are predicted independently to avoid cross-task interference.",
            "response_conformal": response_interval,
            "mortality_conformal": mortality_interval,
            "drug_info": drug_info,
        }

    def predict_batch(self, dataframe: pd.DataFrame, drug_name: str = "Cisplatin") -> List[Dict[str, Any]]:
        """Predict response for each row of a dataframe."""

        records: List[Dict[str, Any]] = []
        for _, row in dataframe.iterrows():
            features = {name: row.get(name, default) for name, default in feature_defaults().items()}
            records.append(self.predict_response(features, drug_name=drug_name))
        return records
