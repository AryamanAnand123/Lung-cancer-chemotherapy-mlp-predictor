"""Flask frontend for the lung cancer project overhaul.

This UI focuses on a clean single-patient form, batch CSV upload, and a status
page. It uses the active feature schema and routes response
predictions through the current model service.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from lungcancer.schema import FEATURE_ORDER, FEATURE_SPECS, feature_defaults
from lungcancer.service import PredictionService


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

service = PredictionService()
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def _shared_feature_names(status: Dict) -> List[str]:
    """Features shared by both deployed task models."""

    response_report = status.get("response_report") or {}
    mortality_report = status.get("mortality_report") or {}
    response_used = response_report.get("used_features")
    mortality_used = mortality_report.get("used_features")

    if isinstance(response_used, list) and isinstance(mortality_used, list):
        shared = set(response_used).intersection(mortality_used)
        filtered = [name for name in FEATURE_ORDER if name in shared]
        if filtered:
            return filtered
    return FEATURE_ORDER


@lru_cache(maxsize=1)
def _shap_top_missing_features(limit: int = 5) -> List[str]:
    """Return top missing schema features ranked by aggregated SHAP importance."""

    status = service.status()
    shared = _shared_feature_names(status)
    missing = [name for name in FEATURE_ORDER if name not in set(shared)]
    if not missing:
        return []

    scores = {name: 0.0 for name in FEATURE_ORDER}
    selected = [
        ("response", status.get("response_algorithm")),
        ("mortality", status.get("mortality_algorithm")),
    ]

    for task, algorithm in selected:
        if not algorithm:
            continue

        shap_path = REPORTS_DIR / f"{task}_{algorithm}_shap_summary.json"
        model_path = MODELS_DIR / f"{task}_{algorithm}_model.joblib"
        if not shap_path.exists() or not model_path.exists():
            continue

        try:
            payload = json.loads(shap_path.read_text(encoding="utf-8"))
            values = payload.get("mean_abs_shap", [])
            model = joblib.load(model_path)
            prep = model.named_steps.get("prep")
            feature_names = prep.get_feature_names_out().tolist() if prep is not None else []
            for transformed_name, shap_value in zip(feature_names, values):
                for raw_name in FEATURE_ORDER:
                    if transformed_name == f"num__{raw_name}" or transformed_name.startswith(
                        f"cat__{raw_name}_"
                    ):
                        scores[raw_name] += float(shap_value)
                        break
        except Exception:
            continue

    ranked = sorted(missing, key=lambda name: scores.get(name, 0.0), reverse=True)
    return ranked[:limit]


def _active_feature_names() -> List[str]:
    """Return features that are actively used by the deployed models.

    If report metadata is unavailable, fall back to the full schema.
    """

    status = service.status()
    shared = _shared_feature_names(status)
    extras = _shap_top_missing_features(limit=5)
    allowed = set(shared).union(extras)
    return [name for name in FEATURE_ORDER if name in allowed]


def _active_feature_specs() -> List:
    active_names = set(_active_feature_names())
    return [spec for spec in FEATURE_SPECS if spec.name in active_names]


def _group_active_features(active_specs: Sequence) -> Dict[str, List]:
    grouped: Dict[str, List] = {}
    for spec in active_specs:
        grouped.setdefault(spec.group, []).append(spec)
    return grouped


def _form_payload(form, active_specs: Sequence) -> dict:
    payload = feature_defaults()
    for spec in active_specs:
        raw_value = form.get(spec.name, "")
        try:
            if raw_value == "":
                payload[spec.name] = spec.default
            elif spec.dtype in {"int", "binary"}:
                payload[spec.name] = int(float(raw_value))
            elif spec.dtype == "float":
                payload[spec.name] = float(raw_value)
            else:
                payload[spec.name] = raw_value
        except (TypeError, ValueError):
            payload[spec.name] = spec.default
    return payload


@app.route("/")
def index():
    active_specs = _active_feature_specs()
    active_names = [spec.name for spec in active_specs]
    active_defaults = {spec.name: spec.default for spec in active_specs}
    return render_template(
        "index.html",
        feature_groups=_group_active_features(active_specs),
        feature_order=active_names,
        defaults=active_defaults,
        active_feature_count=len(active_names),
        response_ready=service.response_ready,
        mortality_ready=service.mortality_ready,
        status=service.status(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    active_specs = _active_feature_specs()
    features = _form_payload(request.form, active_specs)
    drug_name = request.form.get("drug_name", "Cisplatin").strip() or "Cisplatin"
    result = service.predict_response(features, drug_name=drug_name)
    return render_template(
        "result.html",
        result=result,
        features=features,
        drug_name=drug_name,
        response_ready=service.response_ready,
        mortality_ready=service.mortality_ready,
    )


@app.route("/batch", methods=["POST"])
def batch():
    active_names = _active_feature_names()
    uploaded = request.files.get("csv_file")
    drug_name = request.form.get("drug_name", "Cisplatin").strip() or "Cisplatin"
    if not uploaded:
        flash(f"Upload a CSV file with the {len(active_names)} active feature columns.", "error")
        return redirect(url_for("index"))

    try:
        dataframe = pd.read_csv(uploaded)
    except Exception:
        flash("Could not parse CSV. Please upload a valid UTF-8 CSV file.", "error")
        return redirect(url_for("index"))

    overlap = [col for col in active_names if col in dataframe.columns]
    if not overlap:
        flash(
            "CSV does not contain recognized active feature columns. Include at least one active model feature column.",
            "error",
        )
        return redirect(url_for("index"))

    missing = [col for col in active_names if col not in dataframe.columns]
    if missing:
        flash(
            f"CSV missing {len(missing)} active features; defaults will be used for missing columns.",
            "warning",
        )

    results = service.predict_batch(dataframe, drug_name=drug_name)
    preview = pd.DataFrame(results)
    return render_template(
        "batch.html",
        preview_html=preview.to_html(classes="table table-sm table-striped", index=False, border=0, escape=False),
        row_count=len(preview),
        response_ready=service.response_ready,
        mortality_ready=service.mortality_ready,
    )


@app.route("/status")
def status():
    return render_template(
        "status.html",
        status=service.status(),
        response_ready=service.response_ready,
        mortality_ready=service.mortality_ready,
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "response_model_ready": service.response_ready,
        "mortality_model_ready": service.mortality_ready,
        "feature_count": len(FEATURE_ORDER),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
