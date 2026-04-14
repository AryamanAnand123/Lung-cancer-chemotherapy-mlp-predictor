"""Flask frontend for the lung cancer project overhaul.

This UI focuses on a clean single-patient form, batch CSV upload, and a status
page. It uses the active feature schema and routes response
predictions through the current model service.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from lungcancer.schema import FEATURE_ORDER, FEATURE_SPECS, feature_defaults
from lungcancer.service import PredictionService


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

service = PredictionService()
# Ranked from previous SHAP analysis across deployed response/mortality models.
SHAP_TOP_EXTRA_FEATURES: List[str] = [
    "tumor_size_mm",
    "treatment_delay_days",
    "mets_lung_dx",
    "performance_status",
    "egfr_mutation",
]


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


def _shap_top_missing_features(limit: int = 5) -> List[str]:
    """Return cached SHAP-ranked extras without loading model artifacts at request time."""

    return [name for name in SHAP_TOP_EXTRA_FEATURES if name in FEATURE_ORDER][:limit]


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
