"""Feature schema used by the Flask frontend and future model training.

This schema is aligned to the updated SEER export and keeps compatible fields
for cross-source harmonization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class FeatureSpec:
    """Describe a single form field or model input."""

    name: str
    label: str
    group: str
    dtype: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: Sequence[Any] = ()
    help_text: str = ""


FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("age", "Age (years)", "Demographic", "int", 65, 18, 90, 1, help_text="Age at diagnosis."),
    FeatureSpec("sex", "Sex", "Demographic", "categorical", "Male", choices=("Male", "Female")),
    FeatureSpec("race_recode", "Race (SEER recode)", "Demographic", "categorical", "White", choices=("White", "Black", "Asian_or_Pacific_Islander", "American_Indian_Alaska_Native", "Unknown", "Other")),
    FeatureSpec("marital_status", "Marital Status", "Demographic", "categorical", "Married", choices=("Married", "Single", "Divorced", "Widowed", "Separated", "Unmarried_partner", "Unknown")),
    FeatureSpec("histology", "Histology", "Clinical", "categorical", "Adenocarcinoma", choices=("Adenocarcinoma", "Squamous", "Small_cell", "Large_cell", "Other")),
    FeatureSpec("cancer_stage", "TNM Stage", "Clinical", "categorical", "IIIA", choices=("IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV")),
    FeatureSpec("t_stage", "T Stage", "Clinical", "categorical", "T2a", choices=("T0", "T1", "T1a", "T1b", "T1c", "T2", "T2a", "T2b", "T3", "T4", "TX", "Unknown")),
    FeatureSpec("n_stage", "N Stage", "Clinical", "categorical", "N0", choices=("N0", "N1", "N2", "N3", "NX", "Unknown")),
    FeatureSpec("m_stage", "M Stage", "Clinical", "categorical", "M0", choices=("M0", "M1", "M1a", "M1b", "M1c", "MX", "Unknown")),
    FeatureSpec("performance_status", "Performance Status", "Clinical", "int", 1, 0, 4, 1, help_text="ECOG-like status, lower is better."),
    FeatureSpec("tumor_size_mm", "Tumor Size (mm)", "Clinical", "float", 35.0, 0.0, 500.0, 1.0),
    FeatureSpec("mets_lung_dx", "Lung Metastasis at Diagnosis", "Clinical", "binary", 0, choices=(0, 1)),
    FeatureSpec("chemotherapy_cycles", "Chemotherapy Cycles", "Treatment", "int", 1, 0, 10, 1),
    FeatureSpec("radiation_recode", "Radiation Category", "Treatment", "categorical", "None_or_Unknown", choices=("None_or_Unknown", "Beam_radiation", "Radioactive_implants", "Radioisotopes", "Combination_radiation", "Refused", "Recommended_unknown", "Other")),
    FeatureSpec("treatment_delay_days", "Days to Treatment", "Treatment", "float", 30.0, 0.0, 400.0, 1.0),
    FeatureSpec("egfr_mutation", "EGFR Mutation", "Molecular", "categorical", "Wild_type", choices=("Wild_type", "Exon19del", "L858R", "Other")),
]

FEATURE_ORDER: List[str] = [spec.name for spec in FEATURE_SPECS]
FEATURE_GROUPS: Dict[str, List[FeatureSpec]] = {}
for spec in FEATURE_SPECS:
    FEATURE_GROUPS.setdefault(spec.group, []).append(spec)

def feature_defaults() -> Dict[str, Any]:
    """Return default values for the supported form fields."""

    return {spec.name: spec.default for spec in FEATURE_SPECS}


def normalize_value(spec: FeatureSpec, value: Any) -> Any:
    """Convert raw form input into a typed value."""

    if value is None or value == "":
        return spec.default

    if spec.dtype == "int" or spec.dtype == "binary":
        return int(float(value))
    if spec.dtype == "float":
        return float(value)
    return str(value)


def normalize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a raw payload to the schema types."""

    normalized: Dict[str, Any] = {}
    for spec in FEATURE_SPECS:
        normalized[spec.name] = normalize_value(spec, payload.get(spec.name, spec.default))
    return normalized


def group_features() -> Dict[str, List[FeatureSpec]]:
    """Return the grouped schema for rendering the UI."""

    return FEATURE_GROUPS
