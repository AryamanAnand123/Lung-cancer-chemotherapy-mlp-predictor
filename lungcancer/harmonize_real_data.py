from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
DESKTOP_DIR = Path.home() / "OneDrive" / "Desktop"

HARMONIZED_PATH = PROCESSED_DIR / "harmonized_real_world.csv"
MORTALITY_PATH = PROCESSED_DIR / "mortality_training_ready.csv"
RESPONSE_PATH = PROCESSED_DIR / "response_training_ready.csv"
READINESS_JSON = REPORT_DIR / "data_readiness_report.json"
READINESS_CSV = REPORT_DIR / "feature_coverage_report.csv"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "race_recode",
    "marital_status",
    "histology",
    "cancer_stage",
    "t_stage",
    "n_stage",
    "m_stage",
    "performance_status",
    "tumor_size_mm",
    "mets_lung_dx",
    "chemotherapy_cycles",
    "radiation_recode",
    "treatment_delay_days",
    "egfr_mutation",
]

META_COLUMNS = ["source", "source_study", "case_id", "sample_id"]
LABEL_COLUMNS = ["mortality_1yr", "response"]


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "not available", "[not available]", "unknown"}:
        return None
    return text


def _parse_age_recode(value: object) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    match_range = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if match_range:
        low = float(match_range.group(1))
        high = float(match_range.group(2))
        return (low + high) / 2.0
    match_plus = re.search(r"(\d+)\+", text)
    if match_plus:
        low = float(match_plus.group(1))
        return low + 2.0
    match_int = re.search(r"(\d+)", text)
    if match_int:
        return float(match_int.group(1))
    return None


def _normalize_sex(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if "male" in lower:
        return "Male"
    if "female" in lower:
        return "Female"
    return text


def _normalize_stage(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    text = text.replace("stage", "").replace("Stage", "").strip()
    text = text.replace(" ", "")
    if text == "":
        return None
    return text


def _normalize_histology(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if "adeno" in lower:
        return "Adenocarcinoma"
    if "squamous" in lower:
        return "Squamous"
    if "small" in lower and "cell" in lower:
        return "Small_cell"
    if "large" in lower and "cell" in lower:
        return "Large_cell"
    return text


def _normalize_race(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if "white" in lower:
        return "White"
    if "black" in lower:
        return "Black"
    if "asian" in lower or "pacific" in lower:
        return "Asian_or_Pacific_Islander"
    if "indian" in lower or "alaska" in lower:
        return "American_Indian_Alaska_Native"
    if "unknown" in lower:
        return "Unknown"
    return "Other"


def _normalize_marital_status(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if "married" in lower:
        return "Married"
    if "single" in lower and "never" in lower:
        return "Single"
    if "divorc" in lower:
        return "Divorced"
    if "widow" in lower:
        return "Widowed"
    if "separat" in lower:
        return "Separated"
    if "domestic" in lower or "unmarried" in lower:
        return "Unmarried_partner"
    if "unknown" in lower:
        return "Unknown"
    return "Unknown"


def _normalize_tnm(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    upper = text.upper().replace(" ", "")
    if upper in {"BLANK(S)", "BLANK", "UNKNOWN", "N/A", "NA"}:
        return "Unknown"
    return upper


def _parse_three_digit_numeric(value: object) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    if text.lower() in {"blank(s)", "unable to calculate", "unknown"}:
        return None
    match = re.search(r"(\d{1,3})", text)
    if match:
        val = float(match.group(1))
        if val >= 990:
            return None
        return val
    return None


def _normalize_yes_no_unknown(value: object) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if lower.startswith("yes"):
        return 1.0
    if lower.startswith("no"):
        return 0.0
    return None


def _normalize_radiation(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lower = text.lower()
    if "none/unknown" in lower:
        return "None_or_Unknown"
    if "beam" in lower:
        return "Beam_radiation"
    if "implants" in lower or "brachytherapy" in lower:
        return "Radioactive_implants"
    if "radioisotopes" in lower:
        return "Radioisotopes"
    if "combination" in lower:
        return "Combination_radiation"
    if "refused" in lower:
        return "Refused"
    if "recommended" in lower and "unknown" in lower:
        return "Recommended_unknown"
    return "Other"


def _compute_mortality_1yr_from_seer(survival_months: pd.Series, vital_status: pd.Series) -> pd.Series:
    sm = pd.to_numeric(survival_months, errors="coerce")
    vital = vital_status.astype(str).str.lower()
    out = pd.Series([np.nan] * len(sm), dtype="float")
    out[(vital == "dead") & (sm <= 12)] = 1.0
    out[(vital == "dead") & (sm > 12)] = 0.0
    out[(vital == "alive") & (sm >= 12)] = 0.0
    return out


def _extract_gene_symbol(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    match = re.search(r"hugoGeneSymbol=([A-Za-z0-9_\-]+)", text)
    if match:
        return match.group(1).upper()
    if text.isalpha() and len(text) <= 12:
        return text.upper()
    return None


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS)


def _init_output(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS)


def _harmonize_seer() -> pd.DataFrame:
    seer_dir = RAW_DIR / "seer"
    files = [seer_dir / "exportnew.csv", DESKTOP_DIR / "exportnew.csv"]
    data_file = next((f for f in files if f.exists()), None)
    if data_file is None:
        return _empty_frame()

    df = pd.read_csv(data_file, low_memory=False)

    age_col = "Age recode with <1 year olds and 90+"
    sex_col = "Sex"
    race_col = "Race recode (W, B, AI, API)"
    marital_col = "Marital status at diagnosis"
    hist_col = "Histologic Type ICD-O-3"
    stage_col = "Derived AJCC Stage Group, 7th ed (2010-2015)"
    t_col = "Derived AJCC T, 7th ed (2010-2015)"
    n_col = "Derived AJCC N, 7th ed (2010-2015)"
    m_col = "Derived AJCC M, 7th ed (2010-2015)"
    chemo_col = "Chemotherapy recode (yes, no/unk)"
    radiation_col = "Radiation recode"
    delay_col = "Time from diagnosis to treatment in days recode"
    tumor_size_recode_col = "Tumor Size Over Time Recode (1988+)"
    tumor_size_summary_col = "Tumor Size Summary (2016+)"
    mets_lung_col = "SEER Combined Mets at DX-lung (2010+)"
    survival_col = "Survival months"
    vital_col = "Vital status recode (study cutoff used)"

    out = _init_output(df.index)
    out["source"] = "SEER"
    out["source_study"] = "SEER_LUNG_EXPORTNEW"
    out["case_id"] = np.nan
    out["sample_id"] = np.nan
    out["age"] = df[age_col].map(_parse_age_recode) if age_col in df.columns else np.nan
    out["sex"] = df[sex_col].map(_normalize_sex) if sex_col in df.columns else np.nan
    out["race_recode"] = df[race_col].map(_normalize_race) if race_col in df.columns else np.nan
    out["marital_status"] = df[marital_col].map(_normalize_marital_status) if marital_col in df.columns else np.nan
    out["histology"] = df[hist_col].map(_normalize_histology) if hist_col in df.columns else np.nan
    out["cancer_stage"] = df[stage_col].map(_normalize_stage) if stage_col in df.columns else np.nan
    out["t_stage"] = df[t_col].map(_normalize_tnm) if t_col in df.columns else np.nan
    out["n_stage"] = df[n_col].map(_normalize_tnm) if n_col in df.columns else np.nan
    out["m_stage"] = df[m_col].map(_normalize_tnm) if m_col in df.columns else np.nan

    out["performance_status"] = np.nan
    out["tumor_size_mm"] = np.nan
    if tumor_size_recode_col in df.columns:
        out["tumor_size_mm"] = df[tumor_size_recode_col].map(_parse_three_digit_numeric)
    if tumor_size_summary_col in df.columns:
        out["tumor_size_mm"] = out["tumor_size_mm"].fillna(df[tumor_size_summary_col].map(_parse_three_digit_numeric))

    out["mets_lung_dx"] = df[mets_lung_col].map(_normalize_yes_no_unknown) if mets_lung_col in df.columns else np.nan
    out["chemotherapy_cycles"] = df[chemo_col].map(_normalize_yes_no_unknown) if chemo_col in df.columns else np.nan
    out["radiation_recode"] = df[radiation_col].map(_normalize_radiation) if radiation_col in df.columns else np.nan
    out["treatment_delay_days"] = df[delay_col].map(_parse_three_digit_numeric) if delay_col in df.columns else np.nan
    out["egfr_mutation"] = np.nan

    if survival_col in df.columns and vital_col in df.columns:
        out["mortality_1yr"] = _compute_mortality_1yr_from_seer(df[survival_col], df[vital_col])
    else:
        out["mortality_1yr"] = np.nan
    out["response"] = np.nan

    return out


def _harmonize_gdc(cbio_egfr_patient_map: dict[str, str] | None = None) -> pd.DataFrame:
    gdc_file = RAW_DIR / "gdc" / "cases_luad_lusc.csv"
    if not gdc_file.exists():
        return _empty_frame()

    df = pd.read_csv(gdc_file, low_memory=False)
    out = _init_output(df.index)

    out["source"] = "GDC"
    out["source_study"] = df["project_id"].fillna("GDC")
    out["case_id"] = df["case_id"]
    out["sample_id"] = np.nan

    ecog_map, karn_map, prior_treatment_map = _build_gdc_case_maps()

    out["age"] = pd.to_numeric(df.get("age_at_diagnosis_days"), errors="coerce") / 365.25
    out["sex"] = df.get("gender").map(_normalize_sex)
    out["histology"] = df.get("primary_diagnosis").map(_normalize_histology)
    out["cancer_stage"] = df.get("tumor_stage").map(_normalize_stage)
    ecog = pd.to_numeric(df.get("ecog_worst", pd.Series([np.nan] * len(df))), errors="coerce")
    if ecog.isna().all() and ecog_map:
        ecog = df.get("case_id", pd.Series([None] * len(df))).astype(str).map(ecog_map)
    karn = pd.to_numeric(df.get("karnofsky_best", pd.Series([np.nan] * len(df))), errors="coerce")
    if karn.isna().all() and karn_map:
        karn = df.get("case_id", pd.Series([None] * len(df))).astype(str).map(karn_map)
    perf = ecog.copy()
    perf[(perf.isna()) & (karn.notna())] = np.clip(np.round((100.0 - karn[(perf.isna()) & (karn.notna())]) / 20.0), 0, 4)
    out["performance_status"] = perf
    out["chemotherapy_cycles"] = np.nan
    if prior_treatment_map:
        out["chemotherapy_cycles"] = df.get("case_id", pd.Series([None] * len(df))).astype(str).map(prior_treatment_map)
    out["egfr_mutation"] = np.nan
    if cbio_egfr_patient_map:
        submitters = df.get("submitter_id", pd.Series([None] * len(df))).astype(str)
        out["egfr_mutation"] = submitters.map(cbio_egfr_patient_map)

    vital = df.get("vital_status", pd.Series([None] * len(df))).astype(str).str.lower()
    dtd = pd.to_numeric(df.get("days_to_death"), errors="coerce")
    dlfu = pd.to_numeric(df.get("days_to_last_follow_up"), errors="coerce")

    mortality = pd.Series([np.nan] * len(df), dtype="float")
    mortality[(vital == "dead") & (dtd <= 365)] = 1.0
    mortality[(vital == "dead") & (dtd > 365)] = 0.0
    mortality[(vital == "dead") & (dtd.isna()) & (dlfu <= 365)] = 1.0
    mortality[(vital == "dead") & (dtd.isna()) & (dlfu > 365)] = 0.0
    mortality[(vital != "dead") & (dlfu >= 365)] = 0.0
    out["mortality_1yr"] = mortality

    tf_count = pd.to_numeric(df.get("tf_tumor_free_count"), errors="coerce")
    wt_count = pd.to_numeric(df.get("wt_with_tumor_count"), errors="coerce")
    best_response = df.get("best_disease_response", pd.Series([None] * len(df))).astype(str).str.lower()

    response = pd.Series([np.nan] * len(df), dtype="float")
    response[(wt_count > 0) | best_response.str.contains("with tumor", na=False)] = 0.0
    response[(response.isna()) & ((tf_count > 0) | best_response.str.contains("tumor free", na=False))] = 1.0
    out["response"] = response

    return out


def _build_cbio_egfr_map(cbio_dir: Path) -> dict[str, str]:
    egfr_samples: set[str] = set()
    all_samples: set[str] = set()

    for mut_file in cbio_dir.glob("**/mutations_panel_*_mutations.csv"):
        mdf = pd.read_csv(mut_file, low_memory=False)
        if "sampleId" not in mdf.columns or "gene" not in mdf.columns:
            continue

        symbols = mdf["gene"].map(_extract_gene_symbol)
        sample_ids = mdf["sampleId"].astype(str)
        all_samples.update(sample_ids.tolist())
        egfr_samples.update(sample_ids[symbols == "EGFR"].tolist())

    result: dict[str, str] = {}
    for sid in all_samples:
        result[sid] = "Exon19del" if sid in egfr_samples else "Wild_type"
    return result


def _build_cbio_egfr_patient_map(cbio_dir: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for mut_file in cbio_dir.glob("**/mutations_panel_*_mutations.csv"):
        mdf = pd.read_csv(mut_file, low_memory=False)
        if "patientId" not in mdf.columns or "gene" not in mdf.columns:
            continue

        symbols = mdf["gene"].map(_extract_gene_symbol)
        patients = mdf["patientId"].astype(str)
        for pid, sym in zip(patients, symbols):
            if sym is None:
                continue
            prev = status.get(pid)
            if sym == "EGFR":
                status[pid] = "Exon19del"
            elif prev is None:
                status[pid] = "Wild_type"
    return status


def _build_gdc_case_maps() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    gdc_json = RAW_DIR / "gdc" / "cases_luad_lusc.json"
    if not gdc_json.exists():
        return {}, {}, {}

    data = json.loads(gdc_json.read_text(encoding="utf-8-sig"))
    ecog_map: dict[str, float] = {}
    karn_map: dict[str, float] = {}
    prior_treatment_map: dict[str, float] = {}

    for case in data:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            continue

        ecog_vals: list[float] = []
        karn_vals: list[float] = []
        for fu in case.get("follow_ups", []) or []:
            ev = pd.to_numeric(pd.Series([fu.get("ecog_performance_status")]), errors="coerce").iloc[0]
            kv = pd.to_numeric(pd.Series([fu.get("karnofsky_performance_status")]), errors="coerce").iloc[0]
            if pd.notna(ev):
                ecog_vals.append(float(ev))
            if pd.notna(kv):
                karn_vals.append(float(kv))

        if ecog_vals:
            ecog_map[case_id] = float(np.max(ecog_vals))
        if karn_vals:
            karn_map[case_id] = float(np.max(karn_vals))

        prior_vals: list[str] = []
        for dg in case.get("diagnoses", []) or []:
            pv = _clean_text(dg.get("prior_treatment"))
            if pv is not None:
                prior_vals.append(pv.lower())
        if any(v == "yes" for v in prior_vals):
            prior_treatment_map[case_id] = 1.0
        elif any(v == "no" for v in prior_vals):
            prior_treatment_map[case_id] = 0.0

    return ecog_map, karn_map, prior_treatment_map


def _harmonize_cbio() -> pd.DataFrame:
    cbio_dir = RAW_DIR / "cbioportal"
    if not cbio_dir.exists():
        return _empty_frame()

    egfr_map = _build_cbio_egfr_map(cbio_dir)
    frames: list[pd.DataFrame] = []

    for study_dir in cbio_dir.iterdir():
        clinical_path = study_dir / "clinical_data.csv"
        if not clinical_path.exists():
            continue

        cdf = pd.read_csv(clinical_path, low_memory=False)
        needed_cols = {"sampleId", "clinicalAttributeId", "value"}
        if not needed_cols.issubset(set(cdf.columns)):
            continue

        pivot = (
            cdf[["sampleId", "patientId", "studyId", "clinicalAttributeId", "value"]]
            .dropna(subset=["sampleId", "clinicalAttributeId"])
            .pivot_table(
                index=["sampleId", "patientId", "studyId"],
                columns="clinicalAttributeId",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        out = _init_output(pivot.index)
        out["source"] = "cBio"
        out["source_study"] = pivot["studyId"]
        out["case_id"] = pivot["patientId"]
        out["sample_id"] = pivot["sampleId"]

        out["age"] = pd.to_numeric(pivot.get("AGE_AT_SEQ_REPORTED_YEARS"), errors="coerce")
        out["sex"] = np.nan
        out["histology"] = pivot.get("PREDOMINANT_HISTOLOGIC_SUBTYPE", pd.Series([None] * len(pivot))).map(_normalize_histology)
        out["cancer_stage"] = pivot.get("PATHOLOGIC_STAGE", pd.Series([None] * len(pivot))).map(_normalize_stage)
        out["performance_status"] = np.nan
        out["chemotherapy_cycles"] = np.nan
        out["egfr_mutation"] = out["sample_id"].map(lambda x: egfr_map.get(str(x), np.nan))
        out["mortality_1yr"] = np.nan
        out["response"] = np.nan

        frames.append(out)

    if not frames:
        return _empty_frame()

    return pd.concat(frames, ignore_index=True)


def _coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in FEATURE_COLUMNS + LABEL_COLUMNS:
        non_null = int(df[col].notna().sum())
        rows.append({
            "column": col,
            "non_null": non_null,
            "coverage_pct": round((100.0 * non_null / total), 2) if total else 0.0,
        })
    return pd.DataFrame(rows)


def run_harmonization() -> dict[str, object]:
    _ensure_dirs()

    cbio_egfr_patient_map = _build_cbio_egfr_patient_map(RAW_DIR / "cbioportal")

    seer = _harmonize_seer()
    gdc = _harmonize_gdc(cbio_egfr_patient_map=cbio_egfr_patient_map)
    cbio = _harmonize_cbio()

    combined = pd.concat([seer, gdc, cbio], ignore_index=True)

    # Keep rows that contain at least one useful modeling feature.
    useful_cols = [
        "age",
        "sex",
        "race_recode",
        "marital_status",
        "histology",
        "cancer_stage",
        "t_stage",
        "n_stage",
        "m_stage",
        "tumor_size_mm",
        "mets_lung_dx",
        "chemotherapy_cycles",
        "radiation_recode",
        "treatment_delay_days",
        "egfr_mutation",
    ]
    combined = combined[combined[useful_cols].notna().any(axis=1)].copy()

    combined.to_csv(HARMONIZED_PATH, index=False)

    mortality_ready = combined[combined["mortality_1yr"].notna()].copy()
    mortality_ready.to_csv(MORTALITY_PATH, index=False)

    response_ready = combined[combined["response"].notna()].copy()
    response_ready.to_csv(RESPONSE_PATH, index=False)

    coverage = _coverage_report(combined)
    coverage.to_csv(READINESS_CSV, index=False)

    source_counts = combined["source"].value_counts(dropna=False).to_dict()
    mortality_counts = mortality_ready["mortality_1yr"].value_counts(dropna=False).to_dict()
    response_counts = response_ready["response"].value_counts(dropna=False).to_dict()

    report = {
        "total_rows_harmonized": int(len(combined)),
        "rows_mortality_ready": int(len(mortality_ready)),
        "rows_response_ready": int(len(response_ready)),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "mortality_label_counts": {str(k): int(v) for k, v in mortality_counts.items()},
        "response_label_counts": {str(k): int(v) for k, v in response_counts.items()},
        "output_files": {
            "harmonized": str(HARMONIZED_PATH),
            "mortality_ready": str(MORTALITY_PATH),
            "response_ready": str(RESPONSE_PATH),
            "feature_coverage": str(READINESS_CSV),
        },
    }

    READINESS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    summary = run_harmonization()
    print(json.dumps(summary, indent=2))
