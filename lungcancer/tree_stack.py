from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .schema import FEATURE_ORDER, FEATURE_SPECS, normalize_payload

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

MORTALITY_DATA = PROCESSED_DIR / "mortality_training_ready.csv"
RESPONSE_DATA = PROCESSED_DIR / "response_training_ready.csv"

# Resource guardrails for large Windows runs.
MAX_ROWS_FOR_SMOTE = 120000
MAX_ROWS_FOR_GRIDSEARCH = 200000
OUTER_CV_SPLITS = 3
INNER_CV_SPLITS = 2

# Mortality quick-win pruning for extremely sparse features.
DROP_FEATURES_BY_TASK: Dict[str, set[str]] = {
    "mortality": {"performance_status", "egfr_mutation"},
    "response": {"performance_status", "egfr_mutation", "tumor_size_mm", "mets_lung_dx", "treatment_delay_days"},
}


def _coerce_feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for spec in FEATURE_SPECS:
        if spec.name not in out.columns:
            continue
        if spec.dtype in {"int", "float", "binary"}:
            out[spec.name] = pd.to_numeric(out[spec.name], errors="coerce")
        else:
            series = out[spec.name].copy()
            series = series.where(series.notna(), None).astype(str)
            series = series.replace({"None": np.nan, "<NA>": np.nan, "nan": np.nan})
            out[spec.name] = series.astype(object)
    return out


def _feature_order_for_task(task: str) -> List[str]:
    drop = DROP_FEATURES_BY_TASK.get(task, set())
    return [name for name in FEATURE_ORDER if name not in drop]


def _column_groups(feature_order: List[str]) -> Tuple[List[str], List[str]]:
    numeric: List[str] = []
    categorical: List[str] = []
    for spec in FEATURE_SPECS:
        if spec.name not in feature_order:
            continue
        if spec.dtype in {"int", "float", "binary"}:
            numeric.append(spec.name)
        else:
            categorical.append(spec.name)
    return numeric, categorical


def _preprocessor(feature_order: List[str]) -> ColumnTransformer:
    numeric, categorical = _column_groups(feature_order)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _select_data(task: str, max_rows: int | None = None, random_state: int = 42) -> Tuple[pd.DataFrame, str]:
    if task == "mortality":
        path = MORTALITY_DATA
        label_col = "mortality_1yr"
    elif task == "response":
        path = RESPONSE_DATA
        label_col = "response"
    else:
        raise ValueError("task must be 'mortality' or 'response'")

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset for {task}: {path}")

    df = pd.read_csv(path, low_memory=False)
    required = FEATURE_ORDER + [label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {task}: {missing}")

    df = df[required].copy()
    df = df[df[label_col].notna()].copy()

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        # Stratified downsample keeps class balance while reducing runtime.
        df, _ = train_test_split(
            df,
            train_size=max_rows,
            random_state=random_state,
            stratify=df[label_col].astype(int),
        )

    df = _coerce_feature_types(df)
    return df, label_col


def _estimator(algorithm: str, random_state: int = 42):
    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        est = LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=63,
            class_weight="balanced",
            n_jobs=1,
        )
        grid = {
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.03, 0.05],
            "model__n_estimators": [250, 400],
        }
        return est, grid

    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
        )
        grid = {
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05],
            "model__n_estimators": [250, 400],
        }
        return est, grid

    if algorithm == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        est = TabNetClassifier(seed=random_state, verbose=0)
        # TabNet uses fit params heavily; keep tiny grid for first implementation pass.
        grid = {}
        return est, grid

    raise ValueError("algorithm must be lightgbm, xgboost, or tabnet")


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def _auc_ci_95(auc_values: List[float]) -> Dict[str, float | str | int | None]:
    values = np.asarray([v for v in auc_values if np.isfinite(v)], dtype=float)
    n = int(values.size)
    if n < 2:
        return {
            "method": "normal_approx_over_outer_folds",
            "n_folds": n,
            "mean": float(values[0]) if n == 1 else None,
            "lower": None,
            "upper": None,
        }

    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    z = 1.96
    lower = max(0.0, mean - z * se)
    upper = min(1.0, mean + z * se)
    return {
        "method": "normal_approx_over_outer_folds",
        "n_folds": n,
        "mean": mean,
        "lower": float(lower),
        "upper": float(upper),
    }


def _save_calibration_plot(y_true: np.ndarray, prob: np.ndarray, out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt

        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


def _save_roc_plot(y_true: np.ndarray, prob: np.ndarray, out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt

        if len(np.unique(y_true)) < 2:
            return False
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = float(roc_auc_score(y_true, prob))
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Chance")
        ax.plot(fpr, tpr, linewidth=1.8, label=f"ROC (AUC={auc_val:.4f})")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


def _save_probability_histogram(y_true: np.ndarray, prob: np.ndarray, out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        neg = prob[y_true == 0]
        pos = prob[y_true == 1]
        bins = np.linspace(0.0, 1.0, 21)
        if len(neg) > 0:
            ax.hist(neg, bins=bins, alpha=0.55, label="Class 0", color="#457b9d")
        if len(pos) > 0:
            ax.hist(pos, bins=bins, alpha=0.55, label="Class 1", color="#e76f51")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


def _best_threshold(y_true: np.ndarray, prob: np.ndarray, target_metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _metrics(y_true, prob, threshold=best_threshold)
    best_target = float(best_metrics.get(target_metric, float("-inf")))
    best_acc = float(best_metrics.get("accuracy", float("-inf")))

    for threshold in np.linspace(0.2, 0.8, 61):
        metrics = _metrics(y_true, prob, threshold=float(threshold))
        target_value = float(metrics.get(target_metric, float("-inf")))
        acc_value = float(metrics.get("accuracy", float("-inf")))
        if target_value > best_target or (target_value == best_target and acc_value > best_acc):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_target = target_value
            best_acc = acc_value

    return best_threshold, best_metrics


def _fit_mapie_if_available(base_model, X_cal: np.ndarray, y_cal: np.ndarray):
    try:
        from mapie.classification import MapieClassifier

        mapie = MapieClassifier(estimator=base_model, cv="prefit", method="lac")
        mapie.fit(X_cal, y_cal)
        return mapie
    except Exception:
        return None


def _tabnet_fit_kwargs(task: str) -> Dict[str, Any]:
    if task == "response":
        return {
            "model__max_epochs": 50,
            "model__batch_size": 512,
            "model__virtual_batch_size": 128,
            "model__num_workers": 0,
            "model__drop_last": False,
        }
    return {
        "model__max_epochs": 35,
        "model__batch_size": 8192,
        "model__virtual_batch_size": 1024,
        "model__num_workers": 0,
        "model__drop_last": False,
    }


def train_with_nested_cv(
    task: str,
    algorithm: str,
    random_state: int = 42,
    max_rows: int | None = None,
) -> Dict[str, Any]:
    df, label_col = _select_data(task=task, max_rows=max_rows, random_state=random_state)
    notes: List[str] = []
    feature_order = _feature_order_for_task(task)
    if len(feature_order) != len(FEATURE_ORDER):
        notes.append("dropped_features=" + ",".join(sorted(set(FEATURE_ORDER) - set(feature_order))))

    X = df[feature_order]
    y = df[label_col].astype(int).values

    outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=random_state)
    outer_results: List[Dict[str, float]] = []

    estimator, grid = _estimator(algorithm=algorithm, random_state=random_state)
    if len(df) > MAX_ROWS_FOR_GRIDSEARCH:
        grid = {}
        notes.append("gridsearch_skipped_large_dataset=true")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # SMOTE can become prohibitively expensive on very large folds.
        use_smote = len(X_train) <= MAX_ROWS_FOR_SMOTE
        steps = [("prep", _preprocessor(feature_order))]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("model", estimator))

        pipe = ImbPipeline(steps=steps)

        if grid:
            inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=random_state)
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=1,
                pre_dispatch=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model = pipe
            if algorithm == "tabnet":
                model.fit(X_train, y_train, **_tabnet_fit_kwargs(task))
            else:
                model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        fold_metrics = _metrics(y_test, prob)
        fold_metrics["fold"] = float(fold)
        outer_results.append(fold_metrics)

    summary_df = pd.DataFrame(outer_results)
    summary = {
        "task": task,
        "algorithm": algorithm,
        "n_rows": int(len(df)),
        "label_col": label_col,
        "notes": notes,
        "outer_fold_metrics": outer_results,
        "outer_mean": {k: float(v) for k, v in summary_df.mean(numeric_only=True).to_dict().items()},
        "outer_std": {k: float(v) for k, v in summary_df.std(numeric_only=True).to_dict().items()},
        "outer_auc_ci_95": _auc_ci_95(summary_df.get("auc", pd.Series(dtype=float)).tolist()),
    }

    # Final fit for deployment using holdout calibration split for MAPIE/shap.
    X_train, X_cal, y_train, y_cal = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    final_pipe = ImbPipeline(
        steps=(
            [("prep", _preprocessor(feature_order))]
            + ([("smote", SMOTE(random_state=random_state))] if len(X_train) <= MAX_ROWS_FOR_SMOTE else [])
            + [("model", estimator)]
        )
    )
    if algorithm == "tabnet":
        final_pipe.fit(X_train, y_train, **_tabnet_fit_kwargs(task))
    else:
        final_pipe.fit(X_train, y_train)

    cal_prob = final_pipe.predict_proba(X_cal)[:, 1]
    summary["calibration_metrics"] = _metrics(y_cal, cal_prob)
    best_threshold, best_threshold_metrics = _best_threshold(y_cal, cal_prob, target_metric="f1")
    summary["best_threshold"] = {
        "threshold": float(best_threshold),
        "target_metric": "f1",
        "metrics": best_threshold_metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{task}_{algorithm}_model.joblib"
    joblib.dump(final_pipe, model_path)

    mapie = _fit_mapie_if_available(final_pipe, X_cal, y_cal)
    mapie_path = None
    if mapie is not None:
        mapie_path = MODELS_DIR / f"{task}_{algorithm}_mapie.joblib"
        joblib.dump(mapie, mapie_path)

    calibration_plot_path = REPORTS_DIR / f"{task}_{algorithm}_calibration_curve.png"
    calibration_ok = _save_calibration_plot(
        y_true=y_cal,
        prob=cal_prob,
        out_path=calibration_plot_path,
        title=f"Calibration Curve ({task}, {algorithm})",
    )
    if not calibration_ok:
        calibration_plot_path = None

    roc_plot_path = REPORTS_DIR / f"{task}_{algorithm}_roc_curve.png"
    roc_ok = _save_roc_plot(
        y_true=y_cal,
        prob=cal_prob,
        out_path=roc_plot_path,
        title=f"ROC Curve ({task}, {algorithm})",
    )
    if not roc_ok:
        roc_plot_path = None

    prob_hist_plot_path = REPORTS_DIR / f"{task}_{algorithm}_probability_hist.png"
    prob_hist_ok = _save_probability_histogram(
        y_true=y_cal,
        prob=cal_prob,
        out_path=prob_hist_plot_path,
        title=f"Probability Distribution ({task}, {algorithm})",
    )
    if not prob_hist_ok:
        prob_hist_plot_path = None

    shap_path = None
    shap_plot_path = None
    try:
        import shap
        import matplotlib.pyplot as plt

        sample = X_cal.sample(n=min(256, len(X_cal)), random_state=random_state)
        # Compute SHAP on transformed matrix with model component.
        prep = final_pipe.named_steps["prep"]
        model = final_pipe.named_steps["model"]
        X_t = prep.transform(sample)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        feature_names = None
        if hasattr(prep, "get_feature_names_out"):
            feature_names = prep.get_feature_names_out().tolist()
        explainer = shap.Explainer(model)
        values = explainer(X_t)
        shap_payload = {
            "task": task,
            "algorithm": algorithm,
            "mean_abs_shap": np.abs(values.values).mean(axis=0).tolist(),
            "base_values": np.asarray(values.base_values).reshape(-1).tolist(),
            "n_samples": int(X_t.shape[0]),
        }
        shap_path = REPORTS_DIR / f"{task}_{algorithm}_shap_summary.json"
        shap_path.write_text(json.dumps(shap_payload, indent=2), encoding="utf-8")

        shap_plot_path = REPORTS_DIR / f"{task}_{algorithm}_shap_summary.png"
        shap.summary_plot(values.values, X_t, feature_names=feature_names, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=150)
        plt.close()
    except Exception:
        pass

    # Survival analysis only for mortality if event-time columns are available.
    survival_path = None
    if task == "mortality":
        try:
            from lifelines import CoxPHFitter

            merged = df.copy()
            if "days_to_death" in merged.columns and "days_to_last_follow_up" in merged.columns:
                duration = pd.to_numeric(merged["days_to_death"], errors="coerce").fillna(
                    pd.to_numeric(merged["days_to_last_follow_up"], errors="coerce")
                )
                surv = pd.DataFrame({"duration": duration, "event": y})
                surv = surv.dropna()
                if len(surv) > 100:
                    cph = CoxPHFitter()
                    cph.fit(surv, duration_col="duration", event_col="event")
                    survival_path = REPORTS_DIR / "mortality_survival_summary.txt"
                    survival_path.write_text(str(cph.summary), encoding="utf-8")
        except Exception:
            pass

    summary["artifacts"] = {
        "model": str(model_path),
        "mapie": str(mapie_path) if mapie_path else None,
        "shap": str(shap_path) if shap_path else None,
        "shap_plot": str(shap_plot_path) if shap_plot_path else None,
        "calibration_plot": str(calibration_plot_path) if calibration_plot_path else None,
        "roc_plot": str(roc_plot_path) if roc_plot_path else None,
        "probability_hist_plot": str(prob_hist_plot_path) if prob_hist_plot_path else None,
        "survival": str(survival_path) if survival_path else None,
    }

    report_path = REPORTS_DIR / f"{task}_{algorithm}_nested_cv.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass
class TreePredictor:
    model: Any
    mapie: Any | None

    @staticmethod
    def _repair_imputer_fill_dtype(obj: Any) -> None:
        """Best-effort repair for sklearn SimpleImputer state across versions."""

        seen: set[int] = set()

        def _visit(node: Any) -> None:
            if node is None:
                return
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)

            # Newer sklearn versions may expect this private attribute.
            if node.__class__.__name__ == "SimpleImputer":
                if not hasattr(node, "_fill_dtype") and hasattr(node, "statistics_"):
                    try:
                        stats = np.asarray(node.statistics_)
                        node._fill_dtype = stats.dtype  # type: ignore[attr-defined]
                    except Exception:
                        node._fill_dtype = np.asarray([], dtype=object).dtype  # type: ignore[attr-defined]

            if hasattr(node, "steps") and isinstance(getattr(node, "steps"), list):
                for _, step in node.steps:
                    _visit(step)

            if hasattr(node, "transformers") and isinstance(getattr(node, "transformers"), list):
                for transformer in node.transformers:
                    if len(transformer) >= 2:
                        _visit(transformer[1])

            if hasattr(node, "transformers_") and isinstance(getattr(node, "transformers_"), list):
                for transformer in node.transformers_:
                    if len(transformer) >= 2:
                        _visit(transformer[1])

            if hasattr(node, "named_steps"):
                try:
                    for step in node.named_steps.values():
                        _visit(step)
                except Exception:
                    pass

            if hasattr(node, "estimator"):
                _visit(getattr(node, "estimator"))

            if hasattr(node, "estimator_"):
                _visit(getattr(node, "estimator_"))

    @classmethod
    def load(cls, task: str, algorithm: str, strict: bool = False) -> "TreePredictor | None":
        model_path = MODELS_DIR / f"{task}_{algorithm}_model.joblib"
        if not model_path.exists():
            return None
        mapie_path = MODELS_DIR / f"{task}_{algorithm}_mapie.joblib"
        try:
            model = joblib.load(model_path)
            mapie = joblib.load(mapie_path) if mapie_path.exists() else None
            return cls(model=model, mapie=mapie)
        except Exception:
            if strict:
                raise
            # Keep inference service available even if one serialized model
            # cannot be loaded on the current runtime environment.
            return None

    def predict_probability(self, payload: Dict[str, Any]) -> float:
        normalized = normalize_payload(payload)
        X = pd.DataFrame([normalized], columns=FEATURE_ORDER)
        X = _coerce_feature_types(X)
        try:
            return float(self.model.predict_proba(X)[:, 1][0])
        except AttributeError as exc:
            if "_fill_dtype" not in str(exc):
                raise
            self._repair_imputer_fill_dtype(self.model)
            return float(self.model.predict_proba(X)[:, 1][0])

    def predict_interval(self, payload: Dict[str, Any], alpha: float = 0.1) -> Dict[str, float] | None:
        if self.mapie is None:
            return None
        normalized = normalize_payload(payload)
        X = pd.DataFrame([normalized], columns=FEATURE_ORDER)
        X = _coerce_feature_types(X)
        try:
            # MAPIE classification returns set predictions; expose score bounds when available.
            try:
                y_pred, y_ps = self.mapie.predict(X, alpha=alpha)
                prob = float(self.model.predict_proba(X)[:, 1][0])
            except AttributeError as exc:
                if "_fill_dtype" not in str(exc):
                    raise
                self._repair_imputer_fill_dtype(self.model)
                if self.mapie is not None:
                    self._repair_imputer_fill_dtype(self.mapie)
                y_pred, y_ps = self.mapie.predict(X, alpha=alpha)
                prob = float(self.model.predict_proba(X)[:, 1][0])
            return {
                "prediction": float(y_pred[0]),
                "probability": prob,
                "alpha": float(alpha),
                "set_contains_0": float(y_ps[0, 0, 0]),
                "set_contains_1": float(y_ps[0, 1, 0]),
            }
        except Exception:
            return None
