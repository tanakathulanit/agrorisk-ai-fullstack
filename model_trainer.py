"""Training, resampling, hyperparameter search, evaluation and explainability for the risk engine.

Design decisions
----------------
* SMOTE-NC (the mixed-type SMOTE variant) runs inside an imbalanced-learn pipeline, so synthetic
  defaulters are generated only from the training folds of each cross-validation split and from
  the training partition at refit. Validation folds and the held-out test set contain real loans
  only, and samplers are skipped automatically at prediction time.
* Hyperparameters are tuned for ROC-AUC with stratified K-fold cross-validation.
* Monotonic constraints encode credit policy (higher DSCR can never raise risk, higher drought
  exposure can never lower it), which keeps decisions explainable to borrowers and regulators.
* Oversampling inflates the event rate the model sees, so raw scores are mapped back to the true
  portfolio base rate with a prior-shift (odds) correction before being reported as PD. The
  correction factor is solved on out-of-fold training predictions, so it never sees test loans.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from scipy.optimize import brentq
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from config import (
    ARTIFACT_DIR,
    CROPS,
    CV_FOLDS,
    IMPORTANCE_CSV_PATH,
    IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    REJECT_GRADE,
    RISK_GRADE_BANDS,
    SEARCH_ITERATIONS,
    SHAP_PLOT_PATH,
    TEST_SIZE,
)
from data_loader import CATEGORICAL_FEATURES, FEATURE_COLUMNS, generate_dataset, split_features_target

logger = logging.getLogger(__name__)

CROP_DUMMY_PREFIX = "primary_crop_"
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "debt_service_coverage_ratio": -1,
    "off_farm_income_ratio": -1,
    "crop_diversification_index": -1,
    "historical_drought_severity_index": 1,
    "seasonal_cash_flow_gap_months": 1,
}
PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "smote__k_neighbors": randint(3, 8),
    "smote__sampling_strategy": uniform(0.30, 0.70),
    "xgb__n_estimators": randint(150, 601),
    "xgb__max_depth": randint(2, 6),
    "xgb__learning_rate": loguniform(0.01, 0.2),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
    "xgb__min_child_weight": randint(1, 11),
    "xgb__gamma": uniform(0.0, 2.0),
    "xgb__reg_lambda": loguniform(0.5, 10.0),
}


def assign_risk_grade(probability_of_default: float) -> str:
    """Map a calibrated PD to the lender's risk grade (A, B, C, D or Reject)."""
    for grade, upper_bound in RISK_GRADE_BANDS:
        if probability_of_default < upper_bound:
            return grade
    return REJECT_GRADE


def correct_for_resampling(raw_probability: np.ndarray, correction_factor: float) -> np.ndarray:
    """Shift probabilities learned on a resampled class ratio back to the true prior.

    For a model trained with class odds `s` while the population has odds `pi / (1 - pi)`,
    calibrated odds equal raw odds multiplied by `(pi / (1 - pi)) / s`.
    """
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    odds = clipped / (1.0 - clipped) * correction_factor
    return odds / (1.0 + odds)


def fit_prior_correction(out_of_fold_scores: np.ndarray, target_rate: float) -> float:
    """Solve for the odds multiplier that makes mean calibrated PD equal the observed default rate.

    The analytic factor `(pi / (1 - pi)) / sampling_strategy` is exact only for random
    oversampling; solving on out-of-fold scores accounts for SMOTE interpolation and tree fit.
    """
    def calibration_gap(log_factor: float) -> float:
        return float(np.mean(correct_for_resampling(out_of_fold_scores, float(np.exp(log_factor))))) - target_rate

    return float(np.exp(brentq(calibration_gap, -15.0, 15.0)))


def _cast_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast all encoded columns to float64 for XGBoost (SMOTE-NC may emit object dtypes)."""
    return frame.astype(np.float64)


def _original_feature_name(encoded_name: str) -> str:
    """Collapse one-hot crop dummy columns back to their source feature name."""
    return "primary_crop" if encoded_name.startswith(CROP_DUMMY_PREFIX) else encoded_name


def build_pipeline(seed: int = RANDOM_SEED) -> ImbPipeline:
    """Assemble the SMOTE-NC -> one-hot encoding -> XGBoost pipeline."""
    encoder = ColumnTransformer(
        transformers=[
            (
                "crop",
                OneHotEncoder(categories=[list(CROPS)], handle_unknown="ignore", sparse_output=False),
                ["primary_crop"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    to_float = FunctionTransformer(_cast_to_float, feature_names_out="one-to-one").set_output(transform="pandas")
    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=seed,
        n_jobs=1,
    )
    return ImbPipeline(
        steps=[
            ("smote", SMOTENC(categorical_features=CATEGORICAL_FEATURES, random_state=seed)),
            ("encode", encoder),
            ("to_float", to_float),
            ("xgb", classifier),
        ]
    )


@dataclass
class CreditRiskModel:
    """Fitted scoring pipeline plus the calibration and governance metadata it needs."""

    pipeline: ImbPipeline
    prior_correction_factor: float
    feature_columns: list[str]
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame

    @property
    def classifier(self) -> xgb.XGBClassifier:
        """The fitted XGBoost estimator inside the pipeline."""
        return self.pipeline.named_steps["xgb"]

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoding steps (no resampling) to raw model features."""
        encoded = self.pipeline.named_steps["encode"].transform(features[self.feature_columns])
        return self.pipeline.named_steps["to_float"].transform(encoded)

    def predict_raw_score(self, features: pd.DataFrame) -> np.ndarray:
        """Uncalibrated default score as learned on the resampled training distribution."""
        return self.pipeline.predict_proba(features[self.feature_columns])[:, 1]

    def predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of default calibrated to the true portfolio default rate."""
        return correct_for_resampling(self.predict_raw_score(features), self.prior_correction_factor)

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP contributions in log-odds of default, per original feature.

        The returned `bias` column includes the prior-shift correction, so each row sums to
        the logit of the calibrated PD.
        """
        encoded = self.encode(features)
        contributions = self.classifier.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)
        frame = pd.DataFrame(contributions, columns=[*encoded.columns, "bias"], index=features.index)
        bias = frame.pop("bias") + np.log(self.prior_correction_factor)
        grouped = frame.T.groupby(_original_feature_name, sort=False).sum().T
        grouped["bias"] = bias
        return grouped

    def save(self, path: Path = MODEL_PATH) -> None:
        """Persist the model object with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> CreditRiskModel:
        """Load a persisted model and verify its type."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact at {path} is not a {cls.__name__}.")
        return model


@dataclass(frozen=True)
class TrainingResult:
    """Outputs of a full training run."""

    model: CreditRiskModel
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame


def tune_pipeline(
    x_train: pd.DataFrame, y_train: pd.Series, seed: int = RANDOM_SEED, n_iter: int = SEARCH_ITERATIONS
) -> RandomizedSearchCV:
    """Randomised hyperparameter search maximising cross-validated ROC-AUC."""
    search = RandomizedSearchCV(
        estimator=build_pipeline(seed),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
        n_jobs=-1,
        refit=True,
        random_state=seed,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    logger.info("Best CV ROC-AUC %.4f with %s", search.best_score_, search.best_params_)
    return search


def evaluate_model(model: CreditRiskModel, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute discrimination, calibration and policy-cutoff metrics on real held-out loans."""
    pd_hat = model.predict_pd(x_test)
    raw_score = model.predict_raw_score(x_test)
    y_true = y_test.to_numpy()

    roc_auc = float(roc_auc_score(y_true, pd_hat))
    fpr, tpr, _ = roc_curve(y_true, pd_hat)
    reject_cutoff = RISK_GRADE_BANDS[-1][1]
    y_reject = (pd_hat >= reject_cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_reject, labels=[0, 1]).ravel()

    grades = pd.Series([assign_risk_grade(p) for p in pd_hat], name="grade")
    grade_table = (
        pd.DataFrame({"grade": grades, "pd": pd_hat, "default": y_true})
        .groupby("grade")
        .agg(loans=("default", "size"), observed_default_rate=("default", "mean"), mean_pd=("pd", "mean"))
        .reindex([g for g, _ in RISK_GRADE_BANDS] + [REJECT_GRADE])
        .dropna(subset=["loans"])
        .reset_index()
    )

    return {
        "test_roc_auc": roc_auc,
        "test_gini": 2.0 * roc_auc - 1.0,
        "test_pr_auc": float(average_precision_score(y_true, pd_hat)),
        "test_ks_statistic": float(np.max(tpr - fpr)),
        "test_brier_calibrated": float(brier_score_loss(y_true, pd_hat)),
        "test_brier_uncalibrated": float(brier_score_loss(y_true, raw_score)),
        "test_observed_default_rate": float(y_true.mean()),
        "test_mean_predicted_pd": float(pd_hat.mean()),
        "reject_cutoff_pd": reject_cutoff,
        "reject_precision": float(precision_score(y_true, y_reject, zero_division=0)),
        "reject_recall": float(recall_score(y_true, y_reject, zero_division=0)),
        "reject_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "grade_performance": grade_table.to_dict(orient="records"),
    }


def compute_feature_importance(model: CreditRiskModel, reference: pd.DataFrame) -> pd.DataFrame:
    """Gain importance and mean |SHAP| per original feature, each normalised to sum to one."""
    gain_scores = model.classifier.get_booster().get_score(importance_type="gain")
    gain = pd.Series(gain_scores, dtype=float).groupby(_original_feature_name).sum()
    shap_abs = model.explain(reference).drop(columns="bias").abs().mean()
    table = pd.DataFrame({"gain_importance": gain, "mean_abs_shap": shap_abs}).fillna(0.0)
    table = table / table.sum()
    return table.sort_values("mean_abs_shap", ascending=False).rename_axis("feature").reset_index()


def plot_feature_importance(importance: pd.DataFrame, path: Path = IMPORTANCE_PLOT_PATH, top_n: int = 15) -> None:
    """Render a horizontal bar chart of gain importance for the top risk drivers."""
    top = importance.nlargest(top_n, "gain_importance").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["gain_importance"], color="#3d6b35")
    ax.set_xlabel("Share of total gain")
    ax.set_title("XGBoost gain importance: top default risk drivers")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary(model: CreditRiskModel, reference: pd.DataFrame, path: Path = SHAP_PLOT_PATH) -> None:
    """Render a SHAP beeswarm summary using XGBoost's exact TreeSHAP contributions."""
    encoded = model.encode(reference)
    contributions = model.classifier.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)[:, :-1]
    shap.summary_plot(contributions, encoded, max_display=15, show=False, plot_size=(9, 7))
    fig = plt.gcf()
    fig.suptitle("SHAP values: impact on log-odds of default (held-out test set)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_persist(seed: int = RANDOM_SEED, n_iter: int = SEARCH_ITERATIONS) -> TrainingResult:
    """Run the full pipeline end to end and write all artefacts to `ARTIFACT_DIR`."""
    portfolio = generate_dataset(seed=seed)
    features, target = split_features_target(portfolio)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=seed
    )
    logger.info(
        "Train %d loans (%d defaults) | Test %d loans (%d defaults)",
        len(x_train), int(y_train.sum()), len(x_test), int(y_test.sum()),
    )

    search = tune_pipeline(x_train, y_train, seed=seed, n_iter=n_iter)
    train_rate = float(y_train.mean())
    out_of_fold_scores = cross_val_predict(
        clone(search.best_estimator_),
        x_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed + 1),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    correction = fit_prior_correction(out_of_fold_scores, train_rate)
    analytic_correction = (train_rate / (1.0 - train_rate)) / float(search.best_params_["smote__sampling_strategy"])
    logger.info("Prior correction factor %.4f (analytic SMOTE-ratio estimate %.4f)", correction, analytic_correction)

    model = CreditRiskModel(
        pipeline=search.best_estimator_,
        prior_correction_factor=correction,
        feature_columns=list(FEATURE_COLUMNS),
        metrics={},
        feature_importance=pd.DataFrame(),
    )
    metrics: dict[str, Any] = {
        "cv_best_roc_auc": float(search.best_score_),
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "train_default_rate": train_rate,
        "prior_correction_factor": correction,
        **evaluate_model(model, x_test, y_test),
    }
    importance = compute_feature_importance(model, x_test)
    model.metrics = metrics
    model.feature_importance = importance

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    importance.to_csv(IMPORTANCE_CSV_PATH, index=False)
    plot_feature_importance(importance)
    plot_shap_summary(model, x_test)

    logger.info(
        "Test ROC-AUC %.4f | Gini %.4f | PR-AUC %.4f | KS %.4f | Brier %.4f (uncalibrated %.4f)",
        metrics["test_roc_auc"], metrics["test_gini"], metrics["test_pr_auc"], metrics["test_ks_statistic"],
        metrics["test_brier_calibrated"], metrics["test_brier_uncalibrated"],
    )
    logger.info("Artefacts written to %s", ARTIFACT_DIR)
    return TrainingResult(model=model, metrics=metrics, feature_importance=importance)


def main() -> None:
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Train the Agricultural Credit Risk Engine.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for data and model.")
    parser.add_argument("--n-iter", type=int, default=SEARCH_ITERATIONS, help="Hyperparameter search iterations.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train_and_persist(seed=args.seed, n_iter=args.n_iter)
    print(json.dumps({k: v for k, v in result.metrics.items() if k.startswith("test_")}, indent=2))
    print(pd.DataFrame(result.metrics["grade_performance"]).to_string(index=False))


if __name__ == "__main__":
    main()
