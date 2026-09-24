"""Central configuration and agronomic reference tables for the Agricultural Credit Risk Engine.

All monetary values are expressed in USD per year unless stated otherwise. Crop calendars
follow a bimodal East African rainfall pattern (long rains Mar-May, short rains Oct-Dec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

RANDOM_SEED: Final[int] = 42
N_RECORDS: Final[int] = 1_000
TARGET_DEFAULT_RATE: Final[float] = 0.08
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
SEARCH_ITERATIONS: Final[int] = 40
LOSS_GIVEN_DEFAULT: Final[float] = 0.45

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
MODEL_PATH: Final[Path] = ARTIFACT_DIR / "credit_risk_model.joblib"
METRICS_PATH: Final[Path] = ARTIFACT_DIR / "metrics.json"
IMPORTANCE_CSV_PATH: Final[Path] = ARTIFACT_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH: Final[Path] = ARTIFACT_DIR / "feature_importance.png"
SHAP_PLOT_PATH: Final[Path] = ARTIFACT_DIR / "shap_summary.png"

MONTHS: Final[tuple[str, ...]] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

CROPS: Final[tuple[str, ...]] = ("maize", "sorghum", "beans", "cassava", "coffee", "vegetables")

CROP_PREVALENCE: Final[dict[str, float]] = {
    "maize": 0.34, "sorghum": 0.12, "beans": 0.18, "cassava": 0.14, "coffee": 0.12, "vegetables": 0.10,
}

CROP_HARVEST_WEIGHTS: Final[dict[str, tuple[float, ...]]] = {
    "maize":      (0, 0, 0, 0, 0, 0, 3, 5, 2, 0, 0, 0),
    "sorghum":    (0, 0, 0, 0, 0, 0, 0, 3, 5, 2, 0, 0),
    "beans":      (0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 4),
    "cassava":    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    "coffee":     (0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 3),
    "vegetables": (2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3),
}

CROP_INPUT_WEIGHTS: Final[dict[str, tuple[float, ...]]] = {
    "maize":      (0, 1, 4, 3, 1, 1, 0, 0, 0, 0, 0, 0),
    "sorghum":    (0, 0, 2, 4, 2, 1, 0, 0, 0, 0, 0, 0),
    "beans":      (0, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 0),
    "cassava":    (0, 0, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0),
    "coffee":     (1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2),
    "vegetables": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
}

CROP_GROSS_REVENUE_PER_HA: Final[dict[str, float]] = {
    "maize": 650.0, "sorghum": 480.0, "beans": 820.0, "cassava": 720.0, "coffee": 1_900.0, "vegetables": 2_600.0,
}

CROP_OPEX_RATIO: Final[dict[str, float]] = {
    "maize": 0.48, "sorghum": 0.40, "beans": 0.45, "cassava": 0.35, "coffee": 0.42, "vegetables": 0.58,
}

CROP_DROUGHT_SENSITIVITY: Final[dict[str, float]] = {
    "maize": 0.85, "sorghum": 0.45, "beans": 0.75, "cassava": 0.30, "coffee": 0.55, "vegetables": 0.90,
}

REGION_BASE_DROUGHT_RISK: Final[dict[str, float]] = {
    "Highlands": 0.15, "Lake Basin": 0.25, "Central Plateau": 0.40, "Rift Valley": 0.55, "Semi-Arid Lowlands": 0.80,
}

REGION_PREVALENCE: Final[dict[str, float]] = {
    "Highlands": 0.22, "Lake Basin": 0.24, "Central Plateau": 0.24, "Rift Valley": 0.18, "Semi-Arid Lowlands": 0.12,
}

DROUGHT_HISTORY_SEASONS: Final[int] = 20
DROUGHT_LOSS_SCALE: Final[float] = 1.4
EXPECTED_DROUGHT_SHOCK: Final[float] = 2.0 / 7.0
IRRIGATION_DROUGHT_MITIGATION: Final[float] = 0.70
IRRIGATION_REVENUE_UPLIFT: Final[float] = 0.15
HOUSEHOLD_CASH_COST_PER_MEMBER: Final[float] = 60.0

LOAN_TERM_OPTIONS: Final[tuple[int, ...]] = (6, 9, 12, 18, 24)
LOAN_TERM_PREVALENCE: Final[tuple[float, ...]] = (0.10, 0.15, 0.40, 0.20, 0.15)
MIN_LOAN_AMOUNT: Final[float] = 100.0
MAX_LOAN_AMOUNT: Final[float] = 10_000.0
MIN_INTEREST_RATE: Final[float] = 0.18
MAX_INTEREST_RATE: Final[float] = 0.36

RISK_GRADE_BANDS: Final[tuple[tuple[str, float], ...]] = (
    ("A", 0.03), ("B", 0.07), ("C", 0.12), ("D", 0.20),
)
REJECT_GRADE: Final[str] = "Reject"
