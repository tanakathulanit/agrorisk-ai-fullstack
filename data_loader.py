"""Synthetic smallholder loan portfolio generation and agricultural credit-risk feature engineering.

The generator follows the causal chain an agricultural lender faces: regional weather risk
drives expected yields, expected yields drive the cash-flow projection used at underwriting,
and a realised weather shock after disbursement determines whether the season's cash flow
can service the loan. Model features are restricted to information available at origination;
realised outcomes only drive the default label, so the model cannot learn from the future.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import (
    CROP_DROUGHT_SENSITIVITY,
    CROP_GROSS_REVENUE_PER_HA,
    CROP_HARVEST_WEIGHTS,
    CROP_INPUT_WEIGHTS,
    CROP_OPEX_RATIO,
    CROP_PREVALENCE,
    CROPS,
    DROUGHT_HISTORY_SEASONS,
    DROUGHT_LOSS_SCALE,
    EXPECTED_DROUGHT_SHOCK,
    HOUSEHOLD_CASH_COST_PER_MEMBER,
    IRRIGATION_DROUGHT_MITIGATION,
    IRRIGATION_REVENUE_UPLIFT,
    LOAN_TERM_OPTIONS,
    LOAN_TERM_PREVALENCE,
    MAX_INTEREST_RATE,
    MAX_LOAN_AMOUNT,
    MIN_INTEREST_RATE,
    MIN_LOAN_AMOUNT,
    MONTHS,
    N_RECORDS,
    RANDOM_SEED,
    REGION_BASE_DROUGHT_RISK,
    REGION_PREVALENCE,
    TARGET_DEFAULT_RATE,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN: Final[str] = "default_flag"
SHARE_COLUMNS: Final[list[str]] = [f"share_{crop}" for crop in CROPS]
CATEGORICAL_FEATURES: Final[list[str]] = ["primary_crop", "has_irrigation", "cooperative_member"]
NUMERIC_BASE_FEATURES: Final[list[str]] = [
    "farm_size_ha",
    "years_farming",
    "household_size",
    "prior_loans_count",
    "loan_amount",
    "interest_rate",
    "loan_term_months",
]
ENGINEERED_FEATURES: Final[list[str]] = [
    "debt_service_coverage_ratio",
    "off_farm_income_ratio",
    "crop_diversification_index",
    "historical_drought_severity_index",
    "seasonal_cash_flow_gap_months",
]
FEATURE_COLUMNS: Final[list[str]] = CATEGORICAL_FEATURES + NUMERIC_BASE_FEATURES + ENGINEERED_FEATURES


def _normalized(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Scale a non-negative weight vector so that it sums to one."""
    array = np.asarray(weights, dtype=float)
    total = array.sum()
    if total <= 0:
        raise ValueError("Weight vector must have a positive sum.")
    return array / total


CROP_INDEX: Final[dict[str, int]] = {crop: index for index, crop in enumerate(CROPS)}
HARVEST_CALENDAR: Final[np.ndarray] = np.vstack([_normalized(CROP_HARVEST_WEIGHTS[c]) for c in CROPS])
INPUT_CALENDAR: Final[np.ndarray] = np.vstack([_normalized(CROP_INPUT_WEIGHTS[c]) for c in CROPS])
REVENUE_PER_HA: Final[np.ndarray] = np.array([CROP_GROSS_REVENUE_PER_HA[c] for c in CROPS])
OPEX_RATIO: Final[np.ndarray] = np.array([CROP_OPEX_RATIO[c] for c in CROPS])
DROUGHT_SENSITIVITY: Final[np.ndarray] = np.array([CROP_DROUGHT_SENSITIVITY[c] for c in CROPS])


@dataclass(frozen=True)
class ApplicantProfile:
    """Loan application captured by a field officer at origination."""

    farm_size_ha: float
    primary_crop: str
    annual_farm_revenue: float
    historical_drought_severity_index: float
    annual_off_farm_income: float = 0.0
    loan_amount: float = 400.0
    interest_rate: float = 0.27
    loan_term_months: int = 12
    primary_crop_share: float = 1.0
    secondary_crops: tuple[str, ...] = field(default_factory=tuple)
    has_irrigation: bool = False
    cooperative_member: bool = False
    years_farming: int = 10
    household_size: int = 5
    prior_loans_count: int = 0

    def __post_init__(self) -> None:
        """Validate that the application is internally consistent."""
        if self.primary_crop not in CROP_INDEX:
            raise ValueError(f"Unknown primary crop '{self.primary_crop}'.")
        unknown = set(self.secondary_crops) - set(CROPS)
        if unknown:
            raise ValueError(f"Unknown secondary crops: {sorted(unknown)}.")
        if not 0.0 < self.primary_crop_share <= 1.0:
            raise ValueError("primary_crop_share must lie in (0, 1].")
        if not 0.0 <= self.historical_drought_severity_index <= 1.0:
            raise ValueError("historical_drought_severity_index must lie in [0, 1].")
        if self.farm_size_ha <= 0 or self.annual_farm_revenue <= 0 or self.loan_amount <= 0:
            raise ValueError("Farm size, revenue and loan amount must be positive.")
        if self.annual_off_farm_income < 0:
            raise ValueError("Off-farm income cannot be negative.")
        if self.interest_rate <= 0 or self.loan_term_months <= 0 or self.household_size <= 0:
            raise ValueError("Interest rate, loan term and household size must be positive.")

    def crop_shares(self) -> np.ndarray:
        """Return the land share of each crop in `CROPS` order, summing to one."""
        shares = np.zeros(len(CROPS))
        secondary = [crop for crop in dict.fromkeys(self.secondary_crops) if crop != self.primary_crop]
        primary_share = self.primary_crop_share if secondary else 1.0
        shares[CROP_INDEX[self.primary_crop]] = primary_share
        for crop in secondary:
            shares[CROP_INDEX[crop]] = (1.0 - primary_share) / len(secondary)
        return _normalized(shares)


def annuity_installment(
    principal: np.ndarray | float, annual_rate: np.ndarray | float, term_months: np.ndarray | int
) -> np.ndarray:
    """Compute the level monthly installment of an amortising loan."""
    principal_arr = np.asarray(principal, dtype=float)
    monthly_rate = np.asarray(annual_rate, dtype=float) / 12.0
    periods = np.asarray(term_months, dtype=float)
    return principal_arr * monthly_rate / (1.0 - np.power(1.0 + monthly_rate, -periods))


def annual_debt_service(
    principal: np.ndarray | float, annual_rate: np.ndarray | float, term_months: np.ndarray | int
) -> np.ndarray:
    """Total principal and interest falling due within the first 12 months of the loan."""
    return annuity_installment(principal, annual_rate, term_months) * np.minimum(
        np.asarray(term_months, dtype=float), 12.0
    )


def simpson_diversification_index(shares: np.ndarray) -> np.ndarray:
    """Gini-Simpson diversification index (1 - sum of squared crop shares) per farm.

    Zero indicates monoculture; the maximum for k equally weighted crops is 1 - 1/k.
    """
    matrix = np.atleast_2d(np.asarray(shares, dtype=float))
    totals = matrix.sum(axis=1, keepdims=True)
    proportions = matrix / np.where(totals == 0.0, 1.0, totals)
    return 1.0 - np.square(proportions).sum(axis=1)


def expected_yield_factor(
    drought_index: np.ndarray | float, drought_sensitivity: np.ndarray | float, has_irrigation: np.ndarray | bool
) -> np.ndarray:
    """Expected fraction of potential yield retained given historical drought exposure."""
    mitigation = 1.0 - IRRIGATION_DROUGHT_MITIGATION * np.asarray(has_irrigation, dtype=float)
    loss = DROUGHT_LOSS_SCALE * EXPECTED_DROUGHT_SHOCK * np.asarray(drought_index, dtype=float)
    return np.clip(1.0 - loss * np.asarray(drought_sensitivity, dtype=float) * mitigation, 0.15, 1.0)


def estimate_farm_opex(
    expected_revenue: float, crop_shares: np.ndarray, drought_index: float, has_irrigation: bool
) -> float:
    """Estimate operating expenses from expected revenue.

    Inputs are purchased for potential (non-drought) production, so the expected revenue is
    grossed back up to potential before applying crop-specific operating cost ratios.
    """
    shares = _normalized(crop_shares)
    yield_factor = float(expected_yield_factor(drought_index, shares @ DROUGHT_SENSITIVITY, has_irrigation))
    uplift = 1.0 + IRRIGATION_REVENUE_UPLIFT * float(has_irrigation)
    potential_revenue = expected_revenue / (yield_factor * uplift)
    return float(potential_revenue * (shares @ OPEX_RATIO))


def disbursement_month(primary_crop: str) -> int:
    """Month index (0 = January) in which a loan for the given primary crop is disbursed.

    Disbursement is aligned to the peak input-purchase month of the primary crop.
    """
    return int(np.argmax(INPUT_CALENDAR[CROP_INDEX[primary_crop]]))


def monthly_cash_flows(
    crop_shares: np.ndarray,
    annual_farm_revenue: float,
    annual_farm_opex: float,
    annual_off_farm_income: float,
    household_size: int,
    monthly_installment: float,
    loan_term_months: int,
    start_month: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 12 monthly inflows and outflows for a farm household.

    Inflows are crop revenue distributed along each crop's harvest calendar plus off-farm
    income spread evenly. Outflows are farm operating expenses along each crop's input
    calendar, household cash living costs, and loan installments beginning the month after
    disbursement. Loan proceeds are treated as financing the input outlays.
    """
    shares = _normalized(crop_shares)
    inflow = annual_farm_revenue * (shares @ HARVEST_CALENDAR) + annual_off_farm_income / 12.0
    outflow = annual_farm_opex * (shares @ INPUT_CALENDAR) + household_size * HOUSEHOLD_CASH_COST_PER_MEMBER / 12.0
    repayment_months = (start_month + 1 + np.arange(min(int(loan_term_months), 12))) % 12
    np.add.at(outflow, repayment_months, monthly_installment)
    return inflow, outflow


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the agricultural credit-risk features from origination-time raw fields.

    Required columns: crop share columns, `primary_crop`, `annual_farm_revenue`,
    `annual_farm_opex`, `annual_off_farm_income`, `household_size`, `loan_amount`,
    `interest_rate`, `loan_term_months`, and `historical_drought_severity_index`.
    """
    frame = raw.copy()
    shares = frame[SHARE_COLUMNS].to_numpy(dtype=float)

    net_agricultural_income = frame["annual_farm_revenue"] - frame["annual_farm_opex"]
    debt_service = annual_debt_service(frame["loan_amount"], frame["interest_rate"], frame["loan_term_months"])
    installments = annuity_installment(frame["loan_amount"], frame["interest_rate"], frame["loan_term_months"])
    total_revenue = frame["annual_farm_revenue"] + frame["annual_off_farm_income"]

    frame["net_agricultural_income"] = net_agricultural_income
    frame["annual_debt_service"] = debt_service
    frame["debt_service_coverage_ratio"] = np.clip(net_agricultural_income / debt_service, 0.0, None)
    frame["off_farm_income_ratio"] = np.where(
        total_revenue > 0, frame["annual_off_farm_income"] / total_revenue.where(total_revenue > 0, 1.0), 0.0
    )
    frame["crop_diversification_index"] = simpson_diversification_index(shares)

    gap_months = np.empty(len(frame), dtype=int)
    for position, row in enumerate(frame.itertuples(index=False)):
        inflow, outflow = monthly_cash_flows(
            crop_shares=shares[position],
            annual_farm_revenue=row.annual_farm_revenue,
            annual_farm_opex=row.annual_farm_opex,
            annual_off_farm_income=row.annual_off_farm_income,
            household_size=row.household_size,
            monthly_installment=float(installments[position]),
            loan_term_months=row.loan_term_months,
            start_month=disbursement_month(row.primary_crop),
        )
        gap_months[position] = int(np.sum(outflow > inflow))
    frame["seasonal_cash_flow_gap_months"] = gap_months
    return frame


def _simulate_regional_drought_severity(rng: np.random.Generator) -> dict[str, float]:
    """Simulate a multi-season Standardised Precipitation Index history per region.

    Severity is the mean magnitude of negative SPI anomalies; drier regions have both a
    negative mean shift and higher rainfall variability.
    """
    severity: dict[str, float] = {}
    for region, base_risk in REGION_BASE_DROUGHT_RISK.items():
        spi = rng.normal(loc=0.3 - 1.2 * base_risk, scale=0.8 + 0.4 * base_risk, size=DROUGHT_HISTORY_SEASONS)
        severity[region] = float(np.clip(-spi, 0.0, None).mean())
    return severity


def _simulate_crop_portfolios(rng: np.random.Generator, n_records: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw each farm's primary crop and Dirichlet-distributed land shares across its crops."""
    n_crops_total = len(CROPS)
    primary_idx = rng.choice(n_crops_total, size=n_records, p=_normalized([CROP_PREVALENCE[c] for c in CROPS]))
    crops_grown = rng.choice([1, 2, 3, 4], size=n_records, p=[0.25, 0.35, 0.28, 0.12])
    shares = np.zeros((n_records, n_crops_total))
    for row in range(n_records):
        others = np.array([c for c in range(n_crops_total) if c != primary_idx[row]])
        secondary = rng.choice(others, size=crops_grown[row] - 1, replace=False)
        alpha = np.concatenate(([4.0], np.full(crops_grown[row] - 1, 1.2)))
        weights = np.sort(rng.dirichlet(alpha))[::-1]
        shares[row, primary_idx[row]] = weights[0]
        shares[row, secondary] = weights[1:]
    return primary_idx, shares


def _calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Find the logistic intercept that yields the target expected default rate."""
    return float(brentq(lambda b: float(np.mean(1.0 / (1.0 + np.exp(-(logits + b))))) - target_rate, -30.0, 30.0))


def generate_dataset(
    n_records: int = N_RECORDS, seed: int = RANDOM_SEED, target_default_rate: float = TARGET_DEFAULT_RATE
) -> pd.DataFrame:
    """Synthesize a smallholder loan book with engineered features and a default label.

    Returns one row per loan containing identifiers, raw origination fields, engineered
    features, the realised post-disbursement farm revenue (audit only, never a feature)
    and the binary `default_flag`.
    """
    rng = np.random.default_rng(seed)

    regions = np.array(list(REGION_PREVALENCE))
    region = rng.choice(regions, size=n_records, p=_normalized(list(REGION_PREVALENCE.values())))
    regional_severity = _simulate_regional_drought_severity(rng)
    drought_raw = np.array([regional_severity[r] for r in region]) + rng.normal(0.0, 0.06, size=n_records)
    drought_index = (drought_raw - drought_raw.min()) / (drought_raw.max() - drought_raw.min())

    primary_idx, shares = _simulate_crop_portfolios(rng, n_records)
    primary_crop = np.array(CROPS)[primary_idx]

    farm_size = np.clip(rng.lognormal(mean=np.log(1.4), sigma=0.6, size=n_records), 0.2, 15.0).round(2)
    has_irrigation = (rng.random(n_records) < 0.10 + 0.35 * shares[:, CROP_INDEX["vegetables"]]).astype(int)
    cooperative_member = (rng.random(n_records) < 0.45).astype(int)
    years_farming = np.clip(rng.gamma(shape=3.0, scale=5.0, size=n_records), 1, 50).astype(int)
    household_size = np.clip(rng.poisson(4.0, size=n_records) + 1, 1, 14)
    prior_loans = rng.poisson(1.2, size=n_records)

    sensitivity = shares @ DROUGHT_SENSITIVITY
    irrigation_uplift = 1.0 + IRRIGATION_REVENUE_UPLIFT * has_irrigation
    potential_revenue = farm_size * (shares @ REVENUE_PER_HA) * rng.lognormal(0.0, 0.25, size=n_records)
    expected_revenue = potential_revenue * expected_yield_factor(drought_index, sensitivity, has_irrigation) * irrigation_uplift
    farm_opex = potential_revenue * (shares @ OPEX_RATIO) * rng.lognormal(0.0, 0.10, size=n_records)
    off_farm_income = (rng.random(n_records) < 0.55) * rng.lognormal(np.log(550.0), 0.7, size=n_records)

    loan_amount = np.clip(potential_revenue * rng.uniform(0.15, 0.55, size=n_records), MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT)
    interest_rate = rng.uniform(MIN_INTEREST_RATE, MAX_INTEREST_RATE, size=n_records).round(3)
    loan_term = rng.choice(LOAN_TERM_OPTIONS, size=n_records, p=LOAN_TERM_PREVALENCE)

    raw = pd.DataFrame(
        {
            "farmer_id": [f"SHF-{i:05d}" for i in range(1, n_records + 1)],
            "region": region,
            "primary_crop": primary_crop,
            **{column: shares[:, index] for index, column in enumerate(SHARE_COLUMNS)},
            "farm_size_ha": farm_size,
            "has_irrigation": has_irrigation,
            "cooperative_member": cooperative_member,
            "years_farming": years_farming,
            "household_size": household_size,
            "prior_loans_count": prior_loans,
            "annual_farm_revenue": expected_revenue.round(2),
            "annual_farm_opex": farm_opex.round(2),
            "annual_off_farm_income": off_farm_income.round(2),
            "loan_amount": (np.round(loan_amount / 10.0) * 10.0),
            "interest_rate": interest_rate,
            "loan_term_months": loan_term,
            "historical_drought_severity_index": drought_index.round(4),
        }
    )
    frame = engineer_features(raw)

    realised_shock = rng.beta(2.0, 5.0, size=n_records)
    realised_yield = np.clip(
        1.0
        - DROUGHT_LOSS_SCALE * drought_index * sensitivity * realised_shock
        * (1.0 - IRRIGATION_DROUGHT_MITIGATION * has_irrigation),
        0.10,
        1.0,
    ) * rng.lognormal(0.0, 0.15, size=n_records)
    realised_revenue = potential_revenue * realised_yield * irrigation_uplift
    realised_dscr = np.clip(realised_revenue - farm_opex, 0.0, None) / frame["annual_debt_service"].to_numpy()

    logits = (
        -1.3 * np.log(np.clip(realised_dscr, 0.05, None))
        + 0.28 * frame["seasonal_cash_flow_gap_months"].to_numpy()
        - 1.5 * frame["off_farm_income_ratio"].to_numpy()
        - 1.2 * frame["crop_diversification_index"].to_numpy()
        + 0.9 * drought_index
        - 0.35 * cooperative_member
        - 0.10 * np.minimum(prior_loans, 6)
        - 0.015 * years_farming
        + rng.normal(0.0, 0.5, size=n_records)
    )
    intercept = _calibrate_intercept(logits, target_default_rate)
    default_probability = 1.0 / (1.0 + np.exp(-(logits + intercept)))

    frame["realised_farm_revenue"] = realised_revenue.round(2)
    frame[TARGET_COLUMN] = (rng.random(n_records) < default_probability).astype(int)

    validate_dataset(frame)
    logger.info(
        "Generated %d loans | default rate %.2f%% | median DSCR %.2f",
        n_records, 100 * frame[TARGET_COLUMN].mean(), frame["debt_service_coverage_ratio"].median(),
    )
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Enforce schema and domain-range invariants on a portfolio DataFrame."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}.")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing feature values.")
    checks = {
        "historical_drought_severity_index": (0.0, 1.0),
        "off_farm_income_ratio": (0.0, 1.0),
        "crop_diversification_index": (0.0, 1.0),
        "seasonal_cash_flow_gap_months": (0, 12),
    }
    for column, (low, high) in checks.items():
        if not frame[column].between(low, high).all():
            raise ValueError(f"Column '{column}' has values outside [{low}, {high}].")
    if (frame["debt_service_coverage_ratio"] < 0).any():
        raise ValueError("DSCR must be non-negative.")


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model feature matrix and the default target vector."""
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(int).copy()


def _applicant_raw_frame(profile: ApplicantProfile) -> pd.DataFrame:
    """Translate an application into the raw origination schema used by `engineer_features`."""
    shares = profile.crop_shares()
    opex = estimate_farm_opex(
        profile.annual_farm_revenue, shares, profile.historical_drought_severity_index, profile.has_irrigation
    )
    return pd.DataFrame(
        [
            {
                "primary_crop": profile.primary_crop,
                **{column: shares[index] for index, column in enumerate(SHARE_COLUMNS)},
                "farm_size_ha": profile.farm_size_ha,
                "has_irrigation": int(profile.has_irrigation),
                "cooperative_member": int(profile.cooperative_member),
                "years_farming": profile.years_farming,
                "household_size": profile.household_size,
                "prior_loans_count": profile.prior_loans_count,
                "annual_farm_revenue": profile.annual_farm_revenue,
                "annual_farm_opex": opex,
                "annual_off_farm_income": profile.annual_off_farm_income,
                "loan_amount": profile.loan_amount,
                "interest_rate": profile.interest_rate,
                "loan_term_months": profile.loan_term_months,
                "historical_drought_severity_index": profile.historical_drought_severity_index,
            }
        ]
    )


def build_applicant_frame(profile: ApplicantProfile) -> pd.DataFrame:
    """Return a single-row engineered DataFrame for scoring an application."""
    return engineer_features(_applicant_raw_frame(profile))


def applicant_cash_flow_projection(profile: ApplicantProfile) -> pd.DataFrame:
    """Project the applicant's 12-month household cash flow including loan repayments."""
    raw = _applicant_raw_frame(profile).iloc[0]
    inflow, outflow = monthly_cash_flows(
        crop_shares=profile.crop_shares(),
        annual_farm_revenue=float(raw["annual_farm_revenue"]),
        annual_farm_opex=float(raw["annual_farm_opex"]),
        annual_off_farm_income=float(raw["annual_off_farm_income"]),
        household_size=profile.household_size,
        monthly_installment=float(annuity_installment(profile.loan_amount, profile.interest_rate, profile.loan_term_months)),
        loan_term_months=profile.loan_term_months,
        start_month=disbursement_month(profile.primary_crop),
    )
    return pd.DataFrame(
        {"month": list(MONTHS), "inflow": inflow, "outflow": outflow, "net_cash_flow": inflow - outflow}
    )


def benchmark_farm_revenue(profile_crop_shares: np.ndarray, farm_size_ha: float, drought_index: float, has_irrigation: bool) -> float:
    """Regional benchmark of expected annual farm revenue for a given crop mix and farm size."""
    shares = _normalized(profile_crop_shares)
    yield_factor = float(expected_yield_factor(drought_index, shares @ DROUGHT_SENSITIVITY, has_irrigation))
    return float(farm_size_ha * (shares @ REVENUE_PER_HA) * yield_factor * (1.0 + IRRIGATION_REVENUE_UPLIFT * has_irrigation))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    portfolio = generate_dataset()
    print(portfolio[ENGINEERED_FEATURES + [TARGET_COLUMN]].describe().T.round(3))
