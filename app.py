"""Streamlit field-officer dashboard for the Agricultural Credit Risk Engine.

Run with `streamlit run app.py`. If no trained model artefact exists, the dashboard trains
one on first launch and caches it for the session.
"""

from __future__ import annotations

import logging

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import config
from data_loader import load_data
from model_trainer import train_model

# App logic below...

from config import (
    CROPS,
    LOAN_TERM_OPTIONS,
    LOSS_GIVEN_DEFAULT,
    MAX_INTEREST_RATE,
    MIN_INTEREST_RATE,
    MODEL_PATH,
    MONTHS,
    REJECT_GRADE,
    RISK_GRADE_BANDS,
    SHAP_PLOT_PATH,
)
from data_loader import (
    FEATURE_COLUMNS,
    ApplicantProfile,
    applicant_cash_flow_projection,
    benchmark_farm_revenue,
    build_applicant_frame,
    disbursement_month,
    generate_dataset,
)
from model_trainer import CreditRiskModel, assign_risk_grade, train_and_persist

logger = logging.getLogger(__name__)

GRADE_ORDER: list[str] = [grade for grade, _ in RISK_GRADE_BANDS] + [REJECT_GRADE]
GRADE_COLORS: dict[str, str] = {
    "A": "#2e7d32", "B": "#7cb342", "C": "#f9a825", "D": "#ef6c00", REJECT_GRADE: "#c62828",
}
GRADE_GUIDANCE: dict[str, str] = {
    "A": "Approve at standard terms.",
    "B": "Approve with standard monitoring.",
    "C": "Approve with conditions: align installments to harvest months or require a group guarantee.",
    "D": "Refer to credit committee: consider a smaller amount, index-based crop insurance or a co-signer.",
    REJECT_GRADE: "Decline: projected cash flows cannot support this obligation under the expected weather risk.",
}
FEATURE_LABELS: dict[str, str] = {
    "primary_crop": "Primary crop",
    "has_irrigation": "Irrigation access",
    "cooperative_member": "Cooperative membership",
    "farm_size_ha": "Farm size (ha)",
    "years_farming": "Years farming",
    "household_size": "Household size",
    "prior_loans_count": "Prior loans repaid",
    "loan_amount": "Loan amount",
    "interest_rate": "Interest rate",
    "loan_term_months": "Loan term",
    "debt_service_coverage_ratio": "Debt service coverage ratio",
    "off_farm_income_ratio": "Off-farm income ratio",
    "crop_diversification_index": "Crop diversification index",
    "historical_drought_severity_index": "Drought severity index",
    "seasonal_cash_flow_gap_months": "Cash-flow gap months",
}


@st.cache_resource(show_spinner="Training the credit risk model (first launch only)…")
def load_engine() -> CreditRiskModel:
    """Load the persisted model, retraining if the artefact is missing or incompatible."""
    if MODEL_PATH.exists():
        try:
            return CreditRiskModel.load(MODEL_PATH)
        except (OSError, TypeError, AttributeError, ModuleNotFoundError, ValueError) as error:
            logger.warning("Model artefact could not be loaded (%s); retraining.", error)
    return train_and_persist().model


@st.cache_data(show_spinner="Scoring the loan portfolio…")
def load_scored_portfolio(_model: CreditRiskModel, model_key: str) -> pd.DataFrame:
    """Generate the loan book and attach calibrated PD, risk grade and expected loss.

    `model_key` identifies the model version so the cache refreshes after retraining.
    """
    portfolio = generate_dataset()
    portfolio["pd"] = _model.predict_pd(portfolio[FEATURE_COLUMNS])
    portfolio["risk_grade"] = [assign_risk_grade(p) for p in portfolio["pd"]]
    portfolio["expected_loss"] = portfolio["pd"] * LOSS_GIVEN_DEFAULT * portfolio["loan_amount"]
    return portfolio


def render_portfolio_overview(model: CreditRiskModel, portfolio: pd.DataFrame) -> None:
    """Metric cards and grade mix for the current loan book."""
    exposure = float(portfolio["loan_amount"].sum())
    expected_loss = float(portfolio["expected_loss"].sum())
    metrics = model.metrics

    col_risk, col_dscr, col_auc = st.columns(3)
    col_risk.metric(
        "Total portfolio risk (expected loss)",
        f"${expected_loss:,.0f}",
        help=(
            f"Sum of PD × LGD ({LOSS_GIVEN_DEFAULT:.0%}) × exposure across {len(portfolio):,} loans. "
            f"Equals {expected_loss / exposure:.2%} of the ${exposure:,.0f} book; mean PD "
            f"{portfolio['pd'].mean():.2%}."
        ),
    )
    col_dscr.metric(
        "Average DSCR",
        f"{portfolio['debt_service_coverage_ratio'].mean():.2f}×",
        help=(
            f"Median {portfolio['debt_service_coverage_ratio'].median():.2f}×. "
            f"{(portfolio['debt_service_coverage_ratio'] < 1.0).mean():.1%} of loans are below 1.0×."
        ),
    )
    col_auc.metric(
        "Model ROC-AUC (held-out)",
        f"{metrics['test_roc_auc']:.3f}",
        help=(
            f"Gini {metrics['test_gini']:.3f}, PR-AUC {metrics['test_pr_auc']:.3f}, "
            f"KS {metrics['test_ks_statistic']:.3f}. Cross-validated AUC {metrics['cv_best_roc_auc']:.3f}."
        ),
    )

    grade_mix = (
        portfolio.groupby("risk_grade")
        .agg(loans=("farmer_id", "size"), exposure=("loan_amount", "sum"), expected_loss=("expected_loss", "sum"))
        .reindex(GRADE_ORDER, fill_value=0)
        .rename_axis("grade")
        .reset_index()
    )
    chart = (
        alt.Chart(grade_mix)
        .mark_bar()
        .encode(
            x=alt.X("grade:N", sort=GRADE_ORDER, title="Risk grade"),
            y=alt.Y("loans:Q", title="Loans"),
            color=alt.Color(
                "grade:N",
                scale=alt.Scale(domain=list(GRADE_COLORS), range=list(GRADE_COLORS.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("grade:N", title="Grade"),
                alt.Tooltip("loans:Q", title="Loans"),
                alt.Tooltip("exposure:Q", title="Exposure ($)", format=",.0f"),
                alt.Tooltip("expected_loss:Q", title="Expected loss ($)", format=",.0f"),
            ],
        )
        .properties(height=220, title="Portfolio grade distribution")
    )
    st.altair_chart(chart, width="stretch")


def collect_applicant_inputs() -> ApplicantProfile:
    """Render underwriting inputs and return the resulting application profile."""
    farm_size = st.slider("Farm size (hectares)", 0.2, 15.0, 1.5, 0.1)
    primary_crop = st.selectbox("Primary crop", CROPS, index=0, format_func=str.capitalize)
    drought_index = st.slider(
        "Historical drought severity index", 0.0, 1.0, 0.45, 0.01,
        help="Normalised regional weather risk: 0 = lowest drought exposure in the book, 1 = highest.",
    )
    seasonal_income = st.slider(
        "Expected seasonal farm revenue (USD / year)", 100, 20_000, 1_200, 50,
        help="Gross crop sales expected across the season(s), before operating costs.",
    )

    with st.expander("Crop mix, loan terms and household", expanded=False):
        secondary = st.multiselect(
            "Secondary crops", [c for c in CROPS if c != primary_crop], format_func=str.capitalize
        )
        primary_share = (
            st.slider("Land share of primary crop", 0.4, 1.0, 0.7, 0.05) if secondary else 1.0
        )
        off_farm = st.number_input("Off-farm income (USD / year)", 0, 20_000, 300, 50)
        loan_amount = st.number_input("Requested loan (USD)", 100, 10_000, 400, 50)
        rate_pct = st.slider(
            "Annual interest rate (%)", MIN_INTEREST_RATE * 100, MAX_INTEREST_RATE * 100, 27.0, 0.5
        )
        term = st.select_slider("Loan term (months)", options=list(LOAN_TERM_OPTIONS), value=12)
        irrigation = st.toggle("Has irrigation")
        cooperative = st.toggle("Cooperative member", value=True)
        years_farming = st.number_input("Years farming", 1, 50, 10)
        household = st.number_input("Household size", 1, 14, 5)
        prior_loans = st.number_input("Prior loans repaid", 0, 20, 1)

    profile = ApplicantProfile(
        farm_size_ha=float(farm_size),
        primary_crop=primary_crop,
        annual_farm_revenue=float(seasonal_income),
        historical_drought_severity_index=float(drought_index),
        annual_off_farm_income=float(off_farm),
        loan_amount=float(loan_amount),
        interest_rate=float(rate_pct) / 100.0,
        loan_term_months=int(term),
        primary_crop_share=float(primary_share),
        secondary_crops=tuple(secondary),
        has_irrigation=bool(irrigation),
        cooperative_member=bool(cooperative),
        years_farming=int(years_farming),
        household_size=int(household),
        prior_loans_count=int(prior_loans),
    )
    benchmark = benchmark_farm_revenue(profile.crop_shares(), farm_size, drought_index, irrigation)
    ratio = seasonal_income / benchmark
    st.caption(
        f"Regional benchmark for this farm: about ${benchmark:,.0f}/yr. "
        + ("Stated revenue is well above benchmark; verify with sales receipts." if ratio > 1.5 else "")
    )
    return profile


def render_decision(model: CreditRiskModel, profile: ApplicantProfile) -> None:
    """Score the application and render PD, grade, key ratios, cash flow and risk drivers."""
    applicant = build_applicant_frame(profile)
    probability = float(model.predict_pd(applicant)[0])
    grade = assign_risk_grade(probability)
    row = applicant.iloc[0]

    col_pd, col_grade = st.columns(2)
    col_pd.metric("Probability of default", f"{probability:.1%}")
    col_grade.markdown(
        f"<div style='font-size:0.875rem;opacity:0.8'>Risk grade</div>"
        f"<div style='display:inline-block;margin-top:0.25rem;padding:0.15rem 0.9rem;border-radius:0.5rem;"
        f"background:{GRADE_COLORS[grade]};color:white;font-size:2rem;font-weight:700'>{grade}</div>",
        unsafe_allow_html=True,
    )
    (st.error if grade in ("D", REJECT_GRADE) else st.warning if grade == "C" else st.success)(GRADE_GUIDANCE[grade])

    ratio_cols = st.columns(4)
    ratio_cols[0].metric("DSCR", f"{row['debt_service_coverage_ratio']:.2f}×")
    ratio_cols[1].metric("Off-farm ratio", f"{row['off_farm_income_ratio']:.0%}")
    ratio_cols[2].metric("Diversification", f"{row['crop_diversification_index']:.2f}")
    ratio_cols[3].metric("Gap months", f"{int(row['seasonal_cash_flow_gap_months'])} / 12")

    cash_flow = applicant_cash_flow_projection(profile)
    cash_flow["position"] = np.where(cash_flow["net_cash_flow"] < 0, "Deficit", "Surplus")
    start = MONTHS[disbursement_month(profile.primary_crop)]
    cash_chart = (
        alt.Chart(cash_flow)
        .mark_bar()
        .encode(
            x=alt.X("month:N", sort=list(MONTHS), title=None),
            y=alt.Y("net_cash_flow:Q", title="Net cash flow (USD)"),
            color=alt.Color(
                "position:N",
                scale=alt.Scale(domain=["Surplus", "Deficit"], range=["#2e7d32", "#c62828"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("month:N"),
                alt.Tooltip("inflow:Q", format=",.0f"),
                alt.Tooltip("outflow:Q", format=",.0f"),
                alt.Tooltip("net_cash_flow:Q", format=",.0f"),
            ],
        )
        .properties(height=220, title=f"Projected monthly household cash flow (disbursed in {start})")
    )
    st.altair_chart(cash_chart, width="stretch")

    contributions = model.explain(applicant).drop(columns="bias").iloc[0]
    drivers = (
        contributions.rename("contribution")
        .rename_axis("feature")
        .reset_index()
        .assign(
            label=lambda d: d["feature"].map(FEATURE_LABELS).fillna(d["feature"]),
            direction=lambda d: np.where(d["contribution"] > 0, "Raises risk", "Lowers risk"),
            magnitude=lambda d: d["contribution"].abs(),
        )
        .nlargest(8, "magnitude")
    )
    driver_chart = (
        alt.Chart(drivers)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="SHAP contribution to log-odds of default"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#c62828", "#2e7d32"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("label:N", title="Driver"), alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=260, title="Why this score: top drivers for this applicant")
    )
    st.altair_chart(driver_chart, width="stretch")


def render_feature_importance(model: CreditRiskModel) -> None:
    """Global feature importance chart with a choice of gain or mean |SHAP|."""
    measure = st.radio(
        "Importance measure", ["Mean |SHAP|", "XGBoost gain"], horizontal=True,
        help="Mean |SHAP| is consistent and additive; gain reflects split quality inside the trees.",
    )
    column = "mean_abs_shap" if measure == "Mean |SHAP|" else "gain_importance"
    table = model.feature_importance.assign(
        label=lambda d: d["feature"].map(FEATURE_LABELS).fillna(d["feature"])
    ).nlargest(12, column)
    chart = (
        alt.Chart(table)
        .mark_bar(color="#3d6b35")
        .encode(
            x=alt.X(f"{column}:Q", title="Share of total importance", axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip(f"{column}:Q", format=".1%")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")


def render_model_governance(model: CreditRiskModel) -> None:
    """Validation metrics, grade back-test and SHAP summary for model risk reviewers."""
    metrics = model.metrics
    with st.expander("Model validation and governance"):
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Held-out ROC-AUC", "Gini", "PR-AUC", "KS statistic", "Brier score (calibrated PD)",
                    "Observed test default rate", "Mean predicted PD (test)",
                    f"Reject-band precision (PD ≥ {metrics['reject_cutoff_pd']:.0%})", "Reject-band recall",
                ],
                "Value": [
                    f"{metrics['test_roc_auc']:.3f}", f"{metrics['test_gini']:.3f}", f"{metrics['test_pr_auc']:.3f}",
                    f"{metrics['test_ks_statistic']:.3f}", f"{metrics['test_brier_calibrated']:.4f}",
                    f"{metrics['test_observed_default_rate']:.2%}", f"{metrics['test_mean_predicted_pd']:.2%}",
                    f"{metrics['reject_precision']:.2%}", f"{metrics['reject_recall']:.2%}",
                ],
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.markdown("Grade back-test on held-out loans")
        grades = pd.DataFrame(metrics["grade_performance"]).rename(
            columns={"grade": "Grade", "loans": "Loans", "observed_default_rate": "Observed DR", "mean_pd": "Mean PD"}
        )
        st.dataframe(
            grades.style.format({"Loans": "{:.0f}", "Observed DR": "{:.1%}", "Mean PD": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"The test set holds {metrics['n_test']} real loans; SMOTE-NC was applied only to training folds. "
            "Small grade buckets carry wide confidence intervals."
        )
        if SHAP_PLOT_PATH.exists():
            st.image(str(SHAP_PLOT_PATH), caption="SHAP summary on the held-out test set")


def main() -> None:
    """Compose the dashboard."""
    st.set_page_config(page_title="Agricultural Credit Risk Engine", page_icon="🌾", layout="wide")
    st.title("Agricultural Credit Risk Engine")
    st.caption("Smallholder loan underwriting that accounts for harvest seasonality, drought exposure and thin credit files.")

    model = load_engine()
    portfolio = load_scored_portfolio(model, model_key=f"{model.metrics['test_roc_auc']:.6f}")

    st.subheader("Portfolio")
    render_portfolio_overview(model, portfolio)

    st.subheader("Underwriting tool")
    input_col, output_col = st.columns([2, 3], gap="large")
    with input_col:
        profile = collect_applicant_inputs()
    with output_col:
        render_decision(model, profile)

    st.subheader("Top risk drivers across the portfolio")
    render_feature_importance(model)
    render_model_governance(model)


if __name__ == "__main__":
    main()
