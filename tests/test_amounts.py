import numpy as np
import pandas as pd
import pytest

from src.amounts import (
    apply_fixed_non_takeup_calibration,
    assign_guaranteed_amount,
    compute_income_gap,
    finalize_entitlement,
)


def make_amount_hh(**overrides) -> pd.DataFrame:
    defaults = {
        "baseline_main_included": True,
        "baseline_has_listed_schedule": True,
        "guaranteed_amount_listed": 500.0,
        "max_amount": 800.0,
        "max_hh_size_listed": 5,
        "household_size": 3,
        "baseline_amount_topup_factor": np.nan,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestAssignGuaranteedAmount:
    def test_exact_schedule_match(self):
        df = make_amount_hh(guaranteed_amount_listed=500.0)
        result = assign_guaranteed_amount(df)
        assert result["rmi_guaranteed_amount_monthly"].iloc[0] == pytest.approx(500.0)
        assert result["rmi_amount_assignment_type"].iloc[0] == "exact_schedule_match"
        assert result["rmi_amount_rule_available"].iloc[0] == 1.0
        assert result["rmi_amount_approximate"].iloc[0] == 0.0

    def test_topup_factor_applied(self):
        df = make_amount_hh(guaranteed_amount_listed=400.0, baseline_amount_topup_factor=1.25)
        result = assign_guaranteed_amount(df)
        assert result["rmi_guaranteed_amount_monthly"].iloc[0] == pytest.approx(500.0)

    def test_cap_for_household_above_listed_size(self):
        df = make_amount_hh(
            guaranteed_amount_listed=np.nan,
            household_size=6,
            max_hh_size_listed=5,
            max_amount=800.0,
        )
        result = assign_guaranteed_amount(df)
        assert result["rmi_guaranteed_amount_monthly"].iloc[0] == pytest.approx(800.0)
        assert result["rmi_amount_assignment_type"].iloc[0] == "cap_for_above_listed_hhsize"
        assert result["rmi_amount_approximate"].iloc[0] == 1.0

    def test_unassigned_when_not_main_included(self):
        df = make_amount_hh(baseline_main_included=False)
        result = assign_guaranteed_amount(df)
        assert result["rmi_amount_assignment_type"].iloc[0] == "unassigned"
        assert result["rmi_amount_rule_available"].iloc[0] == 0.0

    def test_unassigned_when_no_schedule(self):
        df = make_amount_hh(baseline_has_listed_schedule=False)
        result = assign_guaranteed_amount(df)
        assert result["rmi_amount_assignment_type"].iloc[0] == "unassigned"
        assert pd.isna(result["rmi_guaranteed_amount_monthly"].iloc[0])


def make_income_gap_hh(**overrides) -> pd.DataFrame:
    defaults = {
        "threshold_resources_monthly": 300.0,
        "rmi_guaranteed_amount_monthly": 500.0,
        "rmi_age_eligible": 1.0,
        "rmi_claimant_proxy_eligible": 1.0,
        "rmi_wealth_eligible": 1.0,
        "rmi_hhtype_eligible": 1.0,
        "rmi_threeplus_adults_allowed": 1.0,
        "rmi_labour_eligible": 1.0,
        "rmi_insertion_rule_eligible": 1.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestComputeIncomeGap:
    def test_eligible_with_positive_gap(self):
        df = make_income_gap_hh(threshold_resources_monthly=300.0,
                                rmi_guaranteed_amount_monthly=500.0)
        result = compute_income_gap(df)
        assert result["rmi_income_eligible"].iloc[0] == 1.0
        assert result["rmi_income_gap_entitlement_monthly"].iloc[0] == pytest.approx(200.0)

    def test_not_eligible_when_resources_at_or_above_guarantee(self):
        df = make_income_gap_hh(threshold_resources_monthly=600.0,
                                rmi_guaranteed_amount_monthly=500.0)
        result = compute_income_gap(df)
        assert result["rmi_income_eligible"].iloc[0] == 0.0
        assert result["rmi_income_gap_entitlement_monthly"].iloc[0] == pytest.approx(0.0)

    def test_gap_is_zero_when_resources_equal_guarantee(self):
        df = make_income_gap_hh(threshold_resources_monthly=500.0,
                                rmi_guaranteed_amount_monthly=500.0)
        result = compute_income_gap(df)
        assert result["rmi_income_gap_entitlement_monthly"].iloc[0] == pytest.approx(0.0)

    def test_nan_when_claimant_unit_condition_fails(self):
        df = make_income_gap_hh(rmi_age_eligible=0.0)
        result = compute_income_gap(df)
        assert pd.isna(result["rmi_income_eligible"].iloc[0])
        assert pd.isna(result["rmi_income_gap_entitlement_monthly"].iloc[0])

    def test_nan_when_resources_missing(self):
        df = make_income_gap_hh(threshold_resources_monthly=np.nan)
        result = compute_income_gap(df)
        assert pd.isna(result["rmi_income_eligible"].iloc[0])


def make_entitlement_hh(**overrides) -> pd.DataFrame:
    defaults = {
        "baseline_main_included": True,
        "rmi_amount_rule_available": 1.0,
        "rmi_age_eligible": 1.0,
        "rmi_claimant_proxy_eligible": 1.0,
        "rmi_wealth_eligible": 1.0,
        "rmi_hhtype_eligible": 1.0,
        "rmi_threeplus_adults_allowed": 1.0,
        "rmi_labour_eligible": 1.0,
        "rmi_labour_rule_source": "labour_income_and_context_rule",
        "rmi_income_eligible": 1.0,
        "passes_percentile_filter": 1.0,
        "rmi_insertion_rule_eligible": 1.0,
        "active_inclusion_ok": 1.0,
        "rmi_income_gap_entitlement_monthly": 200.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestFinalizeEntitlement:
    def test_fully_eligible(self):
        df = make_entitlement_hh()
        result = finalize_entitlement(df)
        assert result["rmi_sim_eligible"].iloc[0] == 1.0
        assert result["rmi_positive_entitlement"].iloc[0] == 1.0
        assert result["rmi_exclusion_reason"].iloc[0] == "eligible"

    def test_zero_gap_means_no_positive_entitlement(self):
        df = make_entitlement_hh(rmi_income_gap_entitlement_monthly=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_sim_eligible"].iloc[0] == 1.0
        assert result["rmi_positive_entitlement"].iloc[0] == 0.0

    def test_excluded_when_region_not_main_included(self):
        df = make_entitlement_hh(baseline_main_included=False)
        result = finalize_entitlement(df)
        assert result["rmi_sim_eligible"].iloc[0] == 0.0
        assert result["rmi_exclusion_reason"].iloc[0] == "region_excluded_from_main_baseline"

    def test_excluded_when_no_amount_rule(self):
        df = make_entitlement_hh(rmi_amount_rule_available=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "amount_rule_unavailable"

    def test_excluded_when_age_not_observable(self):
        df = make_entitlement_hh(rmi_age_eligible=np.nan)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "age_rule_not_observable"

    def test_excluded_when_fails_age_rule(self):
        df = make_entitlement_hh(rmi_age_eligible=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "fails_claimant_age_rule"

    def test_excluded_when_fails_percentile_filter(self):
        df = make_entitlement_hh(passes_percentile_filter=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "fails_percentile_filter"

    def test_excluded_when_income_at_or_above_threshold(self):
        df = make_entitlement_hh(rmi_income_eligible=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "income_at_or_above_threshold"

    def test_excluded_when_fails_active_inclusion(self):
        df = make_entitlement_hh(active_inclusion_ok=0.0)
        result = finalize_entitlement(df)
        assert result["rmi_exclusion_reason"].iloc[0] == "fails_active_inclusion_proxy"


def make_ntu_hh(**overrides) -> pd.DataFrame:
    defaults = {
        "baseline_non_takeup_group": "high",
        "rmi_positive_entitlement": 1.0,
        "weight_hh": 100.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestApplyFixedNonTakeupCalibration:
    def test_high_group_rate(self):
        df = make_ntu_hh(baseline_non_takeup_group="high")
        result = apply_fixed_non_takeup_calibration(df, high_ntu=0.70, medium_ntu=0.30)
        assert result["fixed_non_take_up_rate"].iloc[0] == pytest.approx(0.70)
        assert result["fixed_take_up_rate"].iloc[0] == pytest.approx(0.30)

    def test_medium_group_rate(self):
        df = make_ntu_hh(baseline_non_takeup_group="medium")
        result = apply_fixed_non_takeup_calibration(df, high_ntu=0.70, medium_ntu=0.30)
        assert result["fixed_non_take_up_rate"].iloc[0] == pytest.approx(0.30)
        assert result["fixed_take_up_rate"].iloc[0] == pytest.approx(0.70)

    def test_other_group_has_zero_ntu(self):
        df = make_ntu_hh(baseline_non_takeup_group="low")
        result = apply_fixed_non_takeup_calibration(df, high_ntu=0.70, medium_ntu=0.30)
        assert result["fixed_non_take_up_rate"].iloc[0] == pytest.approx(0.0)

    def test_effective_weight_adjusted_by_take_up_rate(self):
        df = make_ntu_hh(baseline_non_takeup_group="high", weight_hh=100.0)
        result = apply_fixed_non_takeup_calibration(df, high_ntu=0.70, medium_ntu=0.30)
        assert result["rmi_effective_recipient_weight"].iloc[0] == pytest.approx(30.0)

    def test_no_entitlement_yields_zero_effective_weight(self):
        df = make_ntu_hh(rmi_positive_entitlement=0.0, weight_hh=100.0)
        result = apply_fixed_non_takeup_calibration(df)
        assert result["rmi_effective_recipient_weight"].iloc[0] == pytest.approx(0.0)

    def test_invalid_high_ntu_raises(self):
        df = make_ntu_hh()
        with pytest.raises(ValueError, match="between 0 and 1"):
            apply_fixed_non_takeup_calibration(df, high_ntu=1.5, medium_ntu=0.30)

    def test_invalid_medium_ntu_raises(self):
        df = make_ntu_hh()
        with pytest.raises(ValueError, match="between 0 and 1"):
            apply_fixed_non_takeup_calibration(df, high_ntu=0.70, medium_ntu=-0.1)
