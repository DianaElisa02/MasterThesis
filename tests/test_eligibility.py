import numpy as np
import pandas as pd
import pytest

from src.eligibility import (
    add_active_inclusion_gate,
    add_multi_nucleus_proxy,
    apply_age_rule,
    apply_claimant_proxy_rule,
    apply_household_type_gate,
    apply_labour_rule,
    apply_region_specific_insertion_rules,
    apply_threeplus_adults_rule,
    apply_wealth_test,
)


def make_hh(**overrides) -> pd.DataFrame:
    """Minimal all-passing household row for eligibility tests."""
    defaults = {
        "responsible_person_proxy_available": 1,
        "rp1_age": 30,
        "rp2_age": np.nan,
        "rp1_claimant_activity_eligible": 1,
        "rp2_claimant_activity_eligible": np.nan,
        "any_responsible_person_claimant_eligible": 1,
        "baseline_age_threshold": 25,
        "baseline_wealth_test": "none",
        "wealth_proxy_strict": 0,
        "baseline_allowed_hh_types": "all_household_types",
        "baseline_threeplus_rule": "allow_all",
        "baseline_exclude_threeplus_adults": False,
        "single_adult": 1,
        "single_parent": 0,
        "two_adults": 0,
        "threeplus_adults": 0,
        "multi_nucleus_proxy": 0,
        "labour_income_hh_monthly": 200.0,
        "any_unemployed_18_64": 1,
        "all_working_age_nonworking": 0,
        "any_responsible_person_active_search": 1,
        "any_social_assistance_income_hh": 0,
        "has_labour_composition": 1,
        "baseline_relax_labour_gate": False,
        "baseline_apply_active_inclusion_gate": False,
        "rmi_claimant_proxy_eligible": 1,
        "all_unemployed_searching": 1,
        "nuts_code": "ES11",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ── Age rule ──────────────────────────────────────────────────────────────────

class TestApplyAgeRule:
    def test_eligible_when_rp1_above_threshold(self):
        df = make_hh(rp1_age=30, baseline_age_threshold=25, rp1_claimant_activity_eligible=1)
        result = apply_age_rule(df)
        assert result["rmi_age_eligible"].iloc[0] == 1.0

    def test_not_eligible_when_rp1_below_threshold(self):
        df = make_hh(rp1_age=20, rp2_age=np.nan, baseline_age_threshold=25,
                     rp1_claimant_activity_eligible=1)
        result = apply_age_rule(df)
        assert result["rmi_age_eligible"].iloc[0] == 0.0

    def test_eligible_via_rp2_when_rp1_too_young(self):
        df = make_hh(rp1_age=20, rp2_age=30, baseline_age_threshold=25,
                     rp1_claimant_activity_eligible=1, rp2_claimant_activity_eligible=1)
        result = apply_age_rule(df)
        assert result["rmi_age_eligible"].iloc[0] == 1.0

    def test_not_eligible_when_rp1_old_enough_but_not_claimant_eligible(self):
        df = make_hh(rp1_age=30, rp2_age=np.nan, baseline_age_threshold=25,
                     rp1_claimant_activity_eligible=0)
        result = apply_age_rule(df)
        assert result["rmi_age_eligible"].iloc[0] == 0.0

    def test_nan_when_no_responsible_person(self):
        df = make_hh(responsible_person_proxy_available=0)
        result = apply_age_rule(df)
        assert pd.isna(result["rmi_age_eligible"].iloc[0])
        assert result["rmi_age_rule_source"].iloc[0] == "not_observed"

    def test_source_label_when_observed(self):
        df = make_hh()
        result = apply_age_rule(df)
        assert result["rmi_age_rule_source"].iloc[0] == "responsible_person_claimant_age_proxy"


# ── Claimant proxy rule ───────────────────────────────────────────────────────

class TestApplyClaimantProxyRule:
    def test_eligible_when_rp_available_and_claimant_eligible(self):
        df = make_hh(responsible_person_proxy_available=1,
                     any_responsible_person_claimant_eligible=1)
        result = apply_claimant_proxy_rule(df)
        assert result["rmi_claimant_proxy_eligible"].iloc[0] == 1.0

    def test_not_eligible_when_rp_available_but_not_claimant_eligible(self):
        df = make_hh(responsible_person_proxy_available=1,
                     any_responsible_person_claimant_eligible=0)
        result = apply_claimant_proxy_rule(df)
        assert result["rmi_claimant_proxy_eligible"].iloc[0] == 0.0

    def test_nan_when_no_responsible_person(self):
        df = make_hh(responsible_person_proxy_available=0)
        result = apply_claimant_proxy_rule(df)
        assert pd.isna(result["rmi_claimant_proxy_eligible"].iloc[0])
        assert result["rmi_claimant_proxy_source"].iloc[0] == "not_observed"


# ── Multi-nucleus proxy ───────────────────────────────────────────────────────

class TestAddMultiNucleusProxy:
    def test_set_when_three_adults_and_two_workers(self):
        df = pd.DataFrame([{"n_adults_18plus": 3, "n_working_18_64": 2, "n_unemployed_18_64": 0}])
        result = add_multi_nucleus_proxy(df)
        assert result["multi_nucleus_proxy"].iloc[0] == 1.0

    def test_set_when_three_adults_and_two_unemployed(self):
        df = pd.DataFrame([{"n_adults_18plus": 3, "n_working_18_64": 0, "n_unemployed_18_64": 2}])
        result = add_multi_nucleus_proxy(df)
        assert result["multi_nucleus_proxy"].iloc[0] == 1.0

    def test_not_set_when_only_two_adults(self):
        df = pd.DataFrame([{"n_adults_18plus": 2, "n_working_18_64": 2, "n_unemployed_18_64": 0}])
        result = add_multi_nucleus_proxy(df)
        assert result["multi_nucleus_proxy"].iloc[0] == 0.0

    def test_not_set_when_three_adults_but_only_one_worker(self):
        df = pd.DataFrame([{"n_adults_18plus": 3, "n_working_18_64": 1, "n_unemployed_18_64": 1}])
        result = add_multi_nucleus_proxy(df)
        assert result["multi_nucleus_proxy"].iloc[0] == 0.0


# ── Wealth test ───────────────────────────────────────────────────────────────

class TestApplyWealthTest:
    def test_eligible_when_no_wealth_test(self):
        df = make_hh(baseline_wealth_test="none")
        assert apply_wealth_test(df)["rmi_wealth_eligible"].iloc[0] == 1.0

    def test_not_eligible_when_strict_and_has_wealth(self):
        df = make_hh(baseline_wealth_test="proxy_asset_exclusion_strict", wealth_proxy_strict=1)
        assert apply_wealth_test(df)["rmi_wealth_eligible"].iloc[0] == 0.0

    def test_eligible_when_strict_and_no_wealth(self):
        df = make_hh(baseline_wealth_test="proxy_asset_exclusion_strict", wealth_proxy_strict=0)
        assert apply_wealth_test(df)["rmi_wealth_eligible"].iloc[0] == 1.0

    def test_both_field_name_variants_accepted(self):
        df = make_hh(baseline_wealth_test="strict_proxy_exclusion", wealth_proxy_strict=0)
        assert apply_wealth_test(df)["rmi_wealth_eligible"].iloc[0] == 1.0

    def test_nan_when_strict_test_but_proxy_unobserved(self):
        df = make_hh(baseline_wealth_test="proxy_asset_exclusion_strict",
                     wealth_proxy_strict=float("nan"))
        assert pd.isna(apply_wealth_test(df)["rmi_wealth_eligible"].iloc[0])


# ── Household-type gate ───────────────────────────────────────────────────────

class TestApplyHouseholdTypeGate:
    def test_all_types_always_eligible(self):
        df = make_hh(baseline_allowed_hh_types="all_household_types", threeplus_adults=1)
        assert apply_household_type_gate(df)["rmi_hhtype_eligible"].iloc[0] == 1.0

    def test_restricted_allows_single_adult(self):
        df = make_hh(
            baseline_allowed_hh_types="single_adult_single_parent_two_adults_only",
            single_adult=1, single_parent=0, two_adults=0, threeplus_adults=0,
        )
        assert apply_household_type_gate(df)["rmi_hhtype_eligible"].iloc[0] == 1.0

    def test_restricted_excludes_threeplus(self):
        df = make_hh(
            baseline_allowed_hh_types="single_adult_single_parent_two_adults_only",
            single_adult=0, single_parent=0, two_adults=0, threeplus_adults=1,
        )
        assert apply_household_type_gate(df)["rmi_hhtype_eligible"].iloc[0] == 0.0

    def test_extended_allows_single_nucleus_threeplus(self):
        df = make_hh(
            baseline_allowed_hh_types=(
                "single_adult_single_parent_two_adults_plus_restricted_threeplus"
            ),
            single_adult=0, single_parent=0, two_adults=0,
            threeplus_adults=1, multi_nucleus_proxy=0,
        )
        assert apply_household_type_gate(df)["rmi_hhtype_eligible"].iloc[0] == 1.0

    def test_extended_excludes_multi_nucleus_threeplus(self):
        df = make_hh(
            baseline_allowed_hh_types=(
                "single_adult_single_parent_two_adults_plus_restricted_threeplus"
            ),
            single_adult=0, single_parent=0, two_adults=0,
            threeplus_adults=1, multi_nucleus_proxy=1,
        )
        assert apply_household_type_gate(df)["rmi_hhtype_eligible"].iloc[0] == 0.0


# ── Three-plus adults rule ────────────────────────────────────────────────────

class TestApplyThreeplusAdultsRule:
    def test_allow_all_rule_passes_threeplus(self):
        df = make_hh(baseline_threeplus_rule="allow_all", threeplus_adults=1)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 1.0

    def test_hard_exclusion_blocks_threeplus(self):
        df = make_hh(baseline_threeplus_rule="other", baseline_exclude_threeplus_adults=True,
                     threeplus_adults=1)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 0.0

    def test_hard_exclusion_passes_non_threeplus(self):
        df = make_hh(baseline_threeplus_rule="other", baseline_exclude_threeplus_adults=True,
                     threeplus_adults=0)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 1.0

    def test_soft_exclusion_allows_single_nucleus_threeplus(self):
        df = make_hh(baseline_threeplus_rule="other", baseline_exclude_threeplus_adults=False,
                     threeplus_adults=1, multi_nucleus_proxy=0)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 1.0

    def test_soft_exclusion_blocks_multi_nucleus_threeplus(self):
        df = make_hh(baseline_threeplus_rule="other", baseline_exclude_threeplus_adults=False,
                     threeplus_adults=1, multi_nucleus_proxy=1)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 0.0

    def test_soft_exclusion_always_passes_non_threeplus(self):
        df = make_hh(baseline_threeplus_rule="other", baseline_exclude_threeplus_adults=False,
                     threeplus_adults=0)
        assert apply_threeplus_adults_rule(df)["rmi_threeplus_adults_allowed"].iloc[0] == 1.0


# ── Labour rule ───────────────────────────────────────────────────────────────

class TestApplyLabourRule:
    def test_eligible_strict_mode_income_and_context_ok(self):
        df = make_hh(labour_income_hh_monthly=200.0, any_unemployed_18_64=1,
                     baseline_relax_labour_gate=False)
        result = apply_labour_rule(df, labour_income_limit=600.0)
        assert result["rmi_labour_eligible"].iloc[0] == 1.0
        assert result["rmi_labour_rule_source"].iloc[0] == "labour_income_and_context_rule"

    def test_fails_when_income_above_limit(self):
        df = make_hh(labour_income_hh_monthly=700.0, any_unemployed_18_64=1,
                     baseline_relax_labour_gate=False)
        result = apply_labour_rule(df, labour_income_limit=600.0)
        assert result["rmi_labour_eligible"].iloc[0] == 0.0
        assert result["rmi_labour_rule_source"].iloc[0] == "fails_labour_income_rule"

    def test_fails_strict_mode_when_context_not_ok(self):
        df = make_hh(
            labour_income_hh_monthly=200.0,
            any_unemployed_18_64=0,
            all_working_age_nonworking=0,
            any_responsible_person_active_search=0,
            any_social_assistance_income_hh=0,
            baseline_relax_labour_gate=False,
            has_labour_composition=1,
        )
        result = apply_labour_rule(df, labour_income_limit=600.0)
        assert result["rmi_labour_eligible"].iloc[0] == 0.0
        assert result["rmi_labour_rule_source"].iloc[0] == "fails_labour_context_rule"

    def test_eligible_relaxed_mode_ignores_context(self):
        df = make_hh(
            labour_income_hh_monthly=200.0,
            any_unemployed_18_64=0,
            all_working_age_nonworking=0,
            any_responsible_person_active_search=0,
            any_social_assistance_income_hh=0,
            baseline_relax_labour_gate=True,
        )
        result = apply_labour_rule(df, labour_income_limit=600.0)
        assert result["rmi_labour_eligible"].iloc[0] == 1.0
        assert result["rmi_labour_rule_source"].iloc[0] == "relaxed_labour_income_only_rule"

    def test_nan_when_income_missing_and_no_context_observable(self):
        df = make_hh(labour_income_hh_monthly=np.nan, has_labour_composition=0,
                     responsible_person_proxy_available=0)
        result = apply_labour_rule(df)
        assert pd.isna(result["rmi_labour_eligible"].iloc[0])
        assert result["rmi_labour_rule_source"].iloc[0] == "labour_rule_not_observable"

    def test_income_at_exact_limit_is_eligible(self):
        df = make_hh(labour_income_hh_monthly=600.0, any_unemployed_18_64=1,
                     baseline_relax_labour_gate=False)
        result = apply_labour_rule(df, labour_income_limit=600.0)
        assert result["rmi_labour_eligible"].iloc[0] == 1.0


# ── Region-specific insertion rules ──────────────────────────────────────────

class TestApplyRegionSpecificInsertionRules:
    def test_not_applicable_for_generic_region(self):
        df = make_hh(nuts_code="ES11")
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 1.0
        assert result["rmi_insertion_rule_source"].iloc[0] == "not_applicable"

    def test_andalusia_passes_with_active_search(self):
        df = make_hh(nuts_code="ES61", any_responsible_person_active_search=1)
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 1.0
        assert result["rmi_insertion_rule_source"].iloc[0] == "andalusia_insertion_proxy"

    def test_andalusia_passes_with_social_assistance(self):
        df = make_hh(nuts_code="ES61", any_responsible_person_active_search=0,
                     any_social_assistance_income_hh=1)
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 1.0

    def test_andalusia_fails_without_either_condition(self):
        df = make_hh(nuts_code="ES61", any_responsible_person_active_search=0,
                     any_social_assistance_income_hh=0)
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 0.0
        assert result["rmi_insertion_rule_source"].iloc[0] == "fails_andalusia_insertion_proxy"

    def test_castilla_la_mancha_requires_both_search_and_claimant_eligible(self):
        df_only_search = make_hh(nuts_code="ES42",
                                 any_responsible_person_active_search=1,
                                 any_responsible_person_claimant_eligible=0)
        result = apply_region_specific_insertion_rules(df_only_search)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 0.0

    def test_castilla_la_mancha_passes_with_both(self):
        df = make_hh(nuts_code="ES42", any_responsible_person_active_search=1,
                     any_responsible_person_claimant_eligible=1)
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 1.0
        assert result["rmi_insertion_rule_source"].iloc[0] == "clm_insertion_proxy"

    def test_madrid_passes_with_unemployed_searching(self):
        df = make_hh(nuts_code="ES30", any_responsible_person_active_search=0,
                     any_social_assistance_income_hh=0,
                     any_unemployed_18_64=1, all_unemployed_searching=1)
        result = apply_region_specific_insertion_rules(df)
        assert result["rmi_insertion_rule_eligible"].iloc[0] == 1.0


# ── Active inclusion gate ─────────────────────────────────────────────────────

class TestAddActiveInclusionGate:
    def test_not_applied_gate_always_passes(self):
        df = make_hh(baseline_apply_active_inclusion_gate=False)
        result = add_active_inclusion_gate(df)
        assert result["active_inclusion_ok"].iloc[0] == 1.0
        assert result["active_inclusion_gate_applied"].iloc[0] == 0.0

    def test_gate_applied_passes_with_active_search(self):
        df = make_hh(baseline_apply_active_inclusion_gate=True,
                     rmi_claimant_proxy_eligible=1,
                     any_responsible_person_active_search=1)
        result = add_active_inclusion_gate(df)
        assert result["active_inclusion_ok"].iloc[0] == 1.0
        assert result["active_inclusion_gate_applied"].iloc[0] == 1.0

    def test_gate_applied_passes_with_social_assistance(self):
        df = make_hh(baseline_apply_active_inclusion_gate=True,
                     rmi_claimant_proxy_eligible=1,
                     any_responsible_person_active_search=0,
                     any_unemployed_18_64=0,
                     any_social_assistance_income_hh=1)
        result = add_active_inclusion_gate(df)
        assert result["active_inclusion_ok"].iloc[0] == 1.0

    def test_gate_applied_fails_without_any_condition(self):
        df = make_hh(
            baseline_apply_active_inclusion_gate=True,
            rmi_claimant_proxy_eligible=1,
            any_responsible_person_active_search=0,
            any_unemployed_18_64=0,
            all_unemployed_searching=0,
            any_social_assistance_income_hh=0,
        )
        result = add_active_inclusion_gate(df)
        assert result["active_inclusion_ok"].iloc[0] == 0.0

    def test_gate_fails_when_claimant_not_proxy_eligible(self):
        df = make_hh(baseline_apply_active_inclusion_gate=True,
                     rmi_claimant_proxy_eligible=0,
                     any_responsible_person_active_search=1)
        result = add_active_inclusion_gate(df)
        assert result["active_inclusion_ok"].iloc[0] == 0.0
