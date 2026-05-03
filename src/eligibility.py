from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats import weighted_quantile

LABOUR_INCOME_MONTHLY_LIMIT_DEFAULT = 600.0


def apply_age_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    threshold = pd.to_numeric(out["baseline_age_threshold"], errors="coerce")

    rp1_candidate_ok = out["rp1_age"].ge(threshold).fillna(False) & out[
        "rp1_claimant_activity_eligible"
    ].eq(1).fillna(False)

    rp2_candidate_ok = out["rp2_age"].ge(threshold).fillna(False) & out[
        "rp2_claimant_activity_eligible"
    ].eq(1).fillna(False)

    out["rmi_age_eligible"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        (rp1_candidate_ok | rp2_candidate_ok).astype(float),
        np.nan,
    )

    out["rmi_age_rule_source"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        "responsible_person_claimant_age_proxy",
        "not_observed",
    )

    return out


def apply_claimant_proxy_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_claimant_proxy_eligible"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        out["any_responsible_person_claimant_eligible"],
        np.nan,
    )

    out["rmi_claimant_proxy_source"] = np.where(
        out["responsible_person_proxy_available"].eq(1),
        "responsible_person_claimant_proxy",
        "not_observed",
    )

    return out


def add_multi_nucleus_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["multi_nucleus_proxy"] = np.where(
        (out["n_adults_18plus"].fillna(0) >= 3)
        & (
            (out["n_working_18_64"].fillna(0) >= 2)
            | (out["n_unemployed_18_64"].fillna(0) >= 2)
        ),
        1.0,
        0.0,
    )

    return out


def apply_wealth_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set rmi_wealth_eligible based on the region's asset exclusion rule.

    Two field-name variants encode the same strict asset-proxy exclusion:
    'proxy_asset_exclusion_strict' and 'strict_proxy_exclusion'. A household
    passes if wealth_proxy_strict == 0 (no detected assets). Regions with no
    wealth test ('none') always pass. Result is NaN when the test type is
    unrecognised or the proxy is unobserved.
    """
    out = df.copy()

    has_strict_wealth_test = out["baseline_wealth_test"].isin(
        ["proxy_asset_exclusion_strict", "strict_proxy_exclusion"]
    )
    no_wealth_test = out["baseline_wealth_test"].eq("none")
    wealth_observable = out["wealth_proxy_strict"].notna()
    passes_wealth_proxy = (out["wealth_proxy_strict"] == 0).astype(float)

    out["rmi_wealth_eligible"] = np.select(
        [has_strict_wealth_test & wealth_observable, no_wealth_test],
        [passes_wealth_proxy, 1.0],
        default=np.nan,
    )

    return out


def apply_household_type_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set rmi_hhtype_eligible based on the region's allowed household-composition rule.

    Three tiers are defined by baseline_allowed_hh_types:
    - 'all_household_types': every composition passes.
    - 'single_adult_single_parent_two_adults_only': only simple types pass.
    - 'single_adult_single_parent_two_adults_plus_restricted_threeplus': simple types
      plus threeplus households with no evidence of multiple cohabiting units
      (multi_nucleus_proxy == 0).
    Result is NaN for unrecognised rule codes.
    """
    out = df.copy()

    is_simple_type = (
        out["single_adult"].eq(1) | out["single_parent"].eq(1) | out["two_adults"].eq(1)
    )
    is_single_nucleus_threeplus = out["threeplus_adults"].eq(1) & out["multi_nucleus_proxy"].eq(0)

    out["rmi_hhtype_eligible"] = np.select(
        [
            out["baseline_allowed_hh_types"].eq("all_household_types"),
            out["baseline_allowed_hh_types"].eq("single_adult_single_parent_two_adults_only"),
            out["baseline_allowed_hh_types"].eq(
                "single_adult_single_parent_two_adults_plus_restricted_threeplus"
            ),
        ],
        [
            1.0,
            is_simple_type.astype(float),
            (is_simple_type | is_single_nucleus_threeplus).astype(float),
        ],
        default=np.nan,
    )

    return out


def apply_threeplus_adults_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set rmi_threeplus_adults_allowed based on the region's three-plus-adults policy.

    Regions with baseline_threeplus_rule == 'allow_all' skip this gate (always 1).
    Others use baseline_exclude_threeplus_adults:
    - True  (hard exclusion): threeplus households are never eligible.
    - False (soft exclusion): threeplus is allowed only for single-nucleus households
      (multi_nucleus_proxy == 0), i.e. no evidence of multiple cohabiting family units.
    Result is NaN when neither condition applies.
    """
    out = df.copy()

    is_threeplus = out["threeplus_adults"].eq(1)
    is_single_nucleus = out["multi_nucleus_proxy"].eq(0)

    not_threeplus = (~is_threeplus).astype(float)
    threeplus_ok_if_single_nucleus = np.where(is_threeplus, is_single_nucleus.astype(float), 1.0)

    out["rmi_threeplus_adults_allowed"] = np.select(
        [
            out["baseline_threeplus_rule"].eq("allow_all"),
            out["baseline_exclude_threeplus_adults"].eq(True),
            out["baseline_exclude_threeplus_adults"].eq(False),
        ],
        [1.0, not_threeplus, threeplus_ok_if_single_nucleus],
        default=np.nan,
    )

    return out


def apply_labour_rule(
    df: pd.DataFrame,
    labour_income_limit: float = LABOUR_INCOME_MONTHLY_LIMIT_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()

    labour_income = pd.to_numeric(out["labour_income_hh_monthly"], errors="coerce")
    labour_income_ok = labour_income.le(labour_income_limit)

    labour_context_ok = (
        out["any_unemployed_18_64"].eq(1)
        | out["all_working_age_nonworking"].eq(1)
        | out["any_responsible_person_active_search"].eq(1)
        | out["any_social_assistance_income_hh"].eq(1)
    )

    out["rmi_labour_income_eligible"] = np.where(
        labour_income.notna(), labour_income_ok.astype(float), np.nan
    )

    out["rmi_labour_context_eligible"] = np.where(
        out["has_labour_composition"].eq(1)
        | out["responsible_person_proxy_available"].eq(1),
        labour_context_ok.astype(float),
        np.nan,
    )

    strict_labour_ok = np.where(
        out["rmi_labour_income_eligible"].eq(1)
        & out["rmi_labour_context_eligible"].eq(1),
        1.0,
        np.where(
            out["rmi_labour_income_eligible"].isna()
            | out["rmi_labour_context_eligible"].isna(),
            np.nan,
            0.0,
        ),
    )

    relaxed_labour_ok = np.where(
        out["rmi_labour_income_eligible"].eq(1),
        1.0,
        np.where(out["rmi_labour_income_eligible"].isna(), np.nan, 0.0),
    )

    out["rmi_labour_eligible"] = np.where(
        out["baseline_relax_labour_gate"].eq(True), relaxed_labour_ok, strict_labour_ok
    )

    out["rmi_labour_rule_source"] = np.select(
        [
            out["rmi_labour_income_eligible"].isna(),
            out["baseline_relax_labour_gate"].eq(True)
            & out["rmi_labour_income_eligible"].eq(1),
            out["rmi_labour_income_eligible"].eq(0),
            out["baseline_relax_labour_gate"].eq(False)
            & out["rmi_labour_context_eligible"].eq(0),
            out["baseline_relax_labour_gate"].eq(False)
            & out["rmi_labour_eligible"].eq(1),
        ],
        [
            "labour_rule_not_observable",
            "relaxed_labour_income_only_rule",
            "fails_labour_income_rule",
            "fails_labour_context_rule",
            "labour_income_and_context_rule",
        ],
        default="other",
    )

    return out


def apply_region_specific_insertion_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_insertion_rule_eligible"] = 1.0
    out["rmi_insertion_rule_source"] = "not_applicable"

    # Andalusia
    mask = out["nuts_code"].eq("ES61")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "andalusia_insertion_proxy", "fails_andalusia_insertion_proxy"
    )

    # Castilla-La Mancha
    mask = out["nuts_code"].eq("ES42")
    ok = out["any_responsible_person_active_search"].eq(1) & out[
        "any_responsible_person_claimant_eligible"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "clm_insertion_proxy", "fails_clm_insertion_proxy"
    )

    # Extremadura
    mask = out["nuts_code"].eq("ES43")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "extremadura_insertion_proxy", "fails_extremadura_insertion_proxy"
    )

    # Madrid
    mask = out["nuts_code"].eq("ES30")
    ok = (
        out["any_responsible_person_active_search"].eq(1)
        | (out["any_unemployed_18_64"].eq(1) & out["all_unemployed_searching"].eq(1))
        | out["any_social_assistance_income_hh"].eq(1)
    )
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "madrid_insertion_proxy", "fails_madrid_insertion_proxy"
    )

    # Castilla y León
    mask = out["nuts_code"].eq("ES41")
    ok = out["any_responsible_person_active_search"].eq(1) | (
        out["any_unemployed_18_64"].eq(1) & out["all_unemployed_searching"].eq(1)
    )
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "cyl_insertion_proxy", "fails_cyl_insertion_proxy"
    )

    # Valencia
    mask = out["nuts_code"].eq("ES52")
    ok = out["any_responsible_person_active_search"].eq(1) | out[
        "any_social_assistance_income_hh"
    ].eq(1)
    out.loc[mask, "rmi_insertion_rule_eligible"] = np.where(ok[mask], 1.0, 0.0)
    out.loc[mask, "rmi_insertion_rule_source"] = np.where(
        ok[mask], "valencia_insertion_proxy", "fails_valencia_insertion_proxy"
    )

    return out


def add_active_inclusion_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_active_inclusion_ok = (
        out["rmi_claimant_proxy_eligible"].eq(1)
        & (
            out["any_responsible_person_active_search"].eq(1)
            | (
                out["any_unemployed_18_64"].eq(1)
                & out["all_unemployed_searching"].eq(1)
            )
            | out["any_social_assistance_income_hh"].eq(1)
        )
    ).astype(float)

    out["active_inclusion_ok"] = np.where(
        out["baseline_apply_active_inclusion_gate"].eq(True),
        base_active_inclusion_ok,
        1.0,
    )

    out["active_inclusion_gate_applied"] = np.where(
        out["baseline_apply_active_inclusion_gate"].eq(True), 1.0, 0.0
    )

    return out


def add_percentile_filter(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = df.copy()

    cutoff_map = (
        out.groupby("year")
        .apply(
            lambda g: weighted_quantile(
                g["pfilter_resources_monthly"], g["weight_hh"], quantile
            )
        )
        .to_dict()
    )

    out["percentile_cutoff_monthly"] = out["year"].map(cutoff_map)

    out["passes_percentile_filter"] = np.select(
        [
            out["pfilter_resources_monthly"].isna()
            | out["percentile_cutoff_monthly"].isna(),
            out["pfilter_resources_monthly"] <= out["percentile_cutoff_monthly"],
            out["pfilter_resources_monthly"] > out["percentile_cutoff_monthly"],
        ],
        [np.nan, 1.0, 0.0],
        default=np.nan,
    )

    out["percentile_rule"] = f"bottom_{int(quantile * 100)}pct"
    return out


def compute_wealth_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three wealth test versions computed as sensitivity columns.
    NOT part of main eligibility — applied on top of rmi_sim_eligible.

    no_test : all households pass
    strict  : fails if any capital income, rental income, or wealth tax paid
    soft    : fails only if capital income > 500 EUR/year
    """
    out = df.copy()

    out["wealth_no_test"] = 1.0

    out["wealth_strict"] = np.where(
        out["any_capital_income"].eq(1)
        | out["any_rental_income"].eq(1)
        | out["any_wealth_tax_paid"].eq(1),
        0.0,
        1.0,
    )

    capital = pd.to_numeric(out["capital_income_annual"], errors="coerce")
    out["wealth_soft"] = np.where(
        capital.gt(500)
        | out["any_rental_income"].eq(1)
        | out["any_wealth_tax_paid"].eq(1),
        0.0,
        1.0,
    )

    return out


def compute_labour_gate_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three labour gate versions computed as sensitivity columns.
    NOT part of main eligibility — applied on top of rmi_sim_eligible.

    Versions derived from baseline_conditionality_profile:
    no_gate     : all households pass
    strict_only : gate applied only where conditionality_profile == 'strict'
                  (Andalusia, Canary Islands, Castilla-La Mancha), which
                  explicitly required unemployed registration as a precondition
                  per Informe RMI 2017, Cuadro 3-2.
    universal   : gate applied to all regions

    Gate condition: at least one responsible person is unemployed AND searching.
    Missing active_job_search is treated as searching — exclusion requires
    positive evidence of non-search (charitable default).
    """
    out = df.copy()

    rp1_unemployed = pd.to_numeric(out["rp1_activity_status_detail"], errors="raise").eq(5)
    rp2_unemployed = pd.to_numeric(out["rp2_activity_status_detail"], errors="raise").eq(5)

    rp1_searching = out["rp1_active_job_search"].eq(1) | out["rp1_active_job_search"].isna()
    rp2_searching = out["rp2_active_job_search"].eq(1) | out["rp2_active_job_search"].isna()

    rp_observed = out["responsible_person_proxy_available"].eq(1)

    gate_passes = (rp1_unemployed & rp1_searching) | (rp2_unemployed & rp2_searching)

    gate_result = np.where(rp_observed, gate_passes.astype(float), 1.0)

    is_strict = out["baseline_conditionality_profile"].eq("strict")

    out["labour_no_gate"] = 1.0
    out["labour_strict_only"] = np.where(is_strict, gate_result, 1.0)
    out["labour_universal"] = gate_result

    return out

def compute_household_type_versions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hhtype_no_restriction"] = 1.0

    n_adults = pd.to_numeric(out["n_adults_18plus"], errors="coerce").fillna(0)
    n_working = pd.to_numeric(out["n_working_18_64"], errors="coerce").fillna(0)
    n_unemployed = pd.to_numeric(out["n_unemployed_18_64"], errors="coerce").fillna(0)

    multi_unit_proxy = (
        (n_adults >= 3) &
        (
            (n_working >= 2) |
            (n_unemployed >= 2)
        )
    )

    is_proxy_region = out["legal_unit_type"].fillna("").str.endswith("_proxy")

    out["hhtype_proxy_restricted"] = np.where(
        is_proxy_region,
        (~multi_unit_proxy).astype(float),
        1.0,
    )

    is_simple = (
        out["single_adult"].eq(1) |
        out["single_parent"].eq(1) |
        out["two_adults"].eq(1)
    )

    out["hhtype_strict_household"] = np.where(
        is_simple.notna(),
        is_simple.astype(float),
        np.nan,
    )

    return out