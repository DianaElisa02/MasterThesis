from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats import weighted_quantile


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
    is_single_nucleus_threeplus = (
        out["threeplus_adults"].eq(1) & out["multi_nucleus_proxy"].eq(0)
    )

    out["rmi_hhtype_eligible"] = np.select(
        [
            out["baseline_allowed_hh_types"].eq("all_household_types"),
            out["baseline_allowed_hh_types"].eq(
                "single_adult_single_parent_two_adults_only"
            ),
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
    out = df.copy()

    is_threeplus = out["threeplus_adults"].eq(1)
    is_single_nucleus = out["multi_nucleus_proxy"].eq(0)

    not_threeplus = (~is_threeplus).astype(float)
    threeplus_ok_if_single_nucleus = np.where(
        is_threeplus, is_single_nucleus.astype(float), 1.0
    )

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


def apply_labour_status_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labour market status gate based on regional conditionality profile.

    Reads labour_gate_profile (one of three values):

    none     — guaranteed income schemes (Basque Country, Navarra, Balearic
               Islands). No labour status condition.
    standard — classic RMI: RP must be unemployed OR all working-age adults
               non-working.
    strict   — explicit insertion conditionality (Andalusia, Canary Islands,
               Castilla-La Mancha): RP unemployed with no evidence of search
               refusal, OR all working-age nonworking.

    Missing active_job_search treated as not refusing (charitable default).
    Where RP not observed, gate passes by default.

    Also produces three sensitivity columns:
      labour_no_gate
      labour_unemployed_or_nonworking
      labour_unemployed_searching
    """
    out = df.copy()

    rp_observed = out["responsible_person_proxy_available"].eq(1)

    rp1_unemployed = out["rp1_activity_status_detail"].astype("string").eq("unemployed")
    rp2_unemployed = out["rp2_activity_status_detail"].astype("string").eq("unemployed")
    any_rp_unemployed = rp1_unemployed | rp2_unemployed

    all_nonworking = out["all_working_age_nonworking"].eq(1)

    rp1_not_refusing = (
        out["rp1_active_job_search"].eq(1) | out["rp1_active_job_search"].isna()
    )
    rp2_not_refusing = (
        out["rp2_active_job_search"].eq(1) | out["rp2_active_job_search"].isna()
    )
    any_rp_unemployed_not_refusing = (
        (rp1_unemployed & rp1_not_refusing) |
        (rp2_unemployed & rp2_not_refusing)
    )

    # Convert all boolean conditions to numpy arrays to avoid np.select dtype issues
    gate_none     = np.ones(len(out), dtype=bool)
    gate_standard = (any_rp_unemployed | all_nonworking).to_numpy()
    gate_strict   = (any_rp_unemployed_not_refusing | all_nonworking).to_numpy()
    rp_observed_np = rp_observed.to_numpy()
    profile_np     = out["labour_gate_profile"].astype("string").to_numpy()

    region_gate = np.select(
        [
            profile_np == "none",
            profile_np == "standard",
            profile_np == "strict",
        ],
        [
            gate_none.astype(float),
            gate_standard.astype(float),
            gate_strict.astype(float),
        ],
        default=np.nan,
    )

    out["rmi_labour_gate"] = np.where(rp_observed_np, region_gate, 1.0)

    is_standard = (profile_np == "standard") & rp_observed_np
    is_strict   = (profile_np == "strict")   & rp_observed_np

    out["rmi_labour_gate_source"] = np.select(
        [
            ~rp_observed_np,
            profile_np == "none",
            is_standard &  gate_standard,
            is_standard & ~gate_standard,
            is_strict   &  gate_strict,
            is_strict   & ~gate_strict,
        ],
        [
            "rp_not_observed",
            "no_conditionality_region",
            "passes_standard_gate",
            "fails_standard_gate",
            "passes_strict_gate",
            "fails_strict_gate",
        ],
        default="unrecognised_profile",
    )

    out["labour_no_gate"] = 1.0
    out["labour_unemployed_or_nonworking"] = np.where(
        rp_observed_np, gate_standard.astype(float), 1.0
    )
    out["labour_unemployed_searching"] = np.where(
        rp_observed_np, gate_strict.astype(float), 1.0
    )

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


def compute_household_type_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three household type versions computed as sensitivity columns.
    NOT part of main eligibility — applied on top of rmi_sim_eligible.

    no_restriction    : all household compositions pass
    proxy_restricted  : 3+ adult households excluded if multi-unit proxy fires
    strict_household  : only single adult, single parent, two-adult households pass
    """
    out = df.copy()

    out["hhtype_no_restriction"] = 1.0

    n_adults     = pd.to_numeric(out["n_adults_18plus"],    errors="coerce").fillna(0)
    n_working    = pd.to_numeric(out["n_working_18_64"],    errors="coerce").fillna(0)
    n_unemployed = pd.to_numeric(out["n_unemployed_18_64"], errors="coerce").fillna(0)

    multi_unit_proxy = (
        (n_adults >= 3) &
        ((n_working >= 2) | (n_unemployed >= 2))
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