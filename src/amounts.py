from __future__ import annotations

import numpy as np
import pandas as pd

HIGH_NTU_DEFAULT = 0.70
MEDIUM_NTU_DEFAULT = 0.30


def assign_guaranteed_amount(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rmi_hhsize_above_listed_schedule"] = np.where(
        out["household_size"].notna()
        & out["max_hh_size_listed"].notna()
        & (out["household_size"] > out["max_hh_size_listed"]),
        1.0,
        0.0,
    )

    exact_listed = (
        out["baseline_main_included"].fillna(False)
        & out["baseline_has_listed_schedule"].fillna(False)
        & out["guaranteed_amount_listed"].notna()
    )

    above_listed_use_cap = (
        out["baseline_main_included"].fillna(False)
        & out["baseline_has_listed_schedule"].fillna(False)
        & out["guaranteed_amount_listed"].isna()
        & out["rmi_hhsize_above_listed_schedule"].eq(1)
        & out["max_amount"].notna()
    )

    base_amount = np.select(
        [exact_listed, above_listed_use_cap],
        [out["guaranteed_amount_listed"], out["max_amount"]],
        default=np.nan,
    )

    out["rmi_guaranteed_amount_monthly"] = np.where(
        pd.notna(base_amount) & out["baseline_amount_topup_factor"].notna(),
        base_amount * out["baseline_amount_topup_factor"],
        base_amount,
    )

    out["rmi_amount_assignment_type"] = np.select(
        [exact_listed, above_listed_use_cap],
        ["exact_schedule_match", "cap_for_above_listed_hhsize"],
        default="unassigned",
    )

    out["rmi_amount_rule_available"] = np.where(
        out["rmi_guaranteed_amount_monthly"].notna(), 1.0, 0.0
    )

    out["rmi_amount_approximate"] = np.select(
        [
            out["rmi_amount_assignment_type"].eq("exact_schedule_match"),
            out["rmi_amount_assignment_type"].eq("cap_for_above_listed_hhsize"),
        ],
        [0.0, 1.0],
        default=np.nan,
    )

    return out


def compute_income_gap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    resources = pd.to_numeric(out["threshold_resources_monthly"], errors="coerce")
    guarantee = pd.to_numeric(out["rmi_guaranteed_amount_monthly"], errors="coerce")

    claimant_unit_ok = (
        out["rmi_age_eligible"].eq(1)
        & out["rmi_claimant_proxy_eligible"].eq(1)
    )

    out["rmi_income_eligible"] = np.where(
        claimant_unit_ok & resources.notna() & guarantee.notna(),
        (resources < guarantee).astype(float),
        np.nan,
    )

    out["rmi_income_gap_entitlement_monthly"] = np.where(
        claimant_unit_ok & resources.notna() & guarantee.notna(),
        np.maximum(guarantee - resources, 0),
        np.nan,
    )

    return out

def finalize_entitlement(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    conditions = (
        out["baseline_main_included"].fillna(False)
        & out["rmi_amount_rule_available"].eq(1)
        & out["rmi_age_eligible"].eq(1)
        & out["rmi_claimant_proxy_eligible"].eq(1)
        & out["rmi_income_eligible"].eq(1)
    )

    out["rmi_sim_eligible"] = np.where(conditions, 1.0, 0.0)

    out["rmi_simulated_benefit_monthly"] = np.where(
        out["rmi_sim_eligible"].eq(1), out["rmi_income_gap_entitlement_monthly"], 0.0
    )

    out["rmi_positive_entitlement"] = np.where(
        out["rmi_simulated_benefit_monthly"] > 0, 1.0, 0.0
    )

    out["rmi_exclusion_reason"] = np.select(
        [
            ~out["baseline_main_included"].fillna(False),
            out["rmi_amount_rule_available"].eq(0),
            out["rmi_age_eligible"].isna(),
            out["rmi_age_eligible"].eq(0),
            out["rmi_claimant_proxy_eligible"].isna(),
            out["rmi_claimant_proxy_eligible"].eq(0),
            out["rmi_income_eligible"].isna(),
            out["rmi_income_eligible"].eq(0),
            out["rmi_sim_eligible"].eq(1),
        ],
        [
            "region_excluded_from_main_baseline",
            "amount_rule_unavailable",
            "age_rule_not_observable",
            "fails_claimant_age_rule",
            "claimant_proxy_not_observable",
            "fails_claimant_proxy_rule",
            "missing_income_or_amount",
            "income_at_or_above_threshold",
            "eligible",
        ],
        default="other",
    )

    return out


def apply_fixed_non_takeup_calibration(
    df: pd.DataFrame,
    high_ntu: float = HIGH_NTU_DEFAULT,
    medium_ntu: float = MEDIUM_NTU_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()

    invalid_rates = [r for r in [high_ntu, medium_ntu] if not (0.0 <= r <= 1.0)]
    if invalid_rates:
        raise ValueError(
            f"All non-take-up rates must be between 0 and 1. "
            f"Got high_ntu={high_ntu}, medium_ntu={medium_ntu}"
        )

    out["fixed_non_take_up_rate"] = np.select(
        [
            out["baseline_non_takeup_group"].eq("high"),
            out["baseline_non_takeup_group"].eq("medium"),
        ],
        [high_ntu, medium_ntu],
        default=0.0,
    )

    out["fixed_take_up_rate"] = 1.0 - out["fixed_non_take_up_rate"]

    out["rmi_effective_recipient_weight"] = np.where(
        out["rmi_positive_entitlement"].eq(1),
        out["weight_hh"] * out["fixed_take_up_rate"],
        0.0,
    )

    out["rmi_positive_entitlement_calibrated"] = np.where(
        out["rmi_positive_entitlement"].eq(1), 1.0, 0.0
    )

    out["non_takeup_calibration_group"] = out["baseline_non_takeup_group"].fillna(
        "none"
    )

    return out
