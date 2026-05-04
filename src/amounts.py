from __future__ import annotations

import numpy as np
import pandas as pd

REGIONAL_MIN_ENTITLEMENT: dict[str, float] = {
    "ES11": 53.78,   # Galicia — Informe RMI 2017, Cuadro 2
    "ES12": 00.00,   # Asturias — Informe RMI 2017, Cuadro 2
    "ES13": 30.32,   # Cantabria — Informe RMI 2017, Cuadro 2
    "ES21": 0.00,    # Basque Country — no minimum listed
    "ES22": 0.00,    # Navarra — no minimum listed
    "ES23": 0.00,    # La Rioja — no minimum listed
    "ES24": 0.00,    # Aragon — no minimum listed
    "ES30": 0.00,    # Madrid — no minimum listed
    "ES41": 0.00,    # Castilla y León — no minimum listed
    "ES42": 100.00,  # Castilla-La Mancha — Informe RMI 2017, Cuadro 2
    "ES43": 100.00,  # Extremadura — Informe RMI 2017, Cuadro 2
    "ES51": 60.93,   # Catalonia — Informe RMI 2017, Cuadro 2
    "ES52": 30.00,   # Valencia — Informe RMI 2017, Cuadro 2
    "ES53": 108.00,  # Balearic Islands — Informe RMI 2017, Cuadro 2
    "ES61": 98.28,   # Andalusia — Informe RMI 2017, Cuadro 2
    "ES62": 0.00,    # Murcia — no minimum listed
    "ES64": 0.00,    # Melilla — no minimum listed
    "ES70": 127.09,  # Canary Islands — Informe RMI 2017, Cuadro 2
}


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

    raw_gap = np.where(
        claimant_unit_ok & resources.notna() & guarantee.notna(),
        np.maximum(guarantee - resources, 0),
        np.nan,
    )

    min_floor = out["nuts_code"].map(REGIONAL_MIN_ENTITLEMENT).fillna(0.0)
    out["rmi_regional_min_floor"] = min_floor

    gap_above_floor = pd.Series(raw_gap, index=out.index) >= min_floor

    out["rmi_income_eligible"] = np.where(
        claimant_unit_ok & resources.notna() & guarantee.notna(),
        (
            (resources < guarantee) & gap_above_floor
        ).astype(float),
        np.nan,
    )

    out["rmi_income_gap_entitlement_monthly"] = np.where(
        out["rmi_income_eligible"].eq(1),
        raw_gap,
        np.nan,
    )

    out["rmi_below_min_floor"] = np.where(
        claimant_unit_ok & resources.notna() & guarantee.notna()
        & (pd.Series(raw_gap, index=out.index) > 0)
        & ~gap_above_floor,
        1.0,
        0.0,
    )

    return out

def compute_income_concept_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two income concept versions computed as sensitivity columns.
    NOT part of main eligibility — applied on top of rmi_sim_eligible.

    All versions use the same regional minimum floor logic as compute_income_gap.

    before_transfers : income before transfers / 12 (main concept)
                       reflects resources the household generates independently
                       of the state — closest to the legal income test intent
    after_transfers  : income after transfers / 12
                       includes all social transfers — more restrictive upper
                       bound since households receiving other benefits appear
                       richer and fewer pass the income test
    """
    out = df.copy()

    guarantee = pd.to_numeric(out["rmi_guaranteed_amount_monthly"], errors="coerce")
    min_floor = out["nuts_code"].map(REGIONAL_MIN_ENTITLEMENT).fillna(0.0)

    claimant_unit_ok = (
        out["rmi_age_eligible"].eq(1)
        & out["rmi_claimant_proxy_eligible"].eq(1)
    )

    for version, col in [
        ("before_transfers", "resources_proxy_baseline_monthly"),
        ("after_transfers",  "income_after_transfers_monthly"),
    ]:
        resources = pd.to_numeric(out[col], errors="coerce")

        raw_gap = np.where(
            claimant_unit_ok & resources.notna() & guarantee.notna(),
            np.maximum(guarantee - resources, 0),
            np.nan,
        )

        gap_series = pd.Series(raw_gap, index=out.index)
        gap_above_floor = gap_series >= min_floor

        out[f"income_{version}_eligible"] = np.where(
            claimant_unit_ok & resources.notna() & guarantee.notna(),
            ((resources < guarantee) & gap_above_floor).astype(float),
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



def finalize_main_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rmi_sim_eligible_main — the main exposure index specification.

    Applies four historically grounded restrictions on top of the permissive
    baseline (rmi_sim_eligible):

    - after_transfers income concept: resources include all transfers
    - wealth_strict: excludes households with detected capital/rental/wealth income
    - labour_region_specific: correct labour condition per region
    - hhtype_region_specific: correct household type rule per region

    rmi_sim_eligible remains the permissive upper bound for sensitivity analysis.
    rmi_sim_eligible_main is used to construct the primary exposure index.
    """
    out = df.copy()

    is_guaranteed = out["nuts_code"].isin(["ES21", "ES22", "ES53", "ES13", "ES41"])
    out["income_concept_main"] = np.where(
        is_guaranteed,
        out["income_before_transfers_eligible"].fillna(0),
        out["income_after_transfers_eligible"].fillna(0),
    )

    check = out.groupby("nuts_code")["income_concept_main"].mean()
    check2 = out.groupby("nuts_code")["labour_gate_profile"].first()
    print(pd.DataFrame({"labour_gate_profile": check2, "mean_income_concept_main": check}))

    conditions = (
        out["baseline_main_included"].fillna(False)
        & out["rmi_amount_rule_available"].eq(1)
        & out["rmi_age_eligible"].eq(1)
        & out["rmi_claimant_proxy_eligible"].eq(1)
        & out["income_concept_main"].eq(1)
        & out["wealth_soft"].eq(1)
        & out["labour_region_specific"].eq(1)
        & out["hhtype_region_specific"].eq(1)
    )

    out["rmi_sim_eligible_main"] = np.where(conditions, 1.0, 0.0)

    out["rmi_positive_entitlement_main"] = np.where(
        out["rmi_sim_eligible_main"].eq(1)
        & out["rmi_income_gap_entitlement_monthly"].notna()
        & out["rmi_income_gap_entitlement_monthly"].gt(0),
        1.0,
        0.0,
    )

    return out