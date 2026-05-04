from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.constants import EXCLUDED_SIMULATION_REGIONS, HOUSEHOLD_REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def load_inputs(
    hh_path: Path,
    rules_path: Path,
    schedule_path: Path,
    coverage_path: Path,
    eligibility_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hh       = pd.read_parquet(hh_path)
    rules    = pd.read_parquet(rules_path)
    schedule = pd.read_parquet(schedule_path)
    coverage = pd.read_parquet(coverage_path)
    eligibility = pd.read_parquet(eligibility_path)

    logger.info("Household rows: %s", len(hh))
    logger.info("Baseline rules rows: %s", len(rules))
    logger.info("Baseline schedule rows: %s", len(schedule))
    logger.info("Coverage rows: %s", len(coverage))
    logger.info("Eligibility rows: %s", len(eligibility))

    return hh, rules, schedule, coverage, eligibility

def prepare_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "nuts_code",
        "year",
        "region_name_policy",
        "titulares",
        "total_perceptors",
        "population_reference",
        "coverage_rate_titular_pct",
        "coverage_rate_total_pct",
        "validation_reference",
    ]
    missing = [c for c in keep if c not in coverage.columns]
    if missing:
        raise KeyError(f"Missing columns in coverage input: {missing}")
    return coverage[keep].copy()


def prepare_households(
    hh: pd.DataFrame,
    years: list[int],
    excluded_regions: frozenset[str] = EXCLUDED_SIMULATION_REGIONS,
) -> pd.DataFrame:
    out = hh.loc[hh["year"].isin(years)].copy()

    if "baseline_sim_data_ok" not in out.columns:
        raise KeyError("Missing baseline_sim_data_ok in household input")

    out = out.loc[out["baseline_sim_data_ok"] == 1].copy()

    if "region_code" not in out.columns:
        raise KeyError("Missing region_code in household input")
    out = out.rename(columns={"region_code": "nuts_code"})

    missing = [c for c in HOUSEHOLD_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns in household input: {missing}")
    
    string_cols = {"rp1_activity_status_detail", "rp2_activity_status_detail"}
    for c in HOUSEHOLD_REQUIRED_COLUMNS:
        if c not in string_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["threshold_resources_monthly"] = out["resources_proxy_baseline_monthly"]
    out["pfilter_resources_monthly"] = out["resources_proxy_excl_capital_monthly"]
    out["hh_size_rule"] = out["household_size"].round().astype("Int64")

    out = out.loc[~out["nuts_code"].isin(excluded_regions)].copy()

    logger.info("Prepared pre-period baseline household rows: %s", len(out))
    return out

def prepare_rules(
    rules: pd.DataFrame,
    eligibility: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    rules = rules.loc[rules["year"].isin(years)].copy()

    keep = [
        "nuts_code",
        "year",
        "region_name_policy",
        "baseline_age_threshold",
        "baseline_age_rule_type",
        "baseline_allowed_hh_types",
        "baseline_exclude_threeplus_adults",
        "baseline_threeplus_rule",
        "baseline_wealth_test",
        "baseline_main_included",
        "baseline_amount_method",
        "program_name",
        "simple_schedule",
        "hh_rule_type",
        "amount_simulable",
        "max_amount",
        "max_hh_size_listed",
        "amount_simulation_notes",
        "baseline_has_listed_schedule",
        "baseline_formula_region",
        "baseline_needs_special_handling",
        "baseline_non_takeup_group",
        "baseline_conditionality_profile",
        "labour_gate_profile",
        "baseline_scheme_structure",
        "baseline_amount_topup_factor",
    ]
    missing = [c for c in keep if c not in rules.columns]
    if missing:
        raise KeyError(f"Missing columns in baseline rules input: {missing}")

    out = rules[keep].copy()
    out["baseline_age_threshold"] = pd.to_numeric(
        out["baseline_age_threshold"], errors="coerce"
    )
    out["baseline_amount_topup_factor"] = pd.to_numeric(
        out["baseline_amount_topup_factor"], errors="coerce"
    )
    
    legal_unit = (
        eligibility[["nuts_code", "year", "legal_unit_type"]]
        .drop_duplicates()
    )
    out = out.merge(legal_unit, on=["nuts_code", "year"], how="left", validate="1:1")

    return out

def prepare_schedule(schedule: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    schedule = schedule.loc[schedule["year"].isin(years)].copy()

    keep = [
        "nuts_code",
        "year",
        "region_name_policy",
        "program_name",
        "hh_size",
        "guaranteed_amount",
        "max_amount",
        "schedule_included_main_baseline",
    ]
    missing = [c for c in keep if c not in schedule.columns]
    if missing:
        raise KeyError(f"Missing columns in baseline schedule input: {missing}")

    out = schedule[keep].copy()
    out["hh_size"] = pd.to_numeric(out["hh_size"], errors="raise").astype("Int64")
    out = out.rename(
        columns={
            "hh_size": "hh_size_rule",
            "guaranteed_amount": "guaranteed_amount_listed",
        }
    )
    return out


def merge_inputs(
    hh: pd.DataFrame,
    rules: pd.DataFrame,
    schedule: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    out = hh.merge(rules, on=["nuts_code", "year"], how="left", validate="m:1")

    out = out.merge(
        schedule,
        on=["nuts_code", "year", "hh_size_rule"],
        how="left",
        validate="m:1",
        suffixes=("", "_sched"),
    )

    out = out.merge(
        coverage,
        on=["nuts_code", "year"],
        how="left",
        validate="m:1",
        suffixes=("", "_cov"),
    )

    return out

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "household_id",
        "year",
        "nuts_code",
        "region_name",
        "region_name_policy",
        "program_name",
        "weight_hh",
        "household_size",
        "hh_size_rule",
        "pfilter_resources_monthly",
        "threshold_resources_monthly",
        "resources_before_transfers",
        "resources_after_transfers",
        "baseline_main_included",
        "baseline_amount_method",
        "baseline_age_threshold",
        "baseline_conditionality_profile",
        "labour_gate_profile",
        "baseline_non_takeup_group",
        "baseline_scheme_structure",
        "rmi_guaranteed_amount_monthly",
        "rmi_amount_assignment_type",
        "rmi_age_eligible",
        "rmi_age_rule_source",
        "rmi_claimant_proxy_eligible",
        "rmi_claimant_proxy_source",
        "rmi_wealth_eligible",
        "rmi_hhtype_eligible",
        "rmi_threeplus_adults_allowed",
        "rmi_labour_gate",
        "rmi_labour_gate_source",
        "rmi_income_eligible",
        "rmi_income_gap_entitlement_monthly",
        "rmi_sim_eligible",
        "rmi_simulated_benefit_monthly",
        "rmi_positive_entitlement",
        "rmi_exclusion_reason",
        "labour_no_gate",
        "labour_unemployed_or_nonworking",
        "labour_unemployed_searching",
        "wealth_no_test",
        "wealth_soft",
        "wealth_strict",
        "hhtype_no_restriction",
        "hhtype_proxy_restricted",
        "hhtype_strict_household",
    ]
    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]