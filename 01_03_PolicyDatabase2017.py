from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas import (
    BaselineAmountSummarySchema,
    BaselineRules2017Schema,
    BaselineRulesSchema,
    BaselineScheduleSchema,
    RegionLookupSchema,
    RmiAmountsSchema,
    RmiCoverageSchema,
)

BASE_PATH = Path(".").resolve()

POLICY_SOURCE_DIR = BASE_PATH / "policy_sources"
POLICY_DIR = BASE_PATH / "policy_db"

ANALYSIS_YEARS = [2017, 2018, 2019]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


POLICY_REGION_TO_NUTS = {
    "Andalusia": "ES61",
    "Aragon": "ES24",
    "Asturias": "ES12",
    "Balearic Islands": "ES53",
    "Canary Islands": "ES70",
    "Cantabria": "ES13",
    "Castilla-La Mancha": "ES42",
    "Castilla y Leon": "ES41",
    "Catalonia": "ES51",
    "Ceuta": "ES63",
    "Extremadura": "ES43",
    "Galicia": "ES11",
    "Madrid": "ES30",
    "Melilla": "ES64",
    "Murcia": "ES62",
    "Navarra": "ES22",
    "Basque Country": "ES21",
    "La Rioja": "ES23",
    "Valencia": "ES52",
}


def read_policy_csv(stem: str) -> pd.DataFrame:
    path = POLICY_SOURCE_DIR / f"{stem}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing policy source file: {path}")

    return pd.read_csv(path)


def build_region_lookup() -> pd.DataFrame:
    region_lookup = pd.DataFrame(
        {
            "region_name_policy": list(POLICY_REGION_TO_NUTS.keys()),
            "nuts_code": list(POLICY_REGION_TO_NUTS.values()),
        }
    )

    return RegionLookupSchema.validate(region_lookup, lazy=True)


def add_region_lookup(
    df: pd.DataFrame,
    region_lookup: pd.DataFrame,
) -> pd.DataFrame:
    out = df.merge(
        region_lookup,
        on="region_name_policy",
        how="left",
        validate="m:1",
    )

    missing = (
        out.loc[out["nuts_code"].isna(), "region_name_policy"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if missing:
        raise ValueError(f"Missing NUTS codes for policy regions: {missing}")

    return out


def expand_years(
    df: pd.DataFrame,
    *,
    years: list[int],
    source_year: int,
    assumption_note: str,
) -> pd.DataFrame:
    out = df.merge(pd.DataFrame({"year": years}), how="cross")

    out["source_year"] = source_year
    out["assumption_no_change"] = True
    out["assumption_note"] = assumption_note

    return out


def build_rmi_amounts(region_lookup: pd.DataFrame) -> pd.DataFrame:
    amounts_2017 = read_policy_csv("rmi_amounts_2017")
    amounts_2017 = add_region_lookup(amounts_2017, region_lookup)

    amounts_2017 = RmiAmountsSchema.validate(amounts_2017, lazy=True)

    amounts_2017["simulation_period"] = "pre_2020"
    amounts_2017["resource_concept_simulation"] = "income_before_transfers_monthly"
    amounts_2017["household_size_variable_simulation"] = "household_size"
    amounts_2017["amount_rule_for_simulation"] = "regional_guaranteed_amount_schedule"
    amounts_2017["separate_formula_required"] = ~amounts_2017["simple_schedule"].fillna(
        False
    )

    return expand_years(
        amounts_2017,
        years=ANALYSIS_YEARS,
        source_year=2017,
        assumption_note="2017 parameters carried forward to 2018-2019",
    )


def build_rmi_coverage(region_lookup: pd.DataFrame) -> pd.DataFrame:
    coverage = read_policy_csv("rmi_coverage")
    coverage = add_region_lookup(coverage, region_lookup)

    coverage["simulation_period"] = "pre_2020"

    coverage["coverage_rate_titular_pct"] = np.where(
        coverage["population_reference"].notna(),
        coverage["titulares"] / coverage["population_reference"] * 100,
        np.nan,
    )

    coverage["coverage_rate_total_pct"] = np.where(
        coverage["population_reference"].notna(),
        coverage["total_perceptors"] / coverage["population_reference"] * 100,
        np.nan,
    )

    return RmiCoverageSchema.validate(coverage, lazy=True)


def build_baseline_schedule(rmi_amounts_full: pd.DataFrame) -> pd.DataFrame:
    schedule = rmi_amounts_full.loc[
        rmi_amounts_full["simple_schedule"].fillna(False),
        [
            "nuts_code",
            "region_name_policy",
            "year",
            "program_name",
            "hh_size",
            "guaranteed_amount",
            "max_amount",
            "source_year",
            "assumption_no_change",
            "assumption_note",
            "source",
            "notes",
        ],
    ].copy()

    schedule = schedule.rename(
        columns={
            "source": "source_amount",
            "notes": "notes_amount",
        }
    )

    schedule["schedule_included_main_baseline"] = True

    return BaselineScheduleSchema.validate(schedule, lazy=True)

def build_rmi_eligibility(region_lookup: pd.DataFrame) -> pd.DataFrame:
    eligibility = read_policy_csv("rmi_eligibility_2017")
    eligibility = add_region_lookup(eligibility, region_lookup)

    return expand_years(
        eligibility,
        years=ANALYSIS_YEARS,
        source_year=2017,
        assumption_note="2017 eligibility rules carried forward to 2018-2019",
    )


def build_baseline_amount_summary(rmi_amounts_full: pd.DataFrame) -> pd.DataFrame:
    amount_summary = rmi_amounts_full.groupby(
        ["nuts_code", "region_name_policy", "year"],
        as_index=False,
    ).agg(
        program_name=("program_name", "first"),
        simple_schedule=("simple_schedule", "first"),
        hh_rule_type=("hh_rule_type", "first"),
        amount_simulable=("amount_simulable", "first"),
        max_amount=("max_amount", "first"),
        max_hh_size_listed=(
            "hh_size",
            lambda x: pd.to_numeric(x, errors="coerce").max(),
        ),
        amount_simulation_notes=("amount_simulation_notes", "first"),
        source_amount=("source", "first"),
        notes_amount=("notes", "first"),
    )

    return BaselineAmountSummarySchema.validate(amount_summary, lazy=True)


def build_baseline_rules(
    region_lookup: pd.DataFrame,
    rmi_amounts_full: pd.DataFrame,
) -> pd.DataFrame:
    rules_2017 = read_policy_csv("baseline_rules_2017")
    rules_2017 = add_region_lookup(rules_2017, region_lookup)

    rules_2017 = BaselineRules2017Schema.validate(rules_2017, lazy=True)

    rules = expand_years(
        rules_2017,
        years=ANALYSIS_YEARS,
        source_year=2017,
        assumption_note="2017 baseline operational rules carried forward to 2018-2019",
    )

    amount_summary = build_baseline_amount_summary(rmi_amounts_full)

    rules = rules.merge(
        amount_summary.drop(columns=["region_name_policy"]),
        on=["nuts_code", "year"],
        how="left",
        validate="1:1",
    )

    rules["baseline_has_listed_schedule"] = rules["simple_schedule"].fillna(False)

    rules["baseline_formula_region"] = rules["baseline_amount_method"].eq("formula")

    rules["baseline_needs_special_handling"] = (
        ~rules["baseline_has_listed_schedule"]
        | rules["baseline_formula_region"]
        | ~rules["baseline_main_included"].fillna(False)
    )

    return BaselineRulesSchema.validate(rules, lazy=True)


def save_policy_outputs(
    baseline_rules: pd.DataFrame,
    baseline_schedule: pd.DataFrame,
    rmi_coverage: pd.DataFrame,
    rmi_eligibility: pd.DataFrame,
) -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rules.to_parquet(
        POLICY_DIR / "rmi_baseline_rules.parquet",
        index=False,
    )

    baseline_schedule.to_parquet(
        POLICY_DIR / "rmi_baseline_schedule.parquet",
        index=False,
    )

    rmi_coverage.to_parquet(
        POLICY_DIR / "rmi_coverage_reference.parquet",
        index=False,
    )

    rmi_eligibility.to_parquet(
        POLICY_DIR / "rmi_eligibility_full.parquet",
        index=False,
    )


def print_summary(
    *,
    region_lookup: pd.DataFrame,
    rmi_amounts_full: pd.DataFrame,
    rmi_coverage: pd.DataFrame,
    baseline_rules: pd.DataFrame,
    baseline_schedule: pd.DataFrame,
) -> None:
    print("\nPolicy database summary")
    print("=======================")
    print("Regions:", region_lookup["region_name_policy"].nunique())
    print("Analysis years:", ", ".join(map(str, ANALYSIS_YEARS)))
    print("Rows - full amounts:", len(rmi_amounts_full))
    print("Rows - coverage:", len(rmi_coverage))
    print("Rows - baseline rules:", len(baseline_rules))
    print("Rows - baseline schedule:", len(baseline_schedule))

    print("\nMain baseline inclusion by region-year:")
    print(
        baseline_rules[
            [
                "region_name_policy",
                "year",
                "baseline_main_included",
                "baseline_amount_method",
                "baseline_age_threshold",
                "max_hh_size_listed",
                "baseline_apply_active_inclusion_gate",
                "baseline_relax_labour_gate",
                "baseline_non_takeup_group",
            ]
        ]
        .sort_values(["year", "region_name_policy"])
        .to_string(index=False)
    )


def main() -> None:
    region_lookup = build_region_lookup()

    rmi_amounts_full = build_rmi_amounts(region_lookup)
    rmi_coverage = build_rmi_coverage(region_lookup)
    rmi_eligibility = build_rmi_eligibility(region_lookup)

    baseline_schedule = build_baseline_schedule(rmi_amounts_full)
    baseline_rules = build_baseline_rules(
        region_lookup=region_lookup,
        rmi_amounts_full=rmi_amounts_full,
    )

    print_summary(
        region_lookup=region_lookup,
        rmi_amounts_full=rmi_amounts_full,
        rmi_coverage=rmi_coverage,
        baseline_rules=baseline_rules,
        baseline_schedule=baseline_schedule,
    )

    save_policy_outputs(
        baseline_rules=baseline_rules,
        baseline_schedule=baseline_schedule,
        rmi_coverage=rmi_coverage,
        rmi_eligibility=rmi_eligibility,
        
    )

    logger.info("Saved policy outputs in: %s", POLICY_DIR)


if __name__ == "__main__":
    main()
