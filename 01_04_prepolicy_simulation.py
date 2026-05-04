from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scipy.stats import spearmanr

from src.amounts import (
    assign_guaranteed_amount,
    compute_income_gap,
    compute_income_concept_versions,
    finalize_entitlement,
)
from src.eligibility import (
    apply_age_rule,
    apply_claimant_proxy_rule,
    compute_wealth_versions,
    compute_labour_gate_versions,
    compute_household_type_versions,
)
from src.io import (
    load_inputs,
    merge_inputs,
    prepare_coverage,
    prepare_households,
    prepare_rules,
    prepare_schedule,
    reorder_columns,
)
from src.stats import print_compact_table
from src.summaries import (
    make_region_diagnostic_table,
    make_region_summary,
    make_year_summary,
    make_wealth_sensitivity_table,
    make_labour_sensitivity_table,
    make_income_sensitivity_table,
    make_household_type_sensitivity_table,
)

BASE_PATH = Path(".").resolve()

INPUT_HH = BASE_PATH / "ecv_household_clean.parquet"
INPUT_RULES = BASE_PATH / "policy_db" / "rmi_baseline_rules.parquet"
INPUT_SCHEDULE = BASE_PATH / "policy_db" / "rmi_baseline_schedule.parquet"
INPUT_COVERAGE = BASE_PATH / "policy_db" / "rmi_coverage_reference.parquet"
INPUT_ELIGIBILITY = BASE_PATH / "policy_db" / "rmi_eligibility_full.parquet"

PRE_YEARS = [2017, 2018, 2019]
RUN_TAG = f"pre_{PRE_YEARS[0]}_{PRE_YEARS[-1]}"

OUTPUT_HH       = BASE_PATH / f"ecv_rmi_baseline_{RUN_TAG}.parquet"
OUTPUT_CSV      = BASE_PATH / f"ecv_rmi_baseline_{RUN_TAG}.csv"
OUTPUT_YEAR     = BASE_PATH / f"rmi_baseline_{RUN_TAG}_year_summary.parquet"
OUTPUT_REGION   = BASE_PATH / f"rmi_baseline_{RUN_TAG}_region_summary.parquet"
OUTPUT_REGION_DIAG = BASE_PATH / f"rmi_baseline_{RUN_TAG}_region_diagnostic.parquet"
OUTPUT_WEALTH   = BASE_PATH / f"rmi_baseline_{RUN_TAG}_wealth_sensitivity.parquet"
OUTPUT_LABOUR   = BASE_PATH / f"rmi_baseline_{RUN_TAG}_labour_sensitivity.parquet"
OUTPUT_INCOME = BASE_PATH / f"rmi_baseline_{RUN_TAG}_income_sensitivity.parquet"
OUTPUT_HHTYPE = BASE_PATH / f"rmi_baseline_{RUN_TAG}_household_type_sensitivity.parquet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    hh, rules, schedule, coverage, eligibility = load_inputs(
    INPUT_HH, INPUT_RULES, INPUT_SCHEDULE, INPUT_COVERAGE, INPUT_ELIGIBILITY
)

    hh = prepare_households(hh, years=PRE_YEARS)
    rules = prepare_rules(rules, eligibility, years=PRE_YEARS)
    schedule = prepare_schedule(schedule, years=PRE_YEARS)
    coverage = prepare_coverage(coverage)
    sim = merge_inputs(hh, rules, schedule, coverage)
    sim = apply_age_rule(sim)
    sim = apply_claimant_proxy_rule(sim)
    sim = assign_guaranteed_amount(sim)
    sim = compute_income_gap(sim)
    sim = finalize_entitlement(sim)
    sim = compute_wealth_versions(sim)
    sim = compute_labour_gate_versions(sim)
    sim = compute_income_concept_versions(sim)
    sim = compute_household_type_versions(sim)

    sim = compute_household_type_versions(sim)

# --- Temporary diagnostic for household type versions ---
    print("\nn_adults_18plus sample values:")
    print(pd.to_numeric(sim["n_adults_18plus"], errors="coerce").describe())
    print("\nn_working_18_64 sample values:")
    print(pd.to_numeric(sim["n_working_18_64"], errors="coerce").describe())
    print("\nHouseholds with 3+ adults:")
    n_adults = pd.to_numeric(sim["n_adults_18plus"], errors="coerce").fillna(0)
    print((n_adults >= 3).value_counts())
    print("\nHouseholds with 2+ working among 3+ adult households:")
    n_working = pd.to_numeric(sim["n_working_18_64"], errors="coerce").fillna(0)
    print(n_working[n_adults >= 3].describe())
# --- End diagnostic ---

    sim = reorder_columns(sim)




    year_summary = make_year_summary(sim)
    region_summary = make_region_summary(sim)
    region_diag = make_region_diagnostic_table(sim)
    wealth_sensitivity = make_wealth_sensitivity_table(sim)
    labour_sensitivity = make_labour_sensitivity_table(sim)
    income_sensitivity = make_income_sensitivity_table(sim)
    hhtype_sensitivity = make_household_type_sensitivity_table(sim)

    print("\n" + "=" * 80)
    print("PRE-POLICY RMI SIMULATION — RAW SIMULATED COUNTS")
    print("=" * 80)
    print_compact_table(
    year_summary,
    title="Year summary: simulated households vs observed titulares",
    columns=[
        "year",
        "weighted_total_simulated_households",
        "observed_titulares",
        "absolute_gap_sim_minus_titulares",
        "pct_gap_vs_titulares",
    ],
    sort_by=["year"],
    ascending=True,
    digits=3,
    )

    for year in sorted(sim["year"].dropna().unique()):
        print_compact_table(
            region_diag.loc[region_diag["year"] == year],
            title=f"Region diagnostic {year}",
            columns=[
                "nuts_code",
                "region_name_policy",
                "observed_titulares",
                "simulated_households",
                "abs_gap",
                "pct_gap",
                "share_simulated",
                "share_age_eligible",
                "share_income_eligible",
            ],
            sort_by=["pct_gap"],
            ascending=False,
            digits=3,
        )

    print("\n" + "=" * 70)
    print("WEALTH TEST SENSITIVITY")
    print("=" * 70)
    print_compact_table(
    wealth_sensitivity,
    title="Simulated households by wealth version and year",
    columns=["wealth_version", "year", "simulated_households", "observed_titulares", "gap_pct"],
    sort_by=["year", "wealth_version"],
    ascending=True,
    digits=3,
    )

    print("\n" + "=" * 70)
    print("LABOUR GATE SENSITIVITY")
    print("=" * 70)
    print_compact_table(
    labour_sensitivity,
    title="Simulated households by labour gate version and year",
    columns=["labour_version", "year", "simulated_households", "observed_titulares", "gap_pct"],
    sort_by=["year", "labour_version"],
    ascending=True,
    digits=3,
    )

    print("\n" + "=" * 70)
    print("INCOME CONCEPT SENSITIVITY")
    print("=" * 70)
    print_compact_table(
    income_sensitivity,
    title="Simulated households by income concept and year",
    columns=["income_version", "year", "simulated_households", "observed_titulares", "gap_pct"],
    sort_by=["year", "income_version"],
    ascending=True,
    digits=3,
    )

    print("\n" + "=" * 70)
    print("HOUSEHOLD TYPE SENSITIVITY")
    print("=" * 70)
    print_compact_table(
    hhtype_sensitivity,
    title="Simulated households by household type version and year",
    columns=["hhtype_version", "year", "simulated_households", "observed_titulares", "gap_pct"],
    sort_by=["year", "hhtype_version"],
    ascending=True,
    digits=3,
    )

    sim.to_parquet(OUTPUT_HH, index=False)
    sim.to_csv(OUTPUT_CSV, index=False)
    year_summary.to_parquet(OUTPUT_YEAR, index=False)
    region_summary.to_parquet(OUTPUT_REGION, index=False)
    region_diag.to_parquet(OUTPUT_REGION_DIAG, index=False)
    wealth_sensitivity.to_parquet(OUTPUT_WEALTH, index=False)
    labour_sensitivity.to_parquet(OUTPUT_LABOUR, index=False)
    income_sensitivity.to_parquet(OUTPUT_INCOME, index=False)
    hhtype_sensitivity.to_parquet(OUTPUT_HHTYPE, index=False)
    logger.info("Saved household simulation file to %s", OUTPUT_HH)
    logger.info("Saved CSV copy to %s", OUTPUT_CSV)
    logger.info("Saved year summary to %s", OUTPUT_YEAR)
    logger.info("Saved region summary to %s", OUTPUT_REGION)
    logger.info("Saved region diagnostic to %s", OUTPUT_REGION_DIAG)
    logger.info("Saved wealth sensitivity to %s", OUTPUT_WEALTH)
    logger.info("Saved labour sensitivity to %s", OUTPUT_LABOUR)
    logger.info("Saved income sensitivity to %s", OUTPUT_INCOME)
    logger.info("Saved household type sensitivity to %s", OUTPUT_HHTYPE)

if __name__ == "__main__":
    main()
