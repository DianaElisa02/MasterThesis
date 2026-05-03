from __future__ import annotations

import logging
from pathlib import Path

from src.amounts import (
    apply_fixed_non_takeup_calibration,
    assign_guaranteed_amount,
    compute_income_gap,
    finalize_entitlement,
)
from src.eligibility import (
    add_active_inclusion_gate,
    add_multi_nucleus_proxy,
    add_percentile_filter,
    apply_age_rule,
    apply_claimant_proxy_rule,
    apply_household_type_gate,
    apply_labour_rule,
    apply_region_specific_insertion_rules,
    apply_threeplus_adults_rule,
    apply_wealth_test,
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
    make_region_summary_calibrated,
    make_year_summary,
    make_year_summary_calibrated,
)

BASE_PATH = Path(".").resolve()

INPUT_HH = BASE_PATH / "ecv_household_clean.parquet"
INPUT_RULES = BASE_PATH / "policy_db" / "rmi_baseline_rules.parquet"
INPUT_SCHEDULE = BASE_PATH / "policy_db" / "rmi_baseline_schedule.parquet"
INPUT_COVERAGE = BASE_PATH / "policy_db" / "rmi_coverage_reference.parquet"

PRE_YEARS = [2017, 2018, 2019]
PERCENTILE_FILTER = 0.30
LABOUR_INCOME_MONTHLY_LIMIT = 600.0
HIGH_NTU = 0.70
MEDIUM_NTU = 0.30

RUN_TAG = f"p{int(PERCENTILE_FILTER * 100)}_pre_{PRE_YEARS[0]}_{PRE_YEARS[-1]}"

OUTPUT_HH = BASE_PATH / f"ecv_rmi_baseline_{RUN_TAG}.parquet"
OUTPUT_CSV = BASE_PATH / f"ecv_rmi_baseline_{RUN_TAG}.csv"
OUTPUT_REGION = BASE_PATH / f"rmi_baseline_{RUN_TAG}_region_summary.parquet"
OUTPUT_YEAR = BASE_PATH / f"rmi_baseline_{RUN_TAG}_year_summary.parquet"
OUTPUT_REGION_DIAG = BASE_PATH / f"rmi_baseline_{RUN_TAG}_region_diagnostic.parquet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    hh, rules, schedule, coverage = load_inputs(
        INPUT_HH, INPUT_RULES, INPUT_SCHEDULE, INPUT_COVERAGE
    )

    hh = prepare_households(hh, years=PRE_YEARS)
    rules = prepare_rules(rules, years=PRE_YEARS)
    schedule = prepare_schedule(schedule, years=PRE_YEARS)
    coverage = prepare_coverage(coverage)

    sim = merge_inputs(hh, rules, schedule, coverage)
    sim = apply_age_rule(sim)
    sim = apply_claimant_proxy_rule(sim)
    sim = assign_guaranteed_amount(sim)
    sim = add_percentile_filter(sim, PERCENTILE_FILTER)
    sim = add_multi_nucleus_proxy(sim)
    sim = apply_wealth_test(sim)
    sim = apply_household_type_gate(sim)
    sim = apply_threeplus_adults_rule(sim)
    sim = apply_labour_rule(sim, labour_income_limit=LABOUR_INCOME_MONTHLY_LIMIT)
    sim = apply_region_specific_insertion_rules(sim)
    sim = compute_income_gap(sim)
    sim = add_active_inclusion_gate(sim)
    sim = finalize_entitlement(sim)
    sim = apply_fixed_non_takeup_calibration(sim, high_ntu=HIGH_NTU, medium_ntu=MEDIUM_NTU)
    sim = reorder_columns(sim)

    year_summary = make_year_summary(sim)
    region_summary = make_region_summary(sim)
    year_summary_calibrated = make_year_summary_calibrated(sim)
    region_summary_calibrated = make_region_summary_calibrated(sim)
    region_diag = make_region_diagnostic_table(sim)

    print("\n" + "=" * 80)
    print(
        f"BASELINE PRE-POLICY RMI SIMULATION WITH {int(PERCENTILE_FILTER * 100)}TH PERCENTILE FILTER"
    )
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
                "share_labour_eligible",
                "share_active_inclusion_ok",
            ],
            sort_by=["pct_gap"],
            ascending=False,
            digits=3,
        )

    print("\nActive inclusion proxy by region:")
    print(
        sim.groupby("nuts_code")["active_inclusion_ok"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )

    print_compact_table(
        year_summary_calibrated,
        title="Year summary after fixed non-take-up calibration",
        columns=[
            "year",
            "weighted_total_calibrated_households",
            "observed_titulares",
            "absolute_gap_calibrated_minus_titulares",
            "pct_gap_vs_titulares",
        ],
        sort_by=["year"],
        ascending=True,
        digits=3,
    )

    print_compact_table(
        region_summary_calibrated.loc[
            region_summary_calibrated["fixed_non_take_up_rate"] > 0
        ],
        title="Regions with non-take-up calibration",
        columns=[
            "year",
            "nuts_code",
            "region_name_policy",
            "fixed_non_take_up_rate",
            "fixed_take_up_rate",
            "weighted_total_calibrated_households",
            "observed_titulares",
            "pct_gap_vs_titulares",
        ],
        sort_by=["year", "region_name_policy"],
        ascending=True,
        digits=3,
    )

    sim.to_parquet(OUTPUT_HH, index=False)
    sim.to_csv(OUTPUT_CSV, index=False)
    year_summary.to_parquet(OUTPUT_YEAR, index=False)
    region_summary.to_parquet(OUTPUT_REGION, index=False)
    region_diag.to_parquet(OUTPUT_REGION_DIAG, index=False)

    logger.info("Saved household simulation file to %s", OUTPUT_HH)
    logger.info("Saved CSV copy to %s", OUTPUT_CSV)
    logger.info("Saved year summary to %s", OUTPUT_YEAR)
    logger.info("Saved region summary to %s", OUTPUT_REGION)
    logger.info("Saved region diagnostic to %s", OUTPUT_REGION_DIAG)


if __name__ == "__main__":
    main()
