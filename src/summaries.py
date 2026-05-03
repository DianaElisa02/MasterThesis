from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats import safe_pct_gap, weighted_share


def make_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year, g in df.groupby("year"):
        simulated_total = g.loc[g["rmi_positive_entitlement"] == 1, "weight_hh"].sum()

        coverage_year = g[["nuts_code", "titulares"]].drop_duplicates().copy()
        if coverage_year["nuts_code"].duplicated().any():
            dup_codes = coverage_year.loc[
                coverage_year["nuts_code"].duplicated(), "nuts_code"
            ].tolist()
            raise ValueError(
                f"Duplicate nuts_code values in year coverage summary for {year}: {dup_codes}"
            )

        titulares_year = coverage_year["titulares"].sum()

        rows.append(
            {
                "year": year,
                "weighted_total_simulated_households": simulated_total,
                "observed_titulares": titulares_year,
                "absolute_gap_sim_minus_titulares": simulated_total - titulares_year,
                "pct_gap_vs_titulares": safe_pct_gap(simulated_total, titulares_year),
            }
        )

    return pd.DataFrame(rows).sort_values("year")


def make_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        simulated_total = g.loc[g["rmi_positive_entitlement"] == 1, "weight_hh"].sum()

        region_names = g["region_name_policy"].dropna().unique()
        if len(region_names) != 1:
            raise ValueError(
                f"Expected exactly one region_name_policy for nuts_code={nuts_code}, year={year}, "
                f"found {region_names.tolist()}"
            )
        region = region_names[0]

        titulares_values = g["titulares"].dropna().unique()
        if len(titulares_values) != 1:
            raise ValueError(
                f"Expected exactly one titulares value for nuts_code={nuts_code}, year={year}, "
                f"found {titulares_values.tolist()}"
            )
        titulares_region_year = float(titulares_values[0])

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "weighted_total_simulated_households": simulated_total,
                "observed_titulares": titulares_region_year,
                "absolute_gap_sim_minus_titulares": simulated_total - titulares_region_year,
                "pct_gap_vs_titulares": safe_pct_gap(
                    simulated_total, titulares_region_year
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "region_name_policy"])


def make_year_summary_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year, g in df.groupby("year"):
        calibrated_total = g["rmi_effective_recipient_weight"].sum()

        coverage_year = g[["nuts_code", "titulares"]].drop_duplicates().copy()
        if coverage_year["nuts_code"].duplicated().any():
            dup_codes = coverage_year.loc[
                coverage_year["nuts_code"].duplicated(), "nuts_code"
            ].tolist()
            raise ValueError(
                f"Duplicate nuts_code values in year coverage summary for {year}: {dup_codes}"
            )

        titulares_year = float(coverage_year["titulares"].sum())

        rows.append(
            {
                "year": year,
                "weighted_total_calibrated_households": calibrated_total,
                "observed_titulares": titulares_year,
                "absolute_gap_calibrated_minus_titulares": calibrated_total
                - titulares_year,
                "pct_gap_vs_titulares": safe_pct_gap(calibrated_total, titulares_year),
            }
        )

    return pd.DataFrame(rows).sort_values("year")


def make_region_summary_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        calibrated_total = g["rmi_effective_recipient_weight"].sum()

        region_names = g["region_name_policy"].dropna().unique()
        if len(region_names) != 1:
            raise ValueError(
                f"Expected exactly one region_name_policy for nuts_code={nuts_code}, year={year}, "
                f"found {region_names.tolist()}"
            )
        region = region_names[0]

        titulares_values = g["titulares"].dropna().unique()
        if len(titulares_values) != 1:
            raise ValueError(
                f"Expected exactly one titulares value for nuts_code={nuts_code}, year={year}, "
                f"found {titulares_values.tolist()}"
            )
        titulares_region_year = float(titulares_values[0])

        non_takeup_values = g["fixed_non_take_up_rate"].dropna().unique()
        if len(non_takeup_values) != 1:
            raise ValueError(
                f"Expected exactly one fixed_non_take_up_rate for nuts_code={nuts_code}, year={year}, "
                f"found {non_takeup_values.tolist()}"
            )

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "fixed_non_take_up_rate": float(non_takeup_values[0]),
                "fixed_take_up_rate": 1.0 - float(non_takeup_values[0]),
                "weighted_total_calibrated_households": calibrated_total,
                "observed_titulares": titulares_region_year,
                "absolute_gap_calibrated_minus_titulares": calibrated_total
                - titulares_region_year,
                "pct_gap_vs_titulares": safe_pct_gap(
                    calibrated_total, titulares_region_year
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "region_name_policy"])


def make_region_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        total_w = g["weight_hh"].sum()

        region_names = g["region_name_policy"].dropna().unique()
        if len(region_names) != 1:
            raise ValueError(
                f"Expected exactly one region_name_policy for nuts_code={nuts_code}, year={year}, "
                f"found {region_names.tolist()}"
            )
        region = region_names[0]

        titulares_values = g["titulares"].dropna().unique()
        if len(titulares_values) != 1:
            raise ValueError(
                f"Expected exactly one titulares value for nuts_code={nuts_code}, year={year}, "
                f"found {titulares_values.tolist()}"
            )
        titulares = float(titulares_values[0])

        simulated = g.loc[g["rmi_positive_entitlement"] == 1, "weight_hh"].sum()

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "observed_titulares": titulares,
                "simulated_households": simulated,
                "abs_gap": simulated - titulares,
                "pct_gap": safe_pct_gap(simulated, titulares),
                "share_simulated": simulated / total_w if total_w > 0 else np.nan,
                "share_income_eligible": weighted_share(
                    g["rmi_income_eligible"], g["weight_hh"], 1.0
                ),
                "share_pass_pfilter": weighted_share(
                    g["passes_percentile_filter"], g["weight_hh"], 1.0
                ),
                "share_labour_eligible": weighted_share(
                    g["rmi_labour_eligible"], g["weight_hh"], 1.0
                ),
                "share_active_inclusion_ok": weighted_share(
                    g["active_inclusion_ok"], g["weight_hh"], 1.0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "pct_gap"], ascending=[True, False])


def make_eligibility_funnel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        total = g["weight_hh"].sum()

        region_names = g["region_name_policy"].dropna().unique()
        if len(region_names) != 1:
            raise ValueError(
                f"Expected exactly one region_name_policy for nuts_code={nuts_code}, year={year}, "
                f"found {region_names.tolist()}"
            )
        region = region_names[0]

        w = g["weight_hh"]

        m_region = g["baseline_main_included"].fillna(False)
        m_amount = m_region & g["rmi_amount_rule_available"].eq(1)
        m_age = m_amount & g["rmi_age_eligible"].eq(1)
        m_claimant = m_age & g["rmi_claimant_proxy_eligible"].eq(1)
        m_wealth = m_claimant & g["rmi_wealth_eligible"].eq(1)
        m_hhtype = m_wealth & g["rmi_hhtype_eligible"].eq(1)
        m_threeplus = m_hhtype & g["rmi_threeplus_adults_allowed"].eq(1)
        m_labour = m_threeplus & g["rmi_labour_eligible"].eq(1)
        m_inclusion = m_labour & g["active_inclusion_ok"].eq(1)
        m_pfilter = m_inclusion & g["passes_percentile_filter"].eq(1)
        m_income = m_pfilter & g["rmi_income_eligible"].eq(1)
        m_final = m_income & g["rmi_sim_eligible"].eq(1)

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "total_households": total,
                "after_region_included": w.loc[m_region].sum(),
                "after_amount_available": w.loc[m_amount].sum(),
                "after_age_rule": w.loc[m_age].sum(),
                "after_claimant_proxy_rule": w.loc[m_claimant].sum(),
                "after_wealth_rule": w.loc[m_wealth].sum(),
                "after_hh_type_rule": w.loc[m_hhtype].sum(),
                "after_threeplus_rule": w.loc[m_threeplus].sum(),
                "after_labour_rule": w.loc[m_labour].sum(),
                "after_active_inclusion": w.loc[m_inclusion].sum(),
                "after_percentile_filter": w.loc[m_pfilter].sum(),
                "after_income_test": w.loc[m_income].sum(),
                "final_simulated": w.loc[m_final].sum(),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "region_name_policy"])


def make_labour_rule_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        total = g["weight_hh"].sum()

        region_names = g["region_name_policy"].dropna().unique()
        if len(region_names) != 1:
            raise ValueError(
                f"Expected exactly one region_name_policy for nuts_code={nuts_code}, year={year}, "
                f"found {region_names.tolist()}"
            )
        region = region_names[0]

        labour_ok = g.loc[g["rmi_labour_eligible"] == 1, "weight_hh"].sum()
        source_relaxed = g.loc[
            g["rmi_labour_rule_source"] == "relaxed_labour_income_only_rule", "weight_hh"
        ].sum()
        source_full = g.loc[
            g["rmi_labour_rule_source"] == "labour_income_and_context_rule", "weight_hh"
        ].sum()
        source_fail_income = g.loc[
            g["rmi_labour_rule_source"] == "fails_labour_income_rule", "weight_hh"
        ].sum()
        source_fail_context = g.loc[
            g["rmi_labour_rule_source"] == "fails_labour_context_rule", "weight_hh"
        ].sum()
        source_missing = g.loc[
            g["rmi_labour_rule_source"] == "labour_rule_not_observable", "weight_hh"
        ].sum()

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "total_households": total,
                "labour_eligible": labour_ok,
                "share_labour_eligible": labour_ok / total if total > 0 else np.nan,
                "share_relaxed_labour_income_only_rule": source_relaxed / total
                if total > 0
                else np.nan,
                "share_labour_income_and_context_rule": source_full / total
                if total > 0
                else np.nan,
                "share_fails_labour_income_rule": source_fail_income / total
                if total > 0
                else np.nan,
                "share_fails_labour_context_rule": source_fail_context / total
                if total > 0
                else np.nan,
                "share_labour_rule_not_observable": source_missing / total
                if total > 0
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "region_name_policy"])


def debug_income_distribution(sim: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("INCOME DISTRIBUTION BY YEAR")
    print("=" * 80)

    def wmean(x, w):
        return (x * w).sum() / w.sum()

    def wpct(x, w, p):
        df2 = pd.DataFrame({"x": x, "w": w}).dropna()
        df2 = df2.sort_values("x")
        df2["cw"] = df2["w"].cumsum() / df2["w"].sum()
        return df2.loc[df2["cw"] >= p, "x"].iloc[0]

    rows = []

    for year, g in sim.groupby("year"):
        w = g["weight_hh"]
        rows.append(
            {
                "year": year,
                "mean_resources": wmean(g["threshold_resources_monthly"], w),
                "p20_resources": wpct(g["threshold_resources_monthly"], w, 0.2),
                "p30_resources": wpct(g["threshold_resources_monthly"], w, 0.3),
            }
        )

    print(pd.DataFrame(rows).sort_values("year").to_string(index=False))
