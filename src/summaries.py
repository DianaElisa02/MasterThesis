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
        "share_age_eligible": weighted_share(
            g["rmi_age_eligible"], g["weight_hh"], 1.0
        ),
        "share_income_eligible": weighted_share(
            g["rmi_income_eligible"], g["weight_hh"], 1.0
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

        m_region    = g["baseline_main_included"].fillna(False)
        m_amount    = m_region   & g["rmi_amount_rule_available"].eq(1)
        m_age       = m_amount   & g["rmi_age_eligible"].eq(1)
        m_claimant  = m_age      & g["rmi_claimant_proxy_eligible"].eq(1)
        m_wealth    = m_claimant & g["rmi_wealth_eligible"].eq(1)
        m_hhtype    = m_wealth   & g["rmi_hhtype_eligible"].eq(1)
        m_threeplus = m_hhtype   & g["rmi_threeplus_adults_allowed"].eq(1)
        m_labour    = m_threeplus & g["rmi_labour_gate"].eq(1)
        m_income    = m_labour   & g["rmi_income_eligible"].eq(1)
        m_final     = m_income   & g["rmi_sim_eligible"].eq(1)

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
                "after_labour_gate": w.loc[m_labour].sum(),
                "after_income_test": w.loc[m_income].sum(),
                "final_simulated": w.loc[m_final].sum(),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "region_name_policy"])

def make_labour_gate_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        total = g["weight_hh"].sum()
        region = g["region_name_policy"].dropna().unique()[0]
        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": year,
                "labour_gate_profile": g["labour_gate_profile"].iloc[0],
                "total_households": total,
                "share_passes_main_gate": weighted_share(
                    g["rmi_labour_gate"], g["weight_hh"], 1.0
                ),
                "share_no_gate": weighted_share(
                    g["labour_no_gate"], g["weight_hh"], 1.0
                ),
                "share_unemployed_or_nonworking": weighted_share(
                    g["labour_unemployed_or_nonworking"], g["weight_hh"], 1.0
                ),
                "share_unemployed_searching": weighted_share(
                    g["labour_unemployed_searching"], g["weight_hh"], 1.0
                ),
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

def make_wealth_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version in ["no_test", "strict", "soft"]:
        col = f"wealth_{version}"
        for year, g in df.groupby("year"):
            eligible_w = g["rmi_sim_eligible"].eq(1) & g[col].eq(1)
            sim_hh = g.loc[eligible_w, "weight_hh"].sum()
            titulares_values = g[["nuts_code", "titulares"]].drop_duplicates()
            obs = float(titulares_values["titulares"].sum())
            rows.append({
                "wealth_version": version,
                "year": year,
                "simulated_households": sim_hh,
                "observed_titulares": obs,
                "gap_pct": safe_pct_gap(sim_hh, obs),
            })
    return pd.DataFrame(rows).sort_values(["year", "wealth_version"])


def make_labour_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version in ["no_gate", "region_specific"]:
        col = f"labour_{version}"
        for year, g in df.groupby("year"):
            eligible_w = g["rmi_sim_eligible"].eq(1) & g[col].eq(1)
            sim_hh = g.loc[eligible_w, "weight_hh"].sum()
            titulares_values = g[["nuts_code", "titulares"]].drop_duplicates()
            obs = float(titulares_values["titulares"].sum())
            rows.append({
                "labour_version": version,
                "year": year,
                "simulated_households": sim_hh,
                "observed_titulares": obs,
                "gap_pct": safe_pct_gap(sim_hh, obs),
            })
    return pd.DataFrame(rows).sort_values(["year", "labour_version"])

def make_income_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version in ["before_transfers", "after_transfers"]:
        col = f"income_{version}_eligible"
        for year, g in df.groupby("year"):
            eligible_w = (
                g["baseline_main_included"].fillna(False)
                & g["rmi_amount_rule_available"].eq(1)
                & g["rmi_age_eligible"].eq(1)
                & g["rmi_claimant_proxy_eligible"].eq(1)
                & g[col].eq(1)
            )
            sim_hh = g.loc[eligible_w, "weight_hh"].sum()
            titulares_values = g[["nuts_code", "titulares"]].drop_duplicates()
            obs = float(titulares_values["titulares"].sum())
            rows.append({
                "income_version": version,
                "year": year,
                "simulated_households": sim_hh,
                "observed_titulares": obs,
                "gap_pct": safe_pct_gap(sim_hh, obs),
            })
    return pd.DataFrame(rows).sort_values(["year", "income_version"])

def make_household_type_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version in ["no_restriction", "region_specific", "strict_household"]:
        col = f"hhtype_{version}"
        for year, g in df.groupby("year"):
            eligible_w = g["rmi_sim_eligible"].eq(1) & g[col].eq(1)
            sim_hh = g.loc[eligible_w, "weight_hh"].sum()
            titulares_values = g[["nuts_code", "titulares"]].drop_duplicates()
            obs = float(titulares_values["titulares"].sum())
            rows.append({
                "hhtype_version": version,
                "year": year,
                "simulated_households": sim_hh,
                "observed_titulares": obs,
                "gap_pct": safe_pct_gap(sim_hh, obs),
            })
    return pd.DataFrame(rows).sort_values(["year", "hhtype_version"])

def make_year_summary_main(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        simulated_total = g.loc[
            g["rmi_positive_entitlement_main"] == 1, "weight_hh"
        ].sum()

        coverage_year = g[["nuts_code", "titulares"]].drop_duplicates()
        titulares_year = coverage_year["titulares"].sum()

        rows.append(
            {
                "year": year,
                "weighted_total_simulated_main": simulated_total,
                "observed_titulares": titulares_year,
                "absolute_gap": simulated_total - titulares_year,
                "pct_gap": safe_pct_gap(simulated_total, titulares_year),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def make_region_summary_main(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nuts_code, year), g in df.groupby(["nuts_code", "year"]):
        simulated_total = g.loc[
            g["rmi_positive_entitlement_main"] == 1, "weight_hh"
        ].sum()

        region = g["region_name_policy"].dropna().unique()[0]
        titulares = float(g["titulares"].dropna().unique()[0])

        rows.append(
            {
                "nuts_code": nuts_code,
                "region_name_policy": region,
                "year": int(year),
                "weighted_total_simulated_main": simulated_total,
                "observed_titulares": titulares,
                "absolute_gap": simulated_total - titulares,
                "pct_gap": safe_pct_gap(simulated_total, titulares),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "pct_gap"], ascending=[True, False])

def diagnose_undersimulated_regions(
    sim: pd.DataFrame,
    regions: list[str],
    years: list[int] | None = None,
) -> None:
    """
    For each region and year, prints a gate-by-gate funnel showing where
    households drop out of the main spec.
    """
    if years is None:
        years = sorted(sim["year"].dropna().unique().astype(int).tolist())

    for year in years:
        g = sim[sim["year"].astype(int) == int(year)].copy()
        if g.empty:
            print(f"No data for year {year}. Available: {sim['year'].unique()}")
            continue

        print(f"\n{'='*80}")
        print(f"GATE-BY-GATE DIAGNOSTIC — {year}")
        print(f"{'='*80}")

        for nuts in regions:
            r = g[g["nuts_code"] == nuts]
            if r.empty:
                print(f"\n  {nuts} — no data for {year}")
                continue

            region_name = r["region_name_policy"].iloc[0]
            w = r["weight_hh"]
            total = w.sum()
            observed = float(r["titulares"].iloc[0])

            print(f"\n{region_name} ({nuts}) — observed: {observed:,.0f}")
            print(f"  {'Gate':<40} {'Households':>12} {'% of total':>10} {'% of obs':>10}")
            print(f"  {'-'*74}")

            gates = [
                ("Total sample",
                    pd.Series(True, index=r.index)),
                ("baseline_main_included",
                    r["baseline_main_included"].fillna(False)),
                ("amount_rule_available",
                    r["rmi_amount_rule_available"].eq(1)),
                ("age_eligible",
                    r["rmi_age_eligible"].eq(1)),
                ("claimant_proxy_eligible",
                    r["rmi_claimant_proxy_eligible"].eq(1)),
                ("income_gap before_transfers",
                    r["rmi_income_eligible"].eq(1)),
                ("income_after_transfers_eligible",
                    r["income_after_transfers_eligible"].eq(1)),
                ("income_before_transfers_eligible",
                    r["income_before_transfers_eligible"].eq(1)),
                ("wealth_soft",
                    r["wealth_soft"].eq(1)),
                ("wealth_strict",
                    r["wealth_strict"].eq(1)),
                ("labour_region_specific",
                    r["labour_region_specific"].eq(1)),
                ("hhtype_region_specific",
                    r["hhtype_region_specific"].eq(1)),
                ("rmi_sim_eligible permissive",
                    r["rmi_sim_eligible"].eq(1)),
                ("rmi_sim_eligible_main",
                    r["rmi_sim_eligible_main"].eq(1)),
            ]

            for label, mask in gates:
                n = w[mask].sum()
                pct_total = 100 * n / total if total > 0 else 0
                pct_obs   = 100 * n / observed if observed > 0 else 0
                print(f"  {label:<40} {n:>12,.0f} {pct_total:>9.1f}% {pct_obs:>9.1f}%")