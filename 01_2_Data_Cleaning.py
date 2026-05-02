from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from schema_loader import load_ecv_schema

BASE_PATH = Path(r".").resolve()
INPUT_DIR = BASE_PATH / "input_data"
DATA_PREFIX = "datos_"
YEARS = list(range(2017, 2025))

PROCESSED_DIR = BASE_PATH / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HOUSEHOLD_OUTPUT = BASE_PATH / "ecv_household_clean.parquet"
PERSON_OUTPUT = BASE_PATH / "ecv_person_clean.parquet"

FORCE_REBUILD = False
STRICT_TR_REQUIRED = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


SCHEMA = load_ecv_schema("ecv_schema.yml")


ECV_FILE_PREFIXES = {
    "td": "ECV_Td",
    "th": "ECV_Th",
    "tr": "ECV_Tr",
    "tp": "ECV_Tp",
}


def ecv_file_path(file_type: str, year: int) -> Path:
    """
    Return path to one ECV .dta file.

    Example:
    ecv_file_path("th", 2021)
    -> input_data/ECV_Th_2021.dta
    """
    try:
        prefix = ECV_FILE_PREFIXES[file_type.lower()]
    except KeyError:
        valid = ", ".join(ECV_FILE_PREFIXES)
        raise ValueError(f"Unknown file_type={file_type!r}. Use one of: {valid}")

    return INPUT_DIR / f"{prefix}_{year}.dta"


def make_paths(year: int) -> dict[str, Path]:
    return {
        file_type: ecv_file_path(file_type, year) for file_type in ECV_FILE_PREFIXES
    }


def hh_cache_path(year: int) -> Path:
    return PROCESSED_DIR / f"household_{year}.parquet"


def person_cache_path(year: int) -> Path:
    return PROCESSED_DIR / f"person_{year}.parquet"


def read_dta(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_stata(path, convert_categoricals=False)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def empty_num(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def empty_str(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise")


def to_id(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return x


def get_series(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    as_id: bool = False,
    numeric: bool = False,
) -> pd.Series:
    col = first_existing(df, candidates)
    if col is None:
        if as_id:
            return empty_str(df.index)
        if numeric:
            return empty_num(df.index)
        return empty_str(df.index)

    s = df[col]
    if as_id:
        return to_id(s)
    if numeric:
        return to_num(s)
    return s.astype("string")


def clean_nonnegative(s: pd.Series) -> pd.Series:
    x = to_num(s)
    return x.mask(x < 0, np.nan)


def ensure_unique(df: pd.DataFrame, key: str, name: str) -> None:
    dup = df[key].duplicated(keep=False)
    if dup.any():
        sample = df.loc[dup, key].drop_duplicates().head(10).tolist()
        raise ValueError(f"{name}: duplicate {key}; sample={sample}")


def safe_left_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    validate: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    n0 = len(left)
    out = left.merge(right, on=on, how="left", validate=validate)
    if len(out) != n0:
        raise ValueError(
            f"Row count changed in merge {left_name} <- {right_name}: {n0} -> {len(out)}"
        )
    return out


def recode_yes_no(code: pd.Series) -> pd.Series:
    x = to_num(code)
    out = np.select([x.eq(1), x.eq(2)], [1.0, 0.0], default=np.nan)
    return pd.Series(out, index=code.index, dtype="float64")


def recode_nationality_foreign(code: pd.Series) -> pd.Series:
    x = to_num(code)
    out = np.where(x.eq(1), 0.0, np.where(x.notna(), 1.0, np.nan))
    return pd.Series(out, index=code.index, dtype="float64")


# =============================================================================
# DOMAIN HELPERS
# =============================================================================


def recode_region_name(region_code: pd.Series) -> pd.Series:
    mapping = {
        "ES11": "Galicia",
        "ES12": "Principado de Asturias",
        "ES13": "Cantabria",
        "ES21": "País Vasco",
        "ES22": "Comunidad Foral de Navarra",
        "ES23": "La Rioja",
        "ES24": "Aragón",
        "ES30": "Comunidad de Madrid",
        "ES41": "Castilla y León",
        "ES42": "Castilla-La Mancha",
        "ES43": "Extremadura",
        "ES51": "Cataluña",
        "ES52": "Comunidad Valenciana",
        "ES53": "Illes Balears",
        "ES61": "Andalucía",
        "ES62": "Región de Murcia",
        "ES63": "Ciudad Autónoma de Ceuta",
        "ES64": "Ciudad Autónoma de Melilla",
        "ES70": "Canarias",
        "ESZZ": "Extra-Regio",
    }
    return region_code.map(mapping).astype("string")


def derive_age(tr: pd.DataFrame, year: int) -> pd.Series:
    if "RB082" in tr.columns:
        age = to_num(tr["RB082"])
    elif "RB081" in tr.columns:
        age = to_num(tr["RB081"])
    elif "RB080" in tr.columns:
        age = year - to_num(tr["RB080"])
    else:
        age = empty_num(tr.index)

    return age.mask((age < 0) | (age > 110), np.nan)


def recode_sex(code: pd.Series) -> pd.Series:
    x = to_num(code)
    out = np.select([x.eq(1), x.eq(2)], ["male", "female"], default=pd.NA)
    return pd.Series(out, index=code.index, dtype="string")


def recode_activity_status(tp: pd.DataFrame) -> pd.Series:
    col = first_existing(tp, SCHEMA["tp"]["labour_status_detail"])
    if col is None:
        return empty_str(tp.index)

    x = to_num(tp[col])

    out = np.select(
        [
            x.eq(1),
            x.eq(2),
            x.eq(3),
            x.eq(4),
            x.eq(5),
            x.eq(6),
            x.eq(7),
            x.eq(8),
            x.eq(9),
            x.eq(10),
            x.eq(11),
        ],
        [
            "employee_full_time",
            "employee_part_time",
            "selfemployed_full_time",
            "selfemployed_part_time",
            "unemployed",
            "student",
            "retired",
            "permanently_disabled",
            "military_service",
            "home_care",
            "other_inactive",
        ],
        default=pd.NA,
    )

    return pd.Series(out, index=tp.index, dtype="string")


def recode_activity_group(activity_status_detail: pd.Series) -> pd.Series:
    s = activity_status_detail.astype("string")
    out = pd.Series(pd.NA, index=s.index, dtype="string")

    working_vals = {
        "employee_full_time",
        "employee_part_time",
        "selfemployed_full_time",
        "selfemployed_part_time",
    }
    unemployed_vals = {"unemployed"}
    inactive_vals = {
        "student",
        "retired",
        "permanently_disabled",
        "military_service",
        "home_care",
        "other_inactive",
    }

    out.loc[s.isin(working_vals)] = "working"
    out.loc[s.isin(unemployed_vals)] = "unemployed"
    out.loc[s.isin(inactive_vals)] = "inactive"

    return out


def derive_household_id_from_person_id(person_id: pd.Series) -> pd.Series:
    pid = to_id(person_id)
    return pid.str[:-2]


def load_td_clean(path: Path) -> pd.DataFrame:
    td = read_dta(path)

    out = pd.DataFrame(
        {
            "household_id": get_series(td, SCHEMA["td"]["household_id"], as_id=True),
            "region_code": get_series(td, SCHEMA["td"]["region_code"], as_id=True),
            "weight_hh": get_series(td, SCHEMA["td"]["weight_hh"], numeric=True),
        }
    )
    out["region_name"] = recode_region_name(out["region_code"])

    ensure_unique(out, "household_id", "td")
    return out


def load_th_clean(path: Path) -> pd.DataFrame:
    th = read_dta(path)

    capital_col = first_existing(th, SCHEMA["th"]["capital_income"])
    capital_income = to_num(th[capital_col]) if capital_col else empty_num(th.index)

    out = pd.DataFrame(
        {
            "household_id": get_series(th, SCHEMA["th"]["household_id"], as_id=True),
            "household_size_raw": clean_nonnegative(
                get_series(th, SCHEMA["th"]["household_size_raw"], numeric=True)
            ),
            "official_hh_type": get_series(
                th, SCHEMA["th"]["official_hh_type"], numeric=True
            ),
            "income_after_transfers_annual": get_series(
                th, SCHEMA["th"]["income_after_transfers"], numeric=True
            ),
            "income_before_transfers_annual": get_series(
                th, SCHEMA["th"]["income_before_transfers"], numeric=True
            ),
            "capital_income_annual": capital_income,
            "rental_income_gross_annual": get_series(
                th, SCHEMA["th"]["rental_income_gross"], numeric=True
            ),
            "mortgage_interest_paid_annual": clean_nonnegative(
                get_series(th, SCHEMA["th"]["mortgage_interest_paid"], numeric=True)
            ),
            "wealth_tax_paid_annual": clean_nonnegative(
                get_series(th, SCHEMA["th"]["wealth_tax_paid"], numeric=True)
            ),
            "tenure_status": get_series(
                th, SCHEMA["th"]["tenure_status"], numeric=True
            ),
            "consumption_units": clean_nonnegative(
                get_series(th, SCHEMA["th"]["consumption_units"], numeric=True)
            ),
            "poverty_raw": get_series(th, SCHEMA["th"]["poverty"], numeric=True),
            "matdep_raw": get_series(th, SCHEMA["th"]["matdep"], numeric=True),
            "responsible_person_1": get_series(
                th, SCHEMA["th"]["responsible_person_1"], as_id=True
            ),
            "responsible_person_2": get_series(
                th, SCHEMA["th"]["responsible_person_2"], as_id=True
            ),
        }
    )

    ensure_unique(out, "household_id", "th")
    return out


def load_person_clean(tr_path: Path, tp_path: Path, year: int) -> pd.DataFrame | None:
    if not tr_path.exists():
        if STRICT_TR_REQUIRED:
            raise FileNotFoundError(f"Missing Tr file for {year}")
        return None

    tr = read_dta(tr_path).copy()
    tr["person_id"] = get_series(tr, SCHEMA["tr"]["person_id"], as_id=True)

    hh_id_direct = get_series(tr, SCHEMA["tr"].get("household_id", []), as_id=True)
    if hh_id_direct.notna().any():
        tr["household_id"] = hh_id_direct
        tr["household_id_source"] = "direct_from_tr"
    else:
        tr["household_id"] = derive_household_id_from_person_id(tr["person_id"])
        tr["household_id_source"] = "derived_from_person_id"

    tr["age"] = derive_age(tr, year)
    tr["sex"] = recode_sex(get_series(tr, SCHEMA["tr"]["sex"], numeric=True))
    tr["partner_id"] = get_series(tr, SCHEMA["tr"]["partner_id"], as_id=True)
    tr["has_partner_id"] = (
        tr["partner_id"].notna() & ~tr["partner_id"].isin(["0", ""])
    ).astype(float)
    tr["weight_r"] = get_series(tr, SCHEMA["tr"]["weight_r"], numeric=True)

    person = tr[
        [
            "person_id",
            "household_id",
            "household_id_source",
            "age",
            "sex",
            "partner_id",
            "has_partner_id",
            "weight_r",
        ]
    ].copy()

    if tp_path.exists():
        tp = read_dta(tp_path).copy()
        tp["person_id"] = get_series(tp, SCHEMA["tp"]["person_id"], as_id=True)

        tp["weight_p"] = get_series(tp, SCHEMA["tp"]["weight_p"], numeric=True)
        tp["weight_selected_resp"] = get_series(
            tp, SCHEMA["tp"]["weight_selected_resp"], numeric=True
        )
        tp["person_weight_preferred"] = tp["weight_selected_resp"].combine_first(
            tp["weight_p"]
        )

        tp["activity_status_detail"] = recode_activity_status(tp)
        tp["activity_group"] = recode_activity_group(tp["activity_status_detail"])

        tp["active_job_search"] = recode_yes_no(
            get_series(tp, SCHEMA["tp"]["active_job_search"], numeric=True)
        )
        tp["currently_in_education"] = recode_yes_no(
            get_series(tp, SCHEMA["tp"]["currently_in_education"], numeric=True)
        )
        tp["foreign_nationality"] = recode_nationality_foreign(
            get_series(tp, SCHEMA["tp"]["nationality"], numeric=True)
        )

        tp["social_assistance_income_annual"] = clean_nonnegative(
            get_series(
                tp, SCHEMA["tp"]["social_assistance_income_annual"], numeric=True
            )
        )
        tp["any_social_assistance_income"] = np.where(
            tp["social_assistance_income_annual"].gt(0),
            1.0,
            np.where(tp["social_assistance_income_annual"].notna(), 0.0, np.nan),
        )

        tp["employee_cash_income_net_annual"] = get_series(
            tp, SCHEMA["tp"]["employee_cash_income_net"], numeric=True
        )
        tp["employee_noncash_income_net_annual"] = get_series(
            tp, SCHEMA["tp"]["employee_noncash_income_net"], numeric=True
        )
        tp["selfemployment_income_net_annual"] = get_series(
            tp, SCHEMA["tp"]["selfemployment_income_net"], numeric=True
        )

        tp["labour_income_person_annual"] = (
            tp["employee_cash_income_net_annual"].fillna(0)
            + tp["employee_noncash_income_net_annual"].fillna(0)
            + tp["selfemployment_income_net_annual"].fillna(0)
        )

        all_income_missing = (
            tp["employee_cash_income_net_annual"].isna()
            & tp["employee_noncash_income_net_annual"].isna()
            & tp["selfemployment_income_net_annual"].isna()
        )
        tp.loc[all_income_missing, "labour_income_person_annual"] = np.nan
        tp["labour_income_person_monthly"] = tp["labour_income_person_annual"] / 12

        tp = tp[
            [
                "person_id",
                "weight_p",
                "weight_selected_resp",
                "person_weight_preferred",
                "activity_status_detail",
                "activity_group",
                "active_job_search",
                "currently_in_education",
                "foreign_nationality",
                "social_assistance_income_annual",
                "any_social_assistance_income",
                "employee_cash_income_net_annual",
                "employee_noncash_income_net_annual",
                "selfemployment_income_net_annual",
                "labour_income_person_annual",
                "labour_income_person_monthly",
            ]
        ].copy()
        ensure_unique(tp, "person_id", "tp")

        person = safe_left_merge(
            person, tp, on="person_id", validate="1:1", left_name="tr", right_name="tp"
        )
        person["labour_file_available"] = 1.0

    else:
        person["weight_p"] = np.nan
        person["weight_selected_resp"] = np.nan
        person["person_weight_preferred"] = np.nan
        person["activity_status_detail"] = pd.Series(
            pd.NA, index=person.index, dtype="string"
        )
        person["activity_group"] = pd.Series(pd.NA, index=person.index, dtype="string")
        person["active_job_search"] = np.nan
        person["currently_in_education"] = np.nan
        person["foreign_nationality"] = np.nan
        person["social_assistance_income_annual"] = np.nan
        person["any_social_assistance_income"] = np.nan
        person["labour_file_available"] = 0.0
        person["employee_cash_income_net_annual"] = np.nan
        person["employee_noncash_income_net_annual"] = np.nan
        person["selfemployment_income_net_annual"] = np.nan
        person["labour_income_person_annual"] = np.nan
        person["labour_income_person_monthly"] = np.nan

    person["working_age_18_64"] = person["age"].between(18, 64, inclusive="both")
    person["activity_group_working_age"] = (
        person["activity_group"]
        .where(person["working_age_18_64"], pd.NA)
        .astype("string")
    )

    person["person_file_available"] = 1.0
    person["year"] = year

    ensure_unique(person, "person_id", f"person_{year}")
    return person


# =============================================================================
# PERSON-HOUSEHOLD LINKAGE CHECK
# =============================================================================


def check_person_household_linkage(
    person: pd.DataFrame | None, hh_ids: pd.Series, year: int
) -> None:
    if person is None or person.empty:
        logger.warning(
            "Year %s: no person file available for household linkage check", year
        )
        return

    hh_ids_clean = pd.Series(hh_ids, dtype="string").dropna().drop_duplicates()
    matched = person["household_id"].isin(hh_ids_clean)
    share_matched = matched.mean()

    logger.info(
        "Year %s: person->household linkage match rate = %.4f", year, share_matched
    )

    if share_matched < 0.98:
        unmatched_sample = person.loc[~matched, ["person_id", "household_id"]].head(10)
        logger.warning(
            "Year %s: low person->household linkage rate. Sample unmatched rows:\n%s",
            year,
            unmatched_sample.to_string(index=False),
        )


def summarise_household(group: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(group["age"], errors="coerce")
    agw = group["activity_group_working_age"].astype("string")

    n_persons = len(group)
    n_age_missing = int(age.isna().sum())
    age_complete = float(n_age_missing == 0)

    n_adults = int((age >= 18).sum(skipna=True))
    n_children = int((age < 18).sum(skipna=True))
    n_adults_18plus = n_adults
    n_adults_23plus = int((age >= 23).sum(skipna=True))
    n_adults_25plus = int((age >= 25).sum(skipna=True))

    n_working = int((agw == "working").sum())
    n_unemployed = int((agw == "unemployed").sum())
    n_inactive = int((agw == "inactive").sum())
    n_missing = int(agw.isna().sum())

    n_working_age = int(((age >= 18) & (age <= 64)).sum(skipna=True))

    labour_income_annual = pd.to_numeric(
        group["labour_income_person_annual"], errors="coerce"
    )
    labour_income_monthly = labour_income_annual / 12

    reciprocal_links = 0
    g2 = group[["person_id", "partner_id"]].copy()
    g2["person_id"] = g2["person_id"].astype("string")
    g2["partner_id"] = g2["partner_id"].astype("string")

    for _, row in g2.dropna().iterrows():
        pid = row["person_id"]
        partner = row["partner_id"]
        partner_row = g2.loc[g2["person_id"] == partner, ["partner_id"]]
        if not partner_row.empty and partner_row["partner_id"].eq(pid).any():
            reciprocal_links += 1

    couple_present_partner_proxy = float(reciprocal_links >= 2)

    single_adult = (
        float((n_adults == 1) and (n_children == 0)) if age_complete == 1 else np.nan
    )
    single_parent = (
        float((n_adults == 1) and (n_children > 0)) if age_complete == 1 else np.nan
    )
    two_adults = float(n_adults == 2) if age_complete == 1 else np.nan
    threeplus_adults = float(n_adults >= 3) if age_complete == 1 else np.nan
    children_present = float(n_children > 0) if age_complete == 1 else np.nan

    labour_observed = float(group["labour_file_available"].eq(1).all())

    active_search = pd.to_numeric(group["active_job_search"], errors="coerce")
    social_assist = pd.to_numeric(
        group["social_assistance_income_annual"], errors="coerce"
    )
    foreign_nat = pd.to_numeric(group["foreign_nationality"], errors="coerce")

    n_students_18_64 = int(
        (
            (group["activity_status_detail"] == "student") & group["working_age_18_64"]
        ).sum()
    )
    n_retired_18_64 = int(
        (
            (group["activity_status_detail"] == "retired") & group["working_age_18_64"]
        ).sum()
    )
    n_disabled_18_64 = int(
        (
            (group["activity_status_detail"] == "permanently_disabled")
            & group["working_age_18_64"]
        ).sum()
    )

    any_active_job_search = (
        float(active_search.eq(1).any()) if active_search.notna().any() else np.nan
    )
    all_unemployed_searching = (
        float(
            active_search[group["activity_group_working_age"] == "unemployed"]
            .eq(1)
            .all()
        )
        if (group["activity_group_working_age"] == "unemployed").any()
        else np.nan
    )

    hh_social_assistance_income_annual = (
        float(social_assist.sum(skipna=True)) if social_assist.notna().any() else np.nan
    )
    any_social_assistance_income_hh = (
        float(social_assist.gt(0).any()) if social_assist.notna().any() else np.nan
    )
    any_foreign_nationality_hh = (
        float(foreign_nat.eq(1).any()) if foreign_nat.notna().any() else np.nan
    )

    return pd.Series(
        {
            "n_persons": n_persons,
            "n_age_missing": n_age_missing,
            "age_composition_complete": age_complete,
            "n_adults": n_adults,
            "n_children": n_children,
            "n_adults_18plus": n_adults_18plus,
            "n_adults_23plus": n_adults_23plus,
            "n_adults_25plus": n_adults_25plus,
            "single_adult": single_adult,
            "single_parent": single_parent,
            "two_adults": two_adults,
            "threeplus_adults": threeplus_adults,
            "children_present": children_present,
            "couple_present_partner_proxy": couple_present_partner_proxy,
            "n_working_18_64": n_working,
            "n_unemployed_18_64": n_unemployed,
            "n_inactive_18_64": n_inactive,
            "n_missing_18_64": n_missing,
            "any_working_18_64": float(n_working > 0),
            "any_unemployed_18_64": float(n_unemployed > 0),
            "all_working_age_nonworking": float(
                (n_working_age > 0) and (n_working == 0)
            ),
            "person_composition_observed": 1.0,
            "labour_observed": labour_observed,
            "labour_income_hh_annual": float(labour_income_annual.sum(skipna=True))
            if labour_income_annual.notna().any()
            else np.nan,
            "labour_income_hh_monthly": float(labour_income_monthly.sum(skipna=True))
            if labour_income_monthly.notna().any()
            else np.nan,
            "any_positive_labour_income": float(labour_income_annual.gt(0).any())
            if labour_income_annual.notna().any()
            else np.nan,
            "n_students_18_64": n_students_18_64,
            "n_retired_18_64": n_retired_18_64,
            "n_disabled_18_64": n_disabled_18_64,
            "any_active_job_search": any_active_job_search,
            "all_unemployed_searching": all_unemployed_searching,
            "hh_social_assistance_income_annual": hh_social_assistance_income_annual,
            "any_social_assistance_income_hh": any_social_assistance_income_hh,
            "any_foreign_nationality_hh": any_foreign_nationality_hh,
        }
    )


def build_household_composition(
    person: pd.DataFrame | None, hh_ids: pd.Series
) -> pd.DataFrame:
    if person is None or person.empty:
        out = pd.DataFrame({"household_id": hh_ids.drop_duplicates()})
        cols = [
            "n_persons",
            "n_age_missing",
            "age_composition_complete",
            "n_adults",
            "n_children",
            "n_adults_18plus",
            "n_adults_23plus",
            "n_adults_25plus",
            "single_adult",
            "single_parent",
            "two_adults",
            "threeplus_adults",
            "children_present",
            "couple_present_partner_proxy",
            "n_working_18_64",
            "n_unemployed_18_64",
            "n_inactive_18_64",
            "n_missing_18_64",
            "any_working_18_64",
            "any_unemployed_18_64",
            "all_working_age_nonworking",
            "labour_income_hh_annual",
            "labour_income_hh_monthly",
            "any_positive_labour_income",
            "n_students_18_64",
            "n_retired_18_64",
            "n_disabled_18_64",
            "any_active_job_search",
            "all_unemployed_searching",
            "hh_social_assistance_income_annual",
            "any_social_assistance_income_hh",
            "any_foreign_nationality_hh",
        ]
        for c in cols:
            out[c] = np.nan
        out["person_composition_observed"] = 0.0
        out["labour_observed"] = 0.0
        return out

    out = (
        person.groupby("household_id", dropna=False)
        .apply(summarise_household)
        .reset_index()
    )
    ensure_unique(out, "household_id", "household_composition")
    return out


def build_responsible_person_proxies(
    household_raw: pd.DataFrame, person: pd.DataFrame | None
) -> pd.DataFrame:
    base = household_raw[
        ["household_id", "responsible_person_1", "responsible_person_2"]
    ].copy()

    if person is None or person.empty:
        for c in [
            "rp1_age",
            "rp1_activity_status_detail",
            "rp1_activity_group",
            "rp1_active_job_search",
            "rp1_currently_in_education",
            "rp1_foreign_nationality",
            "rp1_any_social_assistance_income",
            "rp2_age",
            "rp2_activity_status_detail",
            "rp2_activity_group",
            "rp2_active_job_search",
            "rp2_currently_in_education",
            "rp2_foreign_nationality",
            "rp2_any_social_assistance_income",
            "rp1_found",
            "rp2_found",
        ]:
            base[c] = np.nan
        return base

    lookup = person[
        [
            "person_id",
            "age",
            "activity_status_detail",
            "activity_group",
            "active_job_search",
            "currently_in_education",
            "foreign_nationality",
            "any_social_assistance_income",
        ]
    ].copy()

    rp1 = base[["household_id", "responsible_person_1"]].rename(
        columns={"responsible_person_1": "person_id"}
    )
    rp1 = rp1.merge(lookup, on="person_id", how="left", validate="m:1")
    rp1 = rp1.rename(
        columns={
            "age": "rp1_age",
            "activity_status_detail": "rp1_activity_status_detail",
            "activity_group": "rp1_activity_group",
            "active_job_search": "rp1_active_job_search",
            "currently_in_education": "rp1_currently_in_education",
            "foreign_nationality": "rp1_foreign_nationality",
            "any_social_assistance_income": "rp1_any_social_assistance_income",
        }
    )

    rp2 = base[["household_id", "responsible_person_2"]].rename(
        columns={"responsible_person_2": "person_id"}
    )
    rp2 = rp2.merge(lookup, on="person_id", how="left", validate="m:1")
    rp2 = rp2.rename(
        columns={
            "age": "rp2_age",
            "activity_status_detail": "rp2_activity_status_detail",
            "activity_group": "rp2_activity_group",
            "active_job_search": "rp2_active_job_search",
            "currently_in_education": "rp2_currently_in_education",
            "foreign_nationality": "rp2_foreign_nationality",
            "any_social_assistance_income": "rp2_any_social_assistance_income",
        }
    )

    out = base.copy()

    out = safe_left_merge(
        out,
        rp1[
            [
                "household_id",
                "rp1_age",
                "rp1_activity_status_detail",
                "rp1_activity_group",
                "rp1_active_job_search",
                "rp1_currently_in_education",
                "rp1_foreign_nationality",
                "rp1_any_social_assistance_income",
            ]
        ],
        on="household_id",
        validate="1:1",
        left_name="base",
        right_name="rp1",
    )

    out = safe_left_merge(
        out,
        rp2[
            [
                "household_id",
                "rp2_age",
                "rp2_activity_status_detail",
                "rp2_activity_group",
                "rp2_active_job_search",
                "rp2_currently_in_education",
                "rp2_foreign_nationality",
                "rp2_any_social_assistance_income",
            ]
        ],
        on="household_id",
        validate="1:1",
        left_name="out",
        right_name="rp2",
    )

    out["rp1_found"] = np.where(out["rp1_age"].notna(), 1.0, 0.0)
    out["rp2_found"] = np.where(out["rp2_age"].notna(), 1.0, 0.0)
    return out


def derive_household_variables(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["year"] = year

    out["household_size"] = out["n_persons"].combine_first(out["household_size_raw"])
    out["household_size_source"] = pd.Series(
        np.select(
            [
                out["n_persons"].notna(),
                out["n_persons"].isna() & out["household_size_raw"].notna(),
            ],
            [
                "person_file",
                "household_file",
            ],
            default="missing",
        ),
        index=out.index,
        dtype="string",
    )

    out["income_before_transfers_monthly"] = out["income_before_transfers_annual"] / 12
    out["income_after_transfers_monthly"] = out["income_after_transfers_annual"] / 12
    out["capital_income_monthly"] = out["capital_income_annual"] / 12
    out["rental_income_monthly"] = out["rental_income_gross_annual"] / 12

    out["resources_proxy_baseline_annual"] = out["income_before_transfers_annual"]
    out["resources_proxy_baseline_monthly"] = (
        out["resources_proxy_baseline_annual"] / 12
    )

    out["resources_proxy_excl_capital_annual"] = np.where(
        out["income_before_transfers_annual"].notna()
        & out["capital_income_annual"].notna(),
        np.maximum(
            out["income_before_transfers_annual"] - out["capital_income_annual"], 0
        ),
        out["income_before_transfers_annual"],
    )
    out["resources_proxy_excl_capital_monthly"] = (
        out["resources_proxy_excl_capital_annual"] / 12
    )

    out["any_capital_income"] = np.where(out["capital_income_annual"].gt(0), 1.0, 0.0)
    out["any_rental_income"] = np.where(
        out["rental_income_gross_annual"].gt(0), 1.0, 0.0
    )
    out["any_wealth_tax_paid"] = np.where(out["wealth_tax_paid_annual"].gt(0), 1.0, 0.0)

    out["wealth_proxy_strict"] = np.where(
        out["any_capital_income"].eq(1)
        | out["any_rental_income"].eq(1)
        | out["any_wealth_tax_paid"].eq(1),
        1.0,
        0.0,
    )

    out["homeowner"] = np.select(
        [out["tenure_status"].isin([1, 2]), out["tenure_status"].isin([3, 4, 5])],
        [1.0, 0.0],
        default=np.nan,
    )

    out["poverty"] = np.select(
        [out["poverty_raw"].eq(1), out["poverty_raw"].eq(0)], [1.0, 0.0], default=np.nan
    )
    out["matdep"] = np.select(
        [out["matdep_raw"].eq(1), out["matdep_raw"].eq(0)], [1.0, 0.0], default=np.nan
    )

    out["post"] = np.select(
        [out["year"] >= 2021, out["year"] <= 2019], [1.0, 0.0], default=np.nan
    )
    out["period"] = pd.Series(
        np.select(
            [out["year"] <= 2019, out["year"] == 2020, out["year"] >= 2021],
            ["pre_2020", "covid_2020", "post_2020"],
            default=pd.NA,
        ),
        index=out.index,
        dtype="string",
    )

    out["has_region"] = np.where(out["region_code"].notna(), 1.0, 0.0)
    out["has_household_weight"] = np.where(out["weight_hh"].notna(), 1.0, 0.0)
    out["has_resources_proxy"] = np.where(
        out["resources_proxy_baseline_monthly"].notna(), 1.0, 0.0
    )
    out["has_household_composition"] = np.where(
        out["person_composition_observed"].eq(1), 1.0, 0.0
    )
    out["has_labour_composition"] = np.where(out["labour_observed"].eq(1), 1.0, 0.0)
    out["has_complete_age_composition"] = np.where(
        out["age_composition_complete"].eq(1), 1.0, 0.0
    )

    out["baseline_sim_data_ok"] = np.where(
        out["has_region"].eq(1)
        & out["has_household_weight"].eq(1)
        & out["has_resources_proxy"].eq(1)
        & out["household_size"].notna(),
        1.0,
        0.0,
    )

    out["responsible_person_proxy_available"] = np.where(
        out["rp1_found"].eq(1) | out["rp2_found"].eq(1), 1.0, 0.0
    )

    out["labour_income_observed"] = np.where(
        out["labour_income_hh_annual"].notna(), 1.0, 0.0
    )

    out["has_labour_income_monthly"] = np.where(
        out["labour_income_hh_monthly"].gt(0), 1.0, 0.0
    )

    excluded_claimant_statuses = ["student", "retired", "permanently_disabled"]

    out["rp1_claimant_activity_eligible"] = np.where(
        out["rp1_activity_status_detail"].isin(excluded_claimant_statuses),
        0.0,
        np.where(out["rp1_activity_status_detail"].notna(), 1.0, np.nan),
    )

    out["rp2_claimant_activity_eligible"] = np.where(
        out["rp2_activity_status_detail"].isin(excluded_claimant_statuses),
        0.0,
        np.where(out["rp2_activity_status_detail"].notna(), 1.0, np.nan),
    )

    out["any_responsible_person_claimant_eligible"] = np.where(
        out["rp1_claimant_activity_eligible"].eq(1)
        | out["rp2_claimant_activity_eligible"].eq(1),
        1.0,
        np.where(
            out["rp1_claimant_activity_eligible"].notna()
            | out["rp2_claimant_activity_eligible"].notna(),
            0.0,
            np.nan,
        ),
    )

    out["any_responsible_person_active_search"] = np.where(
        out["rp1_active_job_search"].eq(1) | out["rp2_active_job_search"].eq(1),
        1.0,
        np.where(
            out["rp1_active_job_search"].notna() | out["rp2_active_job_search"].notna(),
            0.0,
            np.nan,
        ),
    )

    ensure_unique(out, "household_id", f"household_final_{year}")
    return out


def process_year(
    year: int, force_rebuild: bool = False
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    hh_cache = hh_cache_path(year)
    person_cache = person_cache_path(year)

    if hh_cache.exists() and person_cache.exists() and not force_rebuild:
        logger.info("Loading cache for %s", year)
        hh = pd.read_parquet(hh_cache)
        person = pd.read_parquet(person_cache)
        return hh, person

    logger.info("Processing raw year %s", year)
    paths = make_paths(year)

    if not paths["td"].exists() or not paths["th"].exists():
        logger.warning("Missing Td or Th for %s", year)
        return None, None

    td = load_td_clean(paths["td"])
    th = load_th_clean(paths["th"])
    person = load_person_clean(paths["tr"], paths["tp"], year)

    check_person_household_linkage(person, th["household_id"], year)

    hh_comp = build_household_composition(person, th["household_id"])
    rp = build_responsible_person_proxies(th, person)

    hh = safe_left_merge(
        th,
        hh_comp,
        on="household_id",
        validate="1:1",
        left_name="th",
        right_name="hh_comp",
    )
    hh = safe_left_merge(
        hh, rp, on="household_id", validate="1:1", left_name="hh", right_name="rp"
    )
    hh = safe_left_merge(
        hh, td, on="household_id", validate="1:1", left_name="hh", right_name="td"
    )

    hh = derive_household_variables(hh, year)
    hh.to_parquet(hh_cache, index=False)

    if person is None:
        person_out = pd.DataFrame()
    else:
        hh_context = hh[
            [
                "household_id",
                "year",
                "region_code",
                "region_name",
                "weight_hh",
                "household_size",
                "n_adults",
                "n_children",
                "income_before_transfers_annual",
                "income_after_transfers_annual",
                "resources_proxy_baseline_monthly",
                "resources_proxy_excl_capital_monthly",
                "labour_income_hh_annual",
                "labour_income_hh_monthly",
                "any_positive_labour_income",
                "wealth_proxy_strict",
                "any_active_job_search",
                "all_unemployed_searching",
                "hh_social_assistance_income_annual",
                "any_social_assistance_income_hh",
                "any_foreign_nationality_hh",
                "any_responsible_person_claimant_eligible",
                "any_responsible_person_active_search",
                "poverty",
                "matdep",
            ]
        ].copy()

        person_out = safe_left_merge(
            person,
            hh_context,
            on="household_id",
            validate="m:1",
            left_name="person",
            right_name="hh_context",
        )

    person_out.to_parquet(person_cache, index=False)
    return hh, person_out


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def weighted_share(x: pd.Series, w: pd.Series, value=1.0) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    return float(np.average((x[m] == value).astype(float), weights=w[m]))


def make_checks(hh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in hh.groupby("year"):
        rows.append(
            {
                "year": year,
                "n_households": len(g),
                "weighted_mean_hhsize": weighted_mean(
                    g["household_size"], g["weight_hh"]
                ),
                "weighted_poverty_rate": 100
                * weighted_share(g["poverty"], g["weight_hh"], 1.0),
                "weighted_matdep_rate": 100
                * weighted_share(g["matdep"], g["weight_hh"], 1.0),
                "unweighted_pct_baseline_sim_data_ok": 100
                * g["baseline_sim_data_ok"].eq(1).mean(),
                "unweighted_pct_has_household_composition": 100
                * g["has_household_composition"].eq(1).mean(),
                "unweighted_pct_has_labour_composition": 100
                * g["has_labour_composition"].eq(1).mean(),
                "unweighted_pct_has_complete_age_composition": 100
                * g["has_complete_age_composition"].eq(1).mean(),
                "unweighted_pct_responsible_person_proxy_available": 100
                * g["responsible_person_proxy_available"].eq(1).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    all_hh = []
    all_person = []

    for year in YEARS:
        hh, person = process_year(year, force_rebuild=FORCE_REBUILD)
        if hh is not None:
            all_hh.append(hh)
        if person is not None and not person.empty:
            all_person.append(person)

    if not all_hh:
        raise RuntimeError("No household datasets were produced.")

    household = pd.concat(all_hh, ignore_index=True)
    person = pd.concat(all_person, ignore_index=True) if all_person else pd.DataFrame()

    checks = make_checks(household)

    print("\nHousehold checks")
    print(checks.to_string(index=False))

    household.to_parquet(HOUSEHOLD_OUTPUT, index=False)
    if not person.empty:
        person.to_parquet(PERSON_OUTPUT, index=False)

    checks.to_csv(BASE_PATH / "cleaning_checks.csv", index=False)

    logger.info("Saved household file: %s", HOUSEHOLD_OUTPUT)
    logger.info("Saved person file: %s", PERSON_OUTPUT)


if __name__ == "__main__":
    main()
