from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import PERSON_HOUSEHOLD_COLUMNS
from src.extractors import extract_section, load_td_clean, load_th_clean, read_dta
from src.schema_loader import load_ecv_schema
from src.schemas import (
    HouseholdCompositionSchema,
    HouseholdFinalSchema,
    PersonSchema,
    TpSchema,
)

BASE_PATH = Path(r".").resolve()
INPUT_DIR = BASE_PATH / "input_data"
DATA_PREFIX = "datos_"
YEARS = list(range(2017, 2025))

PROCESSED_DIR = BASE_PATH / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HOUSEHOLD_OUTPUT = BASE_PATH / "ecv_household_clean.parquet"
PERSON_OUTPUT = BASE_PATH / "ecv_person_clean.parquet"

FORCE_REBUILD = True
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

def add_missing_tp_columns(person: pd.DataFrame) -> pd.DataFrame:
    p = person.copy()
    tp_cols: dict[str, pd.Series] = {
        "weight_p":                          empty_num(p.index),
        "weight_selected_resp":              empty_num(p.index),
        "person_weight_preferred":           empty_num(p.index),
        "activity_status_detail":            empty_str(p.index),
        "activity_group":                    empty_str(p.index),
        "active_job_search":                 empty_num(p.index),
        "currently_in_education":            empty_num(p.index),
        "foreign_nationality":               empty_num(p.index),
        "social_assistance_income_annual":   empty_num(p.index),
        "any_social_assistance_income":      empty_num(p.index),
        "employee_cash_income_net_annual":   empty_num(p.index),
        "employee_noncash_income_net_annual":empty_num(p.index),
        "selfemployment_income_net_annual":  empty_num(p.index),
        "labour_income_person_annual":       empty_num(p.index),
        "labour_income_person_monthly":      empty_num(p.index),
    }
    for col, val in tp_cols.items():
        if col not in p.columns:
            p[col] = val
    return p


def empty_household_composition(hh_ids: pd.Series) -> pd.DataFrame:
    ids = pd.Series(hh_ids, dtype="string").dropna().drop_duplicates()
    out = pd.DataFrame({"household_id": ids})
    num_cols = [
        "n_persons", "n_age_missing", "n_adults", "n_children",
        "n_adults_23plus", "n_adults_25plus", "n_working_18_64",
        "n_unemployed_18_64", "n_inactive_18_64", "n_missing_18_64",
        "n_students_18_64", "n_retired_18_64", "n_disabled_18_64",
        "labour_observed", "any_active_job_search", "any_positive_labour_income",
        "any_social_assistance_income_hh", "any_foreign_nationality_hh",
        "labour_income_hh_annual", "labour_income_hh_monthly",
        "hh_social_assistance_income_annual", "age_composition_complete",
        "n_adults_18plus", "single_adult", "single_parent", "two_adults",
        "threeplus_adults", "children_present", "any_working_18_64",
        "any_unemployed_18_64", "all_working_age_nonworking",
        "all_unemployed_searching", "couple_present_partner_proxy",
        "person_composition_observed",
    ]
    for col in num_cols:
        out[col] = np.nan
    return out

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


def derive_age(tr: pd.DataFrame, year: int) -> pd.Series:
    age = tr["age_current"].combine_first(tr["age_income_ref"])

    if "birth_year" in tr.columns:
        age_from_birth_year = year - tr["birth_year"]
        age = age.combine_first(age_from_birth_year)

    return pd.to_numeric(age, errors="raise")


def recode_sex(code: pd.Series) -> pd.Series:
    x = to_num(code)
    out = np.select([x.eq(1), x.eq(2)], ["male", "female"], default=pd.NA)
    return pd.Series(out, index=code.index, dtype="string")


def recode_activity_status(tp: pd.DataFrame) -> pd.Series:
    if "labour_status_detail" not in tp.columns:
        return empty_str(tp.index)

    x = to_num(tp["labour_status_detail"])

    out = np.select(
        [
            x.eq(1), x.eq(2), x.eq(3), x.eq(4), x.eq(5),
            x.eq(6), x.eq(7), x.eq(8), x.eq(9), x.eq(10), x.eq(11),
        ],
        [
            "employee_full_time", "employee_part_time",
            "selfemployed_full_time", "selfemployed_part_time",
            "unemployed", "student", "retired", "permanently_disabled",
            "military_service", "home_care", "other_inactive",
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

def load_person_clean(tr_path: Path, tp_path: Path, year: int) -> pd.DataFrame | None:
    if not tr_path.exists():
        if STRICT_TR_REQUIRED:
            raise FileNotFoundError(f"Missing Tr file for {year}: {tr_path}")
        return None

    tr_raw = read_dta(tr_path)

    tr = extract_section(
        tr_raw,
        section="tr",
        schema=SCHEMA,
        source_path=tr_path,
    )

    if "household_id" in tr.columns and tr["household_id"].notna().any():
        tr["household_id_source"] = "direct_from_tr"
    else:
        tr["household_id"] = derive_household_id_from_person_id(tr["person_id"])
        tr["household_id_source"] = "derived_from_person_id"

    tr["age"] = derive_age(tr, year)

    tr["sex"] = recode_sex(tr["sex"])

    tr["has_partner_id"] = (
        tr["partner_id"].notna() & ~tr["partner_id"].isin(["0", ""])
    ).astype(float)

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
        tp_raw = read_dta(tp_path)

        tp = extract_section(
            tp_raw,
            section="tp",
            schema=SCHEMA,
            source_path=tp_path,
        )

        tp["person_weight_preferred"] = tp["weight_selected_resp"].combine_first(
            tp["weight_p"]
        )

        # This should use the cleaned column "labour_status_detail",
        # not raw PL031 / PL032 columns.
        tp["activity_status_detail"] = recode_activity_status(tp)
        tp["activity_group"] = recode_activity_group(tp["activity_status_detail"])

        tp["active_job_search"] = recode_yes_no(tp["active_job_search"])
        tp["currently_in_education"] = recode_yes_no(tp["currently_in_education"])
        tp["foreign_nationality"] = recode_nationality_foreign(tp["nationality"])

        tp["any_social_assistance_income"] = np.where(
            tp["social_assistance_income_annual"].gt(0),
            1.0,
            np.where(tp["social_assistance_income_annual"].notna(), 0.0, np.nan),
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

        tp = TpSchema.validate(tp, lazy=True)

        person = safe_left_merge(
            person,
            tp,
            on="person_id",
            validate="1:1",
            left_name="tr",
            right_name="tp",
        )

        person["labour_file_available"] = 1.0

    else:
        person = add_missing_tp_columns(person)
        person["labour_file_available"] = 0.0

    person["working_age_18_64"] = person["age"].between(18, 64, inclusive="both")

    person["activity_group_working_age"] = (
        person["activity_group"]
        .where(person["working_age_18_64"], pd.NA)
        .astype("string")
    )

    person["person_file_available"] = 1.0
    person["year"] = year

    return PersonSchema.validate(person, lazy=True)


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


def household_partner_proxy(person: pd.DataFrame) -> pd.DataFrame:
    pairs = person.loc[
        person["partner_id"].notna() & ~person["partner_id"].isin(["0", ""]),
        ["household_id", "person_id", "partner_id"],
    ].copy()

    if pairs.empty:
        return pd.DataFrame(columns=["household_id", "couple_present_partner_proxy"])

    pairs["household_id"] = pairs["household_id"].astype("string")
    pairs["person_id"] = pairs["person_id"].astype("string")
    pairs["partner_id"] = pairs["partner_id"].astype("string")

    pairs = pairs.loc[
        pairs["person_id"].notna()
        & pairs["partner_id"].notna()
        & ~pairs["partner_id"].isin(["0", ""])
        & pairs["person_id"].ne(pairs["partner_id"])
    ].drop_duplicates()

    reverse = pairs.rename(
        columns={
            "person_id": "partner_id",
            "partner_id": "person_id",
        }
    )

    reciprocal = pairs.merge(
        reverse,
        on=["household_id", "person_id", "partner_id"],
        how="inner",
    )

    out = (
        reciprocal.groupby("household_id")
        .size()
        .rename("n_reciprocal_partner_links")
        .reset_index()
    )

    out["couple_present_partner_proxy"] = (
        out["n_reciprocal_partner_links"].ge(2).astype(float)
    )

    return out[["household_id", "couple_present_partner_proxy"]]


def add_person_flags(person: pd.DataFrame) -> pd.DataFrame:
    p = person.copy()

    age = pd.to_numeric(p["age"], errors="coerce")
    agw = p["activity_group_working_age"].astype("string")
    status = p["activity_status_detail"].astype("string")

    active_search = pd.to_numeric(p["active_job_search"], errors="coerce")
    labour_income = pd.to_numeric(p["labour_income_person_annual"], errors="coerce")
    social_assist = pd.to_numeric(p["social_assistance_income_annual"], errors="coerce")
    foreign_nat = pd.to_numeric(p["foreign_nationality"], errors="coerce")

    p["age_missing"] = age.isna()
    p["is_adult"] = age.ge(18)
    p["is_child"] = age.lt(18)
    p["is_adult_23plus"] = age.ge(23)
    p["is_adult_25plus"] = age.ge(25)
    p["is_working_age"] = age.between(18, 64, inclusive="both")

    p["is_working_18_64"] = agw.eq("working")
    p["is_unemployed_18_64"] = agw.eq("unemployed")
    p["is_inactive_18_64"] = agw.eq("inactive")
    p["activity_missing_18_64"] = agw.isna()

    p["is_student_18_64"] = status.eq("student") & p["is_working_age"]
    p["is_retired_18_64"] = status.eq("retired") & p["is_working_age"]
    p["is_disabled_18_64"] = status.eq("permanently_disabled") & p["is_working_age"]

    p["active_job_search_known"] = active_search.notna().fillna(False)
    p["active_job_search_1"] = active_search.eq(1).fillna(False)

    p["unemployed_search_failure"] = (
    p["is_unemployed_18_64"].fillna(False) &
    p["active_job_search_known"] &
    ~p["active_job_search_1"].fillna(False)
    )


    p["labour_income_person_annual_num"] = labour_income
    p["labour_income_known"] = labour_income.notna()
    p["positive_labour_income"] = labour_income.gt(0)

    p["social_assistance_income_annual_num"] = social_assist
    p["social_assistance_known"] = social_assist.notna()
    p["positive_social_assistance_income"] = social_assist.gt(0)

    p["foreign_nationality_known"] = foreign_nat.notna()
    p["foreign_nationality_1"] = foreign_nat.eq(1)

    p["labour_file_available_1"] = pd.to_numeric(
        p["labour_file_available"], errors="coerce"
    ).eq(1)

    return p


def build_household_composition(
    person: pd.DataFrame | None,
    hh_ids: pd.Series,
) -> pd.DataFrame:
    if person is None or person.empty:
        return empty_household_composition(hh_ids)

    p = add_person_flags(person)
    g = p.groupby("household_id", dropna=False)

    out = g.agg(
        n_persons=("person_id", "size"),
        n_age_missing=("age_missing", "sum"),
        n_adults=("is_adult", "sum"),
        n_children=("is_child", "sum"),
        n_adults_23plus=("is_adult_23plus", "sum"),
        n_adults_25plus=("is_adult_25plus", "sum"),
        n_working_18_64=("is_working_18_64", "sum"),
        n_unemployed_18_64=("is_unemployed_18_64", "sum"),
        n_inactive_18_64=("is_inactive_18_64", "sum"),
        n_missing_18_64=("activity_missing_18_64", "sum"),
        n_working_age=("is_working_age", "sum"),
        n_students_18_64=("is_student_18_64", "sum"),
        n_retired_18_64=("is_retired_18_64", "sum"),
        n_disabled_18_64=("is_disabled_18_64", "sum"),
        labour_observed=("labour_file_available_1", "min"),
        any_active_job_search=("active_job_search_1", "max"),
        active_job_search_known=("active_job_search_known", "max"),
        unemployed_search_failure=("unemployed_search_failure", "max"),
        any_positive_labour_income=("positive_labour_income", "max"),
        labour_income_known=("labour_income_known", "max"),
        any_social_assistance_income_hh=("positive_social_assistance_income", "max"),
        social_assistance_known=("social_assistance_known", "max"),
        any_foreign_nationality_hh=("foreign_nationality_1", "max"),
        foreign_nationality_known=("foreign_nationality_known", "max"),
    )

    # NaN-aware sums. This preserves the old behavior: if all values are missing, household value should be NaN, not 0.
    out["labour_income_hh_annual"] = g["labour_income_person_annual_num"].sum(
        min_count=1
    )
    out["hh_social_assistance_income_annual"] = g[
        "social_assistance_income_annual_num"
    ].sum(min_count=1)

    out = out.reset_index()

    out["age_composition_complete"] = out["n_age_missing"].eq(0).astype(float)
    out["n_adults_18plus"] = out["n_adults"]

    age_complete = out["age_composition_complete"].eq(1)

    out["single_adult"] = np.where(
        age_complete,
        out["n_adults"].eq(1) & out["n_children"].eq(0),
        np.nan,
    ).astype(float)

    out["single_parent"] = np.where(
        age_complete,
        out["n_adults"].eq(1) & out["n_children"].gt(0),
        np.nan,
    ).astype(float)

    out["two_adults"] = np.where(
        age_complete,
        out["n_adults"].eq(2),
        np.nan,
    ).astype(float)

    out["threeplus_adults"] = np.where(
        age_complete,
        out["n_adults"].ge(3),
        np.nan,
    ).astype(float)

    out["children_present"] = np.where(
        age_complete,
        out["n_children"].gt(0),
        np.nan,
    ).astype(float)

    out["any_working_18_64"] = out["n_working_18_64"].gt(0).astype(float)
    out["any_unemployed_18_64"] = out["n_unemployed_18_64"].gt(0).astype(float)

    out["all_working_age_nonworking"] = (
        out["n_working_age"].gt(0) & out["n_working_18_64"].eq(0)
    ).astype(float)

    out["labour_income_hh_monthly"] = out["labour_income_hh_annual"] / 12

    unemployed_search_failure = (
        out["unemployed_search_failure"].fillna(False).astype(bool)
    )

    out["all_unemployed_searching"] = np.where(
        out["n_unemployed_18_64"].gt(0),
        ~unemployed_search_failure,
        np.nan,
    ).astype(float)

    # Restore old NaN behavior for "any" variables.
    out["any_active_job_search"] = np.where(
        out["active_job_search_known"],
        out["any_active_job_search"],
        np.nan,
    ).astype(float)

    out["any_positive_labour_income"] = np.where(
        out["labour_income_known"],
        out["any_positive_labour_income"],
        np.nan,
    ).astype(float)

    out["any_social_assistance_income_hh"] = np.where(
        out["social_assistance_known"],
        out["any_social_assistance_income_hh"],
        np.nan,
    ).astype(float)

    out["any_foreign_nationality_hh"] = np.where(
        out["foreign_nationality_known"],
        out["any_foreign_nationality_hh"],
        np.nan,
    ).astype(float)

    out["person_composition_observed"] = 1.0
    out["labour_observed"] = out["labour_observed"].astype(float)

    # Keep this until partner proxy is vectorized separately.
    partner_proxy = household_partner_proxy(p)

    out = out.merge(
        partner_proxy,
        on="household_id",
        how="left",
        validate="1:1",
    )

    out["couple_present_partner_proxy"] = (
        out["couple_present_partner_proxy"].fillna(0.0).astype(float)
    )

    # Drop internal helper columns.
    out = out.drop(
        columns=[
            "n_working_age",
            "active_job_search_known",
            "unemployed_search_failure",
            "labour_income_known",
            "social_assistance_known",
            "foreign_nationality_known",
        ]
    )

    return HouseholdCompositionSchema.validate(out, lazy=True)


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

    return HouseholdFinalSchema.validate(out)


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

    if person is None:
        raise Exception(f"No person correctly loaded from year {year}")

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
    hh_context = hh[PERSON_HOUSEHOLD_COLUMNS].copy()

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
