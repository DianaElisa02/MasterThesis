from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import REGION_NAME_MAP
from src.schema_loader import ColumnSpec, load_ecv_schema
from src.schemas import TdSchema, ThSchema, TpSchema

SCHEMA = load_ecv_schema("ecv_schema.yml")


def read_dta(path: Path) -> pd.DataFrame:
    return pd.read_stata(path, convert_categoricals=False)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def empty_for_type(index: pd.Index, kind: str) -> pd.Series:
    if kind == "numeric":
        return pd.Series(np.nan, index=index, dtype="float64")

    return pd.Series(pd.NA, index=index, dtype="string")


def to_id(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return x


def convert_series(s: pd.Series, spec: ColumnSpec) -> pd.Series:
    if spec.kind == "id":
        return to_id(s)

    if spec.kind == "numeric":
        out = pd.to_numeric(s, errors="coerce")

        if spec.nonneg:
            out = out.mask(out < 0, np.nan)

        return out

    if spec.kind == "string":
        return s.astype("string")

    raise ValueError(f"Unknown column type: {spec.kind}")


def extract_section(
    df: pd.DataFrame,
    section: str,
    schema: dict[str, dict[str, ColumnSpec]],
    source_path: Path,
) -> pd.DataFrame:
    section_schema = schema[section]
    data = {}

    for clean_name, spec in section_schema.items():
        raw_col = first_existing(df, spec.columns)
        output_name = spec.rename or clean_name

        if raw_col is None:
            if spec.required:
                raise KeyError(
                    f"Missing required column for {section}.{clean_name} in {source_path}. "
                    f"Tried columns: {spec.columns}"
                )

            data[output_name] = empty_for_type(df.index, spec.kind)
            continue

        data[output_name] = convert_series(df[raw_col], spec)

    return pd.DataFrame(data)


def load_tp_clean(path: Path) -> pd.DataFrame:
    tp = read_dta(path)

    out = extract_section(
        tp,
        section="tp",
        schema=SCHEMA,
        source_path=path,
    )

    return TpSchema.validate(out, lazy=True)


def load_td_clean(path: Path) -> pd.DataFrame:
    td = read_dta(path)

    out = extract_section(
        td,
        section="td",
        schema=SCHEMA,
        source_path=path,
    )

    out["region_name"] = out["region_code"].map(REGION_NAME_MAP)
    return TdSchema.validate(out, lazy=True)


def load_th_clean(path: Path) -> pd.DataFrame:
    th = read_dta(path)

    out = extract_section(
        th,
        section="th",
        schema=SCHEMA,
        source_path=path,
    )

    return ThSchema.validate(out, lazy=True)
