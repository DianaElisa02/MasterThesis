from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="raise")
    w = pd.to_numeric(w, errors="raise")
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def weighted_share(x: pd.Series, w: pd.Series, value=1.0) -> float:
    x = pd.to_numeric(x, errors="raise")
    w = pd.to_numeric(w, errors="raise")
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    return float(np.average((x[m] == value).astype(float), weights=w[m]))


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="raise")
    weights = pd.to_numeric(weights, errors="raise")

    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan

    return float(
        np.quantile(
            values[mask].to_numpy(),
            q=quantile,
            weights=weights[mask].to_numpy(),
            method="inverted_cdf",
        )
    )


def safe_pct_gap(simulated: float, observed: float) -> float:
    if pd.isna(simulated) or pd.isna(observed) or observed == 0:
        return np.nan
    return float(100 * (simulated - observed) / observed)


def compact_round(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64", "float32"]).columns
    out[float_cols] = out[float_cols].round(digits)
    return out


def print_compact_table(
    df: pd.DataFrame,
    title: str,
    columns: list[str] | None = None,
    sort_by: list[str] | None = None,
    ascending=True,
    digits: int = 3,
    max_rows: int | None = None,
) -> None:
    out = df.copy()

    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise KeyError(f"Missing columns for compact print: {missing}")
        out = out[columns]

    if sort_by is not None:
        out = out.sort_values(sort_by, ascending=ascending)

    if max_rows is not None:
        out = out.head(max_rows)

    out = compact_round(out, digits=digits)

    print(f"\n{title}")
    print(out.to_string(index=False))
