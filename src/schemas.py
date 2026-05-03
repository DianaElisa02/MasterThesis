from __future__ import annotations

import pandera.pandas as pa

from src.constants import MAX_REASONABLE_AGE

# ── Data-cleaning outputs ─────────────────────────────────────────────────────

TdSchema = pa.DataFrameSchema(
    {
        "household_id": pa.Column(nullable=False, unique=True),
        "weight_hh": pa.Column(nullable=True, checks=pa.Check.ge(0)),
    }
)

ThSchema = pa.DataFrameSchema(
    {
        "household_id": pa.Column(nullable=False, unique=True),
        "household_size_raw": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "mortgage_interest_paid_annual": pa.Column(
            nullable=True, checks=pa.Check.ge(0)
        ),
        "wealth_tax_paid_annual": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "consumption_units": pa.Column(nullable=True, checks=pa.Check.ge(0)),
    }
)

TpSchema = pa.DataFrameSchema(
    {
        "person_id": pa.Column(nullable=False, unique=True),
    }
)

PersonSchema = pa.DataFrameSchema(
    {
        "person_id": pa.Column(nullable=False, unique=True),
        "age": pa.Column(
            nullable=True, checks=[pa.Check.ge(0), pa.Check.le(MAX_REASONABLE_AGE)]
        ),
        "sex": pa.Column(nullable=True, checks=pa.Check.isin(["male", "female"])),
    }
)

HouseholdCompositionSchema = pa.DataFrameSchema(
    {
        "household_id": pa.Column(nullable=False, unique=True),
    }
)

HouseholdFinalSchema = pa.DataFrameSchema(
    {
        "household_id": pa.Column(nullable=False, unique=True),
    }
)

# ── Policy-database outputs ───────────────────────────────────────────────────

RegionLookupSchema = pa.DataFrameSchema(
    {
        "region_name_policy": pa.Column(nullable=False, unique=True),
        "nuts_code": pa.Column(nullable=False, unique=True),
    }
)

RmiAmountsSchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "region_name_policy": pa.Column(nullable=False),
        "program_name": pa.Column(nullable=False),
        "guaranteed_amount": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "max_amount": pa.Column(nullable=True, checks=pa.Check.ge(0)),
    },
    unique=["region_name_policy", "program_name", "hh_size"],
)

RmiEligibilitySchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "region_name_policy": pa.Column(nullable=False, unique=True),
    }
)

RmiCoverageSchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "region_name_policy": pa.Column(nullable=False),
        "year": pa.Column(nullable=False),
    },
    unique=["region_name_policy", "year"],
)

BaselineRules2017Schema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "region_name_policy": pa.Column(nullable=False, unique=True),
    }
)

BaselineScheduleSchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "year": pa.Column(nullable=False),
        "guaranteed_amount": pa.Column(nullable=True, checks=pa.Check.ge(0)),
    },
    unique=["nuts_code", "year", "hh_size"],
)

BaselineAmountSummarySchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "year": pa.Column(nullable=False),
    },
    unique=["nuts_code", "year"],
)

BaselineRulesSchema = pa.DataFrameSchema(
    {
        "nuts_code": pa.Column(nullable=False),
        "region_name_policy": pa.Column(nullable=False),
        "year": pa.Column(nullable=False),
    },
    unique=["nuts_code", "year"],
)
