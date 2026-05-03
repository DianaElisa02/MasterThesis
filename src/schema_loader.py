from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

FileType = Literal["td", "th", "tr", "tp"]
ColumnType = Literal["id", "numeric", "string"]


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    columns: list[str]
    kind: ColumnType = Field(alias="type")
    nonneg: bool = False
    rename: str | None = None
    required: bool = False

    @field_validator("columns")
    @classmethod
    def columns_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("columns must contain at least one raw column name")

        if not all(isinstance(col, str) and col.strip() for col in value):
            raise ValueError("columns must contain non-empty strings")

        return value

    @field_validator("rename")
    @classmethod
    def rename_must_not_be_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("rename cannot be an empty string")
        return value

    @model_validator(mode="after")
    def validate_nonneg_type(self):
        if self.nonneg and self.kind != "numeric":
            raise ValueError("nonneg: true is only valid for type: numeric")
        return self

    @property
    def output_name(self) -> str | None:
        return self.rename


class ECVSchema(RootModel[dict[FileType, dict[str, ColumnSpec]]]):
    @field_validator("root")
    @classmethod
    def validate_sections(cls, value):
        required_sections = {"td", "th", "tr", "tp"}
        missing = required_sections - set(value)

        if missing:
            raise ValueError(f"Missing schema sections: {sorted(missing)}")

        return value


def load_ecv_schema(
    path: str | Path = "ecv_schema.yml",
) -> dict[str, dict[str, ColumnSpec]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    schema = ECVSchema.model_validate(raw)
    return schema.root
