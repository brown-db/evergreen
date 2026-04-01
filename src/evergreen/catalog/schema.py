from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    name: str
    dtype: type[object]
    description: str | None = None

    def __str__(self) -> str:
        if self.description:
            return f"{self.name}: {self.dtype.__name__} [{self.description}]"
        else:
            return f"{self.name}: {self.dtype.__name__}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype.__name__,
            "description": self.description,
        }


@dataclass(frozen=True)
class Schema:
    fields: tuple[Field, ...]
    key: tuple[str, ...]
    _field_name_to_index: dict[str, int] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        for i, f in enumerate(self.fields):
            self._field_name_to_index[f.name] = i

    def __str__(self) -> str:
        return f"Schema({', '.join(str(f) for f in self.fields)})"

    def to_dict(self) -> dict[str, object]:
        return {
            "fields": [field.to_dict() for field in self.fields],
            "key": list(self.key),
        }

    def index_of(self, field_name: str) -> int:
        return self._field_name_to_index[field_name]

    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def key_indices(self) -> tuple[int, ...]:
        return tuple(self.index_of(name) for name in self.key)

    def key_fields(self) -> tuple[Field, ...]:
        return tuple(self.fields[i] for i in self.key_indices())

    def with_fields(self, fields: tuple[Field, ...]) -> Schema:
        return Schema(self.fields + fields, self.key)
