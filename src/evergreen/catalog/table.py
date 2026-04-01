from abc import ABC, abstractmethod

from evergreen.catalog.schema import Schema
from evergreen.data_source import InMemoryDataSource, JsonlDataSource
from evergreen.planner.physical.plan import PhysicalPlan, TableScan
from evergreen.storage.row import Row


class TableProvider(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def scan(self) -> PhysicalPlan:
        pass


class InMemoryTable(TableProvider):
    def __init__(self, rows: list[Row], schema: Schema) -> None:
        self._rows = rows
        self._schema = schema

    def schema(self) -> Schema:
        return self._schema

    def scan(self) -> PhysicalPlan:
        return TableScan(InMemoryDataSource(self._rows), self._schema)


class JsonlTable(TableProvider):
    def __init__(self, path: str, schema: Schema) -> None:
        self._path = path
        self._schema = schema

    def schema(self) -> Schema:
        return self._schema

    def scan(self) -> PhysicalPlan:
        return TableScan(
            JsonlDataSource(self._path, self._schema.field_names()), self._schema
        )
