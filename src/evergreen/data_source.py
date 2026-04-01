import json
from abc import ABC, abstractmethod
from typing import IO

from evergreen.storage.row import Row


class DataSource(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def next(self) -> Row | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class InMemoryDataSource(DataSource):
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows
        self._index: int = 0

    def open(self) -> None:
        pass

    def next(self) -> Row | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def close(self) -> None:
        pass


class JsonlDataSource(DataSource):
    def __init__(self, path: str, field_names: tuple[str, ...]) -> None:
        self._path = path
        self._field_names = field_names
        self._file: IO[str] | None = None

    def open(self) -> None:
        self._file = open(self._path, encoding="utf-8")  # noqa: SIM115

    def next(self) -> Row | None:
        assert self._file is not None
        line = self._file.readline()
        if not line:
            return None
        obj = json.loads(line)
        return Row(tuple(obj[name] for name in self._field_names))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
