import contextlib
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class Cache(ABC):
    @abstractmethod
    def __contains__(self, key: str) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> object:
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: object) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class FileCache(Cache):
    """File-based key-value cache that works reliably on NFS.

    Each entry is stored as a separate pickle file named by its key.
    Writes are atomic via temp-file-then-rename.
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir

    def __contains__(self, key: str) -> bool:
        return (self._cache_dir / key).exists()

    def __getitem__(self, key: str) -> object:
        try:
            return pickle.loads((self._cache_dir / key).read_bytes())
        except FileNotFoundError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: object) -> None:
        path = self._cache_dir / key
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self._cache_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f)
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def __len__(self) -> int:
        if not self._cache_dir.exists():
            return 0
        return sum(1 for _ in self._cache_dir.iterdir())
