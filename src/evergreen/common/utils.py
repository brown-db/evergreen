import shutil

from evergreen.common.constants import CACHE_DIR_ROOT


def clear_all_caches() -> None:
    if CACHE_DIR_ROOT.exists():
        shutil.rmtree(CACHE_DIR_ROOT)
