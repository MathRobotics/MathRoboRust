from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_MATHROBO_ROOT = REPO_ROOT / "vendor" / "MathRobo"


@lru_cache(maxsize=1)
def import_mathrobo():
    try:
        import mathrobo as module
    except ModuleNotFoundError:
        sys.path.insert(0, str(VENDOR_MATHROBO_ROOT))
        import mathrobo as module
    return module
