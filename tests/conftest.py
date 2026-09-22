from __future__ import annotations

import os
import sys
from pathlib import Path


def extension_search_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    configured = os.environ.get("KHOJ_BUILD_DIR")
    candidates = [root / configured] if configured else [root / "build"]
    return [path for path in candidates if path.is_dir()]


for directory in extension_search_paths():
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
