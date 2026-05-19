from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        )
    )


def read_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())


def save_npy(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
