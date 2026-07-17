from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path


def data_file(path: Path) -> Path:
    """Return a local data file, downloading it from LEARNGEETA_DATA_BASE_URL if needed.

    Example:
        LEARNGEETA_DATA_BASE_URL=https://huggingface.co/datasets/user/dataset/resolve/main/data
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    base_url = os.environ.get("LEARNGEETA_DATA_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/{urllib.parse.quote(path.name)}"
    partial = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "LearnGeeta/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(path)
    return path
