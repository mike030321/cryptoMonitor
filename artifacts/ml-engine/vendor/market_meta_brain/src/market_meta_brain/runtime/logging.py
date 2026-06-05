from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None

    def log(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
