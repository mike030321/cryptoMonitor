from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Checkpointer:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
