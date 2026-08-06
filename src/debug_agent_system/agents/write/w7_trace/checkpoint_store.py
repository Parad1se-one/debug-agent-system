"""Content-addressed stage checkpoints for W7 shadow decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

from .contracts import canonical_hash


class CheckpointStore:
    schema_version = "w7.stage_checkpoint.v1"

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root is not None else None

    def key(self, *, stage: str, input_value: Any, version: str) -> str:
        return canonical_hash({
            "stage": stage,
            "version": version,
            "input": input_value,
        })

    def _content_path(self, *, stage: str, key: str) -> Path:
        assert self.root is not None
        return self.root / stage / f"{key}.json"

    def read(self, *, stage: str, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        # The old stage.json path is read-only compatibility.  New writes are
        # genuinely content-addressed so concurrent episode decisions cannot
        # overwrite one another.
        for path in (
            self._content_path(stage=stage, key=key),
            self.root / f"{stage}.json",
        ):
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(value, dict)
                and value.get("schema_version") == self.schema_version
                and value.get("key") == key
                and not value.get("issues")
            ):
                return value
        return None

    def write(
        self,
        *,
        stage: str,
        key: str,
        output: dict[str, Any],
        issues: list[str],
        call: dict[str, Any],
    ) -> None:
        if self.root is None or issues:
            return
        path = self._content_path(stage=stage, key=key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps({
                "schema_version": self.schema_version,
                "key": key,
                "output": output,
                "issues": issues,
                "call": call,
            }, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
