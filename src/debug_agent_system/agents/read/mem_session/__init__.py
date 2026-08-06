from __future__ import annotations

import json
import uuid
from pathlib import Path

from debug_agent_system.core.contracts import SessionState, to_jsonable


class DiagnosticSessionStore:
    """MEM: sole owner of mutable diagnostic session state."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._memory: dict[str, SessionState] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def create(self, query: str, session_id: str | None = None) -> SessionState:
        sid = session_id or f"diag-{uuid.uuid4().hex[:12]}"
        state = SessionState(session_id=sid, query=query)
        self.save(state)
        return state

    def get(self, session_id: str) -> SessionState | None:
        if session_id in self._memory:
            return self._memory[session_id]
        path = self._path(session_id)
        if not path or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = SessionState(**data)
        except Exception:
            return None
        self._memory[session_id] = state
        return state

    def save(self, state: SessionState) -> None:
        self._memory[state.session_id] = state
        path = self._path(state.session_id)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(to_jsonable(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _path(self, session_id: str) -> Path | None:
        if not self.root:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        return self.root / f"{safe}.json"
