from __future__ import annotations

from threading import Lock
from typing import Optional

from apps.insurance_claims.state import SessionState


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            state = self._sessions.get(session_id)
            return state.model_copy(deep=True) if state else None

    def save(self, state: SessionState) -> SessionState:
        state.touch()
        with self._lock:
            self._sessions[state.session_id] = state.model_copy(deep=True)
        return state

    def create(self) -> SessionState:
        return self.save(SessionState())

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
