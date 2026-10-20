from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Optional

from apps.insurance_claims.config import get_settings
from apps.insurance_claims.state import SessionState


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionState]:
        raise NotImplementedError

    @abstractmethod
    def save(self, state: SessionState) -> SessionState:
        raise NotImplementedError

    @abstractmethod
    def create(self) -> SessionState:
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
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
        state = SessionState()
        return self.save(state)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class JsonFileSessionStore(SessionStore):
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory or get_settings().session_store_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def get(self, session_id: str) -> Optional[SessionState]:
        path = self._path(session_id)
        if not path.exists():
            return None
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return SessionState.model_validate(data)

    def save(self, state: SessionState) -> SessionState:
        state.touch()
        path = self._path(state.session_id)
        with self._lock:
            path.write_text(
                state.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return state

    def create(self) -> SessionState:
        state = SessionState()
        return self.save(state)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        with self._lock:
            if path.exists():
                path.unlink()
