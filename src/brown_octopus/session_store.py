"""Replaceable storage for serializable capability-context state."""

from copy import deepcopy
from threading import RLock
from typing import Callable, TypeVar

from brown_octopus.contracts import SessionState


T = TypeVar("T")


class InMemorySessionStore:
    """Zero-configuration process-local session storage.

    Synchronization is private to the store. ``SessionState`` contains only
    serializable data and no runtime Brown Octopus objects.
    """

    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}
        self._locks: dict[str, RLock] = {}
        self._lock = RLock()

    def _get_or_create(self, session_id: str) -> tuple[SessionState, RLock]:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id)
                self._states[session_id] = state
                self._locks[session_id] = RLock()
            return state, self._locks[session_id]

    def get(self, session_id: str) -> SessionState:
        state, lock = self._get_or_create(session_id)
        with lock:
            return deepcopy(state)

    def mutate(self, session_id: str, operation: Callable[[SessionState], T]) -> T:
        """Apply one state operation while holding only that session's lock."""
        state, lock = self._get_or_create(session_id)
        with lock:
            return operation(state)

    def reset(self, session_id: str) -> SessionState:
        def clear(state: SessionState) -> SessionState:
            state.turn = 0
            state.active_state.clear()
            return deepcopy(state)

        return self.mutate(session_id, clear)

    def delete(self, session_id: str) -> None:
        with self._lock:
            state = self._states.get(session_id)
            lock = self._locks.get(session_id)
            if state is None or lock is None:
                return
            with lock:
                self._states.pop(session_id, None)
                self._locks.pop(session_id, None)

    def snapshot(self, session_id: str) -> dict | None:
        with self._lock:
            state = self._states.get(session_id)
            lock = self._locks.get(session_id)
            if state is None or lock is None:
                return None
            with lock:
                state = deepcopy(state)
        return {
            "session_id": state.session_id,
            "turn": state.turn,
            "active_state": dict(state.active_state),
            "active_tool_ids": list(state.active_state),
        }
