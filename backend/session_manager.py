# backend/session_manager.py
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from backend.graph.state import KinnectState
from backend.graph.state_utils import create_initial_state

class SessionManager:
    """
    Manages active patient call sessions in memory.
    Thread/coroutine safe using asyncio.Lock.
    """
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, user_id: str) -> Dict[str, Any]:
        """Create a new session state for a patient."""
        async with self._lock:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            
            # Generate the default state
            state = create_initial_state(user_id=user_id)
            state["session_id"] = session_id
            state["timestamp"] = datetime.now()
            
            self._sessions[session_id] = state
            return state

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the state of an active session."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_session(self, session_id: str, partial_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update part of the session state."""
        async with self._lock:
            if session_id not in self._sessions:
                return None
            
            # Merge dictionary elements carefully
            current = self._sessions[session_id]
            for k, v in partial_state.items():
                if k == "messages" and isinstance(v, list):
                    # Append messages if they exist
                    current["messages"] = current.get("messages", []) + v
                elif k == "errors" and isinstance(v, list):
                    current["errors"] = current.get("errors", []) + v
                elif k == "agent_outputs" and isinstance(v, dict):
                    current["agent_outputs"] = {**current.get("agent_outputs", {}), **v}
                else:
                    current[k] = v
                    
            self._sessions[session_id] = current
            return current

    async def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End and remove a session from memory, returning the final state."""
        async with self._lock:
            return self._sessions.pop(session_id, None)

# Global session manager instance
session_manager = SessionManager()
