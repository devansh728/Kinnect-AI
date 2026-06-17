# backend/websocket/connection_manager.py
from fastapi import WebSocket
from typing import Dict, List
import asyncio

class ConnectionManager:
    """
    Manages active WebSocket connections by user ID.
    Enables targeted messaging for specific patients (e.g. daily call push notifications).
    """
    def __init__(self):
        # Maps user_id to lists of active WebSockets (allows multiple devices if needed, though usually 1)
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept connection and register websocket under user_id."""
        await websocket.accept()
        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
            print(f"🔌 WebSocket connected for user: {user_id} (total active: {len(self.active_connections[user_id])})")

    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Clean up connection registration."""
        async with self._lock:
            if user_id in self.active_connections:
                if websocket in self.active_connections[user_id]:
                    self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            print(f"🔌 WebSocket disconnected for user: {user_id}")

    async def send_to_user(self, user_id: str, message: dict) -> bool:
        """Send a JSON payload to all WebSockets of a specific user."""
        async with self._lock:
            websockets = self.active_connections.get(user_id, [])
            if not websockets:
                return False
            
            # Send concurrently to all active tabs/sockets for this user
            sent = False
            for ws in list(websockets):
                try:
                    await ws.send_json(message)
                    sent = True
                except Exception as e:
                    print(f"⚠️ Failed to send message to user {user_id} on socket: {e}")
                    # Cleanup broken socket
                    try:
                        self.active_connections[user_id].remove(ws)
                    except ValueError:
                        pass
            return sent

    async def broadcast(self, message: dict):
        """Send a JSON payload to every active WebSocket connection."""
        async with self._lock:
            all_sockets = [ws for sockets in self.active_connections.values() for ws in sockets]
            for ws in all_sockets:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    print(f"⚠️ Broadcast fail on socket: {e}")

# Global connection manager instance
connection_manager = ConnectionManager()
