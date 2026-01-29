"""WebSocket service for real-time dashboard updates.

Provides:
- Connection management for multiple clients
- Event broadcasting to all connected clients
- Filtered subscriptions by event type and project
- Heartbeat for connection health
"""
import asyncio
import json
import time
from typing import Dict, Any, Set, Optional, List
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""
    websocket: WebSocket
    connected_at: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    project_filter: Optional[str] = None
    last_ping: float = field(default_factory=time.time)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts.

    Features:
    - Multiple client connections
    - Event type filtering
    - Project-based filtering
    - Automatic reconnection handling
    - Heartbeat monitoring
    """

    def __init__(self):
        self.clients: Dict[str, WebSocketClient] = {}
        self._client_counter = 0
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection.

        Returns:
            Client ID for this connection
        """
        await websocket.accept()
        self._client_counter += 1
        client_id = f"client_{self._client_counter}"

        self.clients[client_id] = WebSocketClient(
            websocket=websocket,
            subscriptions={"*"}  # Subscribe to all events by default
        )

        # Send welcome message
        await self._send_to_client(client_id, {
            "type": "connected",
            "client_id": client_id,
            "message": "Connected to Claude Memory real-time feed",
            "timestamp": time.time()
        })

        return client_id

    async def disconnect(self, client_id: str):
        """Handle client disconnection."""
        if client_id in self.clients:
            del self.clients[client_id]

    async def subscribe(self, client_id: str, event_types: List[str], project_path: Optional[str] = None):
        """Update client subscriptions.

        Args:
            client_id: Client to update
            event_types: List of event types to subscribe to (or ["*"] for all)
            project_path: Optional project filter
        """
        if client_id not in self.clients:
            return

        client = self.clients[client_id]
        client.subscriptions = set(event_types)
        client.project_filter = project_path

        await self._send_to_client(client_id, {
            "type": "subscribed",
            "event_types": event_types,
            "project_filter": project_path,
            "timestamp": time.time()
        })

    async def broadcast(self, event_type: str, data: Dict[str, Any], project_path: Optional[str] = None):
        """Broadcast an event to all subscribed clients.

        Args:
            event_type: Type of event (memory_stored, memory_searched, timeline_logged, etc.)
            data: Event data payload
            project_path: Project this event relates to (for filtering)
        """
        message = {
            "type": event_type,
            "data": data,
            "project_path": project_path,
            "timestamp": time.time()
        }

        # Debug logging
        print(f"[WS] Broadcasting {event_type} to {len(self.clients)} clients, project={project_path}")

        # Send to all matching clients
        disconnected = []
        sent_count = 0
        for client_id, client in self.clients.items():
            # Check if client is subscribed to this event type
            if "*" not in client.subscriptions and event_type not in client.subscriptions:
                print(f"[WS] Skipping {client_id}: not subscribed to {event_type}")
                continue

            # Check project filter
            # If project_path is None, send to all clients (global event)
            # If project_path is set, only send to matching clients
            if project_path and client.project_filter and client.project_filter != project_path:
                print(f"[WS] Skipping {client_id}: project mismatch ({client.project_filter} != {project_path})")
                continue

            try:
                await client.websocket.send_json(message)
                sent_count += 1
                print(f"[WS] Sent {event_type} to {client_id}")
            except Exception as e:
                print(f"[WS] Error sending to {client_id}: {e}")
                disconnected.append(client_id)

        print(f"[WS] Broadcast complete: sent to {sent_count}/{len(self.clients)} clients")

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to a specific client."""
        if client_id not in self.clients:
            return

        try:
            await self.clients[client_id].websocket.send_json(message)
        except Exception:
            await self.disconnect(client_id)

    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming message from client.

        Supported message types:
        - subscribe: Update subscriptions
        - ping: Heartbeat
        """
        msg_type = message.get("type")

        if msg_type == "subscribe":
            await self.subscribe(
                client_id,
                message.get("event_types", ["*"]),
                message.get("project_path")
            )

        elif msg_type == "ping":
            if client_id in self.clients:
                self.clients[client_id].last_ping = time.time()
                await self._send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": time.time()
                })

        elif msg_type == "get_stats":
            # Send current stats
            await self._send_to_client(client_id, {
                "type": "stats",
                "connected_clients": len(self.clients),
                "timestamp": time.time()
            })

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket service statistics."""
        return {
            "connected_clients": len(self.clients),
            "clients": [
                {
                    "id": cid,
                    "connected_at": c.connected_at,
                    "subscriptions": list(c.subscriptions),
                    "project_filter": c.project_filter,
                    "last_ping": c.last_ping
                }
                for cid, c in self.clients.items()
            ]
        }


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


# Event types for memory system
class EventTypes:
    """Standard event types for broadcasting."""
    # Memory events
    MEMORY_STORED = "memory_stored"
    MEMORY_SEARCHED = "memory_searched"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ARCHIVED = "memory_archived"
    MEMORY_RESTORED = "memory_restored"

    # Timeline events
    TIMELINE_LOGGED = "timeline_logged"
    CHECKPOINT_CREATED = "checkpoint_created"

    # Session events
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_SUMMARIZED = "session_summarized"

    # Anchor events
    ANCHOR_MARKED = "anchor_marked"
    ANCHOR_CONFLICT = "anchor_conflict"
    CONFLICT_RESOLVED = "conflict_resolved"

    # Admin events
    CLEANUP_STARTED = "cleanup_started"
    CLEANUP_COMPLETED = "cleanup_completed"
    REINDEX_PROGRESS = "reindex_progress"
    REINDEX_COMPLETED = "reindex_completed"

    # System events
    HEALTH_CHECK = "health_check"
    ERROR = "error"


async def broadcast_event(
    event_type: str,
    data: Dict[str, Any],
    project_path: Optional[str] = None
):
    """Helper function to broadcast an event."""
    manager = get_websocket_manager()
    await manager.broadcast(event_type, data, project_path)
