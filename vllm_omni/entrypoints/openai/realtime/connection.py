import asyncio
import json
from typing import Any, Dict, Optional
from fastapi import WebSocket
from vllm.logger import init_logger

logger = init_logger(__name__)

class RealtimeConnection:
    """Minimal base class for OpenAI Realtime WebSocket connections."""
    
    def __init__(self, websocket: WebSocket, serving: Any):
        self.websocket = websocket
        self.serving = serving
        self.is_connected = True

    async def handle_connection(self):
        """Main loop for handling WebSocket messages."""
        await self.websocket.accept()
        try:
            while self.is_connected:
                message = await self.websocket.receive_text()
                event = json.loads(message)
                await self.handle_event(event)
        except Exception as e:
            logger.error(f"Error in realtime connection: {e}")
        finally:
            self.is_connected = False

    async def handle_event(self, event: Dict[str, Any]):
        """Default event handler (to be overridden)."""
        logger.info(f"Received event: {event.get('type')}")

    async def send(self, event: Any):
        """Send an event to the client."""
        if isinstance(event, dict):
            await self.websocket.send_json(event)
        else:
            # Assume it's a pydantic model from protocol.py
            await self.websocket.send_text(event.model_dump_json())
