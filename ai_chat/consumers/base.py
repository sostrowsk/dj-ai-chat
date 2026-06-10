import json
import logging
from typing import Any

from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


class BaseConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._handling_error: bool = False
        self._connected: bool = False

    async def connect(self) -> None:
        try:
            await super().connect()
            self._connected = True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            await self.close(code=4002)

    async def disconnect(self, close_code: int) -> None:
        try:
            await super().disconnect(close_code)
        except Exception as e:
            logger.error(f"Disconnect failed: {str(e)}")
        finally:
            self._connected = False
            self._handling_error = False

    async def send_json(self, content: Any) -> None:
        if not self._connected:
            return
        try:
            message_content = self._prepare_message_content(content)
            await self.send(text_data=json.dumps(message_content))
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            if not self._handling_error:
                await self._handle_error("Failed to send message", close_connection=True)

    def _prepare_message_content(self, content: Any) -> dict:
        if isinstance(content, dict) and "type" not in content and "message" in content:
            return {"type": "message", "message": content["message"]}
        return content

    async def _handle_error(
        self,
        message: str,
        close_connection: bool = False,
        error_code: int = 4000,
    ) -> None:
        if self._handling_error:
            await self.close(code=1011)
            return
        logger.error(message)
        try:
            self._handling_error = True
            if self._connected:
                error_payload = {"type": "error", "message": message}
                await self.send(text_data=json.dumps(error_payload))
            if close_connection:
                await self.close(code=error_code)
        finally:
            self._handling_error = False
