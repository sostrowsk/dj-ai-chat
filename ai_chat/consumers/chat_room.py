# ai_chat/consumers/chat_room.py
import logging
from dataclasses import dataclass
from typing import Optional

from channels.db import database_sync_to_async
from channels.layers import BaseChannelLayer

from ai_chat import conf

Project = conf.get_project_model()
Document = conf.get_document_model()

logger = logging.getLogger(__name__)


@dataclass
class ChatRoom:
    group_name: str
    project_id: Optional[int] = None
    document_id: Optional[int] = None


class ChatRoomManager:

    def __init__(self, channel_layer: BaseChannelLayer, channel_name: str) -> None:
        self.channel_layer = channel_layer
        self.channel_name = channel_name
        self.chat_room: Optional[ChatRoom] = None

    async def setup_chat_room(self, kwargs: dict) -> None:
        try:
            if project_id := kwargs.get("project_id"):
                await self._setup_project_chat(int(project_id))
            elif document_id := kwargs.get("document_id"):
                await self._setup_document_chat(int(document_id))
            else:
                await self._setup_general_chat()

            await self._join_chat_group()

        except ValueError as e:
            raise ValueError(f"Invalid ID format: {str(e)}")

    @database_sync_to_async
    def _get_project(self, project_id: int) -> Project:
        try:
            return Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            raise ValueError(f"Project with id {project_id} does not exist")

    @database_sync_to_async
    def _get_document(self, document_id: int) -> Document:
        try:
            return Document.objects.select_related("project").get(id=document_id)
        except Document.DoesNotExist:
            raise ValueError(f"Document with id {document_id} does not exist")

    async def _setup_project_chat(self, project_id: int) -> None:
        project = await self._get_project(project_id)
        self.chat_room = ChatRoom(group_name=f"chat_project_{project_id}", project_id=project.id)
        logger.info(f"Project chat initialized: {project_id}")

    async def _setup_document_chat(self, document_id: int) -> None:
        document = await self._get_document(document_id)
        self.chat_room = ChatRoom(
            group_name=f"chat_document_{document_id}",
            project_id=document.project.id,
            document_id=document.id,
        )
        logger.info(f"Document chat initialized: {document_id}")

    async def _setup_general_chat(self) -> None:
        self.chat_room = ChatRoom(group_name="chat_general")
        logger.info("General chat initialized")

    async def _join_chat_group(self) -> None:
        if self.chat_room:
            await self.channel_layer.group_add(self.chat_room.group_name, self.channel_name)
            logger.info(f"Added to channel group: {self.chat_room.group_name}")

    async def leave_chat_group(self) -> None:
        if self.chat_room:
            await self.channel_layer.group_discard(self.chat_room.group_name, self.channel_name)
            logger.info(f"Removed from channel group: {self.chat_room.group_name}")
