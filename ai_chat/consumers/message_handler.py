# ai_chat/consumers/message_handler.py
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class _Message:
    role: str
    content: str


class ChatMessageHistory:
    """Simple list-based chat history — replaces langchain ChatMessageHistory."""

    def __init__(self):
        self.messages: List[_Message] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append(_Message(role="user", content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(_Message(role="assistant", content=content))


class MessageHandler:

    def __init__(self, system_prompt: str):
        self.chat_history = ChatMessageHistory()
        self.system_prompt = system_prompt

    def update_history(self, user_message: str, ai_response: str) -> None:
        self.chat_history.add_user_message(user_message)
        self.chat_history.add_ai_message(ai_response)

    def clear_history(self) -> None:
        self.chat_history = ChatMessageHistory()

    def _prepare_messages(self, query: str, results: List[Tuple[Any, float]]) -> List[Dict[str, str]]:
        results_text = "\n".join(f"<chunk_{doc.id}>{doc.page_content}</chunk_{doc.id}>" for doc, _ in results)

        messages = [{"role": "system", "content": self.system_prompt}]

        messages.extend(self._convert_history_to_messages())

        messages.append(
            {
                "role": "user",
                "content": f"{query}\n<context>\n{results_text}\n</context>",
            }
        )

        return messages

    def _convert_history_to_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.chat_history.messages]
