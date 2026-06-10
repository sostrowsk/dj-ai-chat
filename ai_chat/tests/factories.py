import factory
from factory.django import DjangoModelFactory

from ai_chat.models import ChatMessage, ChatSession, RetrievalLog
from users.tests.factories import ClientFactory


class ChatSessionFactory(DjangoModelFactory):
    class Meta:
        model = ChatSession

    user = factory.SubFactory(ClientFactory)
    scope = ChatSession.Scope.GENERAL
    project = None
    document = None
    title = factory.Sequence(lambda n: f"Chat-Session {n}")
    system_prompt = ""
    is_active = True


class ChatMessageFactory(DjangoModelFactory):
    class Meta:
        model = ChatMessage

    session = factory.SubFactory(ChatSessionFactory)
    role = ChatMessage.Role.USER
    content = factory.Sequence(lambda n: f"Nachricht {n}")


class RetrievalLogFactory(DjangoModelFactory):
    class Meta:
        model = RetrievalLog

    session = factory.SubFactory(ChatSessionFactory)
    user = factory.LazyAttribute(lambda obj: obj.session.user if obj.session else None)
    query = factory.Sequence(lambda n: f"Suchanfrage {n}")
    scope = ChatSession.Scope.GENERAL
    collection = "general_chat"
    candidate_scores = factory.LazyFunction(list)
    final_k = 0
    cutoff_config = factory.LazyFunction(dict)
