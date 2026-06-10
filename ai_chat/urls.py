# ai_chat/urls.py
from django.urls import path

from .views import ChatView

app_name = "ai_chat"

urlpatterns = [
    path("", ChatView.as_view(), name="lobby"),
    path("project/<int:project_id>/", ChatView.as_view(), name="project_chat"),
    path("document/<int:document_id>/", ChatView.as_view(), name="document_chat"),
]
