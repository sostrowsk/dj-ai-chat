# ai_chat/views/permissions.py
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import Http404

from ai_chat import conf

Project = conf.get_project_model()
Document = conf.get_document_model()

User = get_user_model()


class ProjectPermissionMixin:

    def check_project_access(self, user: User, project_id: int) -> Project:
        try:
            project = Project.objects.get(id=project_id)
            if not project.check_permissions(user):
                raise PermissionDenied("No access to this project")
            return project
        except Project.DoesNotExist:
            raise Http404("Project not found")


class DocumentPermissionMixin:

    def check_document_access(self, user: User, document_id: int) -> Document:
        try:
            document = Document.objects.select_related("project").get(id=document_id)
            if not document.project.check_permissions(user):
                raise PermissionDenied("No access to this document")
            return document
        except Document.DoesNotExist:
            raise Http404("Document not found")
