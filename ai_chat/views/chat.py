# ai_chat/views/chat.py
from typing import Any, Dict

from django.db.models import Prefetch

from ai_chat import conf

from .base import SecureView
from .permissions import DocumentPermissionMixin, ProjectPermissionMixin

Project = conf.get_project_model()
ProtectedProjectDocument = conf.get_document_model()


class ChatView(SecureView, ProjectPermissionMixin, DocumentPermissionMixin):
    template_name = "ai_chat/chat.html"
    page_title = "KI Chat"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        projects = self._get_accessible_projects()
        context["project_list"] = projects
        self._handle_specific_context(context, kwargs)
        return context

    def _get_accessible_projects(self):
        """Projects the user may chat about — host hook or package default."""
        accessible_projects_func = conf.get_accessible_projects_func()
        if accessible_projects_func is not None:
            return accessible_projects_func(self.request.user)
        return self._default_accessible_projects()

    def _default_accessible_projects(self):
        user = self.request.user
        if user.type == "admin" and user.is_superuser:
            queryset = Project.objects.all()
        else:
            user_company = user.get_company()
            if user_company:
                if user.type == "client":
                    queryset = Project.objects.filter(client_company=user_company)
                elif user.type == "broker":
                    queryset = Project.objects.filter(broker_company=user_company)
                elif user.type == "partner":
                    queryset = Project.objects.filter(leasing_companies=user_company)
                else:
                    return Project.objects.none()
            else:
                return Project.objects.none()

        indexed_filters = conf.get_indexed_document_filters()
        queryset = queryset.filter(
            **{f"protected_documents__{k}": v for k, v in indexed_filters.items()}
        ).prefetch_related(
            Prefetch(
                "protected_documents",
                queryset=ProtectedProjectDocument.objects.filter(**indexed_filters),
            )
        )

        return queryset.distinct().order_by("name")

    def _handle_specific_context(self, context: dict, kwargs: dict) -> None:
        if project_id := kwargs.get("project_id"):
            self._add_project_context(context, project_id)
        elif document_id := kwargs.get("document_id"):
            self._add_document_context(context, document_id)

    def _add_project_context(self, context: dict, project_id: int) -> None:
        project = self.check_project_access(self.request.user, project_id)
        documents = project.protected_documents.filter(**conf.get_indexed_document_filters())
        context.update(
            {
                "current_project": project,
                "documents": documents,
                "page_title": f"KI Chat - {project.name}",
            }
        )

    def _add_document_context(self, context: dict, document_id: int) -> None:
        document = self.check_document_access(self.request.user, document_id)
        documents = document.project.protected_documents.filter(**conf.get_indexed_document_filters())
        context.update(
            {
                "current_document": document,
                "current_project": document.project,
                "documents": documents,
                "page_title": f"KI Chat - {document.name}",
            }
        )
