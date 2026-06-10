from unittest.mock import Mock, patch

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.test import Client, RequestFactory
from django.urls import reverse

from ai_chat.views.chat import ChatView
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.project_utils import create_project
from users.factories import login_and_verify
from users.tests.factories import (
    AdminFactory,
    BaseUserFactory,
    BrokerCompanyFactory,
    BrokerFactory,
    ClientCompanyFactory,
    ClientFactory,
    LeasingCompanyFactory,
    PartnerFactory,
)

User = get_user_model()


# --- Fixtures ---
@pytest.fixture
def client_company():
    """Create a ClientCompany instance."""
    return ClientCompanyFactory()


@pytest.fixture
def broker_company():
    """Create a BrokerCompany instance."""
    return BrokerCompanyFactory()


@pytest.fixture
def leasing_company():
    """Create a LeasingCompany instance."""
    return LeasingCompanyFactory()


@pytest.fixture
def admin_user():
    """Create an admin superuser."""
    return AdminFactory()


@pytest.fixture
def client_user(client_company):
    """Create a client user with a company."""
    return ClientFactory(client_company=client_company)


@pytest.fixture
def broker_user(broker_company):
    """Create a broker user with a company."""
    return BrokerFactory(broker_company=broker_company)


@pytest.fixture
def partner_user(leasing_company):
    """Create a partner user with a company."""
    return PartnerFactory(leasing_company=leasing_company)


@pytest.fixture
def project(client_user, broker_user, partner_user):
    """Create a project using project_utils.create_project."""
    project = create_project(
        name="Test Project",
        asset_name="Test Asset",
        client_company=client_user.client_company,
        broker_company=broker_user.broker_company,
        leasing_company=partner_user.leasing_company,
        user=client_user,
    )
    project.leasing_companies.add(partner_user.leasing_company)
    return project


@pytest.fixture
def protected_document(project, client_user):
    """Create an indexed, reviewed, enabled protected document."""
    return ProtectedDocumentFactory(
        project=project,
        name="Test Document",
        user=client_user,
        user_company=client_user.client_company.company,
    )


@pytest.fixture
def rf():
    """Provide a RequestFactory instance."""
    return RequestFactory()


@pytest.fixture
def client():
    """Provide a Django test client."""
    return Client()


# --- View Tests for ai_chat/views/chat.py ---
@pytest.mark.django_db
def test_chat_view_get_context_data_admin(rf, admin_user, project, protected_document):
    """Test get_context_data for admin user."""
    request = rf.get(reverse("ai_chat:lobby"))
    request.user = admin_user
    view = ChatView()
    view.request = request

    with patch.object(ChatView, "_handle_specific_context") as mock_handle:
        context = view.get_context_data()
        assert context["project_list"].count() == 1
        assert context["project_list"][0].name == "Test Project"
        assert context["page_title"] == "KI Chat"
        mock_handle.assert_called_once_with(context, {})


@pytest.mark.django_db
@pytest.mark.parametrize(
    "user_type,company_field,expected_count",
    [
        ("client", "client_company", 1),
        ("broker", "broker_company", 1),
        ("partner", "leasing_company", 1),
        ("client", None, 0),
        ("other", "client_company", 0),
    ],
)
def test_get_accessible_projects(
    rf,
    user_type,
    company_field,
    expected_count,
    client_company,
    broker_company,
    leasing_company,
):
    """Test _get_accessible_projects for different user types."""
    user = BaseUserFactory(type=user_type, is_active=True)
    if company_field:
        company = {
            "client_company": client_company,
            "broker_company": broker_company,
            "leasing_company": leasing_company,
        }.get(company_field)
        setattr(user, company_field, company)
        user.save()

    project = create_project(
        name="Test Project",
        asset_name="Test Asset",
        client_company=client_company if user_type == "client" else None,
        broker_company=broker_company if user_type == "broker" else None,
        leasing_company=leasing_company if user_type == "partner" else None,
    )
    if user_type == "partner":
        project.leasing_companies.add(leasing_company)
    ProtectedDocumentFactory(project=project, name="Test Document")

    if company_field is not None:
        company_value = getattr(user, company_field, None)
        mock_get_company = Mock(return_value=company_value)
        user.get_company = mock_get_company
    else:
        mock_get_company = Mock(return_value=None)
        user.get_company = mock_get_company

    request = rf.get(reverse("ai_chat:lobby"))
    request.user = user
    view = ChatView()
    view.request = request

    projects = view._get_accessible_projects()
    assert projects.count() == expected_count
    if expected_count:
        assert projects[0].name == "Test Project"


@pytest.mark.django_db
def test_handle_specific_context_with_project(rf, client_user, project):
    """Test _handle_specific_context with project_id."""
    request = rf.get(reverse("ai_chat:project_chat", kwargs={"project_id": project.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    context = {}
    kwargs = {"project_id": project.id}
    with patch.object(ChatView, "_add_project_context") as mock_add:
        view._handle_specific_context(context, kwargs)
        mock_add.assert_called_once_with(context, project.id)


@pytest.mark.django_db
def test_handle_specific_context_with_document(rf, client_user, protected_document):
    """Test _handle_specific_context with document_id."""
    request = rf.get(reverse("ai_chat:document_chat", kwargs={"document_id": protected_document.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    context = {}
    kwargs = {"document_id": protected_document.id}
    with patch.object(ChatView, "_add_document_context") as mock_add:
        view._handle_specific_context(context, kwargs)
        mock_add.assert_called_once_with(context, protected_document.id)


@pytest.mark.django_db
def test_add_project_context_success(rf, client_user, project, protected_document):
    """Test _add_project_context with valid project access."""
    request = rf.get(reverse("ai_chat:project_chat", kwargs={"project_id": project.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    context = {}
    with patch.object(ChatView, "check_project_access", return_value=project) as mock_check:
        view._add_project_context(context, project.id)
        assert context["current_project"] == project
        assert context["documents"].count() == 1
        assert context["documents"][0].name == protected_document.name
        assert context["page_title"] == f"KI Chat - {project.name}"
        mock_check.assert_called_once_with(client_user, project.id)


@pytest.mark.django_db
def test_add_project_context_permission_denied(rf, client_user, project):
    """Test _add_project_context raises PermissionDenied when no access."""
    request = rf.get(reverse("ai_chat:project_chat", kwargs={"project_id": project.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    with patch.object(ChatView, "check_project_access", side_effect=PermissionDenied):
        with pytest.raises(PermissionDenied):
            view._add_project_context({}, project.id)


@pytest.mark.django_db
def test_add_document_context_success(rf, client_user, protected_document):
    """Test _add_document_context with valid document access."""
    request = rf.get(reverse("ai_chat:document_chat", kwargs={"document_id": protected_document.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    context = {}
    with patch.object(ChatView, "check_document_access", return_value=protected_document) as mock_check:
        view._add_document_context(context, protected_document.id)
        assert context["current_document"] == protected_document
        assert context["current_project"] == protected_document.project
        assert context["documents"].count() == 1
        assert context["documents"][0].name == protected_document.name
        assert context["page_title"] == f"KI Chat - {protected_document.name}"
        mock_check.assert_called_once_with(client_user, protected_document.id)


@pytest.mark.django_db
def test_add_document_context_permission_denied(rf, client_user, protected_document):
    """Test _add_document_context raises PermissionDenied when no access."""
    request = rf.get(reverse("ai_chat:document_chat", kwargs={"document_id": protected_document.id}))
    request.user = client_user
    view = ChatView()
    view.request = request

    with patch.object(ChatView, "check_document_access", side_effect=PermissionDenied):
        with pytest.raises(PermissionDenied):
            view._add_document_context({}, protected_document.id)


@pytest.mark.django_db
def test_chat_view_template_rendering(client, client_user, project, protected_document):
    """Test template rendering with project context."""
    login_and_verify(client_user, client)
    with patch("project.models.Project.check_permissions", return_value=True):
        response = client.get(reverse("ai_chat:project_chat", kwargs={"project_id": project.id}))

        assert response.status_code == 200
        assert "ai_chat/chat.html" in [t.name for t in response.templates]
        assert response.context["current_project"].name == project.name
        assert response.context["documents"].count() == 1
        assert response.context["documents"][0].name == protected_document.name
        assert response.context["page_title"] == f"KI Chat - {project.name}"
        # Verify template content
        content = response.content.decode()
        assert "Test Project" in content
        assert "Test Document" in content
        assert "KI Chat - Test Project" in content


@pytest.mark.django_db
def test_chat_view_unauthenticated(client, project):
    """Test unauthenticated access redirects."""
    response = client.get(reverse("ai_chat:lobby"))
    assert response.status_code == 302  # Redirects to login
    assert "login" in response.url
