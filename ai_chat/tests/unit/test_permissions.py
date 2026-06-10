import pytest
from django.core.exceptions import PermissionDenied
from django.test import Client
from django.urls import reverse

from ai_chat.views.permissions import DocumentPermissionMixin, ProjectPermissionMixin
from data_room.tests.factories import ProtectedDocumentFactory
from project.tests.project_utils import create_project
from users.factories import create_client, login_and_verify


def make_user_verified(user):
    """Helper to make a user verified by adding OTP device."""
    from django_otp.plugins.otp_static.models import StaticDevice

    StaticDevice.objects.create(user=user, confirmed=True)
    return user


@pytest.fixture
def client_user():
    """Create a client user with a company using factory."""
    return create_client()


@pytest.fixture
def verified_client_user(client_user):
    """Create a verified client user with OTP device."""
    return make_user_verified(client_user)


@pytest.fixture
def other_client_user():
    """Create another client user with different company (no access to projects)."""
    return create_client()


@pytest.fixture
def project(verified_client_user):
    """Create a project."""
    return create_project(
        name="Test Project",
        asset_name="Test Asset",
        client_company=verified_client_user.client_company,
        user=verified_client_user,
    )


@pytest.fixture
def protected_document(project, verified_client_user):
    """Create an indexed, reviewed, enabled protected document."""
    return ProtectedDocumentFactory(
        project=project,
        name="Test Document",
        user=verified_client_user,
        user_company=verified_client_user.client_company.company,
    )


@pytest.fixture
def client():
    """Provide a Django test client."""
    return Client()


# Tests for ProjectPermissionMixin
@pytest.mark.django_db
def test_check_project_access_success(verified_client_user, project):
    """Test check_project_access with valid access."""
    mixin = ProjectPermissionMixin()
    result = mixin.check_project_access(verified_client_user, project.id)
    assert result == project
    assert result.name == "Test Project"


@pytest.mark.django_db
def test_check_project_access_no_permission(other_client_user, project):
    """Test check_project_access raises PermissionDenied for unauthorized user."""
    make_user_verified(other_client_user)

    mixin = ProjectPermissionMixin()
    with pytest.raises(PermissionDenied, match="No access to this project"):
        mixin.check_project_access(other_client_user, project.id)


@pytest.mark.django_db
def test_check_project_access_non_existent(client_user):
    """Test check_project_access raises Http404 for non-existent project."""
    from django.http import Http404

    mixin = ProjectPermissionMixin()
    with pytest.raises(Http404, match="Project not found"):
        mixin.check_project_access(client_user, 9999)


# Tests for DocumentPermissionMixin
@pytest.mark.django_db
def test_check_document_access_success(verified_client_user, protected_document):
    """Test check_document_access with valid access."""
    mixin = DocumentPermissionMixin()
    result = mixin.check_document_access(verified_client_user, protected_document.id)
    assert result == protected_document
    assert result.name == "Test Document"


@pytest.mark.django_db
def test_check_document_access_no_permission(other_client_user, protected_document):
    """Test check_document_access raises PermissionDenied for unauthorized user."""
    # Make the other user verified but still no access to document
    make_user_verified(other_client_user)

    mixin = DocumentPermissionMixin()
    with pytest.raises(PermissionDenied, match="No access to this document"):
        mixin.check_document_access(other_client_user, protected_document.id)


@pytest.mark.django_db
def test_check_document_access_non_existent(client_user):
    """Test check_document_access raises Http404 for non-existent document."""
    from django.http import Http404

    mixin = DocumentPermissionMixin()
    with pytest.raises(Http404, match="Document not found"):
        mixin.check_document_access(client_user, 9999)


# Integration Test with ChatView
@pytest.mark.django_db
def test_chat_view_with_permissions(client, verified_client_user, project, protected_document):
    """Test ChatView renders correctly with proper permissions."""
    login_and_verify(verified_client_user, client)
    response = client.get(reverse("ai_chat:project_chat", kwargs={"project_id": project.id}))
    assert response.status_code == 200
    assert "ai_chat/chat.html" in [t.name for t in response.templates]
    assert response.context["current_project"].name == project.name
