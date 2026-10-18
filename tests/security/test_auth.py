"""Tests for authentication and tenancy."""

from fastapi.testclient import TestClient

from rag_pipeline.api import app
from rag_pipeline.security import TenantContext


def test_auth_disabled_by_default():
    """Test that auth is disabled when no API_KEYS configured."""
    client = TestClient(app)

    # Without API key should still work (auth disabled)
    response = client.get("/health")
    assert response.status_code == 200


def test_api_key_parsing():
    """Test API key parsing from environment format."""
    from rag_pipeline.security.auth import parse_api_keys

    # Test empty
    keys = parse_api_keys()
    assert isinstance(keys, dict)

    # Mock environment would need to be set for full test


def test_tenant_context_creation():
    """Test TenantContext creation."""
    context = TenantContext(tenant_id="tenant1", api_key="key123")

    assert context.tenant_id == "tenant1"
    assert context.api_key == "key123"


def test_get_current_tenant_id():
    """Test getting current tenant ID."""
    from rag_pipeline.security.tenancy import (
        get_current_tenant_id,
        set_tenant_context,
    )

    context = TenantContext(tenant_id="test_tenant", api_key="key")
    set_tenant_context(context)

    assert get_current_tenant_id() == "test_tenant"


def test_tenant_context_isolation():
    """Test that tenant contexts are isolated."""
    from rag_pipeline.security.tenancy import (
        get_current_tenant_id,
        set_tenant_context,
    )

    context1 = TenantContext(tenant_id="tenant1", api_key="key1")
    set_tenant_context(context1)
    assert get_current_tenant_id() == "tenant1"

    context2 = TenantContext(tenant_id="tenant2", api_key="key2")
    set_tenant_context(context2)
    assert get_current_tenant_id() == "tenant2"


def test_require_api_key_disabled():
    """Test require_api_key when auth is disabled."""
    import asyncio

    from rag_pipeline.security.auth import require_api_key

    # When no API keys configured, should return public tenant
    context = asyncio.run(require_api_key(x_api_key=None))
    assert context.tenant_id == "public"


def test_health_endpoint_public():
    """Test health endpoint is public."""
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_docs_endpoint_public():
    """Test docs endpoint is public."""
    client = TestClient(app)

    response = client.get("/docs")
    assert response.status_code == 200
