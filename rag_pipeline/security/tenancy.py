"""Tenant context and isolation utilities."""

from contextvars import ContextVar
from typing import Optional

from rag_pipeline.security.auth import TenantContext

# Context variable to store current tenant
_tenant_context: ContextVar[Optional[TenantContext]] = ContextVar(
    "tenant_context", default=None
)


def set_tenant_context(context: TenantContext) -> None:
    """Set tenant context for current request.

    Args:
        context: TenantContext to set.
    """
    _tenant_context.set(context)


def get_tenant_context() -> Optional[TenantContext]:
    """Get current tenant context.

    Returns:
        TenantContext or None if not set.
    """
    return _tenant_context.get()


def get_current_tenant_id() -> str:
    """Get current tenant ID.

    Returns:
        Tenant ID string, defaults to "public".
    """
    context = _tenant_context.get()
    return context.tenant_id if context else "public"
