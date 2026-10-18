"""API key authentication and tenant isolation."""

import os
from typing import Optional

from fastapi import Header, HTTPException

from rag_pipeline.config import settings


class TenantContext:
    """Tenant context for request isolation."""

    def __init__(self, tenant_id: str, api_key: str):
        """Initialize tenant context.

        Args:
            tenant_id: Unique tenant identifier.
            api_key: API key used for authentication.
        """
        self.tenant_id = tenant_id
        self.api_key = api_key


def parse_api_keys() -> dict[str, str]:
    """Parse API keys from environment variable.

    Format: KEY1:TENANT1,KEY2:TENANT2

    Returns:
        Dict mapping API keys to tenant IDs.
    """
    api_keys_str = settings.api_keys or os.getenv("API_KEYS", "")

    if not api_keys_str:
        # Auth disabled - all requests go to "public" tenant
        return {}

    mapping = {}
    for pair in api_keys_str.split(","):
        if ":" in pair:
            key, tenant = pair.strip().split(":", 1)
            mapping[key.strip()] = tenant.strip()

    return mapping


async def require_api_key(
    x_api_key: Optional[str] = Header(None),
) -> TenantContext:
    """Dependency to validate API key and return tenant context.

    Args:
        x_api_key: API key from X-API-Key header.

    Returns:
        TenantContext with tenant_id.

    Raises:
        HTTPException: If API key is invalid or missing (when auth enabled).
    """
    api_key_mapping = parse_api_keys()

    # If no keys configured, auth is disabled
    if not api_key_mapping:
        return TenantContext(tenant_id="public", api_key="")

    # Auth is enabled - require valid API key
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header"
        )

    if x_api_key not in api_key_mapping:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    tenant_id = api_key_mapping[x_api_key]
    return TenantContext(tenant_id=tenant_id, api_key=x_api_key)
