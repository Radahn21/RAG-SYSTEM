"""
Authentication module for Azure services.

Provides InteractiveBrowserCredential for RBAC-based authentication
to Azure services without requiring Azure CLI or API keys.
"""
# 
import os
import logging
from typing import Optional, Union

from azure.identity import (
    InteractiveBrowserCredential,
    DeviceCodeCredential,
    ChainedTokenCredential,
)

logger = logging.getLogger(__name__)

# Singleton credential instance
_credential: Optional[Union[InteractiveBrowserCredential, ChainedTokenCredential]] = None


def get_credential() -> Union[InteractiveBrowserCredential, ChainedTokenCredential]:
    """
    Get or create the singleton credential for Azure RBAC authentication.
    
    This credential will open a browser window for authentication
    on first use. The token is cached for subsequent calls.
    
    For Microsoft corporate accounts, we configure:
    - Multi-tenant sign-in by default to avoid forcing the wrong tenant
    - Optional explicit tenant ID when AZURE_USE_EXPLICIT_TENANT=true
    - Proper redirect URI for browser auth
    
    Required Azure RBAC roles for this pipeline:
    - Storage Blob Data Reader: Read blobs from Azure Storage
    - Search Index Data Contributor: Upload documents to search index
    - Search Index Data Reader: Query the search index
    - Search Service Contributor: Create/manage index schema (if creating index)
    
    Returns:
        Credential: Azure credential for RBAC auth
    """
    global _credential
    
    if _credential is None:
        logger.info("Creating Azure credential...")
        logger.info("A browser window will open for Azure authentication.")
        logger.info("Please sign in with an account that has the required RBAC roles.")
        
        # Force a specific tenant only when explicitly requested.
        # This avoids local dev failures caused by a stale or incorrect tenant ID.
        tenant_id = os.environ.get("AZURE_TENANT_ID")
        use_explicit_tenant = os.environ.get("AZURE_USE_EXPLICIT_TENANT", "").lower() in {
            "1",
            "true",
            "yes",
        }
        effective_tenant_id = tenant_id if tenant_id and use_explicit_tenant else None

        if effective_tenant_id:
            logger.info(f"Using explicit tenant ID: {effective_tenant_id}")
        elif tenant_id:
            logger.info(
                "AZURE_TENANT_ID is set, but interactive auth will not force it by default. "
                "Set AZURE_USE_EXPLICIT_TENANT=true to require that tenant."
            )
        else:
            logger.info("No explicit tenant configured - the sign-in flow will use the selected account tenant")
        
        # Create InteractiveBrowserCredential with proper configuration
        # for Microsoft corporate accounts
        browser_credential = InteractiveBrowserCredential(
            tenant_id=effective_tenant_id,
            # Allow tokens for any tenant (important for corp accounts)
            additionally_allowed_tenants=["*"],
            # Redirect URI for local auth
            redirect_uri="http://localhost:8400",
        )
        
        # Create device code credential as fallback
        device_credential = DeviceCodeCredential(
            tenant_id=effective_tenant_id,
            additionally_allowed_tenants=["*"],
        )
        
        # Chain them - try browser first, fall back to device code
        _credential = ChainedTokenCredential(
            browser_credential,
            device_credential,
        )
        
        logger.info("Credential created successfully.")
    
    return _credential


def clear_credential() -> None:
    """
    Clear the cached credential.
    
    Call this if you need to re-authenticate with a different account.
    """
    global _credential
    _credential = None
    logger.info("Credential cache cleared.")


if __name__ == "__main__":
    # Test authentication
    logging.basicConfig(level=logging.INFO)
    print("Testing Azure authentication...")
    print("A browser window should open for sign-in.")
    
    credential = get_credential()
    
    # Try to get a token to verify the credential works
    try:
        # Request a token for Azure Storage (common scope)
        token = credential.get_token("https://storage.azure.com/.default")
        print(f"Successfully authenticated!")
        print(f"Token expires at: {token.expires_on}")
    except Exception as e:
        print(f"Authentication failed: {e}")
