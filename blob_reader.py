"""
Azure Blob Storage reader module.

Provides functionality to list and download blobs from Azure Blob Storage
using RBAC authentication (no account keys required).
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Generator

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from tqdm import tqdm

from .config import get_config
from .auth import get_credential

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class BlobInfo:
    """Information about a downloaded blob."""
    name: str                    # Blob name (e.g., "folder/document.pdf")
    local_path: Path             # Local file path after download
    blob_path: str               # Full blob path in container
    size_bytes: int              # File size in bytes
    content_type: Optional[str]  # MIME content type
    extension: str               # File extension (e.g., ".pdf")


def get_blob_service_client() -> BlobServiceClient:
    """
    Create a BlobServiceClient with RBAC authentication.
    
    Required RBAC role: Storage Blob Data Reader
    
    Returns:
        BlobServiceClient: Authenticated blob service client
    """
    config = get_config()
    credential = get_credential()
    
    logger.info(f"Connecting to storage account: {config.azure_storage_account_url}")
    
    client = BlobServiceClient(
        account_url=config.azure_storage_account_url,
        credential=credential
    )
    
    return client


def get_container_client() -> ContainerClient:
    """
    Get a ContainerClient for the configured container.
    
    Returns:
        ContainerClient: Client for the blob container
    """
    config = get_config()
    blob_service = get_blob_service_client()
    
    return blob_service.get_container_client(config.azure_storage_container)


def list_blobs(
    prefix: Optional[str] = None,
    extensions: Optional[set] = None
) -> Generator[dict, None, None]:
    """
    List blobs in the configured container.
    
    Args:
        prefix: Optional prefix to filter blobs (overrides config if provided)
        extensions: File extensions to include (default: SUPPORTED_EXTENSIONS)
        
    Yields:
        dict: Blob properties for each matching blob
    """
    config = get_config()
    container = get_container_client()
    
    # Use provided prefix or fall back to config
    blob_prefix = prefix if prefix is not None else config.azure_storage_prefix
    extensions = extensions or SUPPORTED_EXTENSIONS
    
    logger.info(f"Listing blobs in container: {config.azure_storage_container}")
    if blob_prefix:
        logger.info(f"  with prefix: {blob_prefix}")
    
    try:
        blobs = container.list_blobs(name_starts_with=blob_prefix or None)
        
        for blob in blobs:
            # Check file extension
            ext = Path(blob.name).suffix.lower()
            if ext in extensions:
                yield {
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "last_modified": blob.last_modified,
                    "extension": ext
                }
                
    except HttpResponseError as e:
        if e.status_code == 403:
            logger.error(
                "Access denied (403). Please ensure your account has the "
                "'Storage Blob Data Reader' role on the storage account or container."
            )
        raise


def download_blob(blob_name: str, local_dir: Optional[Path] = None) -> BlobInfo:
    """
    Download a single blob to a local directory.
    
    Args:
        blob_name: Name of the blob to download
        local_dir: Local directory to save the file (default: config.download_dir)
        
    Returns:
        BlobInfo: Information about the downloaded blob
    """
    config = get_config()
    container = get_container_client()
    
    local_dir = local_dir or config.download_dir
    
    # Create local file path, preserving directory structure
    local_path = local_dir / blob_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Downloading: {blob_name} -> {local_path}")
    
    try:
        blob_client = container.get_blob_client(blob_name)
        blob_props = blob_client.get_blob_properties()
        
        # Download the blob
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        return BlobInfo(
            name=blob_name,
            local_path=local_path,
            blob_path=f"{config.azure_storage_container}/{blob_name}",
            size_bytes=blob_props.size,
            content_type=blob_props.content_settings.content_type if blob_props.content_settings else None,
            extension=Path(blob_name).suffix.lower()
        )
        
    except ResourceNotFoundError:
        logger.error(f"Blob not found: {blob_name}")
        raise
    except HttpResponseError as e:
        if e.status_code == 403:
            logger.error(
                f"Access denied downloading {blob_name}. "
                "Ensure 'Storage Blob Data Reader' role is assigned."
            )
        raise


def download_all_blobs(
    prefix: Optional[str] = None,
    local_dir: Optional[Path] = None,
    show_progress: bool = True
) -> List[BlobInfo]:
    """
    Download all supported blobs from the container.
    
    Args:
        prefix: Optional prefix to filter blobs
        local_dir: Local directory to save files
        show_progress: Show progress bar
        
    Returns:
        List[BlobInfo]: Information about all downloaded blobs
    """
    config = get_config()
    local_dir = local_dir or config.download_dir
    
    # List all blobs first
    blobs = list(list_blobs(prefix=prefix))
    
    if not blobs:
        logger.warning("No supported files found in the container.")
        return []
    
    logger.info(f"Found {len(blobs)} files to download")
    
    downloaded = []
    
    # Download with optional progress bar
    iterator = tqdm(blobs, desc="Downloading", disable=not show_progress)
    
    for blob in iterator:
        try:
            info = download_blob(blob["name"], local_dir)
            downloaded.append(info)
            
            if show_progress:
                iterator.set_postfix(file=Path(blob["name"]).name[:30])
                
        except Exception as e:
            logger.error(f"Failed to download {blob['name']}: {e}")
            # Continue with other files
            continue
    
    logger.info(f"Downloaded {len(downloaded)} files to {local_dir}")
    return downloaded


if __name__ == "__main__":
    # Test blob operations
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Azure Blob Storage connection...")
    print("=" * 50)
    
    # List blobs
    print("\nListing blobs:")
    count = 0
    for blob in list_blobs():
        print(f"  - {blob['name']} ({blob['size']} bytes)")
        count += 1
        if count >= 10:
            print("  ... (showing first 10)")
            break
    
    if count == 0:
        print("  No supported files found.")
    
    # Optionally download
    response = input("\nDownload all files? (y/n): ")
    if response.lower() == "y":
        downloaded = download_all_blobs()
        print(f"\nDownloaded {len(downloaded)} files.")
