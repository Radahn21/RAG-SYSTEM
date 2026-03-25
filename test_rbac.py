"""Test RBAC propagation for Azure AI Search."""
import time
import logging

logging.basicConfig(level=logging.WARNING)

from src.search_client import get_search_index_client
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

print("Testing Azure AI Search RBAC access...")
print("Will retry every 30 seconds until it works.")
print("Press Ctrl+C to stop.\n")

for attempt in range(1, 21):
    try:
        print(f"Attempt {attempt}...", end=" ", flush=True)
        client = get_search_index_client()
        indexes = list(client.list_index_names())
        print(f"SUCCESS! Found {len(indexes)} indexes: {indexes}")
        break
    except (ClientAuthenticationError, HttpResponseError) as e:
        print(f"Still waiting (RBAC propagation)...")
        if attempt < 20:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nCancelled.")
        break
