import os

import requests

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "benchmark")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_ACCOUNT_ID = os.getenv("QDRANT_ACCOUNT_ID", None)
QDRANT_CLUSTER_ID = os.getenv("QDRANT_CLUSTER_ID", None)
QDRANT_AUTH_TOKEN = os.getenv("QDRANT_AUTH_TOKEN", None)


def get_qdrant_cloud_usage(account_id, cluster_id, token):
    result = {}
    url = f"https://cloud.qdrant.io/api/v1/accounts/{account_id}/clusters/{cluster_id}/metrics"
    headers = {"authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        # Raise an error for bad status codes
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        result = {"error": str(e)}

    return result
