import os
import random
import time

import requests

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "benchmark")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_ACCOUNT_ID = os.getenv("QDRANT_ACCOUNT_ID", None)
QDRANT_CLUSTER_ID = os.getenv("QDRANT_CLUSTER_ID", None)
QDRANT_AUTH_TOKEN = os.getenv("QDRANT_AUTH_TOKEN", None)
QDRANT_MAX_OPTIMIZATION_THREADS = os.getenv("QDRANT_MAX_OPTIMIZATION_THREADS", None)


def get_collection_info(endpoint, collection, api_key):
    result = {}
    url = f"{endpoint}/collections/{collection}"
    headers = {"api-key": f"{api_key}"}

    try:
        response = requests.get(url, headers=headers)
        # Raise an error for bad status codes
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        result = {"error": str(e)}

    return result


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


def retry_with_exponential_backoff(
    func, *args, max_retries=10, base_delay=1, max_delay=90, **kwargs
):
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            delay = min(base_delay * 2**retries + random.uniform(0, 1), max_delay)
            time.sleep(delay)
            retries += 1
            print(f"received the following exception on try #{retries}: {e.__str__}")
            if retries == max_retries:
                raise e
            else:
                print("retrying...")
