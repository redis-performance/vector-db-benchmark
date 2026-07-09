import os
import urllib3
import time
from elasticsearch import Elasticsearch

ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", 9200))
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "bench")
ELASTIC_USER = os.getenv("ELASTIC_USER", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "passwd")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", None)
ELASTIC_TIMEOUT = int(os.getenv("ELASTIC_TIMEOUT", 300))
ELASTIC_INDEX_TIMEOUT = os.getenv("ELASTIC_INDEX_TIMEOUT", "30m")
ELASTIC_INDEX_REFRESH_INTERVAL = os.getenv("ELASTIC_INDEX_REFRESH_INTERVAL", "10s")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_es_client(host, connection_params):
    client: Elasticsearch = None
    init_params = {
        **{
            "verify_certs": False,
            "request_timeout": ELASTIC_TIMEOUT,
            "retry_on_timeout": True,
            "ssl_show_warn": False,
        },
        **connection_params,
    }
    if host.startswith("http"):
        url = ""
    else:
        url = "http://"
    url += f"{host}:{ELASTIC_PORT}"
    if ELASTIC_API_KEY is None:
        client = Elasticsearch(
            url,
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
            **init_params,
        )
    else:
        client = Elasticsearch(
            url,
            api_key=ELASTIC_API_KEY,
            **init_params,
        )
    assert client.ping()
    return client


def _wait_for_es_status(client, status="yellow"):
    print(f"waiting for ES {status} status...")
    for _ in range(100):
        try:
            client.cluster.health(wait_for_status=status)
            return client
        except ConnectionError:
            time.sleep(0.1)
    else:
        # timeout
        raise Exception("Elasticsearch failed to start.")
