import os
from weaviate import Client
from weaviate.auth import AuthApiKey

WEAVIATE_CLASS_NAME = "Benchmark"
WEAVIATE_DEFAULT_PORT = 8090
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None)
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", WEAVIATE_DEFAULT_PORT)


def setup_client(connection_params, host):
    port = connection_params.get("port", WEAVIATE_PORT)
    if host.startswith("http"):
        url = ""
    else:
        url = "http://"
    url += f"{host}:{port}"
    auth_client_secret = None
    if WEAVIATE_API_KEY is not None:
        auth_client_secret = AuthApiKey(WEAVIATE_API_KEY)
    c = Client(url, auth_client_secret, **connection_params)
    # Ping Weaviate's live state.
    assert c.is_live() is True
    return c
