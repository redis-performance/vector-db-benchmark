import os
from weaviate import WeaviateClient, ConnectionParams

WEAVIATE_CLASS_NAME = "Benchmark"
WEAVIATE_DEFAULT_HTTP_PORT = 8080
WEAVIATE_DEFAULT_GRPC_PORT = 50051
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None)
WEAVIATE_HTTP_PORT = os.getenv("WEAVIATE_HTTP_PORT", WEAVIATE_DEFAULT_HTTP_PORT)
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", WEAVIATE_DEFAULT_GRPC_PORT)


def setup_client(connection_params, host):
    port = connection_params.get("port", WEAVIATE_HTTP_PORT)
    if host.startswith("http"):
        url = ""
    else:
        url = "http://"
    url += f"{host}:{port}"
    c = WeaviateClient(
        ConnectionParams.from_url(url, WEAVIATE_GRPC_PORT), skip_init_checks=True
    )
    c.connect()
    # Ping Weaviate's live state.
    assert c.is_live() is True
    return c
