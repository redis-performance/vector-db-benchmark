import os
from databricks.vector_search.client import VectorSearchClient


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import CreateCatalog

# Replace with your Databricks instance URL
instance_url = os.environ["DATABRICKS_HOST"]
# Replace with your Databricks personal access token
token = os.environ["DATABRICKS_TOKEN"]


# Replace with your Databricks instance URL and token
workspace_client = WorkspaceClient(host=instance_url, token=token)
catalog_name = "vector_search_bench"
catalogs = workspace_client.catalogs.list()
found = False
for c in catalogs:
    print(c.name)
    if catalog_name == c.name:
        found = True
        print(f"Catalog found: {catalog_name}")
        break

if found is False:
    print("Catalog not found, creating it")

    workspace_client.catalogs.create(
        name=catalog_name,
        comment="Catalog for vector search",
        properties={"purpose": "vector-search"},
    )

catalog_name = "vector_search_bench"
schema_name = "vector_search"

# Check if schema exists
schemas = workspace_client.schemas.list(catalog_name=catalog_name)
if not any(s.name == schema_name for s in schemas):
    print(f"Schema {catalog_name}.{schema_name} not found, creating it")
    workspace_client.schemas.create(
        name=schema_name,
        catalog_name=catalog_name,
        comment="Schema for vector search benchmark",
    )
else:
    print(f"Schema {catalog_name}.{schema_name} already exists")


client = VectorSearchClient(
    personal_access_token=token,
    disable_notice=True,
)

index_name_short = "bench"
index_name = f"{catalog_name}.{schema_name}.{index_name_short}"

vector_search_endpoint_name = "vector-search-demo-endpoint"

# List existing indexes on the endpoint
existing_indexes = client.list_indexes(name=vector_search_endpoint_name)
if "vector_indexes" in existing_indexes:
    existing_indexes = existing_indexes["vector_indexes"]
else:
    existing_indexes = []
print(f"Existing vector indices: {existing_indexes}")
print(f"Existing indices: {existing_indexes}")
# Check if the index already exists
if any(existing_index["name"] == index_name for existing_index in existing_indexes):
    print(f"Index {index_name} already exists — skipping creation.")
    index = client.get_index(index_name=index_name)
    client.delete_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=index_name,
    )

print(f"Creating index {index_name} on endpoint {vector_search_endpoint_name}")
import time 

index = client.create_direct_access_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=index_name,
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
    schema={
        "id": "int",
        "field2": "string",
        "field3": "float",
        "text_vector": "array<float>",
    },
)


while not index.describe().get("status")["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(30)
print("Index is ready!")
print(index.describe())

index.upsert(
    [
        {"id": 1, "field2": "value2", "field3": 3.0, "text_vector": [1.0] * 1024},
        {"id": 2, "field2": "value2", "field3": 3.0, "text_vector": [1.1] * 1024},
    ]
)

import time

while not index.describe().get("status")["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(30)
print("Index is ready!")
print(index.describe())

# Delta Sync Index with pre-calculated embeddings
results2 = index.similarity_search(
    query_vector=[0.9] * 1024,
    columns=["id", "text_vector"],
    num_results=2,
    disable_notice=True,
)

print(results2)
