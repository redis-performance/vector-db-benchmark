from datasets import load_dataset
import numpy as np
import os
import pickle
from dotenv import load_dotenv
from benchmark import DATASETS_DIR
import h5py
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from tqdm import tqdm
import json

# Load environment variables
load_dotenv()

# Constants
LANG = "en"
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "1000000"))
VECTOR_TYPE = os.getenv("VECTOR_TYPE", "INT8").lower()
QUERIES_NUM = 1000
K = 100

dataset_embed_type_dict = {"float32": "emb", "int8": "emb_int8"}
dataset_vector_dtype_dict = {"float32": np.float32, "int8": np.int8}
dataset_name_type_dict = {
    "float32": "Cohere/wikipedia-2023-11-embed-multilingual-v3",
    "int8": "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
}


def create_redis_index(vector_type):
    client = Redis()
    try:
        client.ft().dropindex(
            delete_documents=True
        )  # Remove existing index if it exists
    except:
        pass
    index_def = IndexDefinition(index_type=IndexType.HASH)
    schema = (
        TextField("_id"),
        TextField("title"),
        TextField("text"),
        VectorField(
            "vector",
            "FLAT",
            {"TYPE": vector_type, "DIM": 1024, "DISTANCE_METRIC": "COSINE"},
        ),
    )
    client.ft().create_index(schema, definition=index_def)
    print("Redis search index created.")


def load_vectors(vector_type, vector_dtype, dataset_name, emb_fieldname):
    embeddings_file = f"{vector_type}_embeddings_{DATASET_SIZE}.pkl"
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as f:
            (vectors, metadata, query_vectors) = pickle.load(f)
            print(
                f"Prepared {len(vectors)} dataset vectors, {len(metadata)} metadata, and {len(query_vectors)} query vectors"
            )
            return vectors, metadata, query_vectors
    dataset = load_dataset(
        dataset_name,
        LANG,
        split="train",
        streaming=True,
    )
    vectors, metadata = [], []
    query_vectors = []
    for num, doc in tqdm(
        enumerate(dataset.take(DATASET_SIZE + QUERIES_NUM)), desc="Loading dataset"
    ):
        vector = doc[emb_fieldname]
        if num >= DATASET_SIZE:
            query_vectors.append(vector)
        else:
            vectors.append(vector)
            metadata.append(
                {
                    "_id": doc["_id"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }
            )
    vectors = np.array(vectors, dtype=vector_dtype)
    with open(embeddings_file, "wb") as f:
        pickle.dump((vectors, metadata, query_vectors), f)
    print(
        f"Prepared {len(vectors)} dataset vectors, {len(metadata)} metadata, and {len(query_vectors)} query vectors"
    )
    return vectors, metadata, query_vectors


def ingest_vectors(vectors, metadata, vector_type):
    client = Redis()
    client.flushdb()  # Clean DB before ingestion
    create_redis_index(vector_type)  # Ensure index is created before ingestion
    pipeline = client.pipeline()
    for i, (vector, meta) in enumerate(
        tqdm(zip(vectors, metadata), desc="Ingesting vectors", total=len(vectors))
    ):
        pipeline.hset(f"{i}", mapping={"vector": vector.tobytes(), **meta})
        if i % 100 == 0:
            pipeline.execute()
    pipeline.execute()
    print("Vector ingestion complete.")


def verify_metadata(vectors):
    client = Redis()
    sample_indices = np.random.choice(len(vectors), 5, replace=False)
    for idx in sample_indices:
        data = client.hgetall(f"{idx}")
        if data:
            print(f"Metadata for vector {idx}: {data}")
        else:
            print(f"No metadata found for vector {idx}")


def run():
    vector_type = VECTOR_TYPE
    dataset_name = dataset_name_type_dict[VECTOR_TYPE]
    vector_dtype = dataset_vector_dtype_dict[VECTOR_TYPE]
    emb_fieldname = dataset_embed_type_dict[VECTOR_TYPE]
    vectors, metadata, queries = load_vectors(
        vector_type, vector_dtype, dataset_name, emb_fieldname
    )
    ingest_vectors(vectors[:DATASET_SIZE], metadata[:DATASET_SIZE], vector_type)
    verify_metadata(vectors[:DATASET_SIZE])
    assert len(queries) == QUERIES_NUM
    assert len(vectors) == DATASET_SIZE
    assert len(metadata) == DATASET_SIZE
    neighbors, distances = [], []
    K = 100
    client_ft = Redis().ft()
    q = (
        Query("*=>[KNN $K @vector $vec_param AS vector_score]")
        .sort_by("vector_score", asc=True)
        .paging(0, K)
        .return_fields("vector_score")
        .dialect(4)
        .timeout(12000000)
    )
    for query_vector in tqdm(queries, desc="Processing queries"):
        params_dict = {
            "vec_param": np.array(query_vector).astype(vector_dtype).tobytes(),
            "K": K,
        }
        results = client_ft.search(q, query_params=params_dict)
        nb = [int(result.id) for result in results.docs]
        ds = [int(result.id) for result in results.docs]
        if len(nb) != K:
            print(f"wrong len {len(nb)}")
            continue

        neighbors.append([int(result.id) for result in results.docs])
        distances.append([float(result.vector_score) for result in results.docs])
    vector_dimension = len(vectors[0])
    output_dir = os.path.join(
        DATASETS_DIR,
        f"cohere-wikipedia-multilingual-{vector_dimension}-angular-{vector_type}",
    )
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(
        output_dir,
        f"cohere-wikipedia-multilingual-{vector_dimension}-angular-{vector_type}.hdf5",
    )

    metadata_json = np.array(
        [json.dumps(meta) for meta in metadata[:DATASET_SIZE]], dtype="S"
    )
    assert len(metadata_json) == len(vectors)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("train", data=vectors, compression=None)
        h5f.create_dataset("test", data=queries, compression=None)
        h5f.create_dataset(
            "neighbors", data=np.array(neighbors, dtype=np.int32), compression=None
        )
        h5f.create_dataset(
            "distances", data=np.array(distances, dtype=np.float32), compression=None
        )
        h5f.create_dataset("metadata", data=metadata_json, compression=None)


if __name__ == "__main__":
    run()
