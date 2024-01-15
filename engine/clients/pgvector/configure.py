from pgvector.psycopg import register_vector
import psycopg

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.pgvector.config import (
    PGVECTOR_PASSWORD,
    PGVECTOR_PORT,
    PGVECTOR_USER,
    PGVECTOR_DBNAME,
    PGVECTOR_HOST,
)


class PgVectorConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "vector_l2_ops",
        Distance.COSINE: "vector_cosine_ops",
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        self._conn = psycopg.connect(
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASSWORD,
            dbname=PGVECTOR_DBNAME,
            autocommit=True,
        )
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

    def clean(self):
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")

    def recreate(self, dataset: Dataset, collection_params):
        self.clean()
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(
            "CREATE TABLE items (id int, embedding vector(%d))"
            % dataset.config.vector_size
        )
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        if dataset.config.distance not in self.DISTANCE_MAPPING:
            raise RuntimeError(f"unknown metric {dataset.config.distance}")
        distance = self.DISTANCE_MAPPING[dataset.config.distance]
        print("creating index...")
        hnsw_config = self.collection_params.get("hnsw_config", {})
        ef_construct = "64"
        if "ef_construct" in hnsw_config:
            ef_construct = hnsw_config["ef_construct"]
        m = "16"
        if "m" in hnsw_config:
            m = hnsw_config["m"]

        cur.execute(
            f"CREATE INDEX ON items USING hnsw (embedding {distance}) WITH (m = {m}, ef_construction = {ef_construct})"
        )
        print("done!")


if __name__ == "__main__":
    pass
