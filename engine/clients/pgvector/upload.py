from typing import List, Optional

import numpy as np
from pgvector.psycopg import register_vector
import psycopg

from engine.base_client.upload import BaseUploader

from engine.clients.pgvector.config import (
    PGVECTOR_PASSWORD,
    PGVECTOR_PORT,
    PGVECTOR_USER,
    PGVECTOR_DBNAME,
    PGVECTOR_HOST,
)


class PgVectorUploader(BaseUploader):
    _conn = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls._conn = psycopg.connect(
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASSWORD,
            dbname=PGVECTOR_DBNAME,
            autocommit=True,
        )
        register_vector(cls._conn)
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        cur = cls._conn.cursor()
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i in range(len(ids)):
                idx = ids[i]
                vec = vectors[i]
                copy.write_row((str(idx), np.array(vec).astype(np.float32)))

    @classmethod
    def post_upload(cls, _distance):
        return {}
