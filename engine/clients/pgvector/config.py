import os

PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", 5432))
PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "localhost")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "ann")
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "ann")
PGVECTOR_DBNAME = os.getenv("PGVECTOR_DBNAME", "ann")
