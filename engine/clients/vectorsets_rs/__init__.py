from vector_db_benchmark_rs import (
    RustVsetConfigurator as _RustConfigurator,
    RustVsetUploader as _RustUploader,
    RustVsetSearcher as _RustSearcher,
)
from engine.base_client.configure import BaseConfigurator
from engine.base_client.upload import BaseUploader
from engine.base_client.search import BaseSearcher


class RustVsetConfigurator(BaseConfigurator):
    """Rust-backed vectorsets configurator."""

    def __init__(self, host, collection_params, connection_params):
        super().__init__(host, collection_params, connection_params)
        self._rust = _RustConfigurator(host, collection_params=collection_params, connection_params=connection_params)

    def clean(self):
        self._rust.clean()

    def recreate(self, dataset, collection_params):
        pass

    def execution_params(self, distance, vector_size):
        return {}

    def delete_client(self):
        pass


class RustVsetUploader(BaseUploader):
    """Rust-backed vectorsets uploader. upload() runs entirely in Rust."""

    def __init__(self, host, connection_params, upload_params):
        super().__init__(host, connection_params, upload_params)
        self._rust = _RustUploader(host, connection_params=connection_params, upload_params=upload_params)

    def upload(self, distance, records):
        """Override BaseUploader.upload — runs the full loop in Rust."""
        return self._rust.upload_all(distance, records)

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        _RustUploader.init_client(host, distance, connection_params, upload_params)

    @classmethod
    def upload_batch(cls, ids, vectors, metadata):
        _RustUploader.upload_batch(ids, vectors, metadata)

    @classmethod
    def post_upload(cls, distance):
        return _RustUploader.post_upload(distance)

    @classmethod
    def get_memory_usage(cls):
        return _RustUploader.get_memory_usage()

    @classmethod
    def delete_client(cls):
        _RustUploader.delete_client()


class RustVsetSearcher(BaseSearcher):
    """Rust-backed vectorsets searcher. search_all runs entirely in Rust."""

    def __init__(self, host, connection_params, search_params):
        super().__init__(host, connection_params, search_params)
        self._rust = _RustSearcher(host, connection_params=connection_params, search_params=search_params)

    @classmethod
    def init_client(cls, host, distance, connection_params, search_params):
        _RustSearcher.init_client(host, distance, connection_params, search_params)

    @classmethod
    def search_one(cls, vector, meta_conditions, top):
        return _RustSearcher.search_one(vector, meta_conditions, top)

    def search_all(self, distance, queries, num_queries=-1):
        """Override BaseSearcher.search_all — runs the full loop in Rust."""
        return self._rust.search_all(distance, queries, num_queries)

    @classmethod
    def delete_client(cls):
        _RustSearcher.delete_client()
