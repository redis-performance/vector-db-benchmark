from typing import Any, Iterable

from ml_dtypes import bfloat16
import numpy as np

from dataset_reader.base_reader import Record


def iter_batches(records: Iterable[Record], n: int) -> Iterable[Any]:
    ids = []
    vectors = []
    metadata = []

    for record in records:
        ids.append(record.id)
        vectors.append(record.vector)
        metadata.append(record.metadata)

        if len(vectors) >= n:
            yield [ids, vectors, metadata]
            ids, vectors, metadata = [], [], []
    if len(ids) > 0:
        yield [ids, vectors, metadata]


def check_data_type(data_type: str):
    valid_data_types = ["FLOAT32", "FLOAT64", "FLOAT16", "BFLOAT16"]
    if data_type.upper() not in valid_data_types:
        raise ValueError(
            f"Invalid data type: {data_type}. Valid options are: {valid_data_types}"
        )
    if data_type == "FLOAT32":
        return np.float32
    if data_type == "FLOAT64":
        return np.float64
    if data_type == "FLOAT16":
        return np.float16
    if data_type == "BFLOAT16":
        return bfloat16
