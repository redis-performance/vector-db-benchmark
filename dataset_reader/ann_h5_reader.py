from typing import Iterator
import json
import h5py
import numpy as np

from benchmark import DATASETS_DIR
from dataset_reader.base_reader import BaseReader, Query, Record


class AnnH5Reader(BaseReader):
    def __init__(self, path, normalize=False):
        self.path = path
        self.normalize = normalize

    def read_queries(self) -> Iterator[Query]:
        data = h5py.File(self.path)

        for vector, expected_result, expected_scores in zip(
            data["test"], data["neighbors"], data["distances"]
        ):
            if self.normalize:
                vector /= np.linalg.norm(vector)
            yield Query(
                vector=vector.tolist(),
                meta_conditions=None,
                expected_result=expected_result.tolist(),
                expected_scores=expected_scores.tolist(),
            )

    def read_data(self, start_idx: int = 0, end_idx: int = None) -> Iterator[Record]:
        data = h5py.File(self.path)
        has_metadata = "metadata" in data  # Check if metadata exists
        for idx, vector in enumerate(data["train"]):
            if self.normalize:
                vector /= np.linalg.norm(vector)
            metadata = None
            if has_metadata:
                try:
                    metadata_str = data["metadata"][idx].decode("utf-8").strip()
                    if metadata_str.startswith("{") and metadata_str.endswith("}"):
                        metadata = json.loads(metadata_str)
                    else:
                        metadata = None
                except (IndexError, AttributeError, UnicodeDecodeError, json.JSONDecodeError) as e :
                    metadata = None  # Handle cases where metadata retrieval fails

            yield Record(id=idx, vector=vector.tolist(), metadata=metadata)


if __name__ == "__main__":
    import os

    # h5py file 4 keys:
    # `train` - float vectors (num vectors 1183514)
    # `test` - float vectors (num vectors 10000)
    # `neighbors` - int - indices of nearest neighbors for test (num items 10k, each item
    # contains info about 100 nearest neighbors)
    # `distances` - float - distances for nearest neighbors for test vectors

    test_path = os.path.join(
        DATASETS_DIR, "glove-100-angular", "glove-100-angular.hdf5"
    )
    record = next(AnnH5Reader(test_path).read_data())
    print(record, end="\n\n")

    query = next(AnnH5Reader(test_path).read_queries())
    print(query)
