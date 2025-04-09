from typing import Iterator, List
import h5py
import numpy as np
import os
from benchmark import DATASETS_DIR
from dataset_reader.base_reader import BaseReader, Query, Record


class AnnH5MultiReader(BaseReader):
    def __init__(
        self,
        data_files: str,
        query_file: str,
        normalize: bool = False,
        skip_upload: bool = False,
        skip_search: bool = False,
    ):
        """
        Args:
            data_dir (str): Directory containing the HDF5 data files.
            query_file (str): Path to the HDF5 query file.
            normalize (bool): Whether to normalize the vectors.
        """
        self.data_files = data_files
        self.query_file = query_file
        self.normalize = normalize
        self.skip_upload = skip_upload
        self.skip_search = skip_search

        # # Load the list of data files (assumes they're named in a consistent format)
        # self.data_files = sorted(
        #     [
        #         os.path.join(self.data_dir, f)
        #         for f in os.listdir(self.data_dir)
        #         if f.endswith(".hdf5")
        #     ]
        # )

    def read_queries(self) -> Iterator[Query]:
        """Reads the queries from the query file."""
        with h5py.File(self.query_file, "r") as data:
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

    def read_data(
        self, start_idx: int = 0, end_idx: int = None, chunk_size: int = 10_000, *args, **kwargs
    ) -> Iterator[Record]:
        """
        Reads the 'train' data vectors from multiple HDF5 files based on the specified range.

        Args:
            start_idx (int): Start index for the range of vectors.
            end_idx (int): End index for the range of vectors.

        Yields:
            Record: A Record object for each vector in the specified range.
        """
        if end_idx is None:
            raise ValueError("You must specify an end index.")

        current_idx = start_idx
        vectors_yielded = 0

        for data_file in self.data_files:
            # Extract the range of vectors covered by this file from the filename
            file_start = data_file["start_idx"]
            file_end = data_file["end_idx"]
            path = data_file["path"]

            if current_idx >= end_idx:
                break

            # Only read the file if it overlaps with the requested range
            if file_start < end_idx and file_end > start_idx:
                with h5py.File(path, "r") as data:
                    train_vectors = data["train"]
                    # Determine the slice to read from the current file
                    file_data_start = max(file_start, start_idx) - file_start
                    file_data_end = min(file_end, end_idx) - file_start

                    # Read in chunks instead of the whole slice
                    for chunk_start in range(
                        file_data_start, file_data_end, chunk_size
                    ):
                        chunk_end = min(chunk_start + chunk_size, file_data_end)
                        vectors_chunk = train_vectors[chunk_start:chunk_end]

                        for vector in vectors_chunk:
                            if self.normalize:
                                vector /= np.linalg.norm(vector)
                            yield Record(
                                id=current_idx + vectors_yielded,
                                vector=vector.tolist(),
                                metadata=None,
                            )
                            vectors_yielded += 1


if __name__ == "__main__":
    # Directory containing the data split into multiple parts
    data_dir = os.path.join(DATASETS_DIR, "laion-1b", "data")

    data_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".hdf5")]
    )

    # Path to the query file
    query_file = os.path.join(
        DATASETS_DIR, "laion-1b", "laion-img-emb-768d-1Billion-cosine-queries.hdf5"
    )

    reader = AnnH5MultiReader(data_dir, query_file, normalize=False)

    # Example of reading queries and counting them
    print("Reading queries...")
    query_count = sum(1 for _ in reader.read_queries())
    print(f"Number of queries: {query_count}")

    # Example of reading vectors from 10M to 30M and asserting the length
    start_idx = 15_000_000
    end_idx = 16_000_001
    print(f"Reading vectors from {start_idx} to {end_idx}...")
    data_vectors = list(reader.read_data(start_idx, end_idx))

    # Assert the length matches the expected range
    expected_length = end_idx - start_idx
    actual_length = len(data_vectors)
    assert (
        actual_length == expected_length
    ), f"Expected {expected_length} vectors, but got {actual_length}"
    print(f"Successfully read {actual_length} vectors.")
