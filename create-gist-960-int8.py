import numpy as np
from ast import Dict
from dataset_reader.ann_h5_reader import AnnH5Reader
from benchmark import DATASETS_DIR
from dataset_reader.base_reader import BaseReader, Query, Record
import tqdm
import time
import h5py
import os

numpy_types_dict = {"float32": np.float32, "int8": np.int8, "uint8": np.uint8}


# quantize vectors pre-dimension
class ScalarQuantization:
    def __init__(self, dim, precision: str = "uint8"):
        self.N = 255  # 2^B - 1
        self.dim = dim
        self.precision = precision
        if precision == "uint8":
            self.offset = 0
        elif precision == "int8":
            self.offset = 128

    def train(self, train_dataset: np.ndarray):
        # Assuming train_dataset is a numpy array with shape (n_train_vec, self.dim)
        self.x_min = train_dataset.min(
            axis=0
        )  # Find the minimum value in each dimension
        self.delta = (
            train_dataset.max(axis=0) - self.x_min
        ) / self.N  # Calculate delta for each dimension

    def quantize(self, dataset: np.ndarray):
        q_vals = np.floor((dataset - self.x_min) / self.delta)
        # use int32 to avoid overflow during offset subtraction
        q_vals = np.clip(q_vals, 0, self.N).astype(np.int32)
        q_vals -= self.offset
        return q_vals.astype(numpy_types_dict[self.precision])

    def decompress(self, x):
        return (self.delta * (x + 0.5 + self.offset).astype(np.float32)) + self.x_min

    def get_quantization_params(self) -> Dict:
        return {"x_min": self.x_min, "delta": self.delta}


if __name__ == "__main__":
    import os

    # h5py file 4 keys:
    # `train` - float vectors (num vectors 1183514)
    # `test` - float vectors (num vectors 10000)
    # `neighbors` - int - indices of nearest neighbors for test (num items 10k, each item
    # contains info about 100 nearest neighbors)
    # `distances` - float - distances for nearest neighbors for test vectors

    test_path = os.path.join(
        DATASETS_DIR, "gist-960-euclidean", "gist-960-euclidean.hdf5"
    )

    data = AnnH5Reader(test_path).read_data()
    queries = AnnH5Reader(test_path).read_queries()

    train_dataset_size = 100000
    full_dataset_size = 1000000
    train_dataset = []
    full_dataset = []
    test = []
    neighbors = []
    distances = []
    for query in tqdm.tqdm(queries):
        test.append(np.array(query.vector).astype(np.float32))
        neighbors.append(query.expected_result)
        distances.append(query.expected_scores)

    for record in tqdm.tqdm(data):
        if len(full_dataset) >= full_dataset_size:
            break
        full_dataset.append(np.array(record.vector).astype(np.float32))
        if len(train_dataset) < train_dataset_size:
            train_dataset.append(np.array(record.vector).astype(np.float32))

    train_dataset = np.array(
        train_dataset
    )  # Convert list of vectors into a single NumPy array
    print("n vectors = ", len(train_dataset))
    print("vector shape = ", train_dataset[0].shape)
    precision = "int8"
    quantizer = ScalarQuantization(train_dataset[0].shape, precision)
    print("Creating quantizer for type = ", precision)

    print("\nTraining dataset ... ")
    start = time.time()
    quantizer.train(train_dataset)
    dur = time.time() - start
    print(
        f"Training took {dur} seconds. \nQuantization params = {quantizer.get_quantization_params()}"
    )

    # quantize dataset
    full_dataset = np.array(full_dataset)
    test = np.array(test)
    print("\Quantizing dataset ... ")
    start = time.time()
    quantized_dataset = quantizer.quantize(full_dataset)
    quantized_queries = quantizer.quantize(test)
    dur = time.time() - start
    print(f"Quantization took {dur} seconds.")
    print("vector 1 shape = ", quantized_dataset[0].shape)
    print("vector 1 sample = ", quantized_dataset[0])

    # Create a new HDF5 file and write the data
    output_path = os.path.join(DATASETS_DIR, "gist-960-euclidean-int8.hdf5")

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("train", data=quantized_dataset, compression=None)
        h5f.create_dataset("test", data=quantized_queries, compression=None)
        h5f.create_dataset("neighbors", data=neighbors, compression=None)
        h5f.create_dataset("distances", data=distances, compression=None)
