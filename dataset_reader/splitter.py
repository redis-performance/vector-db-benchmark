import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse

CHUNK_SIZE = 20000  # Number of records to process at a time


def split_hdf5_file(
    input_path, data_output_dir, start_idx, end_idx, part, normalize=False
):
    """
    Split a specified range of the 'train' dataset from the HDF5 file into a single file.

    Args:
        input_path (str): Path to the input HDF5 file.
        data_output_dir (str): Directory where the output file will be saved.
        start_idx (int): Start index of the dataset.
        end_idx (int): End index of the dataset.
        part (int): Part number for the output file naming.
        normalize (bool): Whether to normalize the dataset or not.
    """
    with h5py.File(input_path, "r") as data_file:
        train_shape = data_file["train"].shape
        print(f"Processing train data part {part}: elements {start_idx} to {end_idx}")

        # Define the output path for this part
        data_output_path = os.path.join(
            data_output_dir,
            f"laion-img-emb-768d-1Billion-cosine-data-part{part}-{start_idx}_to_{end_idx}.hdf5",
        )

        with h5py.File(data_output_path, "w") as data_output:
            train_dset = data_output.create_dataset(
                "train",
                shape=(end_idx - start_idx, train_shape[1]),
                dtype=data_file["train"].dtype,
            )

            # Create a progress bar for the data splitting process
            with tqdm(
                total=end_idx - start_idx,
                unit="vectors",
                desc=f"Processing train data part {part}",
            ) as pbar:
                for i in range(start_idx, end_idx, CHUNK_SIZE):
                    chunk_end = min(i + CHUNK_SIZE, end_idx)
                    train_dset[i - start_idx : chunk_end - start_idx] = data_file[
                        "train"
                    ][i:chunk_end]
                    pbar.update(chunk_end - i)

    print(
        f"Train data part {part} (elements {start_idx} to {end_idx}) saved to {data_output_path}"
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Split HDF5 train dataset into specified parts."
    )
    parser.add_argument(
        "--input_path",
        required=False,
        type=str,
        default="laion-img-emb-768d-1Billion-cosine.hdf5",
        help="Path to the input HDF5 file",
    )
    parser.add_argument(
        "--data_output_dir",
        type=str,
        required=False,
        default="data",
        help="Directory where the split dataset will be saved",
    )
    parser.add_argument(
        "--start_idx", type=int, help="Start index for the dataset range to process"
    )
    parser.add_argument(
        "--end_idx", type=int, help="End index for the dataset range to process"
    )
    parser.add_argument("--part", type=int, help="Part number for the output file")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.data_output_dir, exist_ok=True)

    # Split the dataset into the specified range
    split_hdf5_file(
        args.input_path, args.data_output_dir, args.start_idx, args.end_idx, args.part
    )
