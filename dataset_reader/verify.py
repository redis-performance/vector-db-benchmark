import os
import h5py

EXPECTED_VECTORS = 10_000_000  # Expected number of vectors per file


def verify_hdf5_files(directory):
    """
    Verifies that each HDF5 file in the given directory contains 10 million vectors in the 'train' dataset.

    Args:
        directory (str): Directory containing the HDF5 files to check.
    """
    hdf_files = [f for f in os.listdir(directory) if f.endswith(".hdf5")]

    if not hdf_files:
        print("No HDF5 files found in the directory.")
        return

    all_verified = True

    for hdf_file in sorted(hdf_files):
        file_path = os.path.join(directory, hdf_file)

        with h5py.File(file_path, "r") as data_file:
            if "train" in data_file:
                train_shape = data_file["train"].shape
                num_vectors = train_shape[0]
                print(f"Checking {hdf_file}: contains {num_vectors} vectors.")

                if num_vectors != EXPECTED_VECTORS:
                    print(
                        f"ERROR: {hdf_file} contains {num_vectors} vectors, expected {EXPECTED_VECTORS}."
                    )
                    all_verified = False
            else:
                print(f"ERROR: 'train' dataset not found in {hdf_file}.")
                all_verified = False

    if all_verified:
        print(
            "All HDF5 files verified successfully, each containing 10 million vectors."
        )
    else:
        print("Some files contain discrepancies. Please check the log above.")


if __name__ == "__main__":
    # Define the path to the directory containing the HDF5 files
    directory = "./data"  # Replace with your actual directory

    # Verify the HDF5 files
    verify_hdf5_files(directory)
