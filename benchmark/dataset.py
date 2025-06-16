import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import boto3
import botocore.exceptions
from benchmark import DATASETS_DIR
from dataset_reader.ann_compound_reader import AnnCompoundReader
from dataset_reader.ann_h5_reader import AnnH5Reader
from dataset_reader.ann_h5_multi_reader import AnnH5MultiReader
from dataset_reader.base_reader import BaseReader
from dataset_reader.json_reader import JSONReader
from tqdm import tqdm
from pathlib import Path


@dataclass
class DatasetConfig:
    vector_size: int
    distance: str
    name: str
    type: str
    path: Dict[
        str, List[Dict[str, str]]
    ]  # Now path is expected to handle multi-file structure for h5-multi
    link: Optional[Dict[str, List[Dict[str, str]]]] = None
    schema: Optional[Dict[str, str]] = field(default_factory=dict)


READER_TYPE = {
    "h5": AnnH5Reader,
    "h5-multi": AnnH5MultiReader,
    "jsonl": JSONReader,
    "tar": AnnCompoundReader,
}


# Progress bar for urllib downloads
def show_progress(block_num, block_size, total_size):
    percent = round(block_num * block_size / total_size * 100, 2)
    print(f"{percent} %", end="\r")


# Progress handler for S3 downloads
class S3Progress(tqdm):
    def __init__(self, total_size):
        super().__init__(
            total=total_size, unit="B", unit_scale=True, desc="Downloading from S3"
        )

    def __call__(self, bytes_amount):
        self.update(bytes_amount)


class Dataset:
    def __init__(
        self,
        config: dict,
        skip_upload: bool,
        skip_search: bool,
        upload_start_idx: int,
        upload_end_idx: int,
    ):
        self.config = DatasetConfig(**config)
        self.skip_upload = skip_upload
        self.skip_search = skip_search
        self.upload_start_idx = upload_start_idx
        self.upload_end_idx = upload_end_idx

    def download(self):
        if isinstance(self.config.path, dict):  # Handle multi-file datasets
            if self.skip_search is False:
                # Download query files
                for query in self.config.path.get("queries", []):
                    self._download_file(query["path"], query["link"])
            else:
                print(
                    f"skipping to download query file given skip_search={self.skip_search}"
                )
            if self.skip_upload is False:
                # Download data files
                for data in self.config.path.get("data", []):
                    start_idx = data["start_idx"]
                    end_idx = data["end_idx"]
                    data_path = data["path"]
                    data_link = data["link"]
                    if self.upload_start_idx >= end_idx:
                        print(
                            f"skipping downloading {data_path} from {data_link} given {self.upload_start_idx}>{end_idx}"
                        )
                        continue
                    if self.upload_end_idx < start_idx:
                        print(
                            f"skipping downloading {data_path} from {data_link} given {self.upload_end_idx}<{start_idx}"
                        )
                        continue
                    self._download_file(data["path"], data["link"])
            else:
                print(
                    f"skipping to download data/upload files given skip_upload={self.skip_upload}"
                )

        else:  # Handle single-file datasets
            target_path = DATASETS_DIR / self.config.path

            if target_path.exists():
                print(f"{target_path} already exists")
                return

            if self.config.link:
                downloaded_withboto = False
                if is_s3_link(self.config.link):
                    print("Use boto3 to download from S3. Faster!")
                    try:
                        self._download_from_s3(self.config.link, target_path)
                        downloaded_withboto = True
                    except botocore.exceptions.NoCredentialsError:
                        print("Credentials not found, downloading without boto3")
                if not downloaded_withboto:
                    print(f"Downloading from URL {self.config.link}...")
                    tmp_path, _ = urllib.request.urlretrieve(
                        self.config.link, None, show_progress
                    )
                    self._extract_or_move_file(tmp_path, target_path)

    def _download_file(self, relative_path: str, url: str):
        target_path = DATASETS_DIR / relative_path
        if target_path.exists():
            print(f"{target_path} already exists")
            return

        print(f"Downloading from {url} to {target_path}")
        tmp_path, _ = urllib.request.urlretrieve(url, None, show_progress)
        self._extract_or_move_file(tmp_path, target_path)

    def _extract_or_move_file(self, tmp_path, target_path):
        if tmp_path.endswith(".tgz") or tmp_path.endswith(".tar.gz"):
            print(f"Extracting: {tmp_path} -> {target_path}")
            (DATASETS_DIR / self.config.path).mkdir(exist_ok=True, parents=True)
            with tarfile.open(tmp_path) as file:
                file.extractall(target_path)
            os.remove(tmp_path)
        else:
            print(f"Moving: {tmp_path} -> {target_path}")
            Path(target_path).parent.mkdir(exist_ok=True)
            shutil.copy2(tmp_path, target_path)
            os.remove(tmp_path)

    def _download_from_s3(self, link, target_path):
        s3 = boto3.client("s3")
        bucket_name, s3_key = parse_s3_url(link)
        tmp_path = f"/tmp/{os.path.basename(s3_key)}"

        print(
            f"Downloading from S3: {link}... bucket_name={bucket_name}, s3_key={s3_key}"
        )
        object_info = s3.head_object(Bucket=bucket_name, Key=s3_key)
        total_size = object_info["ContentLength"]

        with open(tmp_path, "wb") as f:
            progress = S3Progress(total_size)
            s3.download_fileobj(bucket_name, s3_key, f, Callback=progress)

        self._extract_or_move_file(tmp_path, target_path)

    def get_reader(self, normalize: bool) -> BaseReader:
        reader_class = READER_TYPE[self.config.type]

        if self.config.type == "h5-multi":
            # For h5-multi, we need to pass both data files and query file
            data_files = self.config.path["data"]
            for data_file_dict in data_files:
                data_file_dict["path"] = DATASETS_DIR / data_file_dict["path"]
            query_file = DATASETS_DIR / self.config.path["queries"][0]["path"]
            return reader_class(
                data_files=data_files,
                query_file=query_file,
                normalize=normalize,
                skip_upload=self.skip_upload,
                skip_search=self.skip_search,
            )
        else:
            # For single-file datasets
            return reader_class(DATASETS_DIR / self.config.path, normalize=normalize)


def is_s3_link(link):
    return link.startswith("s3://") or "s3.amazonaws.com" in link


def parse_s3_url(s3_url):
    if s3_url.startswith("s3://"):
        s3_parts = s3_url.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        s3_key = s3_parts[1] if len(s3_parts) > 1 else ""
    else:
        s3_parts = s3_url.replace("http://", "").replace("https://", "").split("/", 1)

        if ".s3.amazonaws.com" in s3_parts[0]:
            bucket_name = s3_parts[0].split(".s3.amazonaws.com")[0]
            s3_key = s3_parts[1] if len(s3_parts) > 1 else ""
        else:
            bucket_name = s3_parts[0]
            s3_key = s3_parts[1] if len(s3_parts) > 1 else ""

    return bucket_name, s3_key


if __name__ == "__main__":
    dataset_s3_split = Dataset(
        {
            "name": "laion-img-emb-768d-1Billion-cosine",
            "vector_size": 768,
            "distance": "cosine",
            "type": "h5-multi",
            "path": {
                "data": [
                    {
                        "file_number": 1,
                        "path": "laion-1b/data/laion-img-emb-768d-1Billion-cosine-data-part1-0_to_10000000.hdf5",
                        "link": "http://benchmarks.redislabs.s3.amazonaws.com/vecsim/laion-1b/laion-img-emb-768d-1Billion-cosine-data-part1-0_to_10000000.hdf5",
                        "vector_range": "0-10000000",
                        "file_size": "30.7 GB",
                    },
                    {
                        "file_number": 2,
                        "path": "laion-1b/data/laion-img-emb-768d-1Billion-cosine-data-part10-90000000_to_100000000.hdf5",
                        "link": "http://benchmarks.redislabs.s3.amazonaws.com/vecsim/laion-1b/laion-img-emb-768d-1Billion-cosine-data-part10-90000000_to_100000000.hdf5",
                        "vector_range": "90000000-100000000",
                        "file_size": "30.7 GB",
                    },
                    {
                        "file_number": 3,
                        "path": "laion-1b/data/laion-img-emb-768d-1Billion-cosine-data-part100-990000000_to_1000000000.hdf5",
                        "link": "http://benchmarks.redislabs.s3.amazonaws.com/vecsim/laion-1b/laion-img-emb-768d-1Billion-cosine-data-part100-990000000_to_1000000000.hdf5",
                        "vector_range": "990000000-1000000000",
                        "file_size": "30.7 GB",
                    },
                ],
                "queries": [
                    {
                        "path": "laion-1b/laion-img-emb-768d-1Billion-cosine-queries.hdf5",
                        "link": "http://benchmarks.redislabs.s3.amazonaws.com/vecsim/laion-1b/laion-img-emb-768d-1Billion-cosine-queries.hdf5",
                        "file_size": "38.7 MB",
                    },
                ],
            },
        },
        skip_upload=True,
        skip_search=False,
    )

    dataset_s3_split.download()
    reader = dataset_s3_split.get_reader(normalize=False)
    print(reader)  # Outputs the AnnH5MultiReader instance
