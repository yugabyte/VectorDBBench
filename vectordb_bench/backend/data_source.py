import logging
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum
import struct

from tqdm import tqdm

from vectordb_bench import config
import h5py
import numpy as np
import polars as pl
from vectordb_bench.backend.utils import download_file
import os

logging.getLogger("s3fs").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

DatasetReader = typing.TypeVar("DatasetReader")


class DatasetSource(Enum):
    S3 = "S3"
    AliyunOSS = "AliyunOSS"
    Deep1BLocal = "Deep1BLocal"

    def reader(self) -> DatasetReader:
        if self == DatasetSource.S3:
            return AwsS3Reader()

        if self == DatasetSource.AliyunOSS:
            return AliyunOSSReader()

        if self == DatasetSource.Deep1BLocal:
            return Deep1BReader()

        return None


class DatasetReader(ABC):
    source: DatasetSource
    remote_root: str

    @abstractmethod
    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, deep1b_dataset_percentage: float | None = None):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
            deep1b_dataset_percentage(float | None): percentage of Deep1B dataset to use (only for Deep1B)
        """

    @abstractmethod
    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        pass


class AliyunOSSReader(DatasetReader):
    source: DatasetSource = DatasetSource.AliyunOSS
    remote_root: str = config.ALIYUN_OSS_URL

    def __init__(self):
        import oss2

        self.bucket = oss2.Bucket(oss2.AnonymousAuth(), self.remote_root, "benchmark", True)

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        info = self.bucket.get_object_meta(remote.as_posix())

        # check size equal
        remote_size, local_size = info.content_length, local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, deep1b_dataset_percentage: float | None = None):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [
                (
                    pathlib.PurePosixPath("benchmark", dataset, f),
                    local_ds_root.joinpath(f),
                )
                for f in files
            ]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath("benchmark", dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append((remote_file, local_file))

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for remote_file, local_file in tqdm(downloads):
            log.debug(f"downloading file {remote_file} to {local_file}")
            self.bucket.get_object_to_file(remote_file.as_posix(), local_file.absolute())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")


class AwsS3Reader(DatasetReader):
    source: DatasetSource = DatasetSource.S3
    remote_root: str = config.AWS_S3_URL

    def __init__(self):
        import s3fs

        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})

    def ls_all(self, dataset: str):
        dataset_root_dir = pathlib.Path(self.remote_root, dataset)
        log.info(f"listing dataset: {dataset_root_dir}")
        names = self.fs.ls(dataset_root_dir)
        for n in names:
            log.info(n)
        return names

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, deep1b_dataset_percentage: float | None = None):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [pathlib.PurePosixPath(self.remote_root, dataset, f) for f in files]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath(self.remote_root, dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append(remote_file)

        if len(downloads) == 0:
            return

        log.info(f"Start to downloading files, total count: {len(downloads)}")
        for s3_file in tqdm(downloads):
            log.debug(f"downloading file {s3_file} to {local_ds_root}")
            self.fs.download(s3_file, local_ds_root.as_posix())

        log.info(f"Succeed to download all files, downloaded file count = {len(downloads)}")

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        # info() uses ls() inside, maybe we only need to ls once
        info = self.fs.info(remote)

        # check size equal
        remote_size, local_size = info.get("size"), local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True


# Default DEEP1B URL - can be overridden by DEEP1B_URL environment variable
DEFAULT_DEEP1B_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"

# Get DEEP1B_URL from environment variable or use default
DEEP1B_URL = os.getenv("DEEP1B_URL", DEFAULT_DEEP1B_URL)

# Log the source of the URL
if "DEEP1B_URL" in os.environ:
    log.info(f"Using DEEP1B_URL from environment variable: {DEEP1B_URL}")
else:
    log.info(f"Using default DEEP1B_URL (no environment variable set): {DEEP1B_URL}")

def get_file_extension(url: str) -> str:
    """Extract file extension from URL"""
    return pathlib.Path(url).suffix.lower()

def get_filename_from_url(url: str) -> str:
    """Extract filename from URL"""
    return url.split("/")[-1]

def read_fbin_file(file_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read .fbin format file and return train and test vectors.
    
    Args:
        file_path: Path to the .fbin file
        
    Returns:
        Tuple of (train_vectors, test_vectors) as numpy arrays
        
    Note:
        .fbin format typically contains:
        - 4 bytes: number of vectors (int32)
        - 4 bytes: number of dimensions (int32)
        - vector data: num_vectors * num_dimensions * 4 bytes (float32)
    """
    log.info(f"Reading .fbin file: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read header
        num_vectors = struct.unpack('i', f.read(4))[0]
        num_dimensions = struct.unpack('i', f.read(4))[0]
        
        log.info(f"Found {num_vectors:,} vectors with {num_dimensions} dimensions in .fbin file")
        
        # Read all vectors
        vectors = np.fromfile(f, dtype=np.float32, count=num_vectors * num_dimensions)
        vectors = vectors.reshape(num_vectors, num_dimensions)
        
        # For .fbin files, we'll split them into train/test sets
        # Use 90% for training, 10% for testing
        split_point = int(num_vectors * 0.9)
        train_vectors = vectors[:split_point]
        test_vectors = vectors[split_point:]
        
        log.info(f"Split into {len(train_vectors):,} train vectors and {len(test_vectors):,} test vectors")
        
        return train_vectors, test_vectors

DEEP1B_FILENAME = get_filename_from_url(DEEP1B_URL)

class Deep1BReader(DatasetReader):
    source: DatasetSource = None  # Not a remote source
    remote_root: str = DEEP1B_URL

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        return local.exists() and local.stat().st_size > 0

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path, deep1b_dataset_percentage: float | None = None):
        # Detect file type from URL
        file_extension = get_file_extension(DEEP1B_URL)
        filename = get_filename_from_url(DEEP1B_URL)
        
        # Download the file if not present
        downloaded_file_path = local_ds_root.parent.joinpath(filename)
        if not downloaded_file_path.exists():
            local_ds_root.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Downloading {filename} from {DEEP1B_URL}")
            download_file(DEEP1B_URL, str(downloaded_file_path))
        
        # Extract and convert to Parquet if not already done
        if not local_ds_root.exists():
            local_ds_root.mkdir(parents=True)
        
        # Get the percentage configuration - use task config if provided, otherwise use global config
        percentage = deep1b_dataset_percentage if deep1b_dataset_percentage is not None else config.DEEP1B_DATASET_PERCENTAGE
        log.info(f"DEBUG: Dataset preparation reading DEEP1B_DATASET_PERCENTAGE = {percentage} (task_config={deep1b_dataset_percentage}, global_config={config.DEEP1B_DATASET_PERCENTAGE})")
        if percentage <= 0.0 or percentage > 1.0:
            raise ValueError(f"DEEP1B_DATASET_PERCENTAGE must be between 0.0 and 1.0, got {percentage}")
        
        # Create percentage-specific filenames
        train_parquet = local_ds_root.joinpath(f"train_{int(percentage * 100)}p.parquet")
        test_parquet = local_ds_root.joinpath(f"test_{int(percentage * 100)}p.parquet")
        
        if not train_parquet.exists() or not test_parquet.exists():
            log.info(f"Converting {file_extension} file to Parquet format with {percentage*100}% of data to base path: {local_ds_root}")
            
            if file_extension == '.hdf5':
                # Handle HDF5 files (original format)
                self._convert_hdf5_to_parquet(downloaded_file_path, train_parquet, test_parquet, percentage)
            elif file_extension == '.fbin':
                # Handle .fbin files (new format)
                self._convert_fbin_to_parquet(downloaded_file_path, train_parquet, test_parquet, percentage)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .hdf5 and .fbin")
        else:
            log.info(f"Using existing Deep1B dataset files with {percentage*100}% of data")
        
        log.info(f"Deep1B dataset preparation completed with {percentage*100}% of data. Note: No ground truth file is provided.")
    
    def _convert_hdf5_to_parquet(self, hdf5_path: pathlib.Path, train_parquet: pathlib.Path, test_parquet: pathlib.Path, percentage: float):
        """Convert HDF5 file to Parquet format"""
        log.info(f"Converting HDF5 file: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as f:
            # Extract train vectors with percentage sampling
            train_vectors = f["train"][:]
            total_train = train_vectors.shape[0]
            sample_size = int(total_train * percentage)
            
            # Use first N% of the data for consistency
            train_vectors = train_vectors[:sample_size]
            train_ids = list(range(sample_size))
            train_df = pl.DataFrame({
                "id": train_ids,
                "emb": [v.astype("float32") for v in train_vectors],
            })
            log.info(f"Writing train_{int(percentage * 100)}p.parquet with {sample_size:,} vectors")
            train_df.write_parquet(str(train_parquet))
            
            # Extract test vectors with percentage sampling
            test_vectors = f["test"][:]
            total_test = test_vectors.shape[0]
            test_sample_size = int(total_test * percentage)
            
            # Use first N% of the test data for consistency
            test_vectors = test_vectors[:test_sample_size]
            test_ids = list(range(test_sample_size))
            test_df = pl.DataFrame({
                "id": test_ids,
                "emb": [v.astype("float32") for v in test_vectors],
            })
            log.info(f"Writing test_{int(percentage * 100)}p.parquet with {test_sample_size:,} vectors")
            test_df.write_parquet(str(test_parquet))
    
    def _convert_fbin_to_parquet(self, fbin_path: pathlib.Path, train_parquet: pathlib.Path, test_parquet: pathlib.Path, percentage: float):
        """Convert .fbin file to Parquet format"""
        log.info(f"Converting .fbin file: {fbin_path}")
        
        # Read the .fbin file
        all_train_vectors, all_test_vectors = read_fbin_file(fbin_path)
        
        # Apply percentage sampling to training vectors
        total_train = all_train_vectors.shape[0]
        sample_size = int(total_train * percentage)
        train_vectors = all_train_vectors[:sample_size]
        train_ids = list(range(sample_size))
        
        train_df = pl.DataFrame({
            "id": train_ids,
            "emb": [v.astype("float32") for v in train_vectors],
        })
        log.info(f"Writing train_{int(percentage * 100)}p.parquet with {sample_size:,} vectors")
        train_df.write_parquet(str(train_parquet))
        
        # Apply percentage sampling to test vectors
        total_test = all_test_vectors.shape[0]
        test_sample_size = int(total_test * percentage)
        test_vectors = all_test_vectors[:test_sample_size]
        test_ids = list(range(test_sample_size))
        
        test_df = pl.DataFrame({
            "id": test_ids,
            "emb": [v.astype("float32") for v in test_vectors],
        })
        log.info(f"Writing test_{int(percentage * 100)}p.parquet with {test_sample_size:,} vectors")
        test_df.write_parquet(str(test_parquet))

def deep1b_reader():
    return Deep1BReader()
