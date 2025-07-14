import logging
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum

from tqdm import tqdm

from vectordb_bench import config
import h5py
import polars as pl
from vectordb_bench.backend.utils import download_file

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
    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
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

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
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

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
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


DEEP1B_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
DEEP1B_HDF5_FILENAME = "deep-image-96-angular.hdf5"

class Deep1BReader(DatasetReader):
    source: DatasetSource = None  # Not a remote source
    remote_root: str = DEEP1B_URL

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        return local.exists() and local.stat().st_size > 0

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        # Download the HDF5 file if not present
        hdf5_path = local_ds_root.parent.joinpath(DEEP1B_HDF5_FILENAME)
        if not hdf5_path.exists():
            local_ds_root.parent.mkdir(parents=True, exist_ok=True)
            download_file(DEEP1B_URL, str(hdf5_path))
        # Extract and convert to Parquet if not already done
        if not local_ds_root.exists():
            local_ds_root.mkdir(parents=True)
        # Only create train.parquet and test.parquet if not present
        train_parquet = local_ds_root.joinpath("train.parquet")
        test_parquet = local_ds_root.joinpath("test.parquet")
        if not train_parquet.exists() or not test_parquet.exists():
            with h5py.File(hdf5_path, "r") as f:
                # Extract train vectors
                train_vectors = f["train"][:]
                train_ids = list(range(train_vectors.shape[0]))
                train_df = pl.DataFrame({
                    "id": train_ids,
                    "emb": [v.astype("float32") for v in train_vectors],
                })
                train_df.write_parquet(str(train_parquet))
                # Extract test vectors
                test_vectors = f["test"][:]
                test_ids = list(range(test_vectors.shape[0]))
                test_df = pl.DataFrame({
                    "id": test_ids,
                    "emb": [v.astype("float32") for v in test_vectors],
                })
                test_df.write_parquet(str(test_parquet))
        # No ground truth for now (could be added if needed)

def deep1b_reader():
    return Deep1BReader()
