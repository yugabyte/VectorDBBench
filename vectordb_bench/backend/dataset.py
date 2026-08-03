"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import logging
import pathlib
from enum import Enum
from typing import Any, ClassVar, NamedTuple

import pandas as pd
import polars as pl
from pyarrow.parquet import ParquetFile
from pydantic import field_validator

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from . import utils
from .clients import MetricType
from .data_source import DatasetReader, DatasetSource
from .filter import Filter, FilterOp, non_filter

log = logging.getLogger(__name__)


class SizeLabel(NamedTuple):
    size: int
    label: str
    file_count: int


class BaseDataset(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    with_gt: bool = False
    _size_label: ClassVar[dict[int, SizeLabel]]
    is_custom: bool = False
    with_remote_resource: bool = True
    # for label filter cases
    with_scalar_labels: bool = False
    # if True, scalar_labels will be retrieved from a separate parquet file;
    #   otherwise, they will be obtained from train.parquet.
    scalar_labels_file_separated: bool = True
    scalar_labels_file: str = "scalar_labels.parquet"
    scalar_label_percentages: list[float] = []
    scalar_int_rates: list[float] = []
    train_id_field: str = "id"
    train_vector_field: str = "emb"
    test_file: str = "test.parquet"
    test_id_field: str = "id"
    test_vector_field: str = "emb"
    gt_id_field: str = "id"
    gt_neighbors_field: str = "neighbors_id"

    @field_validator("size")
    @classmethod
    def verify_size(cls, v: int):
        if not hasattr(cls, '_size_label') or v not in cls._size_label:
            msg = f"Size {v} not supported for the dataset, expected: {getattr(cls, '_size_label', {}).keys()}"
            raise ValueError(msg)
        return v

    @property
    def label(self) -> str:
        if not hasattr(type(self), '_size_label'):
            return ""
        size_label = type(self)._size_label.get(self.size)
        return size_label.label if size_label else ""

    @property
    def full_name(self) -> str:
        return f"{self.name.capitalize()} ({self.label.capitalize()})"

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

    @property
    def file_count(self) -> int:
        if not hasattr(type(self), '_size_label'):
            return 0
        size_label = type(self)._size_label.get(self.size)
        return size_label.file_count if size_label else 0

    @property
    def train_files(self) -> list[str]:
        return utils.compose_train_files(self.file_count, self.use_shuffled)


class CustomDataset(BaseDataset):
    dir: str
    file_num: int
    is_custom: bool = True
    with_remote_resource: bool = False
    train_file: str = "train"
    train_id_field: str = "id"
    train_vector_field: str = "emb"
    test_file: str = "test.parquet"
    gt_file: str = "neighbors.parquet"
    test_vector_field: str = "emb"
    gt_neighbors_field: str = "neighbors_id"
    with_scalar_labels: bool = True
    scalar_labels_file_separated: bool = True
    scalar_labels_file: str = "scalar_labels.parquet"
    label_percentages: list[float] = []

    @field_validator("size")
    @classmethod
    def verify_size(cls, v: int):
        return v

    @property
    def label(self) -> str:
        return "Custom"

    @property
    def dir_name(self) -> str:
        return self.dir

    @property
    def file_count(self) -> int:
        return self.file_num

    @property
    def train_files(self) -> list[str]:
        train_file = self.train_file
        prefix = f"{train_file}"
        train_files = []
        prefix_s = [item.strip() for item in prefix.split(",") if item.strip()]
        for i in range(len(prefix_s)):
            sub_file = f"{prefix_s[i]}.parquet"
            train_files.append(sub_file)
        return train_files


class LAION(BaseDataset):
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    with_gt: bool = True
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    _size_label: ClassVar[dict] = {
        100_000_000: SizeLabel(100_000_000, "LARGE", 100),
    }


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: ClassVar[dict] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict] = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class Bioasq(BaseDataset):
    name: str = "Bioasq"
    dim: int = 1024
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict] = {
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class Glove(BaseDataset):
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    _size_label: ClassVar[dict] = {1_000_000: SizeLabel(1_000_000, "MEDIUM", 1)}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: ClassVar[dict] = {
        500_000: SizeLabel(
            500_000,
            "SMALL",
            1,
        ),
        5_000_000: SizeLabel(5_000_000, "MEDIUM", 1),
        #  50_000_000: SizeLabel(50_000_000, "LARGE", 50),
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = True
    _size_label: ClassVar[dict] = {
        50_000: SizeLabel(50_000, "SMALL", 1),
        500_000: SizeLabel(500_000, "MEDIUM", 1),
        5_000_000: SizeLabel(5_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    scalar_int_rates: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
    ]


class Deep1B(BaseDataset):
    name: str = "Deep1B"
    dim: int = 96
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    with_gt: bool = True  # S3 version has ground truth (neighbors.parquet)
    # Deep1B has its own dedicated readers (S3 / local), not the standard S3 layout.
    with_remote_resource: bool = False
    _size_label: ClassVar[dict] = {
        1_000_000_000: SizeLabel(1_000_000_000, "LARGE", 100),
    }


class DatasetManager(BaseModel):
    """Download dataset if not in the local directory. Provide data for cases.

    DatasetManager is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere = Dataset.COHERE.manager(100_000)
        >>> for data in cohere:
        >>>    print(data.columns)
    """

    data: BaseDataset
    test_data: list[list[float]] | None = None
    gt_data: list[list[int]] | None = None
    scalar_labels: pl.DataFrame | None = None
    train_files: list[str] = []
    reader: DatasetReader | None = None

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    def __hash__(self) -> int:
        return hash((self.data.name, self.data.label))

    def set_reader(self, reader: DatasetReader):
        self.reader = reader

    @property
    def data_dir(self) -> pathlib.Path:
        """data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.relative_path
            '/tmp/vectordb_bench/dataset/sift/sift_small_500k/'
        """
        return pathlib.Path(
            config.DATASET_LOCAL_DIR,
            self.data.name.lower(),
            self.data.dir_name,
        )

    def __iter__(self):
        return DataSetIterator(self)

    def iter_batches(self, batch_size: int):
        return DataSetIterator(self, batch_size=batch_size)

    # TODO passing use_shuffle from outside
    def prepare(
        self,
        source: DatasetSource = DatasetSource.S3,
        filters: Filter = non_filter,
        with_train_files: bool = False,
        with_scalar_labels: bool = False,
        deep1b_dataset_percentage: float | None = None,
        skip_load: bool = False,
    ) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            filters(Filter): combined with dataset's with_gt to
              compose the correct ground_truth file
            skip_load(bool): whether load phase is skipped - for optimization

        Returns:
            bool: whether the dataset is successfully prepared

        """
        self.train_files = self.data.train_files if with_train_files else []
        gt_file, test_file = None, None
        if self.data.with_gt:
            gt_file, test_file = filters.groundtruth_file, self.data.test_file

        # Deep1B uses dedicated readers (S3 by default, local fallback) and supports
        # percentage-based / skip-load downloads.
        deep1b_reader_source = None
        if self.data.name == "Deep1B":
            deep1b_reader_source = (
                DatasetSource.Deep1BS3 if source != DatasetSource.Deep1BLocal else DatasetSource.Deep1BLocal
            )
            if deep1b_reader_source == DatasetSource.Deep1BS3:
                log.info("Using Deep1B S3 dataset - optimized for pgvector databases")

            download_files = [file for file in self.train_files]
            download_files.extend([gt_file, test_file])
            download_files = [file for file in download_files if file is not None]
            deep1b_reader_source.reader().read(
                dataset=self.data.dir_name.lower(),
                files=download_files,
                local_ds_root=self.data_dir,
                deep1b_dataset_percentage=deep1b_dataset_percentage,
                skip_load=skip_load,
            )
        elif self.data.with_remote_resource:
            download_files = [file for file in self.train_files]
            download_files.extend([gt_file, test_file])
            if self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
                download_files.append(self.data.scalar_labels_file)
            download_files = [file for file in download_files if file is not None]
            source.reader().read(
                dataset=self.data.dir_name.lower(),
                files=download_files,
                local_ds_root=self.data_dir,
            )

        needs_scalar_labels = filters.type == FilterOp.StrEqual or with_scalar_labels

        # read scalar_labels_file if separated
        if needs_scalar_labels and self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
            self.scalar_labels = self._read_file(self.data.scalar_labels_file)

        if self.data.name == "Deep1B":
            # Deep1B stores test/ground-truth differently depending on the reader used.
            if deep1b_reader_source == DatasetSource.Deep1BS3:
                # S3 version: multiple train_<number>.parquet files plus test/neighbors.
                if with_train_files:
                    self.train_files = sorted([f.name for f in self.data_dir.glob("train_*.parquet")])
                test_filename, gt_filename = "test.parquet", "neighbors.parquet"
            else:
                # Local version: percentage-specific filenames, no ground truth.
                percentage = (
                    deep1b_dataset_percentage
                    if deep1b_dataset_percentage is not None
                    else config.DEEP1B_DATASET_PERCENTAGE
                )
                train_filename = f"train_{int(percentage * 100)}p.parquet"
                test_filename = f"test_{int(percentage * 100)}p.parquet"
                gt_filename = None
                if with_train_files:
                    self.train_files = (
                        [train_filename] if self.data_dir.joinpath(train_filename).exists() else []
                    )

            if self.data_dir.joinpath(test_filename).exists():
                self.test_data = self._read_file(test_filename)[self.data.test_vector_field].to_list()
            if gt_filename and self.data_dir.joinpath(gt_filename).exists():
                self.gt_data = self._read_file(gt_filename)[self.data.gt_neighbors_field].to_list()
        elif gt_file is not None and test_file is not None:
            self.test_data = self._read_file(test_file)[self.data.test_vector_field].to_list()
            self.gt_data = self._read_file(gt_file)[self.data.gt_neighbors_field].to_list()

        log.debug(f"{self.data.name}: available train files {self.train_files}")

        return True

    def _read_file(self, file_name: str) -> pl.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = pathlib.Path(self.data_dir, file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pl.DataFrame()

        return pl.read_parquet(p)


class DataSetIterator:
    def __init__(self, dataset: DatasetManager, batch_size: int = config.NUM_PER_BATCH):
        self._ds = dataset
        self._batch_size = batch_size
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [0 for i in range(len(self._ds.train_files))]  # iter num for each file

    def __getstate__(self):
        """Custom pickle support to handle unpicklable generator."""
        state = self.__dict__.copy()
        # Remove the unpicklable generator from ParquetFile.iter_batches()
        state["_cur"] = None
        return state

    def __setstate__(self, state: Any):
        """Restore state after unpickling."""
        self.__dict__.update(state)

    def __iter__(self):
        return self

    def _get_batch_size(self) -> int:
        """Get batch size for the current dataset"""
        # Deep1B dataset uses larger batch size for better performance
        if self._ds.data.name == "Deep1B":
            return 1_000_000
        return self._batch_size

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            msg = f"No such file: {p}"
            log.warning(msg)
            raise IndexError(msg)
        batch_size = self._get_batch_size()
        log.info(f"Using batch size {batch_size} for dataset {self._ds.data.name}")
        return ParquetFile(p, memory_map=True, pre_buffer=True).iter_batches(batch_size)

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            if self._cur is None:
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)

            try:
                return next(self._cur).to_pandas()
            except StopIteration:
                if self._idx == len(self._ds.train_files) - 1:
                    raise StopIteration from None

                self._idx += 1
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)
                return next(self._cur).to_pandas()
        raise StopIteration


class Dataset(Enum):
    """
    Value is Dataset classes, DO NOT use it
    Example:
        >>> all_dataset = [ds.name for ds in Dataset]
        >>> Dataset.COHERE.manager(100_000)
        >>> Dataset.COHERE.get(100_000)
    """

    LAION = LAION
    GIST = GIST
    COHERE = Cohere
    BIOASQ = Bioasq
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI
    DEEP1B = Deep1B

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> DatasetManager:
        return DatasetManager(data=self.get(size))


class DatasetWithSizeType(Enum):
    CohereSmall = "Small Cohere (768dim, 100K)"
    CohereMedium = "Medium Cohere (768dim, 1M)"
    CohereLarge = "Large Cohere (768dim, 10M)"
    LAIONLarge = "Large LAION (768dim, 100M)"
    BioasqMedium = "Medium Bioasq (1024dim, 1M)"
    BioasqLarge = "Large Bioasq (1024dim, 10M)"
    OpenAISmall = "Small OpenAI (1536dim, 50K)"
    OpenAIMedium = "Medium OpenAI (1536dim, 500K)"
    OpenAILarge = "Large OpenAI (1536dim, 5M)"

    @classmethod
    def _missing_(cls, value: object) -> "DatasetWithSizeType | None":
        """Accept the enum member name (e.g. "CohereLarge") in addition to its value.

        Value lookup (e.g. "Large Cohere (768dim, 10M)") is tried first by Enum; this
        fallback lets callers also pass the more ergonomic member name.
        """
        if isinstance(value, str):
            for member in cls:
                if member.name == value:
                    return member
        return None

    def get_manager(self) -> DatasetManager:
        if self not in DatasetWithSizeMap:
            msg = f"wrong ScalarDatasetWithSizeType: {self.name}"
            raise ValueError(msg)
        return DatasetWithSizeMap.get(self)

    def get_load_timeout(self) -> float:
        if self is DatasetWithSizeType.LAIONLarge:
            return config.LOAD_TIMEOUT_768D_100M
        if "small" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_100K
        if "medium" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_10M
        msg = f"No load_timeout for {self.value}"
        raise KeyError(msg)

    def get_optimize_timeout(self) -> float:
        if self is DatasetWithSizeType.LAIONLarge:
            return config.OPTIMIZE_TIMEOUT_768D_100M
        if "small" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_100K
        if "medium" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_10M
        return config.OPTIMIZE_TIMEOUT_DEFAULT


DatasetWithSizeMap = {
    DatasetWithSizeType.CohereSmall: Dataset.COHERE.manager(100_000),
    DatasetWithSizeType.CohereMedium: Dataset.COHERE.manager(1_000_000),
    DatasetWithSizeType.CohereLarge: Dataset.COHERE.manager(10_000_000),
    DatasetWithSizeType.LAIONLarge: Dataset.LAION.manager(100_000_000),
    DatasetWithSizeType.BioasqMedium: Dataset.BIOASQ.manager(1_000_000),
    DatasetWithSizeType.BioasqLarge: Dataset.BIOASQ.manager(10_000_000),
    DatasetWithSizeType.OpenAISmall: Dataset.OPENAI.manager(50_000),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(500_000),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(5_000_000),
}
