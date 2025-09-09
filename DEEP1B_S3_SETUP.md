# Deep1B S3 Dataset Setup (PgVector Only)

This document describes the new Deep1B S3 dataset functionality that allows downloading the Deep1B dataset directly from an S3 bucket. **This implementation is specifically designed for pgvector databases.**

## Overview

The Deep1B dataset is now available in two modes:
1. **Local mode** (existing): Downloads HDF5 file and converts to Parquet - works with all databases
2. **S3 mode** (new): Downloads pre-processed Parquet files directly from S3 - **pgvector only**

## S3 Dataset Structure

The S3 bucket `s3://perf-team/vectordbbench/deep1b/parquet/1b/` contains:

- **Training files**: `learn_split_0.parquet` to `learn_split_99.parquet` (100 files total, 10M vectors each)
- **Test file**: `test.parquet`
- **Ground truth file**: `neighbours.parquet`

## Configuration

### Dataset Percentage

Use the `deep1b_dataset_percentage` parameter to control how much of the dataset to download:

- `1.0` (100%): Downloads all 100 training files (1 billion vectors)
- `0.1` (10%): Downloads 10 training files (100 million vectors)
- `0.01` (1%): Downloads 1 training file (10 million vectors)

### AWS Credentials

The S3 reader uses AWS credentials from `~/.aws/` directory. Ensure your AWS profile has access to the `s3://perf-team/vectordbbench/deep1b/parquet/1b/` bucket.

## Usage

### Using the CLI with PgVector

```bash
# Use S3 source with pgvector (default for Deep1B)
python -m vectordb_bench run --db=pgvector --deep1b-dataset-percentage=0.1

# Force local source (works with any database)
python -m vectordb_bench run --db=pgvector --source=Deep1BLocal --deep1b-dataset-percentage=0.1

# Other databases must use local source
python -m vectordb_bench run --db=milvus --source=Deep1BLocal --deep1b-dataset-percentage=0.1
```

### Using the API

```python
from vectordb_bench.backend.dataset import Dataset
from vectordb_bench.backend.data_source import DatasetSource

# Create dataset manager
deep1b = Dataset.DEEP1B.manager(1_000_000_000)

# Use S3 source with 10% of data
deep1b.prepare(
    source=DatasetSource.Deep1BS3,
    deep1b_dataset_percentage=0.1
)

# Use local source with 10% of data
deep1b.prepare(
    source=DatasetSource.Deep1BLocal,
    deep1b_dataset_percentage=0.1
)
```

## Features

### Automatic Source Selection

- **Default**: Uses `DatasetSource.Deep1BS3` (S3 source) - **pgvector only**
- **Fallback**: Uses `DatasetSource.Deep1BLocal` when explicitly specified - works with all databases

### Ground Truth Support

The S3 version includes ground truth data (`neighbours.parquet`), enabling search quality evaluation.

### File Validation

Files are validated by comparing local and remote file sizes before download.

### Progress Tracking

Downloads show progress bars using tqdm.

### Partial Downloads

Only downloads files that are missing or have different sizes.

## File Naming

### S3 Mode
- Training: `learn_split_0.parquet`, `learn_split_1.parquet`, ..., `learn_split_N.parquet`
- Test: `test.parquet`
- Ground truth: `neighbours.parquet`

### Local Mode (existing)
- Training: `train_1p.parquet`, `train_10p.parquet`, `train_100p.parquet`
- Test: `test_1p.parquet`, `test_10p.parquet`, `test_100p.parquet`

## Implementation Details

### New Classes

- `Deep1BS3Reader`: Handles S3 downloads with AWS credentials
- `DatasetSource.Deep1BS3`: New dataset source enum

### Modified Classes

- `Deep1B`: Updated to support ground truth (`with_gt: bool = True`)
- `DatasetManager.prepare()`: Updated to handle S3 vs local sources

## Testing

Run the test script to verify the setup:

```bash
python test_deep1b_s3.py
```

This will test both S3 and local modes with 1% of the dataset.

## Performance Considerations

- **S3 download speed**: Depends on your internet connection and AWS region
- **Multiple files**: S3 mode downloads multiple smaller files vs one large HDF5 file
- **Storage**: Parquet files are more efficient for columnar access than HDF5
- **Memory**: Uses memory-mapped Parquet files for efficient iteration
