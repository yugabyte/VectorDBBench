# Skip-Load Optimization for Deep1B S3 Dataset

## Overview

The Deep1B S3 dataset implementation now includes a smart optimization that skips downloading training files when the load phase is not needed, significantly reducing download time and storage requirements.

## How It Works

### Detection

The system automatically detects when the load phase is skipped by checking if `TaskStage.LOAD` is present in the task configuration:

```python
# In CaseRunner._pre_run()
skip_load = TaskStage.LOAD not in self.config.stages
```

### Optimization Behavior

**When Load Phase is Included (`--load` or default):**
- Downloads training files: `train_0.parquet`, `train_1.parquet`, ..., `train_N.parquet` (based on percentage)
- Downloads `test.parquet` for search operations
- Downloads `neighbors.parquet` for ground truth evaluation

**When Load Phase is Skipped (`--skip-load`):**
- ⚡ **Skips all training files** - saves significant time and storage
- Downloads only `test.parquet` for search operations
- Downloads only `neighbors.parquet` for ground truth evaluation

## Usage Examples

### CLI Usage

```bash
# Normal operation - downloads all files
python -m vectordb_bench run --db=pgvector --case=Performance96D1B --deep1b-dataset-percentage=0.1

# Skip load - downloads only test and neighbors files (much faster!)
python -m vectordb_bench run --db=pgvector --case=Performance96D1B --skip-load --deep1b-dataset-percentage=0.1
```

### Configuration File

```yaml
# config.yml
load: false  # This will trigger skip-load optimization
search_serial: true
search_concurrent: true
```

## Benefits

### Time Savings
- **With 100% dataset**: Skip downloading 100GB+ of training data
- **With 10% dataset**: Skip downloading 10GB+ of training data  
- **With 1% dataset**: Skip downloading 1GB+ of training data

### Storage Savings
- Only requires space for `test.parquet` (~100MB) and `neighbors.parquet` (~400MB)
- vs full dataset requiring 10GB-1TB+ depending on percentage

### Use Cases
- **Search-only benchmarks**: When you only need to test search performance
- **Existing data**: When training data is already loaded in the database
- **Development/testing**: Quick iterations without full data downloads

## Implementation Details

### Files Modified

1. **`vectordb_bench/backend/data_source.py`**
   - Updated `DatasetReader.read()` interface to accept `skip_load` parameter
   - Enhanced `Deep1BS3Reader.read()` to conditionally skip training files
   - Added logging for optimization behavior

2. **`vectordb_bench/backend/dataset.py`**
   - Updated `DatasetManager.prepare()` to accept and pass `skip_load` parameter
   - Enhanced documentation

3. **`vectordb_bench/backend/task_runner.py`**
   - Added detection logic for `TaskStage.LOAD` in task configuration
   - Pass `skip_load` parameter to dataset preparation

### Backward Compatibility

- All existing readers (`AliyunOSSReader`, `AwsS3Reader`, `Deep1BReader`) accept the new `skip_load` parameter
- Parameter defaults to `False`, maintaining existing behavior
- No changes required for non-Deep1B datasets

## Logging Output

When skip-load is active, you'll see logs like:

```
INFO - Load phase is skipped - only downloading test and neighbors files for search operations
INFO - Skipping training files download since load phase is skipped
INFO - Start downloading 2 files from S3  # Instead of 10+ files
```

## Validation

The optimization has been verified to:
- ✅ Maintain syntax correctness across all modified files
- ✅ Preserve backward compatibility
- ✅ Only affect Deep1B S3 dataset (pgvector-specific)
- ✅ Work with all percentage configurations
- ✅ Maintain proper error handling and logging
