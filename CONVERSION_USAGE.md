# Data Conversion Guide

This guide explains how to convert the Deep1B dataset files to parquet format for use with VectorDB Bench.

## Files to Convert

Based on the Yandex Deep1B dataset format, you need to convert:

1. **`query.public.10K.fbin`** → **`test.parquet`** (test queries)
2. **`groundtruth.public.10K.ibin`** → **`neighbors.parquet`** (ground truth neighbors)
3. **`base.1B.fbin`** → **`train_*.parquet`** (training data - split into chunks)

## Quick Usage

### Option 1: Use the Standalone Script (Recommended)

```bash
# Convert test and neighbors data
python convert_test_neighbors.py

# With custom file names
python convert_test_neighbors.py --query_file custom_query.fbin --gt_file custom_gt.ibin
```

### Option 2: High-Performance Parallel Conversion (Recommended for Large Files)

```bash
# Auto-detects instance type and uses optimal settings
python convert_base1b_parallel.py

# Manual settings for r7i.12xlarge (48 cores, 384GB)
python convert_base1b_parallel.py --workers 44 --chunk-size 20000000

# Manual settings for r7i.8xlarge (32 cores, 256GB)
python convert_base1b_parallel.py --workers 28 --chunk-size 15000000
```

### Option 3: Use Functions Directly

```python
from fbin_to_parquet import fbin_to_test_parquet, ibin_to_neighbors_parquet, fbin_to_parquet_parallel

# Convert test data (10K queries)
fbin_to_test_parquet('query.public.10K.fbin', 'test.parquet')

# Convert neighbors data (10K ground truth)
ibin_to_neighbors_parquet('groundtruth.public.10K.ibin', 'neighbors.parquet')

# Convert training data (1B vectors) - PARALLEL VERSION (Recommended)
# Auto-detects optimal settings, or specify manually:
fbin_to_parquet_parallel('base.1B.fbin', 'train')  # Auto-detect
# fbin_to_parquet_parallel('base.1B.fbin', 'train', max_workers=44)  # r7i.12xlarge

# Convert training data (1B vectors) - SINGLE-THREADED VERSION (Fallback)
# fbin_to_parquet_chunked_with_id('base.1B.fbin', 'train')
```

## File Format Details

### Input Files (Yandex Format)

All files follow the Yandex format from [this reference](https://pastebin.com/BAf6bM5L):

- **`.fbin`**: Float32 vectors
  - Header: 8 bytes (2 int32: nvecs, dim)
  - Data: nvecs × dim float32 values
  
- **`.ibin`**: Int32 vectors  
  - Header: 8 bytes (2 int32: nvecs, dim)
  - Data: nvecs × dim int32 values

### Output Files (Parquet Format)

- **`test.parquet`**: Test queries
  - Columns: `id` (int64), `emb` (array of float32)
  - 10K rows (one per query)
  
- **`neighbors.parquet`**: Ground truth neighbors
  - Columns: `neighbors_id` (array of int32)
  - 10K rows (one per query, containing neighbor IDs)
  
- **`train_*.parquet`**: Training vectors (chunked)
  - Columns: `id` (int64), `emb` (array of float32)
  - Multiple files, each with up to 10M vectors

## Implementation Notes

- **Automatic Header Reading**: Dimensions are read from file headers automatically
- **High-Performance Parallel Processing**: Uses all CPU cores for maximum speed on large files
- **Memory Efficient**: Training data is processed in chunks to handle 1B vectors
- **Format Compliance**: Follows exact Yandex reference implementation
- **AWS r7i Optimized**: Auto-detects and tunes for r7i.12xlarge (48-core) and r7i.8xlarge (32-core) instances
- **Error Handling**: Comprehensive error checking and user feedback
- **Tested**: All functions have been thoroughly tested

## Performance Expectations

### r7i.12xlarge Instance (48 cores, 384GB RAM) - RECOMMENDED
- **Expected conversion time**: 10-20 minutes for 1B vectors
- **Optimal settings**: 44 workers, 20M vectors per chunk
- **Expected throughput**: 800K-1.5M+ vectors/second
- **Memory usage**: ~12-16GB peak during conversion
- **Cost efficiency**: Higher throughput per dollar

### r7i.8xlarge Instance (32 cores, 256GB RAM)
- **Expected conversion time**: 15-30 minutes for 1B vectors
- **Optimal settings**: 28 workers, 15M vectors per chunk
- **Expected throughput**: 500K-1M+ vectors/second
- **Memory usage**: ~8-12GB peak during conversion

### Performance Comparison Summary

| Instance Type | Cores | RAM | Workers | Chunk Size | Time | Throughput | Speedup |
|---------------|-------|-----|---------|------------|------|------------|---------|
| r7i.12xlarge  | 48    | 384GB | 44     | 20M        | 10-20m | 800K-1.5M+/s | 1.6x |
| r7i.8xlarge   | 32    | 256GB | 28     | 15M        | 15-30m | 500K-1M+/s   | 1.0x |
| Single-thread | 1     | Any   | 1      | 10M        | 4-8h   | 50K-100K/s    | 0.1x |

## Expected File Sizes

- `query.public.10K.fbin`: ~3.8MB (10K × 96 × 4 bytes)
- `groundtruth.public.10K.ibin`: ~400KB (10K × 10 × 4 bytes)  
- `base.1B.fbin`: ~384GB (1B × 96 × 4 bytes)
- Output parquet files: Similar sizes with some compression

## Troubleshooting

1. **File not found**: Ensure the input files are in the current directory
2. **Memory errors**: For very large files, increase chunk size or available RAM
3. **Dimension mismatches**: Files should have 96-dimensional vectors for Deep1B
4. **Format errors**: Ensure files follow the exact Yandex fbin/ibin format

## Dependencies

```bash
pip install numpy pandas pyarrow psutil
```

## Quick Start for r7i.12xlarge (RECOMMENDED)

```bash
# 1. Install dependencies
pip install numpy pandas pyarrow psutil

# 2. Convert test and neighbors data (fast, ~30 seconds)
python convert_test_neighbors.py

# 3. Convert 1B training data (parallel, ~10-20 minutes)
python convert_base1b_parallel.py

# 4. Verify output (should show ~100 parquet files)
ls -lh deep1b_parquet/
```

## Quick Start for r7i.8xlarge

```bash
# Same steps as above, but conversion takes ~15-30 minutes
python convert_base1b_parallel.py
```
