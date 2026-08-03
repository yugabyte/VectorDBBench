# VectorDB Benchmark Parameters Reference

This document provides a comprehensive reference for understanding the key parameters used in VectorDB benchmarking, dataset formats, and the relationship between different configuration values.

## Table of Contents
- [Key Parameters Overview](#key-parameters-overview)
- [Parameter Relationships](#parameter-relationships)
- [Dataset File Formats](#dataset-file-formats)
- [Deep1B Dataset Source](#deep1b-dataset-source)
- [Converter Usage](#converter-usage)
- [Configuration Examples](#configuration-examples)

## Key Parameters Overview

### `k` - Number of Search Results
- **Purpose**: Number of nearest neighbors to return in search results
- **Default**: `100` (defined in `config.K_DEFAULT`)
- **Usage**: Final output size - "give me the top 100 most similar vectors"
- **Location**: Used in search functions and ground truth evaluation
- **Example**: `search_embedding(query, k=100)` returns 100 nearest neighbors

### `ef_search` - HNSW Search Width
- **Purpose**: Internal parameter controlling how many candidates to explore during HNSW search
- **Default**: `128` (common in configurations)
- **Usage**: Search quality vs speed tradeoff - "explore 128 candidates to find the best k results"
- **Constraint**: Must be `≥ k` (you can't return 100 results if you only explored 50 candidates)
- **Impact**: Higher `ef_search` = better recall but slower search

### `m` - HNSW Index Connections
- **Purpose**: Maximum number of connections per node in HNSW (Hierarchical Navigable Small World) index
- **Default**: `16` (common default)
- **Usage**: Controls the structure of the HNSW index during build time
- **Impact**: Higher `m` = better recall but more memory usage and slower build time
- **Range**: Valid values between "2" and "100"

### `ef_construction` - HNSW Build Parameter
- **Purpose**: Number of candidate connections to consider during index construction
- **Default**: `128` (common default)
- **Constraint**: Must be `≥ 2 * m`
- **Usage**: Controls index quality vs build time tradeoff
- **Impact**: Higher `ef_construction` = better index quality but slower build time

## Parameter Relationships

```
ef_search ≥ k (always required)
ef_construction ≥ 2 * m (HNSW constraint)
```

### Search Flow:
1. HNSW explores `ef_search=128` candidate vectors
2. From those candidates, it returns the top `k=100` results
3. Results are compared against ground truth with `k=100` neighbors for evaluation

### Index Build Flow:
1. HNSW creates connections with `m=16` max connections per node
2. During construction, considers `ef_construction=128` candidates per node
3. Results in a graph structure optimized for fast search

## Dataset File Formats

### Training Data (`train_*.parquet`)
```
Columns:
- id: int64 (incrementing integer, unique identifier)
- emb: array of float32 (the vector embeddings)

Usage: Contains vector embeddings inserted into the database during benchmarking
```

### Test Data (`test.parquet`)
```
Columns:
- id: int64 (incrementing integer, query identifier)
- emb: array of float32 (query vectors for testing)

Usage: Query vectors used to benchmark search performance and measure latency
Recommendation: Limit to ~1,000 test vectors to avoid memory pressure
```

### Ground Truth (`neighbours.parquet`)
```
Columns:
- id: int64 (corresponds to test query vector IDs)
- neighbors_id: array of int64 (true nearest neighbor IDs)

Usage: Contains ground truth data for evaluating search accuracy
- Used to calculate recall and NDCG metrics
- IDs must correspond exactly to test.parquet IDs
```

### Format Consistency
**Important**: All files should use the `id` + array format (not separate `dim_0`, `dim_1`, ... columns) for compatibility with VectorDB Benchmark.

## Deep1B Dataset Source

### Yandex Research Blog
The authoritative source for Deep1B dataset is the [Yandex research blog on billion-scale similarity search](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search).

### Available Files:
- **`base.1B.fbin`**: 1 billion training vectors (96 dimensions, float32)
- **`query.1B.fbin`**: Test query vectors (96 dimensions, float32)
- **`neighbors.1B.ibin`**: Ground truth neighbors (int32 neighbor IDs)

### Binary File Formats:
- **`.fbin`**: Binary float32 vectors (raw binary data, 4 bytes per dimension)
- **`.ibin`**: Binary int32 neighbor IDs (raw binary data, 4 bytes per ID)

## Converter Usage

### Training Data Converter (`fbin_to_parquet.py`)
```python
# Convert training data
fbin_to_parquet_chunked_with_id(
    input_fbin_file='base.1B.fbin',
    output_prefix='train',
    chunk_size_vectors=10_000_000,  # 10M vectors per file
    dimensions=96,
    output_dir='deep1b_parquet'
)
```

### Test & Neighbors Converter (`test_neighbors_converter.py`)
```python
# Convert both test and neighbors files
convert_test_and_neighbors(
    test_fbin_file='query.1B.fbin',
    neighbors_ibin_file='neighbors.1B.ibin', 
    output_dir='deep1b_parquet',
    dimensions=96,
    k=100  # Number of neighbors per query in ground truth
)
```

### Output Structure:
```
deep1b_parquet/
├── train_0.parquet     # Training vectors (chunk 0)
├── train_1.parquet     # Training vectors (chunk 1)
├── ...
├── test.parquet        # Query vectors
└── neighbours.parquet  # Ground truth neighbors
```

## Configuration Examples

### PgVector HNSW Configuration
```yaml
pgvectorhnsw:
  case_type: Performance96D1B
  m: 16                    # HNSW connections per node
  ef_construction: 128     # Build-time candidate exploration
  ef_search: 128          # Search-time candidate exploration
  deep1b_dataset_percentage: 0.02  # Use 2% of dataset
  create_index_before_load: true
  num_concurrency: 30
```

### Key Relationships in Config:
- `ef_search: 128` ≥ `k: 100` ✓ (valid)
- `ef_construction: 128` ≥ `2 * m: 32` ✓ (valid)
- Ground truth has `k=100` neighbors per query

### Milvus HNSW Configuration
```python
HNSWConfig(
    M=16,                # Same as 'm' in other systems
    efConstruction=128,  # Build-time parameter
    ef=128               # Search-time parameter (same as ef_search)
)
```

## Parameter Verification

### Check Ground Truth Format:
```python
import os
file_size = os.path.getsize('neighbors.1B.ibin')
estimated_queries = file_size // (100 * 4)  # Assuming k=100, 4 bytes per int32
print(f"Estimated number of queries: {estimated_queries}")
```

### Verify Configuration Constraints:
```python
# Check HNSW constraints
assert ef_search >= k, f"ef_search ({ef_search}) must be >= k ({k})"
assert ef_construction >= 2 * m, f"ef_construction ({ef_construction}) must be >= 2*m ({2*m})"
```

## Performance Impact

### Parameter Tuning Guidelines:

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|-----------------|
| `k` | More results returned | Fewer results, faster evaluation |
| `ef_search` | Better recall, slower search | Faster search, lower recall |
| `m` | Better recall, more memory | Less memory, lower recall |
| `ef_construction` | Better index quality, slower build | Faster build, lower quality |

### Recommended Starting Values:
- **`k`**: 100 (benchmark standard)
- **`ef_search`**: 128-256 (good balance)
- **`m`**: 16-32 (good balance)
- **`ef_construction`**: 128-512 (build quality vs time)

## Troubleshooting

### Common Issues:

1. **KeyError: 'id'**: Dataset format mismatch - ensure using `id` + array format
2. **ef_search < k**: Invalid configuration - increase `ef_search`
3. **Memory issues**: Reduce `ef_construction` or `m`, or use smaller test set
4. **Low recall**: Increase `ef_search`, `m`, or `ef_construction`

### Validation Checklist:
- [ ] `ef_search ≥ k`
- [ ] `ef_construction ≥ 2 * m`
- [ ] Test and neighbors files have matching query counts
- [ ] All parquet files use `id` + array format
- [ ] Ground truth `k` matches search `k`

---

*This reference is based on VectorDB Benchmark codebase analysis and Deep1B dataset from Yandex Research.*
