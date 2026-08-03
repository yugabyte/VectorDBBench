I'll help you understand how VectorDBBench uses the train, test, and neighbors dataset files. Let me explore the codebase to gather comprehensive information about this.

[4 tools called]

Now let me explore the utils module to understand how train, test, and neighbors files are composed:

[3 tools called]

Now let me search for how evaluation metrics are calculated using the neighbors data:

[2 tools called]

Now let me examine the metric.py file to get the complete picture of the evaluation metrics:

[2 tools called]

Perfect! Now I have a comprehensive understanding of how VectorDBBench uses the train, test, and neighbors dataset files. Let me create a structured explanation.

[1 tool called]

Based on my analysis of the VectorDBBench codebase, here's a detailed explanation of how the train, test, and neighbors dataset files are used:

## Dataset File Structure and Requirements

VectorDBBench uses three types of Parquet files with strict format requirements:

### 1. **Train Files** (`train.parquet` or `shuffle_train.parquet`)
**Purpose**: Contains the vector embeddings that are inserted into the vector database during benchmarking.

**Structure**:
- **Format**: Parquet files
- **Required columns**:
  - `id`: Incrementing integer (unique identifier)
  - `emb`: Array of float32 values (the vector embeddings)

**Usage**:
- **Data Loading**: Train files are processed by the `DataSetIterator` class which reads them in batches (configured by `config.NUM_PER_BATCH`)
- **Insertion Process**: The `SerialInsertRunner` extracts embeddings and metadata from train files and inserts them into the vector database using `db.insert_embeddings()`
- **Normalization**: Can optionally be normalized (L2 normalization) if the `normalize` parameter is set
- **Multiple Files**: Large datasets can be split into multiple files with naming convention `train-[index]-of-[file_count].parquet` (e.g., `train-01-of-10.parquet`)
- **Shuffled Data**: Alternative shuffled versions can be used with prefix `shuffle_train` instead of `train`

### 2. **Test Files** (`test.parquet`)
**Purpose**: Contains query vectors used to benchmark search performance and measure latency.

**Structure**:
- **Format**: Parquet files  
- **Required columns**:
  - `id`: Incrementing integer (query identifier)
  - `emb`: Array of float32 values (query vectors)

**Usage**:
- **Query Execution**: Test vectors are used by various runners (`MultiProcessingSearchRunner`, `SerialSearchRunner`) to perform searches against the populated vector database
- **Performance Testing**: Each test vector is used as a query to measure:
  - Search latency (including P99 percentile)
  - Queries per second (QPS)
  - Concurrent performance under different load levels
- **Memory Efficiency**: For concurrent testing, complete sets of test queries are prepared for each process to run independently
- **Recommendation**: Limited to ~1,000 test vectors to avoid memory pressure during concurrent testing

### 3. **Neighbors Files** (`neighbors.parquet` or variants)
**Purpose**: Contains ground truth data for evaluating search accuracy and relevance.

**Structure**:
- **Format**: Parquet files
- **Required columns**:
  - `id`: Corresponds to test query vector IDs
  - `neighbors_id`: Array of integers representing the true nearest neighbor IDs

**Usage**:
- **Accuracy Evaluation**: Used to calculate evaluation metrics after search queries:
  - **Recall**: Measures how many of the retrieved results are actually relevant (using `calc_recall()`)
  - **NDCG** (Normalized Discounted Cumulative Gain): Measures ranking quality considering the order of results (using `calc_ndcg()`)
- **Ground Truth Comparison**: For each test query, the search results are compared against the known correct neighbors
- **Filtering Support**: Different variants available:
  - `neighbors.parquet` (default)
  - `neighbors_head_1p.parquet` (top 1% filter)
  - `neighbors_tail_1p.parquet` (bottom 1% filter)

## Workflow Integration

```218:243:vectordb_bench/backend/dataset.py
file_count, use_shuffled = self.data.file_count, self.data.use_shuffled

train_files = utils.compose_train_files(file_count, use_shuffled)
all_files = train_files

gt_file, test_file = None, None
if self.data.with_gt:
    gt_file, test_file = utils.compose_gt_file(filters), "test.parquet"
    all_files.extend([gt_file, test_file])

if not self.data.is_custom:
    source.reader().read(
        dataset=self.data.dir_name.lower(),
        files=all_files,
        local_ds_root=self.data_dir,
    )

if gt_file is not None and test_file is not None:
    self.test_data = self._read_file(test_file)
    self.gt_data = self._read_file(gt_file)

prefix = "shuffle_train" if use_shuffled else "train"
self.train_files = sorted([f.name for f in self.data_dir.glob(f"{prefix}*.parquet")])
```

## Performance Testing Flow

1. **Setup Phase**: Train files are loaded and inserted into the vector database
2. **Query Phase**: Test vectors are used to perform searches with various concurrency levels
3. **Evaluation Phase**: Search results are compared against neighbors data to calculate accuracy metrics

```231:233:vectordb_bench/backend/runner/serial_runner.py
gt = ground_truth["neighbors_id"][idx]
recalls.append(calc_recall(self.k, gt[: self.k], results))
ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
```

This three-file system enables comprehensive benchmarking by separating data insertion (train), query execution (test), and accuracy evaluation (neighbors) concerns, allowing for thorough performance and accuracy assessment of vector databases.

[1 tool called]