import concurrent
import logging
import math
import multiprocessing as mp
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from vectordb_bench.backend.dataset import DatasetManager

from ... import config
from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
from ...models import LoadTimeoutError, PerformanceTimeoutError
from .. import utils
from ..clients import api

NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = 10
WAITTING_TIME = 60

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize

    def task(self) -> int:
        count = 0
        with self.db.init():
            log.info(f"({mp.current_process().name:16}) Start inserting embeddings in batch {config.NUM_PER_BATCH}")
            start = time.perf_counter()
            for data_df in self.dataset:
                all_metadata = data_df["id"].tolist()

                emb_np = np.stack(data_df["emb"])
                if self.normalize:
                    log.debug("normalize the 100k train data")
                    all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
                else:
                    all_embeddings = emb_np.tolist()
                del emb_np
                log.debug(f"batch dataset size: {len(all_embeddings)}, {len(all_metadata)}")

                insert_count, error = self.db.insert_embeddings(
                    embeddings=all_embeddings,
                    metadata=all_metadata,
                )
                if error is not None:
                    raise error

                assert insert_count == len(all_metadata)
                count += insert_count
                if count % 100_000 == 0:
                    log.info(f"({mp.current_process().name:16}) Loaded {count} embeddings into VectorDB")

            log.info(
                f"({mp.current_process().name:16}) Finish loading all dataset into VectorDB, "
                f"dur={time.perf_counter() - start}"
            )
            return count

    def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i + left_id for i in all_metadata]

            num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
            log.info(
                f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
                f"embeddings in batch {NUM_PER_BATCH}"
            )
            count = 0
            for batch_id in range(num_batches):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Start inserting {len(metadata)} embeddings"
                )
                while retry_count < LOAD_MAX_TRY_COUNT:
                    insert_count, error = self.db.insert_embeddings(
                        embeddings=embeddings[already_insert_count:],
                        metadata=metadata[already_insert_count:],
                    )
                    already_insert_count += insert_count
                    if error is not None:
                        retry_count += 1
                        time.sleep(WAITTING_TIME)

                        log.info(f"Failed to insert data, try {retry_count} time")
                        if retry_count >= LOAD_MAX_TRY_COUNT:
                            raise error
                    else:
                        break
                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Finish inserting {len(metadata)} embeddings"
                )

                assert already_insert_count == len(metadata)
                count += already_insert_count
            log.info(
                f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
                f"batch {NUM_PER_BATCH}"
            )
        return count

    @utils.time_it
    def _insert_all_batches(self) -> int:
        """Performance case only"""
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"),
            max_workers=1,
        ) as executor:
            future = executor.submit(self.task)
            try:
                count = future.result(timeout=self.timeout)
            except TimeoutError as e:
                msg = f"VectorDB load dataset timeout in {self.timeout}"
                log.warning(msg)
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(msg) from e
            except Exception as e:
                log.warning(f"VectorDB load dataset error: {e}")
                raise e from e
            else:
                return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = next(iter(self.dataset))
        all_embeddings, all_metadata = (
            np.stack(data_df["emb"]).tolist(),
            data_df["id"].tolist(),
        )

        start_time = time.perf_counter()
        max_load_count, times = 0, 0
        try:
            while time.perf_counter() - start_time < self.timeout:
                count = self.endless_insert_data(
                    all_embeddings,
                    all_metadata,
                    left_id=max_load_count,
                )
                max_load_count += count
                times += 1
                log.info(
                    f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
                    f"{max_load_count}"
                )
        except Exception as e:
            log.info(
                f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
                f"{max_load_count}, err={e}"
            )
            traceback.print_exc()
            return max_load_count
        else:
            raise LoadTimeoutError(self.timeout)

    def run(self) -> int:
        count, dur = self._insert_all_batches()
        return count


class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: pd.DataFrame,
        k: int = 100,
        filters: dict | None = None,
    ):
        self.db = db
        self.k = k
        self.filters = filters

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def search(self, args: tuple[list, pd.DataFrame]) -> tuple[float, float, float]:
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            test_data, ground_truth = args
            ideal_dcg = get_ideal_dcg(self.k)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {ground_truth.columns}, shape: {ground_truth.shape}")

            latencies, recalls, ndcgs = [], [], []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                try:
                    results = self.db.search_embedding(
                        emb,
                        self.k,
                        self.filters,
                    )

                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

                latencies.append(time.perf_counter() - s)

                gt = ground_truth["neighbors_id"][idx]
                recalls.append(calc_recall(self.k, gt[: self.k], results))
                ndcgs.append(calc_ndcg(gt[: self.k], results, ideal_dcg))

                if len(latencies) % 100 == 0:
                    log.debug(
                        f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                        f"latest_latency={latencies[-1]}, latest recall={recalls[-1]}"
                    )

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls), 4)
        avg_ndcg = round(np.mean(ndcgs), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_ndcg={avg_ndcg},"
            f"avg_latency={avg_latency}, "
            f"p99={p99}"
        )
        return (avg_recall, avg_ndcg, p99)

    def _run_in_subprocess(self) -> tuple[float, float]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            return future.result()

    @utils.time_it
    def run(self) -> tuple[float, float, float]:
        """
        Returns:
            tuple[tuple[float, float, float], float]: (avg_recall, avg_ndcg, p99_latency), cost

        """
        return self._run_in_subprocess()


class MultiProcessingInsertRunner:
    """Parallel insertion runner for Deep1B dataset using multiple processes"""
    
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool = False,
        timeout: float | None = None,
        max_workers: int | None = None,
    ):
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.timeout = timeout
        # Use Deep1B specific worker count or fall back to config
        self.max_workers = max_workers or config.DEEP1B_LOAD_WORKERS
        
    @staticmethod
    def _load_single_file(db_config: dict, file_path: str, data_dir: str, normalize: bool) -> tuple[int, float]:
        """Worker function to load a single parquet file"""
        import pathlib
        import time
        from pyarrow import ParquetFile
        import numpy as np
        from vectordb_bench.backend.clients import api
        
        log.info(f"[Worker {mp.current_process().name}] Starting to load file: {file_path}")
        start_time = time.perf_counter()
        
        try:
            # Initialize database connection for this worker
            # Create a copy to avoid modifying the original config
            db_params = db_config['db_params'].copy()
            db = db_config['db_class'](**db_params)
            
            full_path = pathlib.Path(data_dir) / file_path
            if not full_path.exists():
                log.error(f"File not found: {full_path}")
                return 0, 0.0
                
            # Get batch size for Deep1B
            batch_size = 1_000_000  # Deep1B uses larger batches
            
            with db.init():
                count = 0
                # Use ParquetFile with memory mapping for efficiency
                parquet_file = ParquetFile(full_path, memory_map=True, pre_buffer=True)
                
                for batch in parquet_file.iter_batches(batch_size):
                    data_df = batch.to_pandas()
                    all_metadata = data_df["id"].tolist()
                    
                    emb_np = np.stack(data_df["emb"])
                    if normalize:
                        all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
                    else:
                        all_embeddings = emb_np.tolist()
                    del emb_np
                    
                    # Insert with retry logic
                    retry_count = 0
                    while retry_count < LOAD_MAX_TRY_COUNT:
                        insert_count, error = db.insert_embeddings(
                            embeddings=all_embeddings,
                            metadata=all_metadata,
                        )
                        if error is not None:
                            retry_count += 1
                            time.sleep(WAITTING_TIME)
                            log.warning(f"[Worker {mp.current_process().name}] Failed to insert data from {file_path}, retry {retry_count}")
                            if retry_count >= LOAD_MAX_TRY_COUNT:
                                raise error
                        else:
                            break
                    
                    count += insert_count
                    log.debug(f"[Worker {mp.current_process().name}] Loaded {count} embeddings from {file_path}")
                
            duration = time.perf_counter() - start_time
            log.info(f"[Worker {mp.current_process().name}] Completed loading {file_path}: {count:,} embeddings in {duration:.2f}s")
            return count, duration
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            log.error(f"[Worker {mp.current_process().name}] Failed to load {file_path}: {e}")
            raise e
    
    def run(self) -> int:
        """Run parallel loading for Deep1B dataset"""
        if self.dataset.data.name != "Deep1B":
            log.warning("MultiProcessingInsertRunner is optimized for Deep1B dataset. Using serial loading for other datasets.")
            # Fall back to serial loading for non-Deep1B datasets
            serial_runner = SerialInsertRunner(self.db, self.dataset, self.normalize, self.timeout)
            return serial_runner.run()
        
        if len(self.dataset.train_files) <= 1:
            log.info("Single training file detected, using serial loading")
            serial_runner = SerialInsertRunner(self.db, self.dataset, self.normalize, self.timeout)
            return serial_runner.run()
        
        log.info(f"Starting parallel loading of Deep1B dataset with {self.max_workers} workers")
        log.info(f"Files to load: {len(self.dataset.train_files)} files")
        
        # Prepare database configuration for workers
        db_config = {
            'db_class': type(self.db),
            'db_params': self.db.__dict__.copy()
        }
        
        total_count = 0
        start_time = time.perf_counter()
        
        try:
            with concurrent.futures.ProcessPoolExecutor(
                mp_context=mp.get_context("spawn"),
                max_workers=self.max_workers
            ) as executor:
                # Submit all file loading tasks
                future_to_file = {
                    executor.submit(
                        self._load_single_file,
                        db_config,
                        file_name,
                        str(self.dataset.data_dir),
                        self.normalize
                    ): file_name
                    for file_name in self.dataset.train_files
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_file, timeout=self.timeout):
                    file_name = future_to_file[future]
                    try:
                        count, duration = future.result()
                        total_count += count
                        log.info(f"Completed loading {file_name}: {count:,} embeddings")
                    except Exception as e:
                        log.error(f"Failed to load {file_name}: {e}")
                        # Cancel remaining tasks and re-raise
                        for f in future_to_file:
                            f.cancel()
                        raise e
                        
        except concurrent.futures.TimeoutError as e:
            msg = f"Deep1B parallel loading timeout after {self.timeout}s"
            log.error(msg)
            raise PerformanceTimeoutError(msg) from e
        except Exception as e:
            log.error(f"Deep1B parallel loading failed: {e}")
            raise e
        
        total_duration = time.perf_counter() - start_time
        log.info(f"Deep1B parallel loading completed: {total_count:,} embeddings in {total_duration:.2f}s")
        log.info(f"Average loading rate: {total_count / total_duration:.0f} embeddings/second")
        
        return total_count
