import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Following the Yandex reference implementation.
    
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Following the Yandex reference implementation.
    
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def process_chunk_worker(args):
    """
    Worker function for parallel processing of fbin chunks.
    This function runs in a separate process to convert a chunk of vectors to parquet.
    """
    (input_fbin_file, start_vector, num_vectors, dimensions, 
     output_file, chunk_id, total_chunks) = args
    
    try:
        start_time = time.time()
        
        # Calculate file offset (skip 8-byte header + previous vectors)
        header_size = 8  # 2 int32 values
        vector_size = dimensions * 4  # 4 bytes per float32
        file_offset = header_size + (start_vector * vector_size)
        
        # Read the specific chunk of vectors
        with open(input_fbin_file, 'rb') as f:
            f.seek(file_offset)
            chunk_data = f.read(num_vectors * vector_size)
        
        if not chunk_data or len(chunk_data) < num_vectors * vector_size:
            # Handle partial reads at end of file
            actual_vectors = len(chunk_data) // vector_size
            if actual_vectors == 0:
                return None
            num_vectors = actual_vectors
        
        # Convert to numpy array and reshape
        vectors_chunk = np.frombuffer(chunk_data, dtype='float32')
        vectors_chunk = vectors_chunk.reshape(-1, dimensions)
        
        # Create DataFrame with proper IDs
        ids = np.arange(start_vector, start_vector + len(vectors_chunk), dtype=np.int64)
        embeddings = [vec.astype('float32') for vec in vectors_chunk]
        
        df = pd.DataFrame({
            'id': ids,
            'emb': embeddings
        })
        
        # Write to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        
        elapsed = time.time() - start_time
        print(f"✅ Chunk {chunk_id:3d}/{total_chunks}: {output_file} - {len(vectors_chunk):,} vectors in {elapsed:.1f}s")
        
        return {
            'chunk_id': chunk_id,
            'output_file': output_file,
            'vector_count': len(vectors_chunk),
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        print(f"❌ Error processing chunk {chunk_id}: {e}")
        return None

def fbin_to_parquet_parallel(input_fbin_file, output_prefix, chunk_size_vectors=10_000_000, 
                           dimensions=None, output_dir='deep1b_parquet', max_workers=None):
    """
    High-performance parallel conversion of .fbin file to multiple .parquet files.
    Optimized for AWS r7i.8xlarge instance (32 cores, 256GiB RAM).
    
    Args:
        input_fbin_file (str): The path to the input .fbin file.
        output_prefix (str): The prefix for the output .parquet files (e.g., 'train').
        chunk_size_vectors (int): The number of vectors to write per .parquet file.
        dimensions (int): The dimensionality of the vectors. If None, read from fbin header.
        output_dir (str): The directory to store output parquet files.
        max_workers (int): Maximum number of parallel workers. If None, uses CPU count - 2.
    """
    try:
        if not os.path.exists(input_fbin_file):
            raise FileNotFoundError(f"The file '{input_fbin_file}' was not found.")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Read fbin header to get nvecs and dimensions
        with open(input_fbin_file, 'rb') as f:
            nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
            
        # Use dimensions from header if not provided
        if dimensions is None:
            dimensions = dim
        elif dimensions != dim:
            print(f"Warning: Provided dimensions ({dimensions}) differs from fbin header ({dim}). Using header value: {dim}")
            dimensions = dim
        
        total_vectors = nvecs
        file_size_gb = os.path.getsize(input_fbin_file) / (1024**3)
        
        # Calculate optimal parallelism for high-end AWS instances
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        if max_workers is None:
            # Optimize for different instance types
            if cpu_count >= 48:
                # r7i.12xlarge or higher (48+ cores)
                max_workers = min(cpu_count - 4, 44)  # Use 44 workers, leave 4 for system
            elif cpu_count >= 32:
                # r7i.8xlarge (32 cores)
                max_workers = min(cpu_count - 4, 28)  # Use 28 workers, leave 4 for system
            elif cpu_count >= 16:
                # Medium instances (16-31 cores)
                max_workers = min(cpu_count - 2, 24)  # Use most cores, leave 2 for system
            else:
                # Smaller instances
                max_workers = max(2, cpu_count - 1)
        
        # Calculate chunks
        total_chunks = (total_vectors + chunk_size_vectors - 1) // chunk_size_vectors
        
        print("=" * 80)
        print(f"🚀 PARALLEL FBIN TO PARQUET CONVERSION")
        print("=" * 80)
        print(f"Input file: {input_fbin_file} ({file_size_gb:.1f} GB)")
        print(f"Total vectors: {total_vectors:,} ({dimensions}D)")
        print(f"Output directory: {output_dir}")
        print(f"Chunk size: {chunk_size_vectors:,} vectors per file")
        print(f"Total chunks: {total_chunks}")
        print(f"Parallel workers: {max_workers}")
        print(f"System: {cpu_count} cores, {available_ram_gb:.1f} GB available RAM")
        print("-" * 80)
        
        # Prepare work items for parallel processing
        work_items = []
        for chunk_id in range(total_chunks):
            start_vector = chunk_id * chunk_size_vectors
            num_vectors = min(chunk_size_vectors, total_vectors - start_vector)
            output_file = os.path.join(output_dir, f"{output_prefix}_{chunk_id}.parquet")
            
            work_items.append((
                input_fbin_file, start_vector, num_vectors, dimensions,
                output_file, chunk_id, total_chunks
            ))
        
        # Process chunks in parallel
        start_time = time.time()
        completed_chunks = 0
        total_vectors_processed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all work items
            future_to_chunk = {executor.submit(process_chunk_worker, item): item 
                             for item in work_items}
            
            # Process completed chunks
            for future in as_completed(future_to_chunk):
                result = future.result()
                if result:
                    completed_chunks += 1
                    total_vectors_processed += result['vector_count']
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    progress_pct = (completed_chunks / total_chunks) * 100
                    vectors_per_sec = total_vectors_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_vectors - total_vectors_processed) / vectors_per_sec if vectors_per_sec > 0 else 0
                    
                    if completed_chunks % 5 == 0 or completed_chunks == total_chunks:
                        print(f"📊 Progress: {completed_chunks}/{total_chunks} chunks ({progress_pct:.1f}%) | "
                              f"{total_vectors_processed:,}/{total_vectors:,} vectors | "
                              f"{vectors_per_sec:,.0f} vectors/sec | ETA: {eta_seconds/60:.1f}m")
        
        total_time = time.time() - start_time
        vectors_per_sec = total_vectors_processed / total_time
        
        print("=" * 80)
        print(f"🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Vectors processed: {total_vectors_processed:,}")
        print(f"Average speed: {vectors_per_sec:,.0f} vectors/second")
        print(f"Files created: {completed_chunks}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fbin_to_parquet_chunked_with_id(input_fbin_file, output_prefix, chunk_size_vectors=10_000_000, dimensions=None, output_dir='deep1b_parquet'):
    """
    Converts a .fbin file to multiple .parquet files compatible with vectordb benchmark.
    Creates files with 'id' and 'emb' columns where 'emb' contains the full embedding vector.
    
    Follows the standard fbin format used by Yandex with header containing nvecs and dim.

    Args:
        input_fbin_file (str): The path to the input .fbin file.
        output_prefix (str): The prefix for the output .parquet files (e.g., 'train').
        chunk_size_vectors (int): The number of vectors to write per .parquet file.
        dimensions (int): The dimensionality of the vectors. If None, read from fbin header.
        output_dir (str): The directory to store output parquet files.
    """
    try:
        if not os.path.exists(input_fbin_file):
            raise FileNotFoundError(f"The file '{input_fbin_file}' was not found. Please ensure it has been downloaded.")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Read fbin header to get nvecs and dimensions
        with open(input_fbin_file, 'rb') as f:
            nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
            
        # Use dimensions from header if not provided
        if dimensions is None:
            dimensions = dim
        elif dimensions != dim:
            print(f"Warning: Provided dimensions ({dimensions}) differs from fbin header ({dim}). Using header value: {dim}")
            dimensions = dim
        
        total_vectors = nvecs
        print(f"File contains {total_vectors} vectors with {dimensions} dimensions.")
        print(f"Output files will be saved in: {output_dir}")

        chunk_count = 0
        current_id = 0
        vectors_processed = 0
        
        with open(input_fbin_file, 'rb') as f:
            # Skip the header (8 bytes: 2 int32 values)
            f.seek(8)
            
            while vectors_processed < total_vectors:
                # Calculate how many vectors to read in this chunk
                vectors_to_read = min(chunk_size_vectors, total_vectors - vectors_processed)
                
                # Read chunk data
                chunk_data = f.read(vectors_to_read * dimensions * 4)
                if not chunk_data:
                    break

                vectors_chunk = np.frombuffer(chunk_data, dtype='float32')
                vectors_chunk = vectors_chunk.reshape(-1, dimensions)
                
                vectors_processed += len(vectors_chunk)
                
                # Create DataFrame with 'id' and 'emb' columns compatible with vectordb benchmark
                ids = np.arange(current_id, current_id + len(vectors_chunk), dtype=np.int64)
                embeddings = [vec.astype('float32') for vec in vectors_chunk]
                
                df = pd.DataFrame({
                    'id': ids,
                    'emb': embeddings
                })

                output_filename = f"{output_prefix}_{chunk_count}.parquet"
                output_path = os.path.join(output_dir, output_filename)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path)

                print(f"Successfully created '{output_path}' with {len(vectors_chunk)} vectors.")
                current_id += len(vectors_chunk)
                chunk_count += 1
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fbin_to_test_parquet(input_fbin_file, output_file='test.parquet'):
    """
    Converts query.public.10K.fbin file to test.parquet compatible with vectordb benchmark.
    Creates a file with 'id' and 'emb' columns where 'emb' contains the full embedding vector.
    
    Args:
        input_fbin_file (str): The path to the input .fbin file (e.g., 'query.public.10K.fbin').
        output_file (str): The output parquet file name (default: 'test.parquet').
    """
    try:
        if not os.path.exists(input_fbin_file):
            raise FileNotFoundError(f"The file '{input_fbin_file}' was not found.")

        print(f"Converting {input_fbin_file} to {output_file}...")
        
        # Read all vectors from fbin file using Yandex reference implementation
        vectors = read_fbin(input_fbin_file)
        
        print(f"Read {len(vectors)} test vectors with {vectors.shape[1]} dimensions.")
        
        # Create DataFrame with 'id' and 'emb' columns compatible with vectordb benchmark
        ids = np.arange(len(vectors), dtype=np.int64)
        embeddings = [vec.astype('float32') for vec in vectors]
        
        df = pd.DataFrame({
            'id': ids,
            'emb': embeddings
        })

        # Convert to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)

        print(f"Successfully created '{output_file}' with {len(vectors)} test vectors.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def ibin_to_neighbors_parquet(input_ibin_file, output_file='neighbors.parquet'):
    """
    Converts groundtruth.public.10K.ibin file to neighbors.parquet compatible with vectordb benchmark.
    Creates a file with 'neighbors_id' column containing arrays of neighbor IDs.
    
    Args:
        input_ibin_file (str): The path to the input .ibin file (e.g., 'groundtruth.public.10K.ibin').
        output_file (str): The output parquet file name (default: 'neighbors.parquet').
    """
    try:
        if not os.path.exists(input_ibin_file):
            raise FileNotFoundError(f"The file '{input_ibin_file}' was not found.")

        print(f"Converting {input_ibin_file} to {output_file}...")
        
        # Read all vectors from ibin file using Yandex reference implementation
        neighbors = read_ibin(input_ibin_file)
        
        print(f"Read {len(neighbors)} neighbor lists with {neighbors.shape[1]} neighbors each.")
        
        # Create DataFrame with 'neighbors_id' column compatible with vectordb benchmark
        # Each row contains an array of neighbor IDs
        neighbors_lists = [row.astype('int32') for row in neighbors]
        
        df = pd.DataFrame({
            'neighbors_id': neighbors_lists
        })

        # Convert to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)

        print(f"Successfully created '{output_file}' with {len(neighbors)} neighbor lists.")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example Usage:
# Assuming you have downloaded the base.1B.fbin file to your local directory.
# This will split the 1B file into parquet files, each of 10M vectors,
# and save them in the 'deep1b_parquet' directory.
# Dimensions are automatically read from the fbin header.
if __name__ == "__main__":
    # Convert training data (1B vectors) - PARALLEL VERSION (Recommended for large files)
    # This uses all available CPU cores for maximum performance on r7i.8xlarge
    fbin_to_parquet_parallel('base.1B.fbin', 'train')
    
    # Convert training data (1B vectors) - SINGLE-THREADED VERSION (Fallback)
    # fbin_to_parquet_chunked_with_id('base.1B.fbin', 'train')
    
    # Convert test data (10K query vectors)
    fbin_to_test_parquet('query.public.10K.fbin', 'test.parquet')
    
    # Convert neighbors data (10K ground truth neighbor lists)
    ibin_to_neighbors_parquet('groundtruth.public.10K.ibin', 'neighbors.parquet')
