import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

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
    # Convert training data (1B vectors)
    # fbin_to_parquet_chunked_with_id('base.1B.fbin', 'train')
    
    # Convert test data (10K query vectors)
    fbin_to_test_parquet('query.public.10K.fbin', 'test.parquet')
    
    # Convert neighbors data (10K ground truth neighbor lists)
    ibin_to_neighbors_parquet('groundtruth.public.10K.ibin', 'neighbors.parquet')
