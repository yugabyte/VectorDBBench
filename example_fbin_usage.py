#!/usr/bin/env python3
"""
Example of how to use the modified DEEP1B reader with different file types.

This example shows how to modify the DEEP1B_URL to use different file formats.
"""

# Example 1: Using an HDF5 file with environment variable (recommended)
"""
Set environment variable:

export DEEP1B_URL="http://ann-benchmarks.com/deep-image-96-angular.hdf5"

The system will:
1. Read DEEP1B_URL from environment variable
2. Log: "Using DEEP1B_URL from environment variable: http://..."
3. Detect the .hdf5 extension
4. Download the file if not present
5. Use h5py to read "train" and "test" datasets
6. Convert to train_XX.parquet and test_XX.parquet files
"""

# Example 2: Using an FBIN file with environment variable (recommended)
"""
Set environment variable:

export DEEP1B_URL="http://example.com/dataset.fbin"

The system will:
1. Read DEEP1B_URL from environment variable
2. Log: "Using DEEP1B_URL from environment variable: http://..."
3. Detect the .fbin extension  
4. Download the file if not present
5. Read the binary format:
   - 4 bytes: number of vectors (int32)
   - 4 bytes: number of dimensions (int32) 
   - Vector data: vectors × dimensions × 4 bytes (float32)
6. Split into 90% train / 10% test
7. Convert to train_XX.parquet and test_XX.parquet files
"""

# Example 3: Using default URL (when no environment variable is set)
"""
If no DEEP1B_URL environment variable is set:

The system will:
1. Use the default URL from DEFAULT_DEEP1B_URL
2. Log: "Using default DEEP1B_URL (no environment variable set): http://..."
3. Process the file normally
"""

# Example 4: How to set up for your own dataset
"""
Method 1 - Environment Variable (Recommended):

export DEEP1B_URL="https://your-domain.com/your-dataset.fbin"

Method 2 - .env file:

echo "DEEP1B_URL=https://your-domain.com/your-dataset.fbin" >> .env

Method 3 - Code modification:

Edit vectordb_bench/backend/data_source.py:
   
   DEFAULT_DEEP1B_URL = "https://your-domain.com/your-dataset.fbin"

Ensure your .fbin file follows the format:
- Header: 8 bytes (2 int32 values)
  - First 4 bytes: number of vectors
  - Next 4 bytes: number of dimensions
- Data: vector_count × dimension_count × 4 bytes (float32)

Run your VectorDBBench tests as normal - the conversion will happen automatically
"""

# Example 4: File format detection
"""
The file type is detected automatically from the URL extension:

- "dataset.hdf5" -> HDF5 format (uses h5py)
- "dataset.fbin" -> Binary format (uses struct + numpy)
- Other extensions -> Error with supported formats listed

No code changes needed beyond updating the URL!
"""

# Example 5: Percentage sampling still works
"""
Environment variable or task configuration:
DEEP1B_DATASET_PERCENTAGE=0.1  # Use 10% of data

This works with both file formats:
- HDF5: Samples first 10% of train/test datasets
- FBIN: Samples first 10% after 90/10 train/test split
"""

if __name__ == "__main__":
    print("This is an example file showing how to use the modified DEEP1B reader.")
    print("See the comments above for usage instructions.")
    print("\nTo use different file formats:")
    print("1. Edit DEEP1B_URL in vectordb_bench/backend/data_source.py")
    print("2. Point to your .hdf5 or .fbin file")
    print("3. Run VectorDBBench normally")
    print("\nThe file type detection and conversion happens automatically!")
