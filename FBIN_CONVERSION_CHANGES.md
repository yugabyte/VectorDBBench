# FBIN File Type Detection and Conversion Implementation

## Overview

Modified the VectorDBBench codebase to detect file types and add support for converting `.fbin` files to the `.parquet` format required by vectordbbench.

## Changes Made

### 1. Enhanced Imports in `data_source.py`

Added necessary imports for binary file handling:
```python
import struct
import numpy as np
```

### 2. New Utility Functions

#### `get_file_extension(url: str) -> str`
- Extracts file extension from URLs
- Uses `pathlib.Path` to handle URL parsing correctly
- Returns lowercase extension (e.g., ".hdf5", ".fbin")

#### `get_filename_from_url(url: str) -> str`
- Extracts filename from URL
- Handles URLs with paths correctly

#### `read_fbin_file(file_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]`
- Reads `.fbin` format files
- Handles the binary format:
  - 4 bytes: number of vectors (int32)
  - 4 bytes: number of dimensions (int32)
  - Vector data: num_vectors × num_dimensions × 4 bytes (float32)
- Splits data into train/test sets (90%/10% split)
- Returns tuple of (train_vectors, test_vectors)

### 3. Modified Deep1BReader Class

#### Enhanced `read()` method
- **File Type Detection**: Automatically detects file type from DEEP1B_URL
- **Dynamic Processing**: Routes to appropriate conversion method based on file extension
- **Support for Multiple Formats**: Handles both `.hdf5` and `.fbin` files
- **Error Handling**: Provides clear error messages for unsupported formats

#### New Private Methods

##### `_convert_hdf5_to_parquet()`
- Extracted original HDF5 conversion logic into separate method
- Maintains existing functionality for `.hdf5` files
- Applies percentage sampling consistently

##### `_convert_fbin_to_parquet()`
- Handles `.fbin` file conversion to Parquet format
- Uses `read_fbin_file()` to read binary data
- Applies percentage sampling to both train and test vectors
- Creates Polars DataFrames with required schema:
  - `id`: incrementing integer
  - `emb`: array of float32 vectors

### 4. Environment Variable Support

Added support for configuring DEEP1B_URL via environment variable:
```python
# Default DEEP1B URL - can be overridden by DEEP1B_URL environment variable
DEFAULT_DEEP1B_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"

# Get DEEP1B_URL from environment variable or use default
DEEP1B_URL = os.getenv("DEEP1B_URL", DEFAULT_DEEP1B_URL)

# Log the source of the URL
if "DEEP1B_URL" in os.environ:
    log.info(f"Using DEEP1B_URL from environment variable: {DEEP1B_URL}")
else:
    log.info(f"Using default DEEP1B_URL (no environment variable set): {DEEP1B_URL}")
```

### 5. Updated Global Variables

Changed from hard-coded filename to dynamic filename extraction:
```python
# Before
DEEP1B_HDF5_FILENAME = DEEP1B_URL.split("/")[-1]

# After  
DEEP1B_FILENAME = get_filename_from_url(DEEP1B_URL)
```

## Supported File Formats

### 1. HDF5 Files (`.hdf5`)
- **Format**: HDF5 with "train" and "test" datasets
- **Processing**: Direct reading using h5py
- **Example URL**: `http://ann-benchmarks.com/deep-image-96-angular.hdf5`

### 2. FBIN Files (`.fbin`)
- **Format**: Binary format with header + vector data
- **Structure**:
  ```
  [4 bytes: num_vectors] [4 bytes: num_dimensions] [vector_data]
  ```
- **Processing**: Custom binary reader with 90/10 train/test split
- **Example URL**: `http://example.com/dataset.fbin`

## Usage

### Changing the DEEP1B_URL

You can specify a different dataset file in two ways:

#### Method 1: Environment Variable (Recommended)
Set the `DEEP1B_URL` environment variable:

```bash
# For HDF5 files
export DEEP1B_URL="http://ann-benchmarks.com/deep-image-96-angular.hdf5"

# For FBIN files  
export DEEP1B_URL="http://example.com/my-dataset.fbin"

# Or in your .env file
echo "DEEP1B_URL=http://example.com/my-dataset.fbin" >> .env
```

#### Method 2: Modify Code Directly
Edit the `DEFAULT_DEEP1B_URL` variable in `data_source.py`:

```python
# For HDF5 files
DEFAULT_DEEP1B_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"

# For FBIN files  
DEFAULT_DEEP1B_URL = "http://example.com/my-dataset.fbin"
```

The system will automatically:
1. Check for `DEEP1B_URL` environment variable first
2. Fall back to the default URL if environment variable is not set
3. Log which source is being used (environment or default)
4. Detect the file type from the URL extension
5. Download the file if not present locally
6. Convert to Parquet format using the appropriate method
7. Apply the configured percentage sampling

### URL Source Logging

The system automatically logs which URL source is being used:

```
INFO: Using DEEP1B_URL from environment variable: http://example.com/dataset.fbin
```

or 

```
INFO: Using default DEEP1B_URL (no environment variable set): http://ann-benchmarks.com/deep-image-96-angular.hdf5
```

### Configuration

The percentage of data to use is controlled by:
- `DEEP1B_DATASET_PERCENTAGE` environment variable
- `deep1b_dataset_percentage` parameter in task configuration
- Default: 100% (1.0) of the dataset

## Error Handling

The implementation includes comprehensive error handling:

- **Unsupported file formats**: Clear error message listing supported formats
- **Invalid percentage values**: Validation that percentage is between 0.0 and 1.0
- **File reading errors**: Proper exception handling for binary file operations
- **Missing files**: Automatic download with progress indication

## Backwards Compatibility

- **Full compatibility**: Existing HDF5 functionality unchanged
- **Same API**: No changes to public method signatures
- **Same output format**: Parquet files maintain identical schema
- **Same configuration**: All existing configuration options preserved

## Testing

To test the implementation:

1. **HDF5 files**: Use existing DEEP1B_URL (default behavior)
2. **FBIN files**: Change DEEP1B_URL to point to a `.fbin` file
3. **Verification**: Check that train/test Parquet files are generated correctly

The implementation automatically handles the conversion process transparently.

## Dependencies

Required packages (already available in the project):
- `numpy`: For array operations and binary file reading
- `struct`: For binary format parsing
- `polars`: For DataFrame operations and Parquet writing
- `pathlib`: For URL/path manipulation

## Future Enhancements

Potential improvements that could be added:
- Support for additional binary formats (e.g., `.bvecs`, `.fvecs`)
- Configurable train/test split ratios for FBIN files
- Memory-efficient streaming for very large files
- Parallel processing for faster conversion
