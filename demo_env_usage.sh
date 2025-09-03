#!/bin/bash

# Demo script showing how to use DEEP1B_URL environment variable

echo "=== DEEP1B_URL Environment Variable Usage Demo ==="
echo

echo "1. Using default URL (no environment variable set):"
echo "   The system will use: http://ann-benchmarks.com/deep-image-96-angular.hdf5"
echo "   Log message: 'Using default DEEP1B_URL (no environment variable set): http://...'"
echo

echo "2. Setting DEEP1B_URL to use an FBIN file:"
echo "   export DEEP1B_URL=\"http://example.com/dataset.fbin\""
echo "   The system will use: http://example.com/dataset.fbin"
echo "   Log message: 'Using DEEP1B_URL from environment variable: http://example.com/dataset.fbin'"
echo

echo "3. Setting DEEP1B_URL to use a different HDF5 file:"
echo "   export DEEP1B_URL=\"http://example.com/custom-dataset.hdf5\""
echo "   The system will use: http://example.com/custom-dataset.hdf5"
echo "   Log message: 'Using DEEP1B_URL from environment variable: http://example.com/custom-dataset.hdf5'"
echo

echo "4. Using a .env file:"
echo "   echo 'DEEP1B_URL=http://example.com/dataset.fbin' >> .env"
echo "   The system will automatically read from .env file"
echo

echo "=== Commands to try ==="
echo

echo "# Set environment variable for current session:"
echo "export DEEP1B_URL=\"http://example.com/my-dataset.fbin\""
echo

echo "# Add to .env file for persistent configuration:"
echo "echo 'DEEP1B_URL=http://example.com/my-dataset.fbin' >> .env"
echo

echo "# Run VectorDBBench (it will automatically use the environment variable):"
echo "python -m vectordb_bench run --help"
echo

echo "# Check what URL is being used:"
echo "python -c \"import os; print('DEEP1B_URL:', os.getenv('DEEP1B_URL', 'NOT_SET'))\""
echo

echo "=== File Format Support ==="
echo "Supported file extensions:"
echo "  .hdf5 - HDF5 format with 'train' and 'test' datasets"
echo "  .fbin - Binary format with header + vector data"
echo

echo "The system automatically detects the file type and uses the appropriate conversion method."
