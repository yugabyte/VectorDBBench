#!/bin/bash

# Fix for VectorDBBench Memory Allocation Error
# This script sets up environment variables to reduce memory usage when processing Deep1B dataset

echo "=== VectorDBBench Memory Error Fix ==="
echo ""

# Set the Deep1B dataset percentage to a smaller value
# Default is 1.0 (100%), which requires ~125 GiB of RAM
# Common values for testing:
# - 0.01 = 1% (~1.25 GiB)
# - 0.05 = 5% (~6.25 GiB) 
# - 0.1 = 10% (~12.5 GiB)
# - 0.2 = 20% (~25 GiB)

export DEEP1B_DATASET_PERCENTAGE=0.1

echo "Setting DEEP1B_DATASET_PERCENTAGE to $DEEP1B_DATASET_PERCENTAGE (${DEEP1B_DATASET_PERCENTAGE}0% of dataset)"
echo "This will reduce memory usage from ~125 GiB to ~12.5 GiB"
echo ""

# Optional: Set log level to INFO to see memory usage warnings
export LOG_LEVEL=INFO

echo "Environment variables set:"
echo "  DEEP1B_DATASET_PERCENTAGE=$DEEP1B_DATASET_PERCENTAGE"
echo "  LOG_LEVEL=$LOG_LEVEL"
echo ""

# Show the current memory available
if command -v free >/dev/null 2>&1; then
    echo "Current system memory:"
    free -h
elif command -v vm_stat >/dev/null 2>&1; then
    echo "Current system memory (macOS):"
    vm_stat | head -4
fi

echo ""
echo "You can now run your VectorDBBench commands."
echo "The changes will apply for the current terminal session."
echo ""
echo "To make this permanent, add the following to your ~/.bashrc or ~/.zshrc:"
echo "export DEEP1B_DATASET_PERCENTAGE=0.1"
echo ""
echo "Alternative usage: Run your command directly with the environment variable:"
echo "DEEP1B_DATASET_PERCENTAGE=0.1 python -m vectordb_bench ..."
