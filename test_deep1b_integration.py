#!/usr/bin/env python3
"""
Test script to verify Deep1B dataset integration
"""

import sys
import pathlib

# Add the project root to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from vectordb_bench.backend.dataset import Dataset
from vectordb_bench.backend.cases import CaseType, type2case


def test_deep1b_dataset():
    """Test that Deep1B dataset can be created and accessed"""
    print("Testing Deep1B dataset integration...")
    
    # Test dataset creation
    try:
        deep1b_dataset = Dataset.DEEP1B.get(1_000_000_000)
        print(f"✓ Deep1B dataset created successfully")
        print(f"  - Name: {deep1b_dataset.name}")
        print(f"  - Size: {deep1b_dataset.size}")
        print(f"  - Dimensions: {deep1b_dataset.dim}")
        print(f"  - Metric Type: {deep1b_dataset.metric_type}")
        print(f"  - Label: {deep1b_dataset.label}")
        print(f"  - Directory Name: {deep1b_dataset.dir_name}")
        print(f"  - File Count: {deep1b_dataset.file_count}")
    except Exception as e:
        print(f"✗ Failed to create Deep1B dataset: {e}")
        return False
    
    # Test dataset manager
    try:
        deep1b_manager = Dataset.DEEP1B.manager(1_000_000_000)
        print(f"✓ Deep1B dataset manager created successfully")
        print(f"  - Data directory: {deep1b_manager.data_dir}")
    except Exception as e:
        print(f"✗ Failed to create Deep1B dataset manager: {e}")
        return False
    
    # Test case creation
    try:
        case = CaseType.Performance96D1B.case_cls()
        print(f"✓ Deep1B case created successfully")
        print(f"  - Case ID: {case.case_id}")
        print(f"  - Name: {case.name}")
        print(f"  - Description: {case.description[:100]}...")
        print(f"  - Load Timeout: {case.load_timeout}")
        print(f"  - Optimize Timeout: {case.optimize_timeout}")
    except Exception as e:
        print(f"✗ Failed to create Deep1B case: {e}")
        return False
    
    print("✓ All Deep1B integration tests passed!")
    return True


def test_dataset_enum():
    """Test that Deep1B is properly added to the Dataset enum"""
    print("\nTesting Dataset enum...")
    
    # Check if DEEP1B is in the enum
    if hasattr(Dataset, 'DEEP1B'):
        print(f"✓ DEEP1B found in Dataset enum")
    else:
        print(f"✗ DEEP1B not found in Dataset enum")
        return False
    
    # Check if it's properly mapped
    try:
        deep1b_class = Dataset.DEEP1B.value
        print(f"✓ DEEP1B class: {deep1b_class}")
    except Exception as e:
        print(f"✗ Failed to access DEEP1B class: {e}")
        return False
    
    return True


def test_case_enum():
    """Test that Performance96D1B is properly added to the CaseType enum"""
    print("\nTesting CaseType enum...")
    
    # Check if Performance96D1B is in the enum
    if hasattr(CaseType, 'Performance96D1B'):
        print(f"✓ Performance96D1B found in CaseType enum")
    else:
        print(f"✗ Performance96D1B not found in CaseType enum")
        return False
    
    # Check if it's properly mapped in type2case
    if CaseType.Performance96D1B in type2case:
        print(f"✓ Performance96D1B found in type2case mapping")
    else:
        print(f"✗ Performance96D1B not found in type2case mapping")
        return False
    
    return True


if __name__ == "__main__":
    print("Deep1B Dataset Integration Test")
    print("=" * 40)
    
    success = True
    success &= test_deep1b_dataset()
    success &= test_dataset_enum()
    success &= test_case_enum()
    
    if success:
        print("\n🎉 All tests passed! Deep1B integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the integration.")
        sys.exit(1) 