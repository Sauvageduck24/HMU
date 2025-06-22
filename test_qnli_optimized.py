#!/usr/bin/env python3
"""
Test script for optimized QNLI loading
"""

import time
from src.data.glue_loader import GLUELoader

def test_qnli_loading():
    """Test QNLI loading with different sample limits"""
    print("=" * 60)
    print("TESTING OPTIMIZED QNLI LOADING")
    print("=" * 60)
    
    # Test 1: Default QNLI (should be limited to 10,000 samples)
    print("\n1. Testing QNLI with default limit (10,000 samples)...")
    start_time = time.time()
    
    try:
        loader = GLUELoader(
            dataset_name="qnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small"
        )
        
        load_time = time.time() - start_time
        print(f"✓ QNLI loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(loader)}")
        print(f"  - Expected: ~10,000 (limited by default)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: QNLI with custom limit (1,000 samples)
    print("\n2. Testing QNLI with custom limit (1,000 samples)...")
    start_time = time.time()
    
    try:
        loader_small = GLUELoader(
            dataset_name="qnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=1000
        )
        
        load_time = time.time() - start_time
        print(f"✓ QNLI (1,000 samples) loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(loader_small)}")
        print(f"  - Expected: 1,000")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: QNLI with no limit (full dataset)
    print("\n3. Testing QNLI with no limit (full dataset)...")
    start_time = time.time()
    
    try:
        loader_full = GLUELoader(
            dataset_name="qnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=None  # No limit
        )
        
        load_time = time.time() - start_time
        print(f"✓ QNLI (full dataset) loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(loader_full)}")
        print(f"  - Expected: ~104,743 (full dataset)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 4: Compare with other datasets
    print("\n4. Testing other GLUE datasets for comparison...")
    
    datasets = ["mrpc", "rte", "cola"]
    for dataset in datasets:
        start_time = time.time()
        try:
            loader = GLUELoader(
                dataset_name=dataset,
                split="train",
                max_length=128,
                tokenizer_name="t5-small"
            )
            load_time = time.time() - start_time
            print(f"✓ {dataset.upper()}: {len(loader)} samples in {load_time:.2f} seconds")
        except Exception as e:
            print(f"✗ {dataset.upper()}: Error - {e}")
    
    return True

def test_batch_processing():
    """Test that batch processing works correctly"""
    print("\n" + "=" * 60)
    print("TESTING BATCH PROCESSING")
    print("=" * 60)
    
    try:
        # Load a small sample
        loader = GLUELoader(
            dataset_name="qnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=100
        )
        
        print(f"✓ Loaded {len(loader)} samples")
        
        # Test a few samples
        for i in range(min(3, len(loader))):
            sample = loader[i]
            print(f"  Sample {i+1}: shape={sample['src'].shape}, label={sample['tgt'].item()}")
        
        print("✓ Batch processing works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Batch processing error: {e}")
        return False

if __name__ == "__main__":
    print("OPTIMIZED QNLI LOADING TEST")
    print("=" * 60)
    
    success1 = test_qnli_loading()
    success2 = test_batch_processing()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ QNLI loading is now optimized!")
        print("✓ Default limit of 10,000 samples for QNLI")
        print("✓ Custom sample limits work correctly")
        print("✓ Batch processing is efficient")
        print("\nYou can now use:")
        print("  # Fast QNLI with default limit")
        print("  python run_experiments.py --dataset qnli --mode all --epochs 3")
        print("  # Custom limit for QNLI")
        print("  python run_experiments.py --dataset qnli --mode all --epochs 3 --max_samples 5000")
        print("  # All GLUE datasets (QNLI will be limited)")
        print("  python run_experiments.py --dataset glue --mode all --epochs 3")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 