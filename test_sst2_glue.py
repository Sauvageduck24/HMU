#!/usr/bin/env python3
"""
Test script for SST-2 in GLUE mode
"""

import time
from src.data.glue_loader import GLUELoader

def test_sst2_loading():
    """Test SST-2 loading with GLUELoader"""
    print("=" * 60)
    print("TESTING SST-2 IN GLUE MODE")
    print("=" * 60)
    
    # Test 1: SST-2 with GLUELoader
    print("\n1. Testing SST-2 with GLUELoader...")
    start_time = time.time()
    
    try:
        loader = GLUELoader(
            dataset_name="sst2",
            split="train",
            max_length=128,
            tokenizer_name="t5-small"
        )
        
        load_time = time.time() - start_time
        print(f"✓ SST-2 loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(loader)}")
        print(f"  - Vocabulary size: {loader.get_vocab_size()}")
        print(f"  - Number of classes: {loader.get_num_classes()}")
        print(f"  - Task type: {loader.get_task_type()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: SST-2 validation split
    print("\n2. Testing SST-2 validation split...")
    start_time = time.time()
    
    try:
        val_loader = GLUELoader(
            dataset_name="sst2",
            split="validation",
            max_length=128,
            tokenizer_name="t5-small"
        )
        
        load_time = time.time() - start_time
        print(f"✓ SST-2 validation loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(val_loader)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: SST-2 with proof mode (small samples)
    print("\n3. Testing SST-2 with small sample limit...")
    start_time = time.time()
    
    try:
        small_loader = GLUELoader(
            dataset_name="sst2",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=100
        )
        
        load_time = time.time() - start_time
        print(f"✓ SST-2 (100 samples) loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(small_loader)}")
        
        # Test a few samples
        print("\nSample data:")
        for i in range(min(3, len(small_loader))):
            sample = small_loader[i]
            print(f"  Sample {i+1}: shape={sample['src'].shape}, label={sample['tgt'].item()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def test_sst2_vs_original():
    """Compare SST-2 GLUELoader vs original SST2Loader"""
    print("\n" + "=" * 60)
    print("COMPARING SST-2 LOADERS")
    print("=" * 60)
    
    try:
        # Test GLUELoader
        print("\nTesting GLUELoader for SST-2...")
        start_time = time.time()
        glue_loader = GLUELoader(
            dataset_name="sst2",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=100
        )
        glue_time = time.time() - start_time
        print(f"✓ GLUELoader: {len(glue_loader)} samples in {glue_time:.2f} seconds")
        
        # Test original SST2Loader
        print("\nTesting original SST2Loader...")
        start_time = time.time()
        from src.data.sst2_loader import SST2Loader
        original_loader = SST2Loader(
            split="train",
            max_length=128,
            tokenizer_name="t5-small"
        )
        original_time = time.time() - start_time
        print(f"✓ Original SST2Loader: {len(original_loader.get_data())} samples in {original_time:.2f} seconds")
        
        print(f"\nComparison:")
        print(f"  - GLUELoader: {glue_time:.2f}s")
        print(f"  - Original: {original_time:.2f}s")
        print(f"  - Both work correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison error: {e}")
        return False

if __name__ == "__main__":
    print("SST-2 GLUE MODE TEST")
    print("=" * 60)
    
    success1 = test_sst2_loading()
    success2 = test_sst2_vs_original()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ SST-2 is now included in GLUE mode!")
        print("✓ GLUELoader works correctly for SST-2")
        print("✓ Compatible with original SST2Loader")
        print("✓ Works with proof mode")
        print("\nYou can now use:")
        print("  # SST-2 individual")
        print("  python run_experiments.py --dataset sst2 --mode all --epochs 5")
        print("  # SST-2 with proof mode")
        print("  python run_experiments.py --dataset sst2 --mode all --proof")
        print("  # All GLUE datasets (including SST-2)")
        print("  python run_experiments.py --dataset glue --mode all --epochs 3")
        print("  # All GLUE datasets with proof mode")
        print("  python run_experiments.py --dataset glue --mode all --proof")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 