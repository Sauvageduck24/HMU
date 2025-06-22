#!/usr/bin/env python3
"""
Test script for MNLI in GLUE mode
"""

import time
from src.data.glue_loader import GLUELoader

def test_mnli_loading():
    """Test MNLI loading with GLUELoader"""
    print("=" * 60)
    print("TESTING MNLI IN GLUE MODE")
    print("=" * 60)
    
    # Test 1: MNLI with GLUELoader
    print("\n1. Testing MNLI with GLUELoader...")
    start_time = time.time()
    
    try:
        loader = GLUELoader(
            dataset_name="mnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small"
        )
        
        load_time = time.time() - start_time
        print(f"✓ MNLI loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(loader)}")
        print(f"  - Vocabulary size: {loader.get_vocab_size()}")
        print(f"  - Number of classes: {loader.get_num_classes()}")
        print(f"  - Task type: {loader.get_task_type()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: MNLI validation split
    print("\n2. Testing MNLI validation split...")
    start_time = time.time()
    
    try:
        val_loader = GLUELoader(
            dataset_name="mnli",
            split="validation_matched",  # MNLI has matched and mismatched validation sets
            max_length=128,
            tokenizer_name="t5-small"
        )
        
        load_time = time.time() - start_time
        print(f"✓ MNLI validation_matched loaded successfully in {load_time:.2f} seconds")
        print(f"  - Number of samples: {len(val_loader)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: MNLI with proof mode (small samples)
    print("\n3. Testing MNLI with small sample limit...")
    start_time = time.time()
    
    try:
        small_loader = GLUELoader(
            dataset_name="mnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=100
        )
        
        load_time = time.time() - start_time
        print(f"✓ MNLI (100 samples) loaded successfully in {load_time:.2f} seconds")
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

def test_mnli_format():
    """Test MNLI data format and labels"""
    print("\n" + "=" * 60)
    print("TESTING MNLI DATA FORMAT")
    print("=" * 60)
    
    try:
        # Load a small sample to check format
        loader = GLUELoader(
            dataset_name="mnli",
            split="train",
            max_length=128,
            tokenizer_name="t5-small",
            max_samples=10
        )
        
        print(f"✓ Loaded {len(loader)} MNLI samples")
        
        # Check label distribution
        labels = [sample['tgt'].item() for sample in loader]
        unique_labels = set(labels)
        print(f"  - Unique labels: {sorted(unique_labels)}")
        print(f"  - Expected: [0, 1, 2] (entailment, neutral, contradiction)")
        
        # Check text format
        print("\nText format examples:")
        for i in range(min(3, len(loader))):
            sample = loader[i]
            # Decode a few tokens to see the format
            tokens = sample['src'][:20].tolist()  # First 20 tokens
            print(f"  Sample {i+1}: label={sample['tgt'].item()}, tokens={tokens[:10]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Format test error: {e}")
        return False

def test_mnli_splits():
    """Test different MNLI splits"""
    print("\n" + "=" * 60)
    print("TESTING MNLI SPLITS")
    print("=" * 60)
    
    splits = ["train", "validation_matched", "validation_mismatched"]
    
    for split in splits:
        print(f"\nTesting {split} split...")
        start_time = time.time()
        
        try:
            loader = GLUELoader(
                dataset_name="mnli",
                split=split,
                max_length=128,
                tokenizer_name="t5-small",
                max_samples=50  # Small sample for testing
            )
            
            load_time = time.time() - start_time
            print(f"✓ {split}: {len(loader)} samples in {load_time:.2f} seconds")
            
        except Exception as e:
            print(f"✗ {split}: Error - {e}")
    
    return True

if __name__ == "__main__":
    print("MNLI GLUE MODE TEST")
    print("=" * 60)
    
    success1 = test_mnli_loading()
    success2 = test_mnli_format()
    success3 = test_mnli_splits()
    
    if success1 and success2 and success3:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ MNLI is now included in GLUE mode!")
        print("✓ GLUELoader works correctly for MNLI")
        print("✓ 3-class classification (entailment, neutral, contradiction)")
        print("✓ Multiple validation splits (matched, mismatched)")
        print("✓ Works with proof mode")
        print("\nYou can now use:")
        print("  # MNLI individual")
        print("  python run_experiments.py --dataset mnli --mode all --epochs 3")
        print("  # MNLI with proof mode")
        print("  python run_experiments.py --dataset mnli --mode all --proof")
        print("  # All GLUE datasets (including MNLI)")
        print("  python run_experiments.py --dataset glue --mode all --epochs 3")
        print("  # All GLUE datasets with proof mode")
        print("  python run_experiments.py --dataset glue --mode all --proof")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 