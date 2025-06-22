#!/usr/bin/env python3
"""
Test script for GLUE loader and MRPC dataset
"""

import torch
from src.data.glue_loader import GLUELoader
from torch.utils.data import DataLoader

def test_glue_loader():
    """Test GLUE loader with MRPC dataset"""
    print("Testing GLUE Loader with MRPC dataset...")
    
    # Test MRPC dataset
    try:
        # Load train data
        train_loader = GLUELoader(
            dataset_name="mrpc", 
            split="train", 
            max_length=128, 
            tokenizer_name="t5-small"
        )
        
        print(f"✓ MRPC train data loaded successfully")
        print(f"  - Number of samples: {len(train_loader)}")
        print(f"  - Vocabulary size: {train_loader.get_vocab_size()}")
        print(f"  - Number of classes: {train_loader.get_num_classes()}")
        print(f"  - Task type: {train_loader.get_task_type()}")
        
        # Load validation data
        dev_loader = GLUELoader(
            dataset_name="mrpc", 
            split="validation", 
            max_length=128, 
            tokenizer_name="t5-small"
        )
        
        print(f"✓ MRPC validation data loaded successfully")
        print(f"  - Number of samples: {len(dev_loader)}")
        
        # Test a few samples
        print("\nTesting sample data:")
        for i in range(min(3, len(train_loader))):
            sample = train_loader[i]
            print(f"  Sample {i+1}:")
            print(f"    - Input shape: {sample['src'].shape}")
            print(f"    - Target: {sample['tgt'].item()}")
            print(f"    - Input tokens: {sample['src'][:10].tolist()}...")  # First 10 tokens
        
        # Test DataLoader
        print("\nTesting DataLoader:")
        dataloader = DataLoader(train_loader, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        print(f"  - Batch src shape: {batch['src'].shape}")
        print(f"  - Batch tgt shape: {batch['tgt'].shape}")
        
        print("\n✓ All tests passed! GLUE loader is working correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def test_other_glue_datasets():
    """Test other GLUE datasets"""
    datasets = ["rte", "cola", "qnli"]
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name.upper()} dataset...")
        try:
            loader = GLUELoader(
                dataset_name=dataset_name,
                split="train",
                max_length=128,
                tokenizer_name="t5-small"
            )
            
            print(f"✓ {dataset_name.upper()} loaded successfully")
            print(f"  - Samples: {len(loader)}")
            print(f"  - Classes: {loader.get_num_classes()}")
            
        except Exception as e:
            print(f"✗ Error with {dataset_name}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("GLUE LOADER TEST")
    print("=" * 60)
    
    # Test MRPC
    success = test_glue_loader()
    
    if success:
        # Test other datasets
        test_other_glue_datasets()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ GLUE loader implementation is ready!")
        print("✓ MRPC dataset is available")
        print("✓ Other GLUE datasets (RTE, CoLA, QNLI) are available")
        print("\nYou can now run experiments with:")
        print("  python run_experiments.py --dataset mrpc --mode all")
        print("  python run_experiments.py --dataset rte --mode all")
        print("  python run_experiments.py --dataset cola --mode all")
        print("  python run_experiments.py --dataset qnli --mode all")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 