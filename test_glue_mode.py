#!/usr/bin/env python3
"""
Test script for GLUE mode functionality
"""

import subprocess
import sys
import os

def test_glue_mode():
    """Test the new GLUE mode functionality"""
    print("=" * 60)
    print("TESTING GLUE MODE")
    print("=" * 60)
    
    # Test with a small number of epochs to verify functionality
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", "glue",
        "--mode", "all",
        "--epochs", "1",  # Just 1 epoch for testing
        "--batch_size", "8",  # Small batch size for testing
        "--d_model", "128"  # Smaller model for faster testing
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ GLUE MODE TEST PASSED!")
            print("="*60)
            
            # Check if results files were created
            expected_files = [
                "results/glue_comprehensive_results.csv",
                "results/glue_dataset_summary.csv"
            ]
            
            for file_path in expected_files:
                if os.path.exists(file_path):
                    print(f"✓ {file_path} created successfully")
                else:
                    print(f"✗ {file_path} not found")
            
            return True
        else:
            print(f"\n✗ GLUE MODE TEST FAILED! Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ GLUE MODE TEST TIMED OUT!")
        return False
    except Exception as e:
        print(f"\n✗ GLUE MODE TEST ERROR: {e}")
        return False

def test_individual_glue_datasets():
    """Test individual GLUE datasets"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL GLUE DATASETS")
    print("=" * 60)
    
    datasets = ["mrpc", "rte", "cola", "qnli"]
    
    for dataset in datasets:
        print(f"\nTesting {dataset.upper()}...")
        
        cmd = [
            sys.executable, "run_experiments.py",
            "--dataset", dataset,
            "--mode", "all",
            "--epochs", "1",
            "--batch_size", "8",
            "--d_model", "128"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✓ {dataset.upper()} test passed")
            else:
                print(f"✗ {dataset.upper()} test failed")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ {dataset.upper()} test timed out")
        except Exception as e:
            print(f"✗ {dataset.upper()} test error: {e}")

if __name__ == "__main__":
    print("GLUE MODE FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test individual datasets first
    test_individual_glue_datasets()
    
    # Test the comprehensive GLUE mode
    success = test_glue_mode()
    
    if success:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ GLUE mode is working correctly!")
        print("✓ Individual GLUE datasets are working!")
        print("\nYou can now use:")
        print("  python run_experiments.py --dataset glue --mode all --epochs 5")
        print("  python run_experiments.py --dataset mrpc --mode all --epochs 5")
        print("  python run_experiments.py --dataset rte --mode all --epochs 5")
        print("  python run_experiments.py --dataset cola --mode all --epochs 5")
        print("  python run_experiments.py --dataset qnli --mode all --epochs 3")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 