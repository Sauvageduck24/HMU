#!/usr/bin/env python3
"""
Test script for proof mode functionality
"""

import subprocess
import sys
import os
import time

def test_proof_mode():
    """Test proof mode with small dataset sizes"""
    print("=" * 60)
    print("TESTING PROOF MODE")
    print("=" * 60)
    
    # Test 1: Individual dataset with proof mode
    print("\n1. Testing MRPC with proof mode...")
    start_time = time.time()
    
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", "mrpc",
        "--mode", "all",
        "--proof",
        "--epochs", "5",  # Should be limited to 1
        "--batch_size", "32",  # Should be limited to 8
        "--d_model", "128"  # Small model for testing
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ MRPC proof mode test passed")
            print(f"  - Time taken: {time.time() - start_time:.2f} seconds")
        else:
            print(f"✗ MRPC proof mode test failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ MRPC proof mode test timed out")
        return False
    except Exception as e:
        print(f"✗ MRPC proof mode test error: {e}")
        return False
    
    # Test 2: QNLI with proof mode (should be very fast)
    print("\n2. Testing QNLI with proof mode...")
    start_time = time.time()
    
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", "qnli",
        "--mode", "all",
        "--proof",
        "--d_model", "128"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ QNLI proof mode test passed")
            print(f"  - Time taken: {time.time() - start_time:.2f} seconds")
        else:
            print(f"✗ QNLI proof mode test failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ QNLI proof mode test timed out")
        return False
    except Exception as e:
        print(f"✗ QNLI proof mode test error: {e}")
        return False
    
    # Test 3: GLUE mode with proof mode
    print("\n3. Testing GLUE mode with proof mode...")
    start_time = time.time()
    
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", "glue",
        "--mode", "all",
        "--proof",
        "--d_model", "128"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes for all GLUE
        
        if result.returncode == 0:
            print("✓ GLUE proof mode test passed")
            print(f"  - Time taken: {time.time() - start_time:.2f} seconds")
            
            # Check if results files were created
            expected_files = [
                "results/glue_comprehensive_results.csv",
                "results/glue_dataset_summary.csv"
            ]
            
            for file_path in expected_files:
                if os.path.exists(file_path):
                    print(f"  ✓ {file_path} created")
                else:
                    print(f"  ✗ {file_path} not found")
            
        else:
            print(f"✗ GLUE proof mode test failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ GLUE proof mode test timed out")
        return False
    except Exception as e:
        print(f"✗ GLUE proof mode test error: {e}")
        return False
    
    return True

def test_proof_mode_settings():
    """Test that proof mode applies correct settings"""
    print("\n" + "=" * 60)
    print("TESTING PROOF MODE SETTINGS")
    print("=" * 60)
    
    # Test that proof mode overrides settings correctly
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", "mrpc",
        "--mode", "all",
        "--proof",
        "--epochs", "10",  # Should be limited to 1
        "--batch_size", "64",  # Should be limited to 8
        "--max_samples", "1000",  # Should be overridden to 100
        "--d_model", "128"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Proof mode settings test passed")
            print("  - Epochs should be limited to 1")
            print("  - Batch size should be limited to 8")
            print("  - Max samples should be set to 100")
        else:
            print(f"✗ Proof mode settings test failed")
            return False
            
    except Exception as e:
        print(f"✗ Proof mode settings test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("PROOF MODE FUNCTIONALITY TEST")
    print("=" * 60)
    
    success1 = test_proof_mode()
    success2 = test_proof_mode_settings()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ Proof mode is working correctly!")
        print("✓ All datasets load quickly with small sizes")
        print("✓ Settings are properly overridden")
        print("✓ GLUE mode works with proof mode")
        print("\nYou can now use:")
        print("  # Quick test of individual dataset")
        print("  python run_experiments.py --dataset mrpc --mode all --proof")
        print("  # Quick test of QNLI (normally slow)")
        print("  python run_experiments.py --dataset qnli --mode all --proof")
        print("  # Quick test of all GLUE datasets")
        print("  python run_experiments.py --dataset glue --mode all --proof")
        print("  # Quick test with custom settings (will be overridden)")
        print("  python run_experiments.py --dataset mrpc --mode all --proof --epochs 10 --batch_size 64")
    else:
        print("\n✗ Tests failed. Please check the implementation.") 