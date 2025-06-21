import pytest
import torch
import numpy as np
from src.models.transformer import StandardTransformer
from src.models.htransformer import HTransformer
from src.evaluation.metrics import TransformerEvaluator

@pytest.fixture
def models():
    """Initialize both transformer models for testing."""
    standard_transformer = StandardTransformer(
        d_model=64,  # Smaller model for testing
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    htransformer = HTransformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        memory_size=128
    )
    
    return standard_transformer, htransformer

@pytest.fixture
def evaluator():
    """Initialize the evaluator."""
    return TransformerEvaluator()

def generate_test_sequence(seq_len: int, vocab_size: int, batch_size: int = 1) -> torch.Tensor:
    """Generate a random test sequence."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))

def test_memory_retention(models, evaluator):
    """Test memory retention capabilities of both models."""
    standard_transformer, htransformer = models
    
    # Generate test sequences
    seq_len = 100
    vocab_size = 1000
    test_sequences = generate_test_sequence(seq_len, vocab_size, batch_size=10)
    
    # Evaluate both models
    standard_results = evaluator.evaluate_long_term_memory(
        standard_transformer,
        test_sequences,
        []  # No memory tasks for this test
    )
    
    htransformer_results = evaluator.evaluate_long_term_memory(
        htransformer,
        test_sequences,
        []  # No memory tasks for this test
    )
    
    # Compare results
    for time_step in [10, 50, 100, 200]:
        standard_retention = standard_results[f'retention_{time_step}']
        htransformer_retention = htransformer_results[f'retention_{time_step}']
        
        # HTransformer should show better retention over longer time steps
        if time_step > 50:
            assert htransformer_retention >= standard_retention, \
                f"HTransformer should have better retention at time step {time_step}"

def test_memory_retrieval(models, evaluator):
    """Test memory retrieval capabilities of both models."""
    standard_transformer, htransformer = models
    
    # Generate memory tasks
    seq_len = 50
    vocab_size = 1000
    num_tasks = 10
    
    memory_tasks = []
    for _ in range(num_tasks):
        query = generate_test_sequence(seq_len, vocab_size)
        target = generate_test_sequence(seq_len, vocab_size)
        memory_tasks.append((query, target))
    
    # Generate test sequences
    test_sequences = generate_test_sequence(seq_len, vocab_size, batch_size=10)
    
    # Evaluate both models
    standard_results = evaluator.evaluate_long_term_memory(
        standard_transformer,
        test_sequences,
        memory_tasks
    )
    
    htransformer_results = evaluator.evaluate_long_term_memory(
        htransformer,
        test_sequences,
        memory_tasks
    )
    
    # Compare retrieval accuracy
    assert htransformer_results['retrieval_accuracy'] >= standard_results['retrieval_accuracy'], \
        "HTransformer should have better retrieval accuracy"

def test_sequence_length_impact(models, evaluator):
    """Test how sequence length affects memory performance."""
    standard_transformer, htransformer = models
    
    # Test different sequence lengths
    seq_lengths = [50, 100, 200, 400]
    vocab_size = 1000
    
    for seq_len in seq_lengths:
        test_sequences = generate_test_sequence(seq_len, vocab_size, batch_size=5)
        
        standard_results = evaluator.evaluate_long_term_memory(
            standard_transformer,
            test_sequences,
            []
        )
        
        htransformer_results = evaluator.evaluate_long_term_memory(
            htransformer,
            test_sequences,
            []
        )
        
        # Compare retention at different sequence lengths
        for time_step in [10, 50, 100, 200]:
            if time_step <= seq_len:
                standard_retention = standard_results[f'retention_{time_step}']
                htransformer_retention = htransformer_results[f'retention_{time_step}']
                
                # HTransformer should maintain better performance with longer sequences
                if seq_len > 100:
                    assert htransformer_retention >= standard_retention, \
                        f"HTransformer should have better retention for sequence length {seq_len} at time step {time_step}"

def test_memory_consistency(models, evaluator):
    """Test consistency of memory retention over multiple runs."""
    standard_transformer, htransformer = models
    
    # Generate test sequence
    seq_len = 100
    vocab_size = 1000
    test_sequence = generate_test_sequence(seq_len, vocab_size)
    
    # Run multiple evaluations
    num_runs = 5
    standard_retentions = []
    htransformer_retentions = []
    
    for _ in range(num_runs):
        standard_results = evaluator.evaluate_long_term_memory(
            standard_transformer,
            test_sequence.unsqueeze(0),
            []
        )
        
        htransformer_results = evaluator.evaluate_long_term_memory(
            htransformer,
            test_sequence.unsqueeze(0),
            []
        )
        
        standard_retentions.append(standard_results['retention_100'])
        htransformer_retentions.append(htransformer_results['retention_100'])
    
    # Calculate standard deviations
    standard_std = np.std(standard_retentions)
    htransformer_std = np.std(htransformer_retentions)
    
    # HTransformer should show more consistent results
    assert htransformer_std <= standard_std, \
        "HTransformer should show more consistent memory retention across runs" 