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

def test_sequence_generation(models, evaluator):
    """Test the quality of generated sequences."""
    standard_transformer, htransformer = models
    
    # Generate test sequences
    seq_len = 100
    vocab_size = 1000
    test_sequences = generate_test_sequence(seq_len, vocab_size, batch_size=10)
    reference_sequences = generate_test_sequence(seq_len, vocab_size, batch_size=10)
    
    # Evaluate both models
    standard_results = evaluator.evaluate_imagination(
        standard_transformer,
        test_sequences,
        reference_sequences
    )
    
    htransformer_results = evaluator.evaluate_imagination(
        htransformer,
        test_sequences,
        reference_sequences
    )
    
    # Compare results
    assert htransformer_results['coherence'] >= standard_results['coherence'], \
        "HTransformer should generate more coherent sequences"
    
    assert htransformer_results['diversity'] >= standard_results['diversity'], \
        "HTransformer should generate more diverse sequences"
    
    assert htransformer_results['perplexity'] <= standard_results['perplexity'], \
        "HTransformer should have lower perplexity (better quality)"

def test_future_prediction(models, evaluator):
    """Test the ability to predict future states."""
    standard_transformer, htransformer = models
    
    # Generate sequences with future states
    seq_len = 50
    vocab_size = 1000
    num_sequences = 10
    
    all_results = []
    for _ in range(num_sequences):
        # Generate input sequence
        input_seq = generate_test_sequence(seq_len, vocab_size)
        
        # Generate future states
        future_states = generate_test_sequence(seq_len, vocab_size)
        
        # Evaluate both models
        with torch.no_grad():
            standard_output = standard_transformer(input_seq.unsqueeze(0))
            htransformer_output = htransformer(input_seq.unsqueeze(0))
        
        standard_accuracy = evaluator.compute_sequence_accuracy(
            standard_output,
            future_states.unsqueeze(0)
        )
        
        htransformer_accuracy = evaluator.compute_sequence_accuracy(
            htransformer_output,
            future_states.unsqueeze(0)
        )
        
        all_results.append((standard_accuracy, htransformer_accuracy))
    
    # Compare average accuracies
    standard_avg = np.mean([r[0] for r in all_results])
    htransformer_avg = np.mean([r[1] for r in all_results])
    
    assert htransformer_avg >= standard_avg, \
        "HTransformer should have better future state prediction accuracy"

def test_counterfactual_reasoning(models, evaluator):
    """Test the ability to generate counterfactual scenarios."""
    standard_transformer, htransformer = models
    
    # Generate base sequences
    seq_len = 50
    vocab_size = 1000
    num_scenarios = 10
    
    all_results = []
    for _ in range(num_scenarios):
        # Generate base sequence
        base_seq = generate_test_sequence(seq_len, vocab_size)
        
        # Generate counterfactual target
        counterfactual_target = generate_test_sequence(seq_len, vocab_size)
        
        # Evaluate both models
        with torch.no_grad():
            standard_output = standard_transformer(base_seq.unsqueeze(0))
            htransformer_output = htransformer(base_seq.unsqueeze(0))
        
        # Compute quality metrics
        standard_metrics = evaluator.compute_imagination_quality(
            standard_output,
            counterfactual_target.unsqueeze(0)
        )
        
        htransformer_metrics = evaluator.compute_imagination_quality(
            htransformer_output,
            counterfactual_target.unsqueeze(0)
        )
        
        all_results.append((standard_metrics, htransformer_metrics))
    
    # Compare average metrics
    for metric in ['coherence', 'diversity']:
        standard_avg = np.mean([r[0][metric] for r in all_results])
        htransformer_avg = np.mean([r[1][metric] for r in all_results])
        
        assert htransformer_avg >= standard_avg, \
            f"HTransformer should have better {metric} in counterfactual scenarios"

def test_creative_generation(models, evaluator):
    """Test the ability to generate creative and novel sequences."""
    standard_transformer, htransformer = models
    
    # Generate seed sequences
    seq_len = 50
    vocab_size = 1000
    num_seeds = 10
    
    all_results = []
    for _ in range(num_seeds):
        # Generate seed sequence
        seed_seq = generate_test_sequence(seq_len, vocab_size)
        
        # Generate multiple variations
        num_variations = 5
        standard_variations = []
        htransformer_variations = []
        
        for _ in range(num_variations):
            with torch.no_grad():
                standard_output = standard_transformer(seed_seq.unsqueeze(0))
                htransformer_output = htransformer(seed_seq.unsqueeze(0))
            
            standard_variations.append(standard_output)
            htransformer_variations.append(htransformer_output)
        
        # Compute diversity between variations
        standard_diversity = evaluator._compute_sequence_diversity(
            torch.cat(standard_variations, dim=0)
        )
        
        htransformer_diversity = evaluator._compute_sequence_diversity(
            torch.cat(htransformer_variations, dim=0)
        )
        
        all_results.append((standard_diversity, htransformer_diversity))
    
    # Compare average diversities
    standard_avg = np.mean([r[0] for r in all_results])
    htransformer_avg = np.mean([r[1] for r in all_results])
    
    assert htransformer_avg >= standard_avg, \
        "HTransformer should generate more diverse and creative variations" 