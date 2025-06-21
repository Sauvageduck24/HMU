import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random

class DataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.set_seed(seed)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def generate_synthetic_data(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        pattern_type: str = "random"
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic data with different patterns.
        
        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences
            vocab_size: Size of vocabulary
            pattern_type: Type of pattern to generate
                - "random": Random sequences
                - "temporal": Sequences with temporal dependencies
                - "hierarchical": Sequences with hierarchical structure
                - "repetitive": Sequences with repeating patterns
        
        Returns:
            List of dictionaries containing source and target sequences
        """
        data = []
        for _ in tqdm(range(num_samples), desc=f"Generating {pattern_type} synthetic data"):
            if pattern_type == "random":
                src, tgt = self._generate_random_sequence(seq_len, vocab_size)
            elif pattern_type == "temporal":
                src, tgt = self._generate_temporal_sequence(seq_len, vocab_size)
            elif pattern_type == "hierarchical":
                src, tgt = self._generate_hierarchical_sequence(seq_len, vocab_size)
            elif pattern_type == "repetitive":
                src, tgt = self._generate_repetitive_sequence(seq_len, vocab_size)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            
            data.append({
                'src': src,
                'tgt': tgt,
                'pattern_type': pattern_type
            })
        
        return data
    
    def _generate_random_sequence(self, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random sequences."""
        src = torch.randint(0, vocab_size, (seq_len,))
        tgt = torch.randint(0, vocab_size, (seq_len,))
        return src, tgt
    
    def _generate_temporal_sequence(self, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences with temporal dependencies."""
        # Create a sequence where each token depends on previous tokens
        src = torch.zeros(seq_len, dtype=torch.long)
        tgt = torch.zeros(seq_len, dtype=torch.long)
        
        # Generate initial tokens
        src[0] = torch.randint(0, vocab_size, (1,))
        tgt[0] = torch.randint(0, vocab_size, (1,))
        
        # Generate dependent tokens
        for i in range(1, seq_len):
            # Source depends on previous token
            src[i] = (src[i-1] + torch.randint(1, 5, (1,))) % vocab_size
            # Target depends on current source and previous target
            tgt[i] = (src[i] + tgt[i-1]) % vocab_size
        
        return src, tgt
    
    def _generate_hierarchical_sequence(self, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences with hierarchical structure."""
        # Create a sequence with nested patterns
        src = torch.zeros(seq_len, dtype=torch.long)
        tgt = torch.zeros(seq_len, dtype=torch.long)
        
        # Generate base pattern
        pattern_length = max(seq_len // 4, 1)  # Ensure pattern_length is at least 1
        base_pattern = torch.randint(0, vocab_size, (pattern_length,))
        
        # Repeat and modify pattern
        for i in range(0, seq_len, pattern_length):
            end_idx = min(i + pattern_length, seq_len)  # Ensure we don't exceed sequence length
            current_length = end_idx - i
            src[i:end_idx] = base_pattern[:current_length]
            tgt[i:end_idx] = (base_pattern[:current_length] + 1) % vocab_size
        
        return src, tgt
    
    def _generate_repetitive_sequence(self, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences with repeating patterns."""
        # Create a sequence with repeating patterns
        pattern_length = seq_len // 5
        pattern = torch.randint(0, vocab_size, (pattern_length,))
        
        src = pattern.repeat(seq_len // pattern_length + 1)[:seq_len]
        tgt = (pattern + 1).repeat(seq_len // pattern_length + 1)[:seq_len] % vocab_size
        
        return src, tgt
    
    def generate_semi_synthetic_data(
        self,
        base_data: List[Dict[str, torch.Tensor]],
        pattern_type: str,
        injection_probability: float = 0.3
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate semi-synthetic data by injecting patterns into real data.
        
        Args:
            base_data: List of real data samples
            pattern_type: Type of pattern to inject
            injection_probability: Probability of injecting pattern into each sample
        
        Returns:
            List of dictionaries containing modified sequences
        """
        modified_data = []
        for sample in tqdm(base_data, desc="Generating semi-synthetic data"):
            if random.random() < injection_probability:
                # Generate new pattern
                new_patterns = self.generate_synthetic_data(
                    num_samples=1,
                    seq_len=len(sample['src']),
                    vocab_size=torch.max(sample['src']).item() + 1,
                    pattern_type=pattern_type
                )
                new_pattern = new_patterns[0]  # Get the first (and only) pattern
                
                # Inject pattern into random position
                pos = random.randint(0, len(sample['src']) - len(new_pattern['src']))
                sample['src'][pos:pos+len(new_pattern['src'])] = new_pattern['src']
                sample['tgt'][pos:pos+len(new_pattern['tgt'])] = new_pattern['tgt']
            
            modified_data.append(sample)
        
        return modified_data
    
    def add_noise(
        self,
        data: List[Dict[str, torch.Tensor]],
        noise_type: str = "random",
        noise_level: float = 0.1
    ) -> List[Dict[str, torch.Tensor]]:
        """Add noise to the data.
        
        Args:
            data: List of data samples
            noise_type: Type of noise to add
                - "random": Random token replacement
                - "swap": Random token swapping
                - "delete": Random token deletion
            noise_level: Probability of applying noise
        
        Returns:
            List of dictionaries containing noisy sequences
        """
        noisy_data = []
        for sample in tqdm(data, desc=f"Adding {noise_type} noise"):
            src = sample['src'].clone()
            tgt = sample['tgt'].clone()
            
            if noise_type == "random":
                # Random token replacement
                mask = torch.rand(len(src)) < noise_level
                src[mask] = torch.randint(0, torch.max(src).item() + 1, (mask.sum(),))
                tgt[mask] = torch.randint(0, torch.max(tgt).item() + 1, (mask.sum(),))
            
            elif noise_type == "swap":
                # Random token swapping
                for i in range(len(src)):
                    if random.random() < noise_level:
                        j = random.randint(0, len(src)-1)
                        src[i], src[j] = src[j], src[i]
                        tgt[i], tgt[j] = tgt[j], tgt[i]
            
            elif noise_type == "delete":
                # Random token deletion (replace with padding token)
                mask = torch.rand(len(src)) < noise_level
                src[mask] = 0  # Assuming 0 is padding token
                tgt[mask] = 0
            
            noisy_data.append({
                'src': src,
                'tgt': tgt,
                'pattern_type': sample.get('pattern_type', 'noisy')
            })
        
        return noisy_data 