import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm

class GLUELoader:
    """
    General loader for GLUE benchmark datasets using HuggingFace datasets.
    Supports: MRPC, RTE, CoLA, QNLI, SST-2, etc.
    """
    
    def __init__(self, dataset_name: str, split: str = "train", max_length: int = 128, tokenizer_name: str = "t5-small", max_samples: int = None):
        """
        Args:
            dataset_name: "mrpc", "rte", "cola", "qnli", "sst2", etc.
            split: "train", "validation", or "test"
            max_length: Maximum sequence length
            tokenizer_name: HuggingFace tokenizer name
            max_samples: Maximum number of samples to load (None for all)
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.max_samples = max_samples
        
        # Dataset configurations for each GLUE task
        self.dataset_configs = {
            "mrpc": {
                "hf_name": "glue",
                "hf_config": "mrpc",
                "num_classes": 2,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": None  # Use all samples
            },
            "rte": {
                "hf_name": "glue",
                "hf_config": "rte",
                "num_classes": 2,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": None  # Use all samples
            },
            "cola": {
                "hf_name": "glue",
                "hf_config": "cola",
                "num_classes": 2,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": None  # Use all samples
            },
            "qnli": {
                "hf_name": "glue",
                "hf_config": "qnli",
                "num_classes": 2,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": 10000  # Limit QNLI by default
            },
            "sst2": {
                "hf_name": "glue",
                "hf_config": "sst2",
                "num_classes": 2,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": None  # Use all samples
            },
            "mnli": {
                "hf_name": "glue",
                "hf_config": "mnli",
                "num_classes": 3,
                "task_type": "classification",
                "has_labels": True,
                "default_max_samples": 15000  # Limit MNLI by default (it's very large)
            }
        }
        
        # Check if the dataset is supported
        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.dataset_configs.keys())}")
        
        self.config = self.dataset_configs[self.dataset_name]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad token exists
        
        # Load data
        self.data = self._load_glue_data()
        
    def _load_glue_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load GLUE data using HuggingFace datasets and convert to tensor format."""
        print(f"Loading {self.dataset_name.upper()} dataset from HuggingFace...")
        
        # Handle special split names for MNLI
        actual_split = self.split
        if self.dataset_name == "mnli":
            if self.split == "validation":
                # For MNLI, use validation_matched as the default validation split
                actual_split = "validation_matched"
                print(f"MNLI: Using {actual_split} split (default validation split)")
            elif self.split == "test":
                # For MNLI test, use test_matched as the default test split
                actual_split = "test_matched"
                print(f"MNLI: Using {actual_split} split (default test split)")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(
            self.config["hf_name"], 
            self.config["hf_config"],
            split=actual_split
        )
        
        # Limit samples if specified by user or default config
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples")
        elif self.config["default_max_samples"] is not None:
            dataset = dataset.select(range(min(self.config["default_max_samples"], len(dataset))))
            print(f"Using default limit of {len(dataset)} samples")
        
        data = []
        print(f"Processing {actual_split} split of {self.dataset_name.upper()} with {len(dataset)} samples...")
        
        # Process all samples with progress bar
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
            # Handle different dataset formats for each GLUE task
            if self.dataset_name == "mrpc":
                sentence1 = example['sentence1']
                sentence2 = example['sentence2']
                label = example['label']
                # Combine sentences for classification
                text = f"{sentence1} [SEP] {sentence2}"
            elif self.dataset_name == "rte":
                sentence1 = example['sentence1']
                sentence2 = example['sentence2']
                label = example['label']
                text = f"{sentence1} [SEP] {sentence2}"
            elif self.dataset_name == "cola":
                sentence = example['sentence']
                label = example['label']
                text = sentence
            elif self.dataset_name == "qnli":
                question = example['question']
                sentence = example['sentence']
                label = example['label']
                text = f"{question} [SEP] {sentence}"
            elif self.dataset_name == "sst2":
                sentence = example['sentence']
                label = example['label']
                text = sentence
            elif self.dataset_name == "mnli":
                premise = example['premise']
                hypothesis = example['hypothesis']
                label = example['label']
                # Combine premise and hypothesis for classification
                text = f"{premise} [SEP] {hypothesis}"
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
            # Tokenize the input text with padding and truncation
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Create sample dictionary with input and label tensors
            sample = {
                'src': encoding['input_ids'].squeeze(0),  # Input tensor
                'tgt': torch.tensor(label, dtype=torch.long)  # Target label tensor
            }
            data.append(sample)
        
        print(f"Loaded {len(data)} samples from {self.dataset_name.upper()} {actual_split} split")
        return data
    
    def get_data(self) -> List[Dict[str, torch.Tensor]]:
        """Return the loaded data."""
        return self.data
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def get_num_classes(self) -> int:
        """Return number of classes for this dataset."""
        return self.config["num_classes"]
    
    def get_task_type(self) -> str:
        """Return task type for this dataset."""
        return self.config["task_type"]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx] 