import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
from typing import List, Dict, Tuple
import requests
import zipfile
from tqdm import tqdm

class SST2Loader:
    """
    Loader for Stanford Sentiment Treebank (SST-2) dataset.
    SST-2 is a binary sentiment classification dataset with ~67K sentences.
    """
    
    def __init__(self, split: str = "train", max_length: int = 128, tokenizer_name: str = "t5-small"):
        """
        Args:
            split: "train", "validation", or "test"
            max_length: Maximum sequence length
            tokenizer_name: HuggingFace tokenizer name
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad token exists
        
        # Load data
        self.data = self._load_sst2_data()
        
    def _download_sst2(self, data_dir: str = "data/sst2") -> str:
        """Download SST-2 dataset if not already present."""
        os.makedirs(data_dir, exist_ok=True)  # Create directory if needed
        
        # URLs for SST-2 (all splits in same zip)
        urls = {
            "train": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
            "dev": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
            "test": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
        }
        
        zip_path = os.path.join(data_dir, "SST-2.zip")
        
        # Download if not exists
        if not os.path.exists(zip_path):
            print("Downloading SST-2 dataset...")
            response = requests.get(urls["train"], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract
            print("Extracting SST-2 dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        
        return os.path.join(data_dir, "SST-2")
    
    def _load_sst2_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load SST-2 data and convert to tensor format."""
        data_dir = self._download_sst2()  # Ensure data is downloaded
        
        # File paths for each split
        if self.split == "train":
            file_path = os.path.join(data_dir, "train.tsv")
        elif self.split == "validation":
            file_path = os.path.join(data_dir, "dev.tsv")
        else:  # test
            file_path = os.path.join(data_dir, "test.tsv")
        
        # Read TSV file into DataFrame
        df = pd.read_csv(file_path, sep='\t')
        
        # For test set, labels might not be available
        if self.split == "test":
            # Create dummy labels for test set (all 0s)
            df['label'] = 0
        
        data = []
        print(f"Processing {self.split} split with {len(df)} samples...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            sentence = row['sentence']  # Extract sentence
            label = row['label']        # Extract label (0 or 1)
            
            # Tokenize sentence with padding and truncation
            encoding = self.tokenizer(
                sentence,
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
        
        print(f"Loaded {len(data)} samples from SST-2 {self.split} split")
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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx] 