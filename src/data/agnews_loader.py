import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import List, Dict

class AGNewsLoader:
    """
    Data loader for the AG News dataset using HuggingFace Datasets and T5 Tokenizer.
    This class prepares the dataset for classification tasks, mapping labels to their string names.
    """
    def __init__(self, split='train', max_length=32, tokenizer_name='t5-small'):
        """
        Initialize the AGNewsLoader.
        Args:
            split (str): Dataset split to load ('train', 'test', etc.).
            max_length (int): Maximum sequence length for tokenization.
            tokenizer_name (str): Name of the pretrained tokenizer to use.
        """
        self.dataset = load_dataset('ag_news', split=split)  # Load the specified split of AG News
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)  # Initialize the tokenizer
        self.max_length = max_length
        self.label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}  # Map label ids to names

    def get_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize the dataset and return a list of dictionaries with input tensors.
        Returns:
            List[Dict[str, torch.Tensor]]: List of tokenized samples with input ids and labels.
        """
        data = []
        for example in self.dataset:
            src_text = example['text']  # Extract the news text
            label_id = example['label']  # Extract the label id
            # Tokenize the text with padding and truncation
            src_enc = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            data.append({
                'src': src_enc['input_ids'].squeeze(0),  # Input tensor
                'tgt': torch.tensor(label_id, dtype=torch.long)  # Target label tensor
            })
        return data

    def get_vocab_size(self):
        """
        Return the size of the tokenizer vocabulary.
        """
        return len(self.tokenizer)

    def get_tokenizer(self):
        """
        Return the tokenizer instance.
        """
        return self.tokenizer 