import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import List, Dict

class CommonGenLoader:
    """
    Data loader for the CommonGen dataset using HuggingFace Datasets and T5 Tokenizer.
    This class prepares the dataset for text generation tasks, encoding both concepts and target sentences.
    """
    def __init__(self, split='train', max_length=32, tokenizer_name='t5-small'):
        """
        Initialize the CommonGenLoader.
        Args:
            split (str): Dataset split to load ('train', 'validation', etc.).
            max_length (int): Maximum sequence length for tokenization.
            tokenizer_name (str): Name of the pretrained tokenizer to use.
        """
        self.dataset = load_dataset('allenai/common_gen', split=split)  # Load the specified split of CommonGen
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)  # Initialize the tokenizer
        self.max_length = max_length

    def get_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize the dataset and return a list of dictionaries with input and target tensors.
        Returns:
            List[Dict[str, torch.Tensor]]: List of tokenized samples with input and target ids.
        """
        data = []
        for example in self.dataset:
            src_text = ', '.join(example['concepts'])  # Join concepts as input string
            tgt_text = example['target']  # Target sentence
            # Tokenize both input and target
            src_enc = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            tgt_enc = self.tokenizer(tgt_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            data.append({
                'src': src_enc['input_ids'].squeeze(0),  # Input tensor
                'tgt': tgt_enc['input_ids'].squeeze(0)   # Target tensor
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