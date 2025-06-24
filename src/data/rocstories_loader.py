import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import List, Dict

class ROCStoriesLoader:
    """
    Data loader for the ROCStories dataset using HuggingFace Datasets and T5 Tokenizer.
    This class prepares the dataset for story generation or understanding tasks.
    """
    def __init__(self, split='train', max_length=32, tokenizer_name='t5-small'):
        """
        Initialize the ROCStoriesLoader.
        Args:
            split (str): Dataset split to load ('train', 'test', etc.).
            max_length (int): Maximum sequence length for tokenization.
            tokenizer_name (str): Name of the pretrained tokenizer to use.
        """
        self.dataset = load_dataset('mintujupally/ROCStories', split=split)  # Load the specified split of ROCStories
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
            # The dataset may have fields 'sentence1', ..., 'sentence5'
            sentences = [example.get(f'sentence{i}', '') for i in range(1, 6)]  # Extract sentences
            story = ' '.join(sentences).strip()  # Concatenate sentences to form the story
            src_text = story  # Use the story as both input and target
            tgt_text = story
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