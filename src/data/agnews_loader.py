import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import List, Dict

class AGNewsLoader:
    def __init__(self, split='train', max_length=32, tokenizer_name='t5-small'):
        self.dataset = load_dataset('ag_news', split=split)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

    def get_data(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        for example in self.dataset:
            src_text = example['text']
            label_id = example['label']
            src_enc = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            data.append({
                'src': src_enc['input_ids'].squeeze(0),
                'tgt': torch.tensor(label_id, dtype=torch.long)
            })
        return data

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_tokenizer(self):
        return self.tokenizer 