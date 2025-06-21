import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import List, Dict

class CommonGenLoader:
    def __init__(self, split='train', max_length=32, tokenizer_name='t5-small'):
        self.dataset = load_dataset('allenai/common_gen', split=split)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def get_data(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        for example in self.dataset:
            src_text = ', '.join(example['concepts'])
            tgt_text = example['target']
            src_enc = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            tgt_enc = self.tokenizer(tgt_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            data.append({
                'src': src_enc['input_ids'].squeeze(0),
                'tgt': tgt_enc['input_ids'].squeeze(0)
            })
        return data

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_tokenizer(self):
        return self.tokenizer 