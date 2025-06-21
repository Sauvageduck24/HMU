import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Asegura que la entrada tenga 3 dimensiones (B, L, d_model)
        added_batch = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            added_batch = True
        B, Lq, _ = query.size()
        B, Lk, _ = key.size()
        # Linear projections
        Q = self.q_linear(query).view(B, Lq, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, Lq, d_k)
        K = self.k_linear(key).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, nhead, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (B, nhead, Lq, d_k)
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(context)
        if added_batch:
            out = out.squeeze(0)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (B, S, d_model)
        attn_mask = src_mask
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (B, S) -> (B, 1, 1, S)
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        src2 = self.self_attn(src, src, src, attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ff(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        attn_mask = tgt_mask
        if tgt_key_padding_mask is not None:
            attn_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        cross_mask = memory_mask
        if memory_key_padding_mask is not None:
            cross_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)
        tgt2 = self.cross_attn(tgt, memory, memory, cross_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ff(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
        return self.norm(src)

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(tgt)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, d_model) or (L, B, d_model)
        if x.dim() != 3:
            raise ValueError(f"Input to PositionalEncoding must be 3D, got shape {x.shape}")
        if x.size(1) <= self.max_len:
            # batch_first: (B, L, d_model)
            x = x + self.pe[:, :x.size(1), :]
        elif x.size(0) <= self.max_len:
            # not batch_first: (L, B, d_model)
            x = x + self.pe[:, :x.size(0), :].transpose(0, 1)
        else:
            raise ValueError(f"Input sequence length {x.size(1)} or {x.size(0)} exceeds maximum positional encoding length {self.max_len}.")
        return self.dropout(x)

class StandardTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
        vocab_size: int = 1000,
        max_len: int = 5000,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        # Asegura que src y tgt sean 3D (batch, seq_len, d_model)
        if src.dim() == 2:
            src = src.unsqueeze(0)
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.output_projection(output)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        if src.dim() == 2:
            src = src.unsqueeze(0)
        src = self.pos_encoder(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.output_projection(output) 