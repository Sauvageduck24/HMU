import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for sequence models.
    Adds position information to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts (B, L, D) or (L, D)
        if x.dim() == 2:
            # (L, D) -> (1, L, D)
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Input to PositionalEncoding must be 3D, got shape {x.shape}")
        if x.size(1) <= self.pe.size(1):
            # batch_first: (B, L, d_model)
            x = x + self.pe[:, :x.size(1), :]
        else:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds maximum positional encoding length {self.pe.size(1)}.")
        return x

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism.
    """
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
        # query/key/value: (B, L, D)
        B, Lq, _ = query.size()
        _, Lk, _ = key.size()
        # Linear projections and split into heads
        Q = self.q_linear(query).view(B, Lq, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            # mask: (B, 1, 1, Lk) or broadcastable
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(context)

class FeedForward(nn.Module):
    """
    Position-wise feedforward network used in Transformer blocks.
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Single encoder layer: self-attention + feedforward + normalization.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(src2))
        # Feedforward block
        src2 = self.ff(src)
        src = self.norm2(src + self.dropout2(src2))
        return src

class DecoderLayer(nn.Module):
    """
    Single decoder layer: self-attention, cross-attention, feedforward, normalization.
    """
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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention on target
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # Cross-attention with encoder memory
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # Feedforward
        tgt2 = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

class Encoder(nn.Module):
    """
    Stacks multiple encoder layers.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        if src.dim() == 2:
            src = src.unsqueeze(0)
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

class Decoder(nn.Module):
    """
    Stacks multiple decoder layers.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)
    
class StandardTransformer(nn.Module):
    """
    Standard Transformer model implementation (no memory module).
    Follows the theoretical pipeline in the pseudocode.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        latent_dim: int = 256,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layer = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # 1) Ensure batch-first
        src = self._ensure_batch_first(src)
        tgt = self._ensure_batch_first(tgt)
        # 2) Embedding if input is indices
        src_embed = self._maybe_embed(src)          # (B, L, D)
        tgt_embed = self._maybe_embed(tgt)          # (B, L, D)
        # 3) Positional encodings
        src_embed = self.pos_encoder(src_embed)
        tgt_embed = self.pos_decoder(tgt_embed)
        # 4) Encoder → Decoder
        memory = self.encoder_layer(src_embed)      # (B, L, D)
        dec_out = self.decoder(tgt_embed, memory)   # (B, L, D)
        # 5) Output logits
        return self.output_proj(dec_out)            # (B, L, vocab)

    def encode(self, src):
        # src: (L,) or (B, L) of token indices (LongTensor)
        if src.dim() == 1:
            src = src.unsqueeze(0)
        if not torch.is_floating_point(src):
            src_embed = self.embedding(src)  # (B, L, D)
            src_embed = self.pos_encoder(src_embed)
        else:
            src_embed = src  # already embedded
        v = self.encoder_layer(src_embed)  # (B, L, D)
        return v

    def decode(self, tgt, memory):
        # tgt: (L,) or (B, L) of token indices (LongTensor)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        if not torch.is_floating_point(tgt):
            tgt_embed = self.embedding(tgt)
            tgt_embed = self.pos_decoder(tgt_embed)
        else:
            tgt_embed = tgt
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        dec_out = self.decoder(tgt_embed, memory)
        return self.output_proj(dec_out)
    
    # Internal utilities
    def _ensure_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures x has shape (B, L) [long] or (B, L, D) [float].
        - (L,)    → (1, L)
        """
        return x.unsqueeze(0) if x.dim() == 1 else x

    def _maybe_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        If x is Long → applies embedding;  If x is float (already embedded) → returns as is.
        """
        return self.embedding(x) if x.dtype == torch.long else x