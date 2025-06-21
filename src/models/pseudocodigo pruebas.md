## 🧠 Pseudocódigo — HMU **fuera del encoder**

```python
# INPUT: x (source), y (target)
x_embed = Embed(x) + PositionalEncoding
y_embed = Embed(shift(y)) + PositionalEncoding

# Encoder
v = TransformerEncoder(x_embed)         # v: contextual embedding

# HMU (sin gating)
v_auto = VAE(v)                          # compressed latent memory
values = MLP(concat([v, v_auto], dim=1))  # fused encoder output

# Decoder
decoder_out = TransformerDecoder(y_embed, memory=values)

# Output logits
logits = Linear(decoder_out)
```