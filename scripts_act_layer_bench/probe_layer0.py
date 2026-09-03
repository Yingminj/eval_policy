"""Probe what the decoder's layer-0 self-attention can actually do when tgt=0."""
import sys

import torch

sys.path[:0] = ["/home/kewei/YING/ACT_rosbag2", "/home/kewei/YING/ACT_rosbag2/detr"]
from detr.models.transformer import TransformerDecoderLayer  # noqa: E402

torch.manual_seed(0)
K, B, D = 100, 2, 512
layer = TransformerDecoderLayer(D, 8, 3200, dropout=0.0).eval()
for p in layer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

tgt = torch.zeros(K, B, D)
query_pos = torch.randn(K, B, D)
memory = torch.randn(902, B, D)
pos = torch.randn(902, B, D)

# --- reproduce the first two lines of forward_post by hand ---
def self_attn_out(layer, tgt, query_pos):
    q = k = tgt + query_pos
    return layer.self_attn(q, k, value=tgt)[0]

out_init = self_attn_out(layer, tgt, query_pos)
print(f"[init]  layer0 self-attn output: max|x| = {out_init.abs().max():.3e}")

# after training the attention biases are no longer zero; simulate that
with torch.no_grad():
    layer.self_attn.in_proj_bias.normal_(0, 0.02)
    layer.self_attn.out_proj.bias.normal_(0, 0.02)
out_trained = self_attn_out(layer, tgt, query_pos)
spread = (out_trained - out_trained.mean(dim=0, keepdim=True)).abs().max()
print(f"[trained-like] max|x| = {out_trained.abs().max():.3e}, "
      f"spread across the {K} queries = {spread:.3e}")

# --- and confirm the cross-attention is where the query index starts to matter ---
tgt1 = layer.norm1(tgt + out_trained)
cross = layer.multihead_attn(query=tgt1 + query_pos, key=memory + pos, value=memory)[0]
cspread = (cross - cross.mean(dim=0, keepdim=True)).abs().max()
print(f"[cross-attn]   spread across the {K} queries = {cspread:.3e}")

# --- ablate memory: does the decoder output still depend on the sample? ---
mem_a, mem_b = torch.randn(902, B, D), torch.randn(902, B, D)
ca = layer(tgt, mem_a, pos=pos, query_pos=query_pos)
cb = layer(tgt, mem_b, pos=pos, query_pos=query_pos)
print(f"[memory swap]  max|out_a - out_b| = {(ca - cb).abs().max():.3e}")
