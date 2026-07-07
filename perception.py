"""MULTIMODAL on top of the FROZEN spin LM -- built as an ADAPTER, not a retrain (LLaVA-style).

The spin LM stays exactly as trained. A small `PerceptionAdapter` maps a perception feature vector (from ANY
image/audio encoder) into the LM's token-embedding space, as a few 'perception tokens' that prepend to the text
stream and flow through `SpinAttentionLM.forward_embeds`. Only the adapter trains; the LM is frozen.

What is live NOW: the architecture + the end-to-end interface (feature -> perception tokens -> combined stream
-> logits). What is NOT yet done: the adapter is untrained, so the model does not yet 'see' -- making it see is
a later GPU job (a real vision encoder + image-text pairs to train just this adapter). We build the socket now,
not the eye. `encode_image_stub` is a deterministic placeholder encoder so the pipeline runs end-to-end today.
"""
import numpy as np
import torch
import torch.nn as nn


class PerceptionAdapter(nn.Module):
    """Project a perception feature [B, feat_dim] into `n_tokens` embeddings in the LM's d-space [B, n_tokens, d].
    Prepended to text embeddings, these are the model's 'view' of a non-text input. Frozen LM, trainable adapter."""

    def __init__(self, feat_dim, d, n_tokens=4):
        super().__init__()
        self.n_tokens, self.d = n_tokens, d
        self.proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d * n_tokens))

    def forward(self, feat):                       # feat [B, feat_dim] -> [B, n_tokens, d]
        return self.proj(feat).view(feat.shape[0], self.n_tokens, self.d)


def encode_image_stub(img, feat_dim=64):
    """Placeholder 'encoder': pool an image (H,W or H,W,C array) into a fixed intensity-grid feature. Deterministic,
    no training -- it exists so the multimodal pipeline is runnable end-to-end today. A real vision encoder
    (CLIP / ViT) drops in here later; the adapter above is what learns to align its features with the LM."""
    a = np.asarray(img, dtype="float32")
    if a.ndim == 3: a = a.mean(-1)
    H, W = a.shape
    gh, gw = max(1, H // 8), max(1, W // 8)
    grid = [a[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() for i in range(8) for j in range(8)]
    v = np.array((grid + [0.0] * feat_dim)[:feat_dim], dtype="float32")
    return v / (np.linalg.norm(v) + 1e-6)
