"""MULTIMODAL on top of the FROZEN spin LM -- built as an ADAPTER, not a retrain (LLaVA-style).

The spin LM stays exactly as trained. A modality encoder turns an image / audio clip into a few 'perception
tokens' in the LM's d-space; those prepend to the text stream and flow through `SpinAttentionLM.forward_embeds`.
Only the encoder (+ a thin align layer) trains; the LM is frozen.

OUR OWN ENCODERS (no external CLIP/Whisper stacked): the encoder backbone is the SAME spin recurrence that is
the LM's core token-mixer -- a diagonal-complex LRU run as a parallel scan (see s6_hybrid.SpinBlock). A sequence
of image PATCHES or audio MEL-FRAMES is just a sequence, so the spin mixer encodes it in O(T) (linear, O(1)/step
at inference) instead of an attention encoder's O(T^2). Front-ends are pure DSP, not models: patchify for vision,
log-mel for voice (the standard speech input -- a fixed transform, not a pretrained net). So the *learned* part
is entirely ours, and one mechanism (spin) spans text, vision and voice.

Status: encoders + end-to-end interface are LIVE and CPU-smoke-tested (see train_perception.py). They start COLD
(from-scratch, unlike a pretrained CLIP), so real perception needs a paired-data training budget -- that is the
Phase-C GPU job, to run AFTER the 1B text model lands. `PerceptionAdapter` / `encode_image_stub` are kept for
back-compat with the original stub pipeline.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from s6_hybrid import SpinBlock          # reuse the REAL spin token-mixer as the encoder backbone


# ----------------------------------------------------------------------------------------------------------
# shared spin backbone + pooling
# ----------------------------------------------------------------------------------------------------------
class SpinEncoder(nn.Module):
    """A stack of spin blocks over a feature sequence [B,T,d]. If `bidir`, alternate layers scan right-to-left
    (flip in, flip out) so the stack gets a bidirectional receptive field -- right for offline image/audio
    understanding. `bidir=False` keeps it causal (streaming, O(1)/frame)."""
    def __init__(self, d, n_layer=4, mlp_mult=4, bidir=True):
        super().__init__()
        self.blocks = nn.ModuleList([SpinBlock(d, mlp_mult) for _ in range(n_layer)])
        self.bidir = bidir

    def forward(self, x):
        for i, b in enumerate(self.blocks):
            if self.bidir and (i % 2 == 1):
                x = b(x.flip(1)).flip(1)          # odd layers see the sequence reversed
            else:
                x = b(x)
        return x


class AttnPool(nn.Module):
    """Learned-query attention pool: `n_tokens` query vectors cross-attend to the encoded sequence -> a fixed
    set of `n_tokens` perception tokens (Perceiver/attention-pool style), independent of the input length."""
    def __init__(self, d, n_tokens=8, n_head=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, seq):                        # seq [B,T,d] -> [B,n_tokens,d]
        B = seq.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, seq, seq)
        return self.ln(out)


def _sinusoidal_pos(T, d, device, dtype):
    """Fixed sinusoidal positions -> handles variable-length sequences (audio) with no learned max length."""
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang[:, : (d - d // 2)])
    return pe.to(dtype)


# ----------------------------------------------------------------------------------------------------------
# VISION: patchify -> spin encoder -> perception tokens
# ----------------------------------------------------------------------------------------------------------
class SpinVisionEncoder(nn.Module):
    """Image -> perception tokens, spin-based (our own, no CLIP). Conv patch-embed (ViT-style, a linear
    projection of non-overlapping patches) -> learned positions -> bidirectional spin encoder -> attention pool."""
    def __init__(self, d, n_tokens=8, img_size=128, patch=16, in_ch=3, n_layer=4, mlp_mult=4, n_head=8):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch"
        self.img_size, self.patch = img_size, patch
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)   # [B,d,H/p,W/p]
        self.pos = nn.Parameter(torch.randn(1, n_patches, d) * 0.02)
        self.encoder = SpinEncoder(d, n_layer, mlp_mult, bidir=True)
        self.pool = AttnPool(d, n_tokens, n_head)

    def forward(self, img):                        # img [B,in_ch,img_size,img_size] in ~[0,1] -> [B,n_tokens,d]
        h = self.patch_embed(img)                  # [B,d,gh,gw]
        h = h.flatten(2).transpose(1, 2)           # [B,n_patches,d]
        h = h + self.pos
        h = self.encoder(h)
        return self.pool(h)


# ----------------------------------------------------------------------------------------------------------
# VOICE: log-mel front-end -> spin encoder -> perception tokens
# ----------------------------------------------------------------------------------------------------------
def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-style triangular mel filterbank [n_mels, n_fft//2+1]. Pure DSP -- no learned params, no external
    model (this is the same fixed transform every speech system uses as its input)."""
    fmax = fmax or sr / 2
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    f = mel2hz(m)
    bins = np.floor((n_fft + 1) * f / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype="float32")
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l: fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[i - 1, k] = (r - k) / (r - c)
    return torch.from_numpy(fb)


class SpinVoiceEncoder(nn.Module):
    """Audio waveform -> perception tokens, spin-based (our own, no Whisper). log-mel front-end -> per-frame
    linear embed -> sinusoidal positions -> bidirectional spin encoder -> attention pool."""
    def __init__(self, d, n_tokens=8, sr=16000, n_fft=400, hop=160, n_mels=80,
                 n_layer=4, mlp_mult=4, n_head=8, bidir=True):
        super().__init__()
        self.sr, self.n_fft, self.hop, self.n_mels = sr, n_fft, hop, n_mels
        self.register_buffer("mel_fb", _mel_filterbank(sr, n_fft, n_mels), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.frame_embed = nn.Linear(n_mels, d)
        self.encoder = SpinEncoder(d, n_layer, mlp_mult, bidir=bidir)
        self.pool = AttnPool(d, n_tokens, n_head)

    def log_mel(self, wav):                        # wav [B, n_samples] -> [B, n_frames, n_mels]
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window.to(wav.device),
                          return_complex=True)                       # [B, n_fft/2+1, n_frames]
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.matmul(self.mel_fb.to(wav.device), power)        # [B, n_mels, n_frames]
        return torch.log(mel.clamp_min(1e-10)).transpose(1, 2)       # [B, n_frames, n_mels]

    def forward(self, wav):                        # wav [B, n_samples] -> [B, n_tokens, d]
        h = self.log_mel(wav)
        h = self.frame_embed(h)
        h = h + _sinusoidal_pos(h.shape[1], h.shape[2], h.device, h.dtype)[None]
        h = self.encoder(h)
        return self.pool(h)


# ----------------------------------------------------------------------------------------------------------
# back-compat: the original stub pipeline (kept so nothing that imported these breaks)
# ----------------------------------------------------------------------------------------------------------
class PerceptionAdapter(nn.Module):
    """Project a perception feature [B, feat_dim] into `n_tokens` embeddings in the LM's d-space [B, n_tokens, d]."""
    def __init__(self, feat_dim, d, n_tokens=4):
        super().__init__()
        self.n_tokens, self.d = n_tokens, d
        self.proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d * n_tokens))

    def forward(self, feat):
        return self.proj(feat).view(feat.shape[0], self.n_tokens, self.d)


def encode_image_stub(img, feat_dim=64):
    """Deterministic 8x8 intensity-grid placeholder (superseded by SpinVisionEncoder; kept for back-compat)."""
    a = np.asarray(img, dtype="float32")
    if a.ndim == 3: a = a.mean(-1)
    H, W = a.shape
    gh, gw = max(1, H // 8), max(1, W // 8)
    grid = [a[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean() for i in range(8) for j in range(8)]
    v = np.array((grid + [0.0] * feat_dim)[:feat_dim], dtype="float32")
    return v / (np.linalg.norm(v) + 1e-6)
