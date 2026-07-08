"""LoRA continual learning (T5.2) -- fixes teach()'s OOM on the 1B / 8GB. Instead of backprop through the whole
model (which needs full-precision grads + optimizer state ~5.6GB for 706M), freeze the base and train tiny
low-rank adapters (a few MB): the base forward stays bf16 (~2GB), only the LoRA A/B matrices + their optimizer
live in memory. So continual weight-learning fits on the small GPU.

  from lora import inject_lora, lora_params
  n = inject_lora(brain.lm, r=8)                 # wrap the MLP/attn Linears; base frozen, LoRA trainable
  opt = torch.optim.AdamW(lora_params(brain.lm), lr=1e-4)   # optimizes only the adapters
  ... normal teach loop ...                      # gradients flow only into A/B
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen nn.Linear + a trainable low-rank update: y = W x + scale * (B @ A) x. B is zero-init, so at start
    the wrapped layer is IDENTICAL to the original (function-preserving) and only diverges as the adapter learns."""

    def __init__(self, lin: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.lin = lin
        for p in self.lin.parameters():
            p.requires_grad = False
        d_in, d_out = lin.in_features, lin.out_features
        dev, dt = lin.weight.device, lin.weight.dtype
        self.A = nn.Parameter((torch.randn(r, d_in, device=dev) * 0.01).to(dt))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=dt))
        self.scale = alpha / r

    def forward(self, x):
        return self.lin(x) + ((x @ self.A.t()) @ self.B.t()) * self.scale


def inject_lora(model, r=8, min_features=64):
    """Wrap the model's Linear layers (MLP + attention projections) with LoRA. Base frozen, adapters trainable.
    Returns how many layers were wrapped."""
    to_wrap = []                                             # collect first -- never mutate the tree while walking it
    for mod in model.modules():
        for attr, child in mod.named_children():
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_features:
                to_wrap.append((mod, attr, child))
    for mod, attr, child in to_wrap:
        setattr(mod, attr, LoRALinear(child, r))
    return len(to_wrap)


def lora_params(model):
    return [p for name, p in model.named_parameters() if name.endswith(".A") or name.endswith(".B")]


def merge_and_unload(model):
    """Fold the learned adapters back into the base weights (W += scale * B@A) and restore plain Linears, so the
    consolidated knowledge lives in the weights with no inference overhead."""
    for mod in model.modules():
        for attr, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.lin.weight += (child.scale * (child.B @ child.A)).to(child.lin.weight.dtype)
                setattr(mod, attr, child.lin)
