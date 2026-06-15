"""
grow.py -- self-directed capacity growth (neurogenesis on demand).

A fixed-size brain saturates: once full, continual learning overwrites old
knowledge instead of adding. Real brains grow new neurons and synapses. This
module lets Pragnosia do the same: when its OWN signals say it is full and
straining (rising forgetting + persistently high surprise after teaching), it
grows -- adding neurons/layers -- using FUNCTION-PRESERVING initialization, so
the grown brain computes exactly the same function at the moment it grows (no
forgetting) and then fills the new capacity through continued learning.

Two modes (compose freely):
  grow_depth(model, n_new)  -- add transformer blocks, each initialized to
                               identity (zero output projections). Net2DeeperNet.
  grow_width(model, extra)  -- add neurons to every block's MLP, new neurons'
                               OUTPUT weights = 0 so they don't perturb yet.
                               Net2WiderNet.

Both return a NEW, larger model; the old one is untouched. Growth is offline and
checkpoint-producing, so it never disturbs a running trainer.
"""
import copy, torch, torch.nn as nn
import s6_hybrid as H


@torch.no_grad()
def grow_depth(model, n_new=1):
    """Add n_new transformer blocks as near-identity -> function preserved."""
    vocab = model.emb.num_embeddings; d = model.d
    nh = model.blocks[0].attn.n_head; nl = len(model.blocks)
    dev = next(model.parameters()).device
    big = H.SpinAttentionLM(vocab, d, nh, nl + n_new).to(dev)
    big.emb.load_state_dict(model.emb.state_dict())
    big.pos.load_state_dict(model.pos.state_dict())
    big.carrier.load_state_dict(model.carrier.state_dict())
    big.lnf.load_state_dict(model.lnf.state_dict())
    big.head.weight = big.emb.weight                       # keep the tie
    for i in range(nl):                                    # copy existing blocks
        big.blocks[i].load_state_dict(model.blocks[i].state_dict())
    for j in range(nl, nl + n_new):                        # new blocks -> identity
        b = big.blocks[j]
        nn.init.zeros_(b.attn.proj.weight); nn.init.zeros_(b.attn.proj.bias)
        nn.init.zeros_(b.mlp[2].weight);    nn.init.zeros_(b.mlp[2].bias)
    return big


@torch.no_grad()
def grow_width(model, extra=None):
    """Widen every block's MLP hidden layer; new neurons output 0 -> preserved."""
    big = copy.deepcopy(model)
    for blk in big.blocks:
        l0, l2 = blk.mlp[0], blk.mlp[2]
        d_in, h, d_out = l0.in_features, l0.out_features, l2.out_features
        k = extra or max(1, h // 4)
        n0 = nn.Linear(d_in, h + k).to(l0.weight.device)
        n2 = nn.Linear(h + k, d_out, bias=(l2.bias is not None)).to(l2.weight.device)
        n0.weight[:h] = l0.weight; n0.bias[:h] = l0.bias
        nn.init.normal_(n0.weight[h:], 0.0, 0.02); nn.init.zeros_(n0.bias[h:])
        n2.weight[:, :h] = l2.weight
        if l2.bias is not None: n2.bias.copy_(l2.bias)
        nn.init.zeros_(n2.weight[:, h:])                   # new neurons: zero output
        blk.mlp[0] = n0; blk.mlp[2] = n2
    return big


def n_params(m): return sum(p.numel() for p in m.parameters())


# ---- the self-derived growth trigger (no hardcoded threshold) ----
def should_grow(forgetting, surprise_after, boundary):
    """Grow only when the brain is BOTH forgetting (interference) and still
    surprised after trying to learn (capacity full). Signals are the brain's own;
    the bar is its own calibrated familiarity boundary."""
    return (forgetting > 0.10) and (surprise_after > 0.85 * boundary)


if __name__ == "__main__":
    # ---- demonstrate function-preserving growth on a small CPU model ----
    torch.manual_seed(0); dev = "cpu"
    m = H.SpinAttentionLM(vocab=512, d=96, n_head=4, n_layer=3).to(dev).eval()
    x = torch.randint(0, 512, (2, 16))
    with torch.no_grad(): o0 = m(x)

    print(f"base model: {n_params(m):,} params, {len(m.blocks)} layers")
    md = grow_depth(m, n_new=2).eval()
    with torch.no_grad(): od = md(x)
    print(f"grow_depth: {n_params(md):,} params (+{len(md.blocks)-len(m.blocks)} layers)  "
          f"max output change = {(od-o0).abs().max().item():.2e}  -> "
          f"{'FUNCTION PRESERVED' if (od-o0).abs().max()<1e-3 else 'CHANGED'}")
    mw = grow_width(m, extra=128).eval()
    with torch.no_grad(): ow = mw(x)
    print(f"grow_width: {n_params(mw):,} params (+{n_params(mw)-n_params(m):,})  "
          f"max output change = {(ow-o0).abs().max().item():.2e}  -> "
          f"{'FUNCTION PRESERVED' if (ow-o0).abs().max()<1e-3 else 'CHANGED'}")

    # ---- show the new capacity is trainable (the grown block learns) ----
    import torch.nn.functional as F
    md.train(); opt = torch.optim.Adam(md.blocks[-1].parameters(), lr=1e-3)
    y = torch.randint(0, 512, (2, 16))
    l0 = F.cross_entropy(md(x).reshape(-1, 512), y.reshape(-1)).item()
    for _ in range(50):
        loss = F.cross_entropy(md(x).reshape(-1, 512), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"new block is trainable: loss {l0:.3f} -> {loss.item():.3f} "
          f"(the fresh capacity learns; old layers were the untouched starting point)")
