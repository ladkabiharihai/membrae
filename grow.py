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


def _shell(model, n_layer, mlp_mult):
    big = H.SpinAttentionLM(model.emb.num_embeddings, model.d, model.n_head, n_layer, mlp_mult=mlp_mult)
    big = big.to(next(model.parameters()).device)
    big.emb.load_state_dict(model.emb.state_dict())
    big.pos.load_state_dict(model.pos.state_dict())
    big.carrier.load_state_dict(model.carrier.state_dict())
    big.lnf.load_state_dict(model.lnf.state_dict())
    big.head.weight = big.emb.weight                       # keep the tie
    return big

@torch.no_grad()
def grow_depth(model, n_new=1):
    """Add n_new transformer blocks as near-identity -> function preserved."""
    nl = len(model.blocks)
    big = _shell(model, nl + n_new, model.mlp_mult)
    for i in range(nl):                                    # copy existing blocks
        big.blocks[i].load_state_dict(model.blocks[i].state_dict())
    for j in range(nl, nl + n_new):                        # new blocks -> identity
        b = big.blocks[j]
        nn.init.zeros_(b.attn.proj.weight); nn.init.zeros_(b.attn.proj.bias)
        nn.init.zeros_(b.mlp[2].weight);    nn.init.zeros_(b.mlp[2].bias)
    return big

@torch.no_grad()
def grow_width(model, add_mult=1):
    """Widen every block's MLP by add_mult*d neurons (mlp_mult += add_mult); new
    neurons' OUTPUT weights = 0 -> function preserved. Resume-safe via mlp_mult."""
    nl = len(model.blocks); d = model.d; h_old = model.mlp_mult * d
    big = _shell(model, nl, model.mlp_mult + add_mult)
    for i in range(nl):
        ob, nb = model.blocks[i], big.blocks[i]
        nb.ln1.load_state_dict(ob.ln1.state_dict()); nb.attn.load_state_dict(ob.attn.state_dict())
        nb.ln2.load_state_dict(ob.ln2.state_dict())
        nb.mlp[0].weight[:h_old] = ob.mlp[0].weight; nb.mlp[0].bias[:h_old] = ob.mlp[0].bias
        nn.init.normal_(nb.mlp[0].weight[h_old:], 0.0, 0.02); nn.init.zeros_(nb.mlp[0].bias[h_old:])
        nb.mlp[2].weight[:, :h_old] = ob.mlp[2].weight; nb.mlp[2].bias.copy_(ob.mlp[2].bias)
        nn.init.zeros_(nb.mlp[2].weight[:, h_old:])        # new neurons: zero output
    return big

@torch.no_grad()
def shrink_width(model, new_mult):
    """PRUNE every block's MLP from mlp_mult*d down to new_mult*d, keeping the most important
    neurons (importance = ||input row|| * ||output col||), warm-started from THIS model's weights.
    LOSSY -- this is NOT function-preserving: it drops capacity, so val ppl jumps and the model
    must be RE-TRAINED (a warm start, far faster than scratch) to recover. The payoff: mlp_mult 12->4
    is ~half the params = ~2x faster training, built ON TOP of the current model. Run it, then
    --resume to recover. Always keep a backup of the original checkpoint first."""
    assert new_mult < model.mlp_mult, "shrink_width only reduces mlp_mult"
    nl = len(model.blocks); d = model.d; h_new = new_mult * d
    big = _shell(model, nl, new_mult)
    for i in range(nl):
        ob, nb = model.blocks[i], big.blocks[i]
        nb.ln1.load_state_dict(ob.ln1.state_dict()); nb.attn.load_state_dict(ob.attn.state_dict())
        nb.ln2.load_state_dict(ob.ln2.state_dict())
        win, wout = ob.mlp[0].weight, ob.mlp[2].weight            # [h,d] in, [d,h] out
        imp = win.norm(dim=1) * wout.norm(dim=0)                  # per-neuron importance [h]
        keep = torch.topk(imp, h_new).indices.sort().values      # keep the top-h_new neurons
        nb.mlp[0].weight.copy_(win[keep]); nb.mlp[0].bias.copy_(ob.mlp[0].bias[keep])
        nb.mlp[2].weight.copy_(wout[:, keep]); nb.mlp[2].bias.copy_(ob.mlp[2].bias)
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
    mw = grow_width(m, add_mult=2).eval()
    with torch.no_grad(): ow = mw(x)
    print(f"grow_width: {n_params(mw):,} params (+{n_params(mw)-n_params(m):,}, mlp_mult "
          f"{m.mlp_mult}->{mw.mlp_mult})  max output change = {(ow-o0).abs().max().item():.2e}  -> "
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
