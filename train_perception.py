"""Phase-B scaffold: train an OUR-OWN spin modality encoder (vision or voice) to feed the FROZEN spin LM.

Objective = captioning: prepend the encoder's perception tokens to the caption/transcript embeddings, run the
frozen LM via forward_embeds, and cross-entropy the LM's predictions against the caption tokens. Only the encoder
(+ attention pool + frame/patch embed) trains; the LM is frozen (no_grad, requires_grad_(False)).

    perception tokens  ─┐
                        ├─ forward_embeds(frozen LM) ─> logits ─> CE vs caption tokens
    caption embeds ─────┘

This file is CPU-smoke-tested end-to-end (`python train_perception.py --smoke`). The REAL GPU training (a real
frozen 1B + image-caption / audio-transcript corpora) is Phase C, to run AFTER the text 1B lands -- deliberately
NOT started here.
"""
import argparse, sys, torch, torch.nn as nn, torch.nn.functional as F
import s6_hybrid as H
from perception import SpinVisionEncoder, SpinVoiceEncoder


def build_lm(cfg_path, ckpt=None, device="cpu"):
    import json
    cfg = json.load(open(cfg_path))
    H.VOC, H.L = cfg["vocab"], cfg["ctx"]
    m = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"],
                          mlp_mult=cfg["mlp_mult"], carrier=cfg["carrier"])
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)                    # FROZEN LM
    return m, cfg


def caption_loss(lm, encoder, x_modality, cap_ids):
    """cap_ids [B,T] (int). Returns scalar CE over the caption positions."""
    ptok = encoder(x_modality)                     # [B,P,d]  (trains)
    with torch.no_grad():
        cap_emb = lm.emb(cap_ids)                  # [B,T,d]  (frozen embedding)
    P = ptok.shape[1]
    h = torch.cat([ptok, cap_emb], dim=1)          # [B,P+T,d]
    pos = lm.pos(torch.arange(h.shape[1], device=h.device))[None]
    logits = lm.forward_embeds(h + pos)            # [B,P+T,V]
    pred = logits[:, P - 1: P - 1 + cap_ids.shape[1], :]   # position P-1+t predicts cap_ids[:,t]
    return F.cross_entropy(pred.reshape(-1, logits.shape[-1]), cap_ids.reshape(-1))


def smoke():
    """End-to-end CPU check: tiny frozen LM + BOTH encoders on synthetic image/audio. Verifies (1) the combined
    perception+text stream flows through forward_embeds, (2) grads reach ONLY the encoder, LM stays frozen,
    (3) a few optimizer steps actually lower the loss (a real learning signal exists)."""
    torch.manual_seed(0)
    dev = "cpu"
    V, d = 256, 128
    H.VOC, H.L = V, 256
    lm = H.SpinAttentionLM(V, d, 4, 4, mlp_mult=4, carrier="spin_dominant").eval().to(dev)
    for p in lm.parameters(): p.requires_grad_(False)

    B, P, T = 4, 8, 12
    vis = SpinVisionEncoder(d, n_tokens=P, img_size=64, patch=16, n_layer=4).to(dev)
    voc = SpinVoiceEncoder(d, n_tokens=P, n_mels=80, n_layer=4).to(dev)

    img = torch.rand(B, 3, 64, 64)                 # synthetic image
    wav = torch.randn(B, 16000)                    # 1s synthetic audio @16kHz
    cap = torch.randint(0, V, (B, T))              # synthetic caption ids

    for name, enc, x in [("VISION", vis, img), ("VOICE", voc, wav)]:
        opt = torch.optim.AdamW(enc.parameters(), lr=3e-3)
        losses = []
        for step in range(6):
            opt.zero_grad()
            loss = caption_loss(lm, enc, x, cap)
            loss.backward()
            # frozen-LM check on the FIRST step
            if step == 0:
                enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
                lm_grad = any(p.grad is not None for p in lm.parameters())
                assert enc_grad, f"{name}: encoder got NO gradient"
                assert not lm_grad, f"{name}: LM is NOT frozen (got gradient)"
            opt.step()
            losses.append(loss.item())
        ptok = enc(x)
        assert ptok.shape == (B, P, d), f"{name}: bad perception-token shape {tuple(ptok.shape)}"
        drop = losses[0] - losses[-1]
        print(f"  {name:6s} perception tokens {tuple(ptok.shape)} | loss {losses[0]:.3f} -> {losses[-1]:.3f} "
              f"(down {drop:+.3f}) | encoder-only grad OK, LM frozen OK")
        assert drop > 0, f"{name}: loss did not decrease -- no learning signal"
    print("[smoke] PASS -- both spin encoders train end-to-end against the frozen LM (CPU).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU end-to-end sanity check (default if no mode given)")
    ap.add_argument("--modality", choices=["vision", "voice"])
    ap.add_argument("--lm-config"); ap.add_argument("--lm-ckpt")
    a = ap.parse_args()
    if a.smoke or not a.modality:
        smoke(); sys.exit(0)
    print("Real training is Phase C (after the 1B). Scaffold ready; wire the data loader + frozen 1B then.",
          flush=True)
