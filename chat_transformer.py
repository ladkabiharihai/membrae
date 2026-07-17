"""Chat with the 288M crux transformer (BASE model, not instruction-tuned) -- CPU only, so it never touches
the running GPU training or prod. Wraps input in the <user>/<assistant> template the corpus used and samples a
reply. It's a base model: fluent but factually unreliable, no persona, short 256-token context.

  interactive:  python chat_transformer.py
  one-shot demo: python chat_transformer.py --demo "hello" "what is the capital of japan?"
"""
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/opt/code/membrae"); os.chdir("/opt/code/membrae")
import s6_hybrid as H
from tokenizers import Tokenizer

CKPT = os.environ.get("CHAT_CKPT", "pragnosia_baseline_best.pt")
H.VOC, H.L, H.DEVICE = 16384, 256, "cpu"
tok = Tokenizer.from_file("data/bpe.json"); CTX = 256
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
nl = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.") and k.split(".")[1].isdigit())
mm = next(sd[k].shape[0] for k in sd if k.endswith("mlp.0.weight")) // 768
lm = H.SpinAttentionLM(16384, 768, 12, nl, mlp_mult=mm, carrier="none").eval()
lm.load_state_dict(sd)


@torch.no_grad()
def reply(history, msg, n=60, temp=0.7, rep=1.3):
    """history = running transcript string. Returns the assistant's reply text."""
    prompt = history + f"<user> {msg}\n<assistant>"
    ids = tok.encode(prompt).ids
    out = []
    for _ in range(n):
        logits = lm(torch.tensor([(ids + out)[-CTX:]]))[0, -1].float()
        for t in set((ids + out)[-40:]): logits[t] /= rep          # repetition penalty
        nx = int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
        if nx == 0: break
        out.append(nx)
        txt = tok.decode(out)
        if "<user>" in txt:                                        # stop at the next turn marker
            return txt.split("<user>")[0].strip()
    return tok.decode(out).strip()


def main():
    print(f"[chat] {sum(p.numel() for p in lm.parameters())/1e6:.0f}M transformer (BASE, not instruction-tuned) on CPU.")
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        hist = ""
        for msg in sys.argv[2:]:
            r = reply(hist, msg)
            print(f"\nYou: {msg}\nModel: {r}")
            hist += f"<user> {msg}\n<assistant> {r}\n"
            hist = tok.decode(tok.encode(hist).ids[-200:])          # keep context short
        return
    print("Type your message (Ctrl-C or 'quit' to exit).")
    hist = ""
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if msg.lower() in ("quit", "exit"): return
        if not msg: continue
        r = reply(hist, msg)
        print(f"Model: {r}")
        hist += f"<user> {msg}\n<assistant> {r}\n"
        hist = tok.decode(tok.encode(hist).ids[-200:])


if __name__ == "__main__":
    main()
