"""Long-context needle-in-a-haystack (Tier-2 breadth, NMI expects long-range evidence). Insert a one-line
fact ("the secret word is X") at a fractional depth inside filler text of a target token length, then ask
for it and check retrieval. Varies context length x depth. The model is trained at ctx=256 with an O(T)
cross-window carry, so this is an honest test of whether the damped carry gives SHARP retrieval (it likely
degrades with length -- which supports, not contradicts, the paper's stated limitation).

Run (GPU, Unreal closed):  python3 needle_eval.py [ckpt]
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, brain, s6_hybrid as H

CKPT = sys.argv[1] if len(sys.argv) > 1 else None      # e.g. pragnosia_spin_h100.pt; default = config ckpt
LENGTHS = [128, 256, 512, 1024, 2048]
DEPTHS = [0.1, 0.5, 0.9]
NEEDLES = [("banana", "the secret word is banana."), ("crimson", "the secret word is crimson."),
           ("tiger", "the secret word is tiger."), ("comet", "the secret word is comet."),
           ("maple", "the secret word is maple.")]
QUERY = " the secret word is"


def main():
    b = brain.Brain(lm_ckpt=CKPT, learn=False)
    tok, ctxmax = b.tok, b.cfg["ctx"]
    # filler = decoded spans of the training corpus (in-distribution neutral text)
    vd = H.load(b.cfg["train_bin"])
    def filler_ids(n):
        i = random.randint(0, len(vd) - n - 1)
        return [int(t) for t in vd[i:i + n]]

    results = {}
    for L in LENGTHS:
        for depth in DEPTHS:
            hit = 0
            for word, sent in NEEDLES:
                nid = tok.encode(" " + sent).ids                       # needle tokens
                qid = tok.encode(QUERY).ids
                budget = max(16, L - len(nid) - len(qid))
                pre = int(budget * depth); post = budget - pre
                ids = filler_ids(pre) + nid + filler_ids(post) + qid
                ids = ids[-max(L, ctxmax):]                            # keep the tail (query always present)
                out = b.generate_text(tok.decode(ids), n=6).lower()    # does it recall the word?
                hit += (word in out)
            results[f"L{L}_d{int(depth*100)}"] = {"acc": round(hit / len(NEEDLES), 3), "hit": hit, "n": len(NEEDLES)}

    # aggregate by length (retrieval vs context length is the headline curve)
    by_len = {}
    for L in LENGTHS:
        hs = [results[f"L{L}_d{int(d*100)}"]["hit"] for d in DEPTHS]
        by_len[f"L{L}"] = round(sum(hs) / (len(DEPTHS) * len(NEEDLES)), 3)

    out = {"ckpt": CKPT or b.cfg["ckpt"], "ctx_trained": ctxmax,
           "retrieval_by_length": by_len, "detail": results}
    os.makedirs("eval_registry", exist_ok=True)
    tag = (CKPT or "config").replace("/", "_").replace(".pt", "")
    json.dump(out, open(f"eval_registry/needle_{tag}.json", "w"), indent=2)
    print(json.dumps({"retrieval_by_length": by_len}, indent=2), flush=True)
    print("(retrieval should be ~1.0 within the 256 train window and fall off beyond it -- honest long-ctx limit)",
          flush=True)


if __name__ == "__main__":
    main()
