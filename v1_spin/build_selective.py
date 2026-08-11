"""Corpus for the SELECTIVE-STATE + long-form mixed-W continue-train. Writes to sft_selective.bin (NOT
sft_combined.bin -- v4 is live on that). Mix: v4's short-form SFT tasks (keep the skills) + LONG-RANGE
recall (teach the selective carrier to latch a fact across >256 tokens) + long coherent docs (coherence)
+ 50% replay (general). Trained at mixed-W (W up to 32) so the carry is actually exercised. No hardcoding
-- templated/parametrized data, same as build_sft."""
import os, random, numpy as np
import build_sft as B
random.seed(1)
OUT, CTX, EOS, enc, tok = B.OUT, B.CTX, B.EOS, B.enc, B.tok

def long_recall(n):
    """Fact planted EARLY, long filler, fact required again LATE -> predicting the late mention needs the
    cross-window carry (the latching signal the selective decay learns). Filler length varies to span depths."""
    FILL = "The weather stayed calm and the hours passed slowly while the routine work continued without much event. "
    out = []
    for _ in range(n):
        key, val = random.choice(B.KEYS), random.choice(B.VALS)
        filler = FILL * random.choice([8, 16, 24, 32, 48])          # ~120 - ~750 words -> ~180-1100 tokens
        text = (f"Please remember this fact carefully: the {key} is {val}. {filler} "
                f"Now, recalling the fact stated at the very beginning, the {key} is {val}.")
        out.append(text)                                            # plain passage (LM objective carries the fact)
    return out

def main():
    parts = []
    # keep the v4 short-form skills
    parts += B.cot_math(20000) + B.multihop(20000) + B.multiop(15000) + B.instructions(20000) + B.abstain(12000) + B.recall(20000)
    # the NEW long-range latching signal
    long_r = long_recall(6000)
    chat = B.stream_chat(60000) or B.mine_chat(f"{OUT}/expanded2_train.bin", 20000)
    out = f"{OUT}/sft_selective.bin"; tot = 0
    with open(out, "wb") as f:
        for ex in parts:                       # short-form SFT tasks
            b = enc(ex); f.write(b); tot += len(b) // 2
        print(f"  short-form SFT: {tot/1e6:.1f}M tok", flush=True)
        c0 = tot
        for ex in long_r:                       # long-range recall passages (can exceed CTX -> truncated to CTX per enc; keep full)
            ids = tok.encode(ex).ids            # DO NOT truncate these -> the whole passage must be one contiguous span
            f.write(np.asarray(ids, dtype=np.uint16).tobytes()); f.write(EOS); tot += len(ids) + 1
        print(f"  long-recall: {(tot-c0)/1e6:.1f}M tok ({len(long_r)} passages)", flush=True)
        c1 = tot
        for ex in chat:
            b = enc(ex); f.write(b); tot += len(b) // 2
        print(f"  chat-real: {(tot-c1)/1e6:.1f}M tok", flush=True)
        # long coherent docs (real reasoning-CoT) for long-form coherence
        cot = B.cot_train_slice(18_000_000); f.write(cot); tot += len(cot) // 2
        print(f"  cot_train(long docs): {len(cot)//2/1e6:.1f}M tok", flush=True)
        sft_content = tot
        # 50% REPLAY (general knowledge) -- anchors quality, per the standing recipe
        d = np.memmap(f"{OUT}/window2_train.bin", dtype=np.uint16, mode="r")
        got = 0
        while got < sft_content:
            i = random.randint(0, len(d) - CTX - 2)
            f.write(np.asarray(d[i:i + CTX], dtype=np.uint16).tobytes()); f.write(EOS); got += CTX + 1
        tot += got
        print(f"  replay(window2): {got/1e6:.1f}M tok ({100*got//tot}%)", flush=True)
    print(f"[selective] wrote {out}: {tot/1e6:.1f}M tokens (sft {sft_content/1e6:.1f}M + replay {(tot-sft_content)/1e6:.1f}M)", flush=True)
    open(f"{OUT}/sft_selective_DONE.flag", "w").write(str(tot))

if __name__ == "__main__":
    main()
