"""Corpus for the COMBINED run = selective-state (long-context) + weight-internalization in ONE mixed-W
supervised train. Corrected recipe (RESULTS #22): long-range RECALL is the DOMINANT signal (teaches the
selective decay to latch), plus KNOWLEDGE-dense text (internalization -> the model knows more), plus the v4
SFT skills (keep abilities), plus 50% replay (anti-forgetting). Writes sft_combined_run.bin. CPU only."""
import os, random, numpy as np
import build_sft as B, build_selective as S
random.seed(2)
OUT, CTX, EOS, enc, tok = B.OUT, B.CTX, B.EOS, B.enc, B.tok

def main():
    parts = []
    # SKILLS (keep v4's abilities) -- a modest slice
    parts += (B.cot_math(15000) + B.multihop(15000) + B.multiop(12000) + B.instructions(15000)
              + B.abstain(10000) + B.recall(15000))
    chat = B.stream_chat(50000) or B.mine_chat(f"{OUT}/expanded2_train.bin", 20000)
    out = f"{OUT}/sft_combined_run.bin"; tot = 0
    with open(out, "wb") as f:
        for ex in parts:
            b = enc(ex); f.write(b); tot += len(b) // 2
        print(f"  skills: {tot/1e6:.1f}M tok", flush=True); c0 = tot
        for ex in chat:
            b = enc(ex); f.write(b); tot += len(b) // 2
        print(f"  chat: {(tot-c0)/1e6:.1f}M tok", flush=True); c1 = tot
        # DOMINANT long-range recall (the latching signal -- corrected selective recipe: recall-dominated)
        for ex in S.long_recall(20000):
            ids = tok.encode(ex).ids
            f.write(np.asarray(ids, dtype=np.uint16).tobytes()); f.write(EOS); tot += len(ids) + 1
        print(f"  long-recall (DOMINANT): {(tot-c1)/1e6:.1f}M tok", flush=True); c2 = tot
        # KNOWLEDGE-dense text for internalization (real reasoning/edu text -> the model learns more facts)
        for src, n in [("cot_train", 24_000_000)]:
            d = np.memmap(f"{OUT}/{src}.bin", dtype=np.uint16, mode="r")
            st = random.randint(0, max(0, len(d) - n - 1))
            f.write(np.asarray(d[st:st + n], dtype=np.uint16).tobytes()); tot += n
        print(f"  knowledge(cot_train): {(tot-c2)/1e6:.1f}M tok", flush=True)
        sft_content = tot
        # 50% REPLAY (general knowledge) -- anti-forgetting
        d = np.memmap(f"{OUT}/window2_train.bin", dtype=np.uint16, mode="r")
        got = 0
        while got < sft_content:
            i = random.randint(0, len(d) - CTX - 2)
            f.write(np.asarray(d[i:i + CTX], dtype=np.uint16).tobytes()); f.write(EOS); got += CTX + 1
        tot += got
        print(f"  replay(window2): {got/1e6:.1f}M tok ({100*got//tot}%)", flush=True)
    print(f"[combined] wrote {out}: {tot/1e6:.1f}M tok (sft {sft_content/1e6:.1f}M + replay {(tot-sft_content)/1e6:.1f}M)", flush=True)
    open(f"{OUT}/sft_combined_run_DONE.flag", "w").write(str(tot))

if __name__ == "__main__":
    main()
