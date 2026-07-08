"""SFT-for-chat data prep (T5.1) -- assemble a chat-format SFT corpus so the core learns to ANSWER, not
continue (the 'Hi -> WW2 text' problem). Combines the self/persona corpus + identity + (optionally) mined
<user>/<assistant> spans from the main corpus, tokenizes to data/sft.bin.

Run:  python3 prepare_sft.py
Then fine-tune on the H100 (recipe at the bottom of this file).
"""
import json, os
import numpy as np
from tokenizers import Tokenizer

CFG = json.load(open("pragnosia.json"))
tok = Tokenizer.from_file(CFG["tokenizer"])
EOS = 0


def load_lines(path):
    out = []
    for line in open(path):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def mine_chat_spans(bin_name, n=2000, ctx=256):
    """Pull existing <user>...<assistant>... spans out of the main corpus so the SFT set is dominated by REAL
    dialogue structure, not just the small self-corpus. Best-effort; skipped if the bin is missing."""
    path = f"data/{bin_name}.bin"
    if not os.path.exists(path):
        return []
    data = np.memmap(path, dtype=np.int16, mode="r")
    spans, L, step = [], len(data), max(ctx, len(data) // 20000)
    for i in range(0, L - ctx, step):
        txt = tok.decode([int(x) for x in data[i:i + ctx]])
        if "<user>" in txt and "<assistant>" in txt:
            j = txt.find("<user>")
            spans.append(txt[j:j + 600])
            if len(spans) >= n:
                break
    return spans


def main():
    examples = []
    for src in ("pragnosia_self.txt", "identity_sentences.txt"):
        if os.path.exists(src):
            examples += load_lines(src)
    n_self = len(examples)
    examples += mine_chat_spans(CFG["train_bin"])          # real dialogue from the corpus (bulk of the SFT set)

    ids = []
    for ex in examples:
        ids += tok.encode(ex).ids[:CFG["ctx"]] + [EOS]
    os.makedirs("data", exist_ok=True)
    np.array(ids, dtype=np.int16).tofile("data/sft.bin")
    print(f"wrote data/sft.bin: {len(examples)} examples ({n_self} self/identity + "
          f"{len(examples)-n_self} mined dialogue), {len(ids):,} tokens", flush=True)


# ---- FINE-TUNE RECIPE (H100) ----------------------------------------------------------------------------
# A SHORT, LOW-LR pass -- enough to teach the answer-shape without eroding knowledge:
#   * resume from the current checkpoint (pragnosia_spin.pt)
#   * train_bin = sft (this file's output), LR ~1e-5 (10-20x below pretrain), 1-2 epochs over sft.bin
#   * KEEP a replay fraction of the main corpus in the mix (say 20%) so it doesn't forget while specializing
#   * growth OFF, long-context OFF (pure chat-shape SFT)
# Sketch:  CONFIG=pragnosia.json PRAGNOSIA_SELF=pragnosia_self.txt python3 train_pragnosia.py --resume  (with
#   the trainer pointed at sft.bin + a low LR); or a dedicated 1-2 epoch loop over data/sft.bin at lr 1e-5.
# Expect: 'Hi' -> a greeting, 'what is X?' -> an answer, without losing the facts/arithmetic already learned.
if __name__ == "__main__":
    main()
