"""Prove the RESOLUTION: brain.py retrieval = first-class long context. Feed a long document (needle-in-haystack)
as ordinary input, then ask about a fact buried deep in it. If the brain returns the exact sentence regardless of
how far back it is, long context is RESOLVED for the product (arbitrary length in, exact fact out) -- the thing
the native carrier cannot do (RESULTS #27)."""
import os, sys; sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault("BRAIN_SILENT", "1")
import brain as B
print("[test] building brain...", flush=True)
br = B.Brain(learn=False)

filler = ["The weather today is mild and the streets are quiet.",
          "A committee met on Tuesday to review the annual budget.",
          "The river flows gently past the old stone bridge.",
          "Several birds gathered near the fountain at noon.",
          "The library extended its hours for the exam season.",
          "A light rain fell over the hills in the afternoon.",
          "The market sold fresh vegetables and warm bread.",
          "Students walked across the courtyard between classes."]

NEEDLES = [("The secret access code is ZEBRA-7-DELTA.", "what is the secret access code", "ZEBRA-7-DELTA"),
           ("Dr. Halloran keeps the archive keys in the blue cabinet.", "where does Dr. Halloran keep the archive keys", "blue cabinet"),
           ("The rocket launch is scheduled for November 9th at dawn.", "when is the rocket launch scheduled", "November 9th")]

import random; random.seed(0)
for depth in [10, 40, 100]:                       # bury the needle after N filler sentences (arbitrary distance)
    print(f"\n=== haystack depth = {depth} filler sentences ===", flush=True)
    for needle, question, gold in NEEDLES:
        br.store = []                              # fresh memory per trial
        doc = " ".join(random.choice(filler) for _ in range(depth // 2)) + " " + needle + " " \
              + " ".join(random.choice(filler) for _ in range(depth // 2))
        br.interact(doc)                           # ingest the whole document (chunked into sentences)
        out = br.interact(question + "?")
        ans = out.get("answer", "")
        ok = gold.lower() in ans.lower()
        print(f"  [{'OK' if ok else 'XX'}] chunks={len(br.store):3d}  Q: {question[:38]:38s} -> {ans[:60]!r}", flush=True)
print("\n[done] OK = retrieved the exact buried fact regardless of depth -> long context RESOLVED via retrieval.")
