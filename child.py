"""
child.py -- raise Pragnosia like a child.

It OBSERVES what you tell it (and learns it if the content is new), it gets CURIOUS
and asks its OWN question about a piece of it, it tries to ANSWER from itself, and if
it does not know, it LOOKS THE ANSWER UP (Wikipedia) and learns it. When it keeps
failing to learn, it GROWS itself. Press Enter with no input to let it follow its own
train of thought (chase its last question on its own).

  python3 child.py
  you> The Eiffel Tower is a famous landmark in Paris.   # tell it something
  you> What is the capital of France?                    # ask it something
  you>                                                   # (Enter) let it explore on its own
  you> quit                                              # save what it learned and exit
"""
import json, brain as B

def show(tr):
    if tr.get("learned"):          print("   · took it in and learned it (new content)")
    if tr.get("wonders"):          print(f"   · it WONDERS: {tr['wonders']}")
    if tr.get("knows"):            print(f"   · it already knows: {tr['knows'][:90]}")
    if tr.get("didnt_know"):       print("   · it did NOT know -> went to look it up")
    if tr.get("learned_answer"):   print(f"   · learned [{tr.get('source')}]: {tr['learned_answer']}")
    if tr.get("pursuing"):         print(f"   · chasing its own question: {tr['pursuing']}")
    if tr.get("learned_from_web"): print(f"   · read & learned: {tr['learned_from_web']}")
    if tr.get("now_wonders"):      print(f"   · now it WONDERS: {tr['now_wonders']}")
    if tr.get("couldnt_find"):     print(f"   · couldn't find anything on: {tr['couldnt_find']}")
    if tr.get("grew"):             print(f"   · !! it GREW its brain: {tr['grew']}")
    if tr.get("idle"):             print(f"   · {tr['idle']}")

def main():
    print("waking Pragnosia ...", flush=True)
    brain = B.Brain()
    print(f"awake ({brain.n_params():,} params). thresholds it set for itself: "
          f"knows>={brain.consistency_min:.2f}, new>{brain.novelty_min:.2f}\n")
    print("Tell it things, ask it questions, or just press Enter to let it think. 'quit' to save+exit.\n")
    while True:
        try: msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if msg.lower() in ("quit", "exit"): break
        if not msg:                                   # let it follow its own curiosity
            print("Pragnosia> (thinking on my own ...)"); show(brain.explore()); print(); continue
        if msg.endswith("?"):                         # a question -> answer or honestly abstain
            reply, decision = brain.respond(msg)
            print(f"Pragnosia> {reply}\n   · [{decision}]\n"); continue
        tr = brain.think(msg)                          # a statement -> observe, wonder, look up
        print("Pragnosia> (took that in)"); show(tr); print()
    brain.persist(); print("\n[saved what it learned this session]")

if __name__ == "__main__":
    main()
