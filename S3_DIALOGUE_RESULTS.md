# Phase S3 — conversation at toy scale — RESULTS (2026-06-11)

## Verdict
The spinning model **crosses from "continues text" to "converses"**: it does
real turn-taking and frequently lands the right emotional register. It is NOT
yet content-specific — replies are appropriate but generic, the expected
ceiling for 3.3M params on a 1.66M-token corpus. The key lesson stands:
**conversation was a DATA problem, not architecture** — same spinning core as
the story model, only the data changed.

## Setup
- Architecture: GainSpin2LM d=512 (unchanged spinning core, warm-started from
  the TinyStories backbone s2_gspin2.pt).
- Data: EmpatheticDialogues (19.5K everyday conversations) →
  `<user> … <bot> …` format, BPE-2048 with 3 special tokens.
  1.66M train tokens / 653K bot-target tokens. Tokenizer `data/dlg_bpe.json`.
- Training = SFT with **loss masked to `<bot>` reply tokens only** (learn to
  respond, not to echo the user). Code: `s3_dialogue.py`.

## The overfitting fix (recorded — method rule #1 & #6)
First run (Adam, lr 1.5e-3, 20K steps, dialogue-only) **diverged**: train loss
4.09→1.35 while val ppl rose monotonically 88→825. Root cause: 1.66M tokens =
~45 steps/epoch, so 1K steps was already 22 epochs of memorization.
Fix that worked: (1) **mix 50% story batches** as a fluency regularizer,
(2) AdamW weight-decay 0.1 + lr 5e-4, (3) eval every 200 steps with early-stop.
Result: stable convergence, **best bot-token val ppl 90.11** (plateau 90–93, no
divergence).

## Conversation transcripts (scripted, temp 0.7)
```
you> Hello! How are you today?      bot> Nice, what did you do?
you> I just got a new job.          bot> wow thats great!
you> I'm feeling really sad today.   bot> Oh wow. I hope they didn't get too bad.
you> My dog passed away.             bot> Nice! I'm sorry to hear that. ...
you> I won the lottery!              bot> Nice! I bet you got a lot of fun! ...
```
Turn-taking: reliable. Emotional register: often correct. Grammar: fractures
mid-sentence on longer replies. Content-specificity: weak.

## Causal control (brain-swap probe on continuations)
continuation NLL: own_state 5.95 ≈ swapped_state 5.94 ≪ random_state 6.92.
- **random ≫ own/swapped**: the spinning state is causally carrying real
  conversational information (not memorized weights).
- **own ≈ swapped**: replies are weakly conditioned on the *specific* prior
  turn → generic-response ceiling. This is the honest limit and the next target.

## Next levers (in priority)
1. More dialogue data (combine DailyDialog/others) — biggest lever for
   content-specificity; corpus size is the binding constraint here.
2. Longer context + more capacity (d, layers).
3. The clean entity-register (s2_coherent.py, proven on stories) applied to
   dialogue state for referent tracking across turns.

## Artifacts
`s3_dialogue.py`, `data/dlg_bpe.json`, `data/d{train,valid}.bin` (+_mask),
`s3_dialogue.pt`, `s3_dialogue_result.json`, `s3_dialogue2.log`.
Run chat: `python3 s3_dialogue.py chat`.
