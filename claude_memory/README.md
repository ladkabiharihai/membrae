# Claude memory backup (survives a laptop reset)

Claude's persistent memory lives OUTSIDE this repo at:
`~/.claude/projects/-home-ashish-Downloads-membrae/memory/`

That path is wiped by a machine reset, so the memory is mirrored here.

## Restore after a reset
```bash
mkdir -p ~/.claude/projects/-home-ashish-Downloads-membrae/memory
cp claude_memory/MEMORY.md claude_memory/spinning-brain-project.md \
   ~/.claude/projects/-home-ashish-Downloads-membrae/memory/
```
Then start Claude Code in this repo; it will load the memory index automatically.

## What's here
- `MEMORY.md` — the index Claude loads each session.
- `spinning-brain-project.md` — the full project memory. **Read the
  "CURRENT STATE (2026-07-17) — SUPERSEDES ANY CONFLICTING CLAIM ABOVE" section at the end first**:
  earlier passages contain claims that later measurement RETRACTED (notably "spin-dominant wins by 21%"
  and the 306x/767x/1886x ablation progression).

## Not backed up here (deliberately)
- `title_page/` + `pragnosia_title_page.pdf` — carry the author's postal address; gitignored on purpose.
  **Back these up separately (not to a public repo).** Regenerable from the address if lost.
- `*.pt` checkpoints — large; the canonical copies live on the H100 (`/opt/code/membrae`).
