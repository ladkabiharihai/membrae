"""
lm-eval-harness adapter + runner for the Pragnosia spin model, so it can be scored on
the same tasks (IFEval, BBH, MATH, GPQA, MUSR, MMLU-PRO) as any HF model. HF comparison
models (gpt2, SmolLM-360M, Qwen2.5-0.5B) run on lm-eval's built-in `hf` backend; this
wraps OUR custom SpinAttentionLM.

NOTE: our model's context is only 256 tokens, so long few-shot prompts are left-
truncated to the last 255 — an honest handicap on the long-context tasks.

  /opt/code/eval_env/bin/python eval_spin.py --tasks leaderboard --limit 100
"""
import argparse, json, torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval import simple_evaluate
import s6_hybrid as H
from tokenizers import Tokenizer


class SpinLM(LM):
    def __init__(self, ckpt="pragnosia.pt", config="pragnosia.json", device="cuda", batch_size=1):
        super().__init__()
        cfg = json.load(open(config)); H.VOC = cfg["vocab"]; H.L = cfg["ctx"]
        self.ctx, self.vocab, self.device = cfg["ctx"], cfg["vocab"], device
        self.tok = Tokenizer.from_file(cfg["tokenizer"])
        self.m = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"],
                                   mlp_mult=cfg["mlp_mult"]).to(device).eval()
        self.m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        self.nparams = sum(p.numel() for p in self.m.parameters())

    def _enc(self, s): return self.tok.encode(s).ids

    @torch.no_grad()
    def loglikelihood(self, requests):
        out = []
        for req in requests:
            ctx, cont = req.args
            cids, tids = self._enc(ctx), self._enc(cont)
            if not tids: out.append((0.0, True)); continue
            tids = tids[-self.ctx + 1:]                          # cont must fit
            full = (cids + tids)[-self.ctx:]
            n = len(tids)
            x = torch.tensor([full], device=self.device)
            logits = self.m(x)[0].float()                       # (T, V)
            lp = F.log_softmax(logits, -1)
            tgt = torch.tensor(full[-n:], device=self.device)
            pos = torch.arange(len(full) - n - 1, len(full) - 1, device=self.device)
            ll = float(lp[pos, tgt].sum())
            greedy = bool((logits[pos].argmax(-1) == tgt).all())
            out.append((ll, greedy))
        return out

    @torch.no_grad()
    def generate_until(self, requests):
        out = []
        for req in requests:
            ctx, kw = req.args
            until = kw.get("until", []) or []
            if isinstance(until, str): until = [until]
            maxg = kw.get("max_gen_toks", 256)
            ids = self._enc(ctx); gen = []
            for _ in range(maxg):
                x = torch.tensor([(ids + gen)[-self.ctx:]], device=self.device)
                nx = int(self.m(x)[0, -1].argmax())
                if nx == 0: break
                gen.append(nx)
                txt = self.tok.decode(gen)
                if any(u in txt for u in until):
                    for u in until: txt = txt.split(u)[0]
                    break
            else:
                txt = self.tok.decode(gen)
            out.append(txt)
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        res = []
        for req in requests:
            ids = self._enc(req.args[0])[: self.ctx]
            if len(ids) < 2: res.append(0.0); continue
            x = torch.tensor([ids], device=self.device)
            lp = F.log_softmax(self.m(x)[0].float(), -1)
            tgt = torch.tensor(ids[1:], device=self.device)
            res.append(float(lp[torch.arange(len(ids) - 1, device=self.device), tgt].sum()))
        return res


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", default="pragnosia.pt"); pa.add_argument("--config", default="pragnosia.json")
    pa.add_argument("--tasks", default="leaderboard")
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--out", default="eval_spin_results.json")
    a = pa.parse_args()
    lm = SpinLM(a.ckpt, a.config)
    print(f"[spin] {lm.nparams/1e6:.0f}M params, ctx={lm.ctx}", flush=True)
    r = simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit, bootstrap_iters=0)
    res = {k: v for k, v in r["results"].items()}
    json.dump(res, open(a.out, "w"), indent=2)
    for task, m in res.items():
        acc = {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
        print(f"  {task}: {acc}", flush=True)
