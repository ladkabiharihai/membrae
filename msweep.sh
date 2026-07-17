#!/usr/bin/env bash
# Multi-seed sweep orchestrator: runs seeds 2 and 3 automatically after seed 1 finishes.
# Each seed = a spin+attn PAIR trained in parallel (identical GPU conditions), matched tokens/lr.
# Seed 1 is already running; this waits for it, then does 2 then 3, one pair at a time.
cd /opt/code/membrae
PY=/opt/code/parakeet_env/bin/python
COMMON="--steps 70000 --lr 3e-3 --bs 32 --no-grow"

echo "[msweep] $(date) waiting for seed 1 to finish..."
while pgrep -f "run_fast.py --steps 70000" >/dev/null; do sleep 120; done
echo "[msweep] $(date) seed 1 done. running seeds 2,3."

for s in 2 3; do
  echo "[msweep] $(date) launching seed $s pair"
  CONFIG=pragnosia_100m_spin.json SEED=$s WARM=2000 BEST_CKPT=p100_spin_s$s.pt \
    nohup $PY run_fast.py $COMMON > p100_spin_s$s.log 2>&1 & P1=$!
  sleep 3
  CONFIG=pragnosia_100m_attn.json SEED=$s WARM=2000 BEST_CKPT=p100_attn_s$s.pt \
    nohup $PY run_fast.py $COMMON > p100_attn_s$s.log 2>&1 & P2=$!
  wait $P1 $P2
  echo "[msweep] $(date) seed $s complete"
done
echo "[msweep] $(date) ALL SEEDS DONE" | tee msweep_DONE.flag
