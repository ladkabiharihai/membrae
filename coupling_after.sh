#!/usr/bin/env bash
# Coupling Stage A, auto-launched AFTER the multi-seed sweep frees the GPU.
# From-scratch coupled-vs-control at 284M, matched budget + matched explicit lr (the two arms differ ONLY in
# the coupling). Per H100_TODO.md priority 3 / RESULTS_MEASURED #15 (do NOT warm-start; train from random init).
cd /opt/code/membrae
PY=/opt/code/parakeet_env/bin/python
COMMON="--steps 100000 --lr 6e-3 --bs 32 --no-grow"

echo "[coupling] $(date) waiting for multiseed sweep (msweep_DONE.flag)..."
while [ ! -f msweep_DONE.flag ]; do sleep 180; done
echo "[coupling] $(date) multiseed done. launching Stage A pair (from scratch, matched lr 6e-3)."

CONFIG=pragnosia_284m_scratch.json WARM=4000 BEST_CKPT=pragnosia_284m_scratch_best.pt \
  nohup $PY run_fast.py $COMMON > coupling_control.log 2>&1 & P1=$!
sleep 3
CONFIG=pragnosia_coupled_284m.json WARM=4000 BEST_CKPT=pragnosia_coupled_284m_best.pt \
  nohup $PY run_fast.py $COMMON > coupling_coupled.log 2>&1 & P2=$!
wait $P1 $P2
echo "[coupling] $(date) Stage A DONE (compare pragnosia_284m_scratch_best.pt vs pragnosia_coupled_284m_best.pt)" | tee coupling_DONE.flag
