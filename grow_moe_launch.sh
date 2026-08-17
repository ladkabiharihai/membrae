#!/usr/bin/env bash
# GROW-MoE: constant-compute growth (add experts, top-1 -> active params flat as total grows) + COMPILE (~1.5x) + bf16
# + ckpt. Gives BOTH: constant compute-per-token as it scales AND the compile speedup. window2 176B + recall mix.
cd /opt/code/membrae
exec >> /opt/code/membrae/brainlogs/grow_moe.log 2>&1
export D=1024 L=14 E0=4 EMAX=16 BS=4 RECALL_FRAC=0.15 GROW_EVERY=30000 VRAM_STOP_GB=14 COMPILE=1
exec /opt/code/parakeet_env/bin/python -u /opt/code/membrae/grow_moe_train.py
