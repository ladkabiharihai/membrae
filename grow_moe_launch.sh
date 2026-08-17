#!/usr/bin/env bash
cd /opt/code/membrae
D=1024 L=14 E0=4 EMAX=16 BS=4 RECALL_FRAC=0.15 GROW_EVERY=30000 VRAM_STOP_GB=14 \
  exec /opt/code/parakeet_env/bin/python -u grow_moe_train.py
