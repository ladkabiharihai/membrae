#!/usr/bin/env bash
# GROW-MoE v3 (user design): START lean ~136M dense (E=1, d=896 L=14). Train the BASE to genuine SATURATION first
# (loss-EMA plateau, not entropy noise), THEN grow experts one by one up to E=13 (~1.22B total, ~136M active FLAT).
# At E=13, STOP growing and keep training to convergence (EMAX cap). Fixed trigger + longer base training (GROW_EVERY
# big) so growth happens AFTER saturation, not early. COMPILE(~1.5x)+bf16+ckpt. window2 176B + recall mix.
cd /opt/code/membrae
exec >> /opt/code/membrae/brainlogs/grow_moe.log 2>&1
export D=896 L=14 E0=1 EMAX=13 BS=4 RECALL_FRAC=0.15 GROW_EVERY=40000 VRAM_STOP_GB=16 COMPILE=1
exec /opt/code/parakeet_env/bin/python -u /opt/code/membrae/grow_moe_train.py
