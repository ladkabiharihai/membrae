#!/usr/bin/env bash
# GROW-MoE v2 (user design): START as a LEAN single-expert dense model (~363M, E=1), then GROW EXPERTS one at a time
# on entropy saturation. Active compute stays FLAT at ~363M as total grows E=1->6 (363M -> ~1.67B). Cleaner+leaner
# than starting E=4 (no dead experts carried early). COMPILE(~1.5x)+bf16+ckpt. window2 176B + recall mix.
cd /opt/code/membrae
exec >> /opt/code/membrae/brainlogs/grow_moe.log 2>&1
export D=1280 L=20 E0=1 EMAX=6 BS=4 RECALL_FRAC=0.15 GROW_EVERY=30000 VRAM_STOP_GB=16 COMPILE=1
exec /opt/code/parakeet_env/bin/python -u /opt/code/membrae/grow_moe_train.py
