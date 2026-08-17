#!/usr/bin/env bash
# GROW-TO-1B dense, speed stack: COMPILE(1.45x #78)+BF16+symmetric-Taylor(1.2x #77)+GRAD_CKPT(fits w/ prod).
cd /opt/code/membrae
exec >> /opt/code/membrae/brainlogs/grow1b_compiled.log 2>&1
export RECALL_FRAC=0.15 D=2048 LSTART=6 NOGROW=0 COMPILE=1 BF16=1 GRAD_CKPT=1 BS=4 FEAT_CAP=8
export VRAM_STOP_GROW_GB=14 WARM=2000 LR=3e-4 STEPS=100000000
exec /opt/code/parakeet_env/bin/python -u /opt/code/membrae/scale_train.py
