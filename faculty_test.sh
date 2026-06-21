#!/bin/bash
# Test the LATEST model on ALL brain faculties WITHOUT interrupting training.
# Snapshots the live checkpoint (consistent read) and runs CPU-only (CUDA hidden), so it
# never competes with the GPU training run.  Usage:  bash faculty_test.sh
cd /opt/code/membrae
SNAP=/tmp/faculty_ckpt.pt
cp pragnosia.pt "$SNAP" || { echo "no pragnosia.pt yet"; exit 1; }
echo "snapshot taken from live checkpoint -> $SNAP"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/opt/code/membrae PYTHONUNBUFFERED=1 \
  /opt/code/eval_env/bin/python -u faculty_test.py "$SNAP" 2>/dev/null
