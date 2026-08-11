#!/usr/bin/env bash
cd /opt/code/membrae
D=1024 LSTART=18 NOGROW=1 BS=4 WARM=2000 LR=3e-4 STEPS=100000000 exec /opt/code/parakeet_env/bin/python -u scale_train.py
