#!/usr/bin/env bash
cd /opt/code/membrae
# MANUAL 400M: grew 216M/18L/feat8 -> 399M/35L/feat12 in ONE function-preserving jump (grow_to_400m.py), now TRAIN
# it (NOGROW=1, no auto-grow -- one jump + sustained training beats the thrashing spree #48). #53: the wall is
# language capacity -> a bigger model trained properly is the bet. BS=2 (35L feat12 fwd+bwd peak 16.3G, fits w/ prod).
# FEAT_STEP=2 must match so RESUME replays the GroupFeatNorm structure to reconstruct feat12.
RESUME=1 RECALL_FRAC=0.20 D=1024 LSTART=35 NOGROW=1 BS=2 WARM=2000 LR=3e-4 STEPS=100000000 FEAT_STEP=2 exec /opt/code/parakeet_env/bin/python -u scale_train.py
