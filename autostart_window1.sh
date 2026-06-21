#!/bin/bash
# Detached watcher: when the Window-1 build finishes, point the trainer at it and
# launch a clean balanced run (resume from the post-trained 276M, growth OFF —
# growth is reserved for the Jun19-21 freed window). Survives independent of any session.
cd /opt/code/membrae
LOG=scale_build.log
echo "[autostart] watching $LOG for window1 completion @ $(date)" >> autostart.log

# 1) wait for window1 to finish building (the BUILT line prints after the file is closed)
until grep -q "\[window1\] BUILT" "$LOG" 2>/dev/null; do
  sleep 60
done
echo "[autostart] window1 BUILT detected @ $(date)" >> autostart.log
grep "\[window1\] BUILT" "$LOG" | tail -1 >> autostart.log

# 2) point the trainer at window1 (keep big_valid so ppl stays comparable)
/opt/code/eval_env/bin/python -c "import json;p='pragnosia.json';c=json.load(open(p));c['train_bin']='window1_train';json.dump(c,open(p,'w'),indent=2);print('train_bin ->',c['train_bin'])" >> autostart.log 2>&1

# 3) launch the balanced training run, resumed from pragnosia.pt
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
nohup /opt/code/parakeet_env/bin/python run_fast.py --resume --no-grow \
      --bs 32 --lr 2e-4 --steps 1500000 > window1_train.log 2>&1 &
echo "[autostart] training launched PID $! @ $(date)" >> autostart.log
