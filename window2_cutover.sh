#!/bin/bash
# ARMED CUTOVER: when the science/code top-up finishes, free the whole H100 and start the
# Window 2 full-throttle GROWTH run. Stops Window 1 training + all 3 prod services, then
# launches W2 on the freed card (fit_batch fills 80GB), compile on, growth ON, identity
# injection on, resuming the W1-trained 276M. Detached; survives the session.
cd /opt/code/membrae
LOG=window2_cutover.log
echo "[cutover] ARMED, watching top-up @ $(date)" > $LOG

# 1) wait for the top-up to finish AND append onto window2_train.bin
until grep -q "\[topup1\] done @" topup_build.log 2>/dev/null; do sleep 30; done
grep "APPENDED" topup_build.log | tail -1 >> $LOG
echo "[cutover] top-up done, window2 ready @ $(date)" >> $LOG

# 2) stop Window 1 training (last checkpoint is <=500 steps old; W2 resumes from it)
for pid in $(pgrep -f "run_fast.py"); do kill -TERM "$pid" 2>/dev/null; done
sleep 8; pkill -9 -f "run_fast.py" 2>/dev/null
echo "[cutover] Window 1 training stopped @ $(date)" >> $LOG

# 3) stop ALL prod services to free the H100 (user-authorized for the Jun19-21 window)
sudo systemctl stop gemma-api.service kokoro-tts.service parakeet-stt.service 2>>$LOG
echo "[cutover] prod services stopped (gemma/kokoro/parakeet) @ $(date)" >> $LOG

# 4) wait for VRAM to actually release (vLLM frees slowly)
freed=0
for i in $(seq 1 90); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -gt 72000 ]; then freed=1; break; fi
  sleep 5
done
echo "[cutover] GPU free ${free}MiB (freed=$freed) @ $(date)" >> $LOG

# 5) point the trainer at the big window2 corpus
/opt/code/eval_env/bin/python -c "import json;p='pragnosia.json';c=json.load(open(p));c['train_bin']='window2_train';json.dump(c,open(p,'w'),indent=2);print('train_bin ->',c['train_bin'])" >> $LOG 2>&1

# 6) launch Window 2 FULL THROTTLE: full-H100 batch (fit_batch), compile, GROWTH on,
#    identity injection on, long horizon to consume as much data as the window allows.
export PRAGNOSIA_IDENTITY=identity_sentences.txt IDENTITY_EVERY=50 IDENTITY_LR=1e-4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
nohup /opt/code/parakeet_env/bin/python run_fast.py --resume --bs 0 --lr 2e-4 --steps 4000000 \
      > window2_train.log 2>&1 &
echo "[cutover] Window 2 training LAUNCHED PID $! (full H100, growth ON, identity ON) @ $(date)" >> $LOG
