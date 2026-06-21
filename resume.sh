#!/bin/bash
# Resume the Window 2 growth run from the latest checkpoint (pragnosia.pt), full H100,
# compile + prefetch + growth ON. Safe to run after any stop/crash/reboot — it picks up
# the grown arch from pragnosia.json and continues. Usage:  bash resume.sh
cd /opt/code/membrae

if pgrep -f "run_fast.py" >/dev/null; then
  echo "training already RUNNING (pid $(pgrep -f run_fast.py | head -1)) — not starting a second one."
  exit 0
fi
if [ ! -f pragnosia.pt ]; then echo "no checkpoint pragnosia.pt to resume from"; exit 1; fi

# show what we're resuming
/opt/code/eval_env/bin/python -c "import json,torch;c=json.load(open('pragnosia.json'));sd=torch.load('pragnosia.pt',map_location='cpu',weights_only=True);print('resuming:',round(sum(v.numel() for v in sd.values())/1e6),'M params,',c['layers'],'layers, mlp_mult',c['mlp_mult'],'| train_bin',c['train_bin'])"

export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PREFETCH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup /opt/code/parakeet_env/bin/python run_fast.py --resume --no-grow --bs 48 --lr 1e-4 --steps 4000000 \
      > window2_refine.log 2>&1 &
echo "RESUMED — pid $! — logging to window2_refine.log"
