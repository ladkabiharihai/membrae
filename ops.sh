#!/bin/bash
# Pragnosia H100 training ops — one script (merged from status.sh / resume.sh / faculty_test.sh).
# Runs on the training box (/opt/code/membrae). Usage:
#   bash ops.sh status   — training alive? latest step / ppl / growth / GPU
#   bash ops.sh resume   — resume the run from the latest checkpoint (compile + prefetch, no-grow refine)
#   bash ops.sh test     — test ALL faculties on the live model, CPU-only (never touches training)
# The one-off data-building ops (window cutover, shard top-up, autostart) are in git history;
# the corpora are already built (data/window2_train.bin, ~317GB).
cd /opt/code/membrae
PY=/opt/code/eval_env/bin/python

case "${1:-status}" in
status)
  echo "==================== PRAGNOSIA STATUS ===================="
  pgrep -f run_fast.py >/dev/null && echo "  RUNNING (pid $(pgrep -f run_fast.py | head -1))" || echo "  NOT running"
  LOG=$(ls -t *_refine.log *_train.log 2>/dev/null | head -1)
  echo "-- step / loss / ppl / speed (log: $LOG) --"
  tail -c 4000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -oE "[0-9]+/[0-9]+.*epoch=[0-9.]+" | tail -1
  echo "-- last validations --"
  grep "VAL_PPL" "$LOG" 2>/dev/null | tail -3 | grep -oE "it=[0-9]+ +VAL_PPL=[0-9.]+ +\(best [0-9.]+\)"
  echo "-- growth events --"
  grep -E "GREW|saturated" "$LOG" 2>/dev/null | tail -5 | sed 's/^ *//'
  $PY -c "import json;c=json.load(open('pragnosia.json'));print('-- arch: layers='+str(c['layers'])+' mlp_mult='+str(c['mlp_mult']))" 2>/dev/null
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader | sed 's/^/  GPU /'
  echo "========================================================="
  ;;
resume)
  pgrep -f run_fast.py >/dev/null && { echo "already RUNNING (pid $(pgrep -f run_fast.py|head -1)) — not starting a second"; exit 0; }
  [ -f pragnosia.pt ] || { echo "no checkpoint pragnosia.pt to resume from"; exit 1; }
  $PY -c "import json,torch;c=json.load(open('pragnosia.json'));sd=torch.load('pragnosia.pt',map_location='cpu',weights_only=True);print('resuming:',round(sum(v.numel() for v in sd.values())/1e6),'M params,',c['layers'],'layers mlp_mult',c['mlp_mult'],'| train_bin',c['train_bin'])"
  export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PREFETCH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup /opt/code/parakeet_env/bin/python run_fast.py --resume --no-grow --bs 48 --lr 1e-4 --steps 4000000 \
        > window2_refine.log 2>&1 &
  echo "RESUMED — pid $! — logging to window2_refine.log"
  ;;
test)
  SNAP=/tmp/faculty_ckpt.pt
  cp pragnosia.pt "$SNAP" || { echo "no pragnosia.pt yet"; exit 1; }
  echo "snapshot from live checkpoint -> $SNAP ; running CPU-only (training untouched)"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/opt/code/membrae PYTHONUNBUFFERED=1 $PY -u faculty_test.py "$SNAP" 2>/dev/null
  ;;
*)
  echo "usage: bash ops.sh {status|resume|test}"
  ;;
esac
