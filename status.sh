#!/bin/bash
# One-shot status of the Window 2 growth run. Usage:  bash status.sh
cd /opt/code/membrae
echo "==================== PRAGNOSIA W2 STATUS ===================="
echo "-- training alive? --"
pgrep -f "run_fast.py" >/dev/null && echo "  RUNNING (pid $(pgrep -f run_fast.py | head -1))" || echo "  NOT running"
echo "-- latest step / loss / ppl / speed --"
tail -c 4000 window2_train.log | tr '\r' '\n' | grep -oE "[0-9]+/4000000.*epoch=[0-9.]+" | tail -1
echo "-- last validation (perplexity) --"
grep "VAL_PPL" window2_train.log | tail -3 | grep -oE "it=[0-9]+ +VAL_PPL=[0-9.]+ +\(best [0-9.]+\)"
echo "-- GROWTH events (model scaling up) --"
grep -E "GREW|saturated" window2_train.log | tail -5 | sed 's/^ *//' || echo "  none yet (still at 276M)"
echo "-- current model size (pragnosia.json) --"
/opt/code/eval_env/bin/python -c "import json;c=json.load(open('pragnosia.json'));print('  layers='+str(c['layers'])+' mlp_mult='+str(c['mlp_mult']))" 2>/dev/null
echo "-- GPU --"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu --format=csv,noheader | sed 's/^/  /'
echo "-- RESUME (if training is stopped) --"
if pgrep -f "run_fast.py" >/dev/null; then
  echo "  running — to resume after a stop:  bash resume.sh"
else
  echo "  STOPPED — resume from last checkpoint with:  bash resume.sh"
fi
echo "  test all faculties (no training impact):  bash faculty_test.sh"
echo "============================================================"
