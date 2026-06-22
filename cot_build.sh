#!/bin/bash
# Build the long multi-step CoT corpus (~18B: OpenMathReasoning + OpenThoughts2 + OpenMathInstruct
# + NuminaMath) across 4 shards, then APPEND it into the existing window1 + window2 corpora so the
# next training run sees it. Runs detached.
cd /opt/code/membrae
export HF_HOME=/mnt/kv_cache/hf_home TOKENIZERS_PARALLELISM=true
D=/mnt/kv_cache/pragnosia_data
K=4
echo "[cot] launching $K shards @ $(date)" > cot_build.log
pids=()
for i in $(seq 0 $((K-1))); do
  RAYON_NUM_THREADS=8 nohup /opt/code/parakeet_env/bin/python prepare_scale.py \
      --only cot --shards $K --shard-id $i > cot_s${i}.log 2>&1 &
  pids+=($!)
done
echo "[cot] shard PIDs: ${pids[*]}" >> cot_build.log
for p in "${pids[@]}"; do wait "$p"; done
echo "[cot] shards finished, assembling @ $(date)" >> cot_build.log
cat $D/cot_s*.bin > $D/cot_train.bin
csz=$(stat -c %s $D/cot_train.bin)
# APPEND into the existing data (both windows) — non-destructive; trainer memmaps the whole file
cat $D/cot_train.bin >> $D/window2_train.bin
cat $D/cot_train.bin >> $D/window1_train.bin
w1=$(stat -c %s $D/window1_train.bin); w2=$(stat -c %s $D/window2_train.bin)
echo "[cot] BUILT $((csz/2/1000000000))B CoT -> appended into window1 ($((w1/2/1000000000))B) + window2 ($((w2/2/1000000000))B) @ $(date)" >> cot_build.log
rm -f $D/cot_s*.bin
echo "[cot] done @ $(date)" >> cot_build.log
