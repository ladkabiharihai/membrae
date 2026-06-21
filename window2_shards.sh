#!/bin/bash
# Orchestrate Window 2 as K disjoint shard-workers (parallel producers -> uses idle cores),
# then concatenate the parts into window2_train.bin. Order across parts is irrelevant
# (training samples random windows). Runs detached.
cd /opt/code/membrae
export HF_HOME=/mnt/kv_cache/hf_home TOKENIZERS_PARALLELISM=true
K=4
echo "[window2] launching $K shards @ $(date)" > window2_build.log
pids=()
for i in $(seq 0 $((K-1))); do
  RAYON_NUM_THREADS=6 nohup /opt/code/parakeet_env/bin/python prepare_scale.py \
      --only window2 --shards $K --shard-id $i > window2_s${i}.log 2>&1 &
  pids+=($!)
done
echo "[window2] shard PIDs: ${pids[*]}" >> window2_build.log
for p in "${pids[@]}"; do wait "$p"; done
echo "[window2] all shards finished, assembling @ $(date)" >> window2_build.log
cat /mnt/kv_cache/pragnosia_data/window2_s*.bin > /mnt/kv_cache/pragnosia_data/window2_train.bin
sz=$(stat -c %s /mnt/kv_cache/pragnosia_data/window2_train.bin)
echo "[window2] ASSEMBLED $((sz/2/1000000000))B tok @ $(date)" >> window2_build.log
rm -f /mnt/kv_cache/pragnosia_data/window2_s*.bin
echo "[window2] shard parts removed, done @ $(date)" >> window2_build.log
