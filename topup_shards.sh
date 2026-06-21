#!/bin/bash
# Build the science+code top-up (4 disjoint shards) and APPEND it onto window2_train.bin,
# repairing Window 2's failed science/code sections. Runs while Window 1 training continues.
cd /opt/code/membrae
export HF_HOME=/mnt/kv_cache/hf_home TOKENIZERS_PARALLELISM=true
K=4
echo "[topup] launching $K shards @ $(date)" > topup_build.log
pids=()
for i in $(seq 0 $((K-1))); do
  RAYON_NUM_THREADS=6 nohup /opt/code/parakeet_env/bin/python prepare_scale.py \
      --only topup --shards $K --shard-id $i > topup_s${i}.log 2>&1 &
  pids+=($!)
done
echo "[topup] shard PIDs: ${pids[*]}" >> topup_build.log
for p in "${pids[@]}"; do wait "$p"; done
echo "[topup] shards done, assembling @ $(date)" >> topup_build.log
cat /mnt/kv_cache/pragnosia_data/topup_s*.bin > /mnt/kv_cache/pragnosia_data/topup_train.bin
tsz=$(stat -c %s /mnt/kv_cache/pragnosia_data/topup_train.bin)
# append onto window2 (training samples random windows, so order is irrelevant)
cat /mnt/kv_cache/pragnosia_data/topup_train.bin >> /mnt/kv_cache/pragnosia_data/window2_train.bin
wsz=$(stat -c %s /mnt/kv_cache/pragnosia_data/window2_train.bin)
echo "[topup] APPENDED $((tsz/2/1000000000))B -> window2 now $((wsz/2/1000000000))B @ $(date)" >> topup_build.log
rm -f /mnt/kv_cache/pragnosia_data/topup_s*.bin
echo "[topup] done @ $(date)" >> topup_build.log
