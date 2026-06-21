#!/bin/bash
# Single-process science+code top-up (common-pile arXiv fills science; non-gated code is the
# ceiling). Streams sources fully (no split_dataset_by_node slicing), then appends onto window2.
cd /opt/code/membrae
export HF_HOME=/mnt/kv_cache/hf_home TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=24
echo "[topup1] start @ $(date)" > topup_build.log
/opt/code/parakeet_env/bin/python prepare_scale.py --only topup >> topup_build.log 2>&1
tsz=$(stat -c %s /mnt/kv_cache/pragnosia_data/topup_train.bin)
cat /mnt/kv_cache/pragnosia_data/topup_train.bin >> /mnt/kv_cache/pragnosia_data/window2_train.bin
wsz=$(stat -c %s /mnt/kv_cache/pragnosia_data/window2_train.bin)
echo "[topup1] APPENDED $(echo "scale=2;$tsz/2/1000000000"|bc)B -> window2 now $(echo "scale=2;$wsz/2/1000000000"|bc)B @ $(date)" >> topup_build.log
echo "[topup1] done @ $(date)" >> topup_build.log
