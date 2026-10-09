#!/bin/bash
# ncu full-metrics profile of the FA4 SM120 fwd kernel, decode b1 k4096.
# Usage:  sudo /home/danielk/code/fa4_sm120/benchmarks/ncu_decode_b1.sh [num_splits]
NS="${1:-1}"
cd /home/danielk/code/vllm || exit 1
exec env PYTHONPATH=/home/danielk/code \
  /usr/local/cuda-13/bin/ncu \
  -k "regex:FlashAttentionForwardSm120" \
  --launch-count 1 --launch-skip 2 \
  --set full \
  --export "/tmp/fa4_b1_k4096_s${NS},force-overwrite" \
  .venv/bin/python /home/danielk/code/fa4_sm120/benchmarks/profile_decode_b1.py "$NS"
