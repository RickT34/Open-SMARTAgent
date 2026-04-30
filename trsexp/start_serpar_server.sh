#!/bin/bash
/share/miniconda3/envs/vllm/bin/python trsexp/serpar_server.py \
  --dataset data_inference/domain_time_smart.json \
  --dataset data_inference/ood_mint_smart.json \
  --secret-file secret.json \
  --model gpt-4o \
  --host 127.0.0.1 \
  --port 8765