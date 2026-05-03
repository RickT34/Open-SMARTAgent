#!/bin/bash
/share/miniconda3/envs/vllm/bin/python trsexp/serpar_server.py \
  --secret-file secret.json \
  --model gpt-4o \
  --host 127.0.0.1 \
  --port 8765