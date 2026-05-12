#!/bin/bash
# set -e
PYTHON=/share/miniconda3/envs/vllm/bin/python
$PYTHON ./trsexp/run_exp_prompt.py
$PYTHON ./trsexp/vllm_openai_server.py &
PID=$!
sleep 600
$PYTHON ./trsexp/run_exp_prompt.py --judge
kill $PID
