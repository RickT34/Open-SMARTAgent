#!/bin/bash
# set -e
PYTHON=/share/miniconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
$PYTHON ./trsexp/run_exp_prompt.py
$PYTHON ./trsexp/vllm_openai_server.py &
PID=$!
sleep 600
$PYTHON ./trsexp/run_exp_prompt.py --judge
kill $PID
