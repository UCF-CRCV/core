#!/bin/bash

METHOD=${METHOD:-'low_confidence'}
STEPS=${STEPS:-128}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
SEED=${SEED:-1234}

TASK=$1

python eval.py \
    --llada_seed ${SEED} \
    --tasks ${TASK} --confirm_run_unsafe_code --num_fewshot ${NUM_FEWSHOT} \
    --model llada_dist --log_samples \
    --output_path "./logs/jsonl_${REMASKING_METHOD}" \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=512,steps=$STEPS,block_length=512,remasking="${METHOD}"

