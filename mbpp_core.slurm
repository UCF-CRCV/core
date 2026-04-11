#!/bin/bash
#SBATCH --gres gpu:ampere:1
#SBATCH -C gmem48
#SBATCH -c 12
#SBATCH --mem 88G
#SBATCH --job-name dream_mbpp_core
#SBATCH --output logs/mbpp_core_%A.out
#SBATCH --error logs/mbpp_core_%A.err

module load cuda
source activate dream

cd /home/sa162103/workspace/dream-cluster

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="/home/sa162103/workspace/dream-cluster/eval_instruct${PYTHONPATH:+:${PYTHONPATH}}"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=UTF-8

accelerate launch --main_process_port 29513 /home/sa162103/workspace/dream-cluster/eval/eval.py --model dream \
  --model_args pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=512,diffusion_steps=128,add_bos_token=true,temperature=0.2,core_style=true,alg=core,block_length=512,remask_k=1,revise_every=8,candidate_m=32,base_masking=confidence \
  --tasks mbpp \
  --num_fewshot 3 \
  --batch_size 1 \
  --output_path eval/evals_results/mbpp-core-ns3-steps128-len512 \
  --log_samples \
  --confirm_run_unsafe_code
