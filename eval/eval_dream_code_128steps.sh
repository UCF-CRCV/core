model=Dream-org/Dream-v0-Base-7B
common_args="pretrained=${model},max_new_tokens=512,diffusion_steps=128,add_bos_token=true,temperature=0.2,top_p=0.95"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
eval_script="${repo_root}/eval/eval.py"

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${repo_root}/eval_instruct${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29510 "${eval_script}" --model dream \
    --model_args ${common_args},escape_until=true \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path evals_results/humaneval-ns0-steps128-len512 \
    --log_samples \
    --confirm_run_unsafe_code &

# NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29511 "${eval_script}" --model dream \
    --model_args ${common_args} \
    --tasks mbpp \
    --num_fewshot 3 \
    --batch_size 1 \
    --output_path evals_results/mbpp-ns3-steps128-len512 \
    --log_samples \
    --confirm_run_unsafe_code

wait
