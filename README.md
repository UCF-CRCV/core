# CORE: Context-Robust Remasking for Diffusion Language Models

**Kevin Zhai, Sabbir Mollah, Zhenyi Wang, Mubarak Shah**  
University of Central Florida

[![arXiv](https://img.shields.io/badge/arXiv-2602.04096-b31b1b.svg)](https://arxiv.org/abs/2602.04096)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ucf-crcv.github.io/core/)

CORE is a training-free, inference-time revision method for Masked Diffusion Models. Standard decoders freeze a token once it is unmasked, even when later context exposes it as wrong. Instead of trusting static/stale confidence, CORE identifies *context-brittle* tokens by stress-testing them: it masks a small candidate set, measures each token's instability (drop in likelihood) under that perturbed context, and remasks the most unstable ones for resampling. The method plugs into the LLaDA/Dream sampler and adds only a handful of extra forward passes.

## Setup

Requires Python 3.12 and a CUDA GPU (experiments use a single A100-80GB).

```bash
# 1) Install torch from the appropriate CUDA index (see requirements.txt for notes)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Install the rest of the pinned dependencies
pip install -r requirements.txt
```

> **Dependency compatibility (important):**
> - `transformers` **must** be 4.46.x. LLaDA's `trust_remote_code` modeling file predates the
>   transformers 5.x loader refactor and fails to load on 5.x.
> - `lm-eval` 4.x exposes `LM.device` as a read-only property; `eval.py` handles this compatibly.

The base model `GSAI-ML/LLaDA-8B-Base` (~15 GB) is fetched from the Hugging Face Hub on first run.
Set `HF_HOME` to control the cache location, and after the initial download you can run cache-only
with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/path/to/hf_cache
hf download GSAI-ML/LLaDA-8B-Base   # optional explicit prefetch
```

## Quick Start

Reproduce the main-results table (runs `low_confidence` and `core` across GSM8K, HumanEval,
Minerva-MATH, BBH, MBPP with the paper's few-shot counts):

```bash
bash run_all.sh
```

## Running a Single Task

`run.sh` takes the task name as its argument and reads settings from environment variables:

```bash
METHOD=core STEPS=128 NUM_FEWSHOT=3 SEED=1234 bash run.sh mbpp
```

| Variable | Default | Description |
|----------|---------|-------------|
| `METHOD` | `low_confidence` | Unmasking / remasking strategy (see below) |
| `STEPS` | `128` | Number of diffusion steps |
| `NUM_FEWSHOT` | `0` | Few-shot examples |
| `SEED` | `1234` | Random seed |

Under the hood this calls `eval.py` (an [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
plugin registered as `llada_dist`) with `gen_length=512, block_length=512`. To pass other harness
flags (e.g. `--limit` for a quick smoke test), call `eval.py` directly:

```bash
python eval.py --llada_seed 1234 --tasks gsm8k --num_fewshot 4 --limit 2 \
  --confirm_run_unsafe_code --model llada_dist --log_samples \
  --output_path ./logs/smoke_gsm8k_core \
  --model_args model_path=GSAI-ML/LLaDA-8B-Base,gen_length=256,steps=64,block_length=256,remasking=core
```

### Remasking strategies (`remasking=` / `METHOD`)

| Value | Description |
|-------|-------------|
| `low_confidence` | Standard LLaDA unmasking baseline |
| `topk_margin` | Top-2 probability margin unmasking |
| `random` | Random unmasking |
| `core` | **CORE** — instability-based context-robust remasking (ours) |
| `margin_remask` | Compute-matched control: remask by smallest margin |
| `random_remask` | Compute-matched control: remask at random |

### CORE knobs (environment variables, read in `generate.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `REVISE_EVERY` | `8` | Run a revision pass every *E* steps (0 disables) |
| `CANDIDATE_M` | `32` | Candidate set size *m* stress-tested per revision pass |
| `BASE_MASKING` | `confidence` | Base unmasking score: `confidence` or `margin` |
| `JOINT_REEVAL` | `0` | If `1`, add a forward pass on the corrected context (ablation; +1 NFE) |
| `MECH_SAVE_DIR` | _(unset)_ | If set, dump per-token instability/mechanism stats here |
| `CORE_DEBUG` | `0` | If `1`, print verbose per-step `[budget]`/`[remask]` traces |
| `TEMPERATURE` | `0.0` | Sampling temperature (>0 enables stochastic decoding + per-example seeding) |

Revision is active only in the intermediate step window `[0.25, 0.75)` and revises at most
`k_rm = 1` token per pass (matching the paper).

## Citation
If you use this code or find the paper useful, please cite:
```bibtex
@inproceedings{zhai2026core,
  title={{CORE}: Context-Robust Remasking for Diffusion Language Models},
  author={Kevin Zhai and Sabbir Mollah and Zhenyi Wang and Mubarak Shah},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=bmKHxLWkz9}
}
```

## Acknowledgements

We thank colleagues and collaborators for discussions and feedback that improved this work. We also acknowledge the authors and maintainers of the open-source libraries, pretrained models (e.g., LLaDA), and evaluation suites (e.g., lm-eval) used in this repository, along with the creators of the benchmarks and datasets used in our experiments. Finally, we appreciate the compute infrastructure and operational support that enabled the runs reported here.
