#!/bin/bash

set -euo pipefail

MODE=${MODE:-"main_results"}

declare -a DATASETS=(
    "gsm8k 4"
    "humaneval 0"
    "minerva_math 4"
    "bbh 3"
    "mbpp 3"
)

declare -a METHODS=("low_confidence" "core")

case "${MODE}" in
    main_results)
        for entry in "${DATASETS[@]}"; do
            read -r TASK FEWSHOT <<< "${entry}"
            echo "[Main Results] Running ${TASK} with ${FEWSHOT} few-shot examples"
            bash run.sh "${TASK}" "NUM_FEWSHOT=${FEWSHOT}"
        done
        ;;

    *)
        exit 1
        ;;
esac