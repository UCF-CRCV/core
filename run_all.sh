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
        for METHOD in "${METHODS[@]}"; do
            for entry in "${DATASETS[@]}"; do
                read -r TASK FEWSHOT <<< "${entry}"
                echo "[Main Results] method=${METHOD} task=${TASK} fewshot=${FEWSHOT}"
                METHOD="${METHOD}" NUM_FEWSHOT="${FEWSHOT}" bash run.sh "${TASK}"
            done
        done
        ;;

    *)
        exit 1
        ;;
esac