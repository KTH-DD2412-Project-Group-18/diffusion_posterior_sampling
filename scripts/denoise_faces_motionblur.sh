#!/bin/bash

MEASUREMENT_MODEL="MotionBlur"
DATASETS=("eval_imgs_celeba_hq" "eval_imgs_ffhq")
DATASET_NAMES=("celebA" "ffhq")
STEP_SIZE="0.15"
INPAINTING_NOISE_LEVEL="0.92"


START_TIME=$(date +%s)

for i in "${!DATASETS[@]}"; do
    bash ./scripts/generate_faces.sh "${DATASETS[i]}" "${DATASET_NAMES[i]}" "$MEASUREMENT_MODEL" "$STEP_SIZE" "$INPAINTING_NOISE_LEVEL"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "Total elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"