#!/bin/bash
MEASUREMENT_MODEL="RandomInpainting"
MODEL_PATH="models/256x256_diffusion_uncond.pt"
DATASETS=("eval_imgs_imagenet")
DATASET_NAMES=("imagenet")
STEP_SIZE="0.5"
INPAINTING_NOISE_LEVEL="0.92"

START_TIME=$(date +%s)

for i in "${!DATASETS[@]}"; do
    bash ./scripts/generate_imagenet.sh "${DATASETS[i]}" "${DATASET_NAMES[i]}" "$MEASUREMENT_MODEL" "$STEP_SIZE" "$INPAINTING_NOISE_LEVEL"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "Total elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"