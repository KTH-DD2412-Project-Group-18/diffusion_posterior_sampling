#!/bin/bash

EVAL_DIR="./datasets/imagenet/eval_imgs"
MODEL_PATH="models/256x256_diffusion_uncond.pt"
START_TIME=$(date +%s)
echo "Starting inverse sampling at $(date)"

echo "Processing one image from each class folder in $EVAL_DIR"
for class_folder in "$EVAL_DIR"/*; do
    if [ ! -d "$class_folder" ]; then
        continue
    fi
    echo "Processing class folder: $class_folder"
    echo "Current time: $(date)"
    poetry run python scripts/image_sample.py \
        --attention_resolutions "32,16,8" \
        --class_cond "False" \
        --diffusion_steps "1000" \
        --dropout "0.0" \
        --image_size "256" \
        --learn_sigma "True" \
        --noise_schedule "linear" \
        --num_channels "256" \
        --num_head_channels "64" \
        --num_heads "4" \
        --num_res_blocks "2" \
        --resblock_updown "True" \
        --use_fp16 "False" \
        --use_scale_shift_norm "True" \
        --model_path "$MODEL_PATH" \
        --num_samples "1" \
        --batch_size "1" \
        --timestep_respacing "1000" \
        --dps_update "True" \
        --measurement_model "RandomInpainting" \
        --noise_model "gaussian" \
        --sigma "0.05" \
        --step_size "1.0" \
        --data_path "$class_folder" \
        --sampling_batch_size "1" \
        --single_image_data "True"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"