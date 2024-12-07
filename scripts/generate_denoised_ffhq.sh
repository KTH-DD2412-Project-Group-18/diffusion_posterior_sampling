#!/bin/bash

OOD_DIR="./datasets/celebahq/celeba_hq_256"
MODEL_PATH="models/ffhq_baseline.pt"
START_TIME=$(date +%s)
echo "Starting inverse sampling at $(date)"

for image_file in "$OOD_DIR"/*.jpg; do
    [ -e "$image_file" ] || continue
    
    echo "Processing image: $image_file"
    echo "Current time: $(date)"
    
    temp_dir=$(mktemp -d)
    cp "$image_file" "$temp_dir/"
    
    poetry run python scripts/image_sample.py \
        --attention_resolutions "16" \
        --class_cond "False" \
        --diffusion_steps "1000" \
        --dropout "0.0" \
        --image_size "256" \
        --learn_sigma "True" \
        --noise_schedule "linear" \
        --num_channels "128" \
        --num_head_channels "64" \
        --num_res_blocks "1" \
        --resblock_updown "True" \
        --use_fp16 "False" \
        --use_scale_shift_norm "True" \
        --model_path "$MODEL_PATH" \
        --num_samples "1" \
        --batch_size "1" \
        --timestep_respacing "1000" \
        --dps_update "True" \
        --measurement_model "RandomInpainting" \
        --inpainting_noise_level "0.92" \
        --noise_model "gaussian" \
        --sigma "0.05" \
        --step_size "5.5" \
        --data_path "$temp_dir" \
        --sampling_batch_size "1" \
        --single_image_data "True" 

    rm -rf "$temp_dir"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"