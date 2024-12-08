#!/bin/bash
MEASUREMENT_MODEL="MotionBlur"
MODEL_PATH="models/ffhq_baseline.pt"
DATASETS=("eval_imgs_celeba_hq" "eval_imgs_ffhq")
DATASET_NAMES=("celebA" "ffhq")
STEP_SIZE="0.15"
INPAINTING_NOISE_LEVEL="0.92"

process_dataset() {
    local dir="./datasets/$1"
    local dataset_name="$2"
    local output_dir="./output/$MEASUREMENT_MODEL/$dataset_name/"
    echo "================================="
    echo "Processing dataset: $dataset_name"
    echo "Input directory: $dir"
    echo "Output directory: $output_dir"
    echo "Starting DPS-sampling at $(date)"
    echo "================================="
    for image_file in "$dir"/*.jpg; do
        [ -e "$image_file" ] || continue
        
        image_name=$(basename "$image_file")
        temp_dir=$(mktemp -d)
        echo ""
        echo "temp_dir: $temp_dir"
        echo "Processing image: $image_file"
        echo "Current time: $(date)"
        
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
            --measurement_model "$MEASUREMENT_MODEL" \
            --inpainting_noise_level "$INPAINTING_NOISE_LEVEL" \
            --noise_model "gaussian" \
            --sigma "0.05" \
            --step_size "$STEP_SIZE" \
            --data_path "$temp_dir" \
            --sampling_batch_size "1" \
            --single_image_data "True" \
            --img_name "$image_name" \
            --output_dir "$output_dir"
            
        rm -rf "$temp_dir"
    done
}

START_TIME=$(date +%s)

for i in "${!DATASETS[@]}"; do
    process_dataset "${DATASETS[i]}" "${DATASET_NAMES[i]}"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "Total elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"