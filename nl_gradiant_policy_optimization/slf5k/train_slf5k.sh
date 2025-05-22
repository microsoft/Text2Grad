#!/bin/bash
# Set strict error handling
set -e

# Model paths
REWARD_MODEL_NAME="" # Please enter the reward model path here
BASE_MODEL_ADAPTER_MODEL="" # If using an adapter model, enter the path here
BASE_MODEL_NAME="" # Please enter the base model path here

# Training configuration
LOG_WITH="wandb"
LEARNING_RATE=1e-6
LR_SCHEDULER_TYPE="linear"
OUTPUT_MAX_LENGTH=90
MINI_BATCH_SIZE=1
BATCH_SIZE=12
PPO_EPOCHS=4
GRADIENT_ACCUMULATION_STEPS=12
ADAFACTOR=True
EARLY_STOPPING=false
TARGET_KL=6
BATCHED_GEN=true
SAVE_FREQ=10
REWARD_MODE=1
OUTPUT_DIR="ckpt/text2grad_slf5k"
SEED=42
TRAIN_EPOCHS=4
STEPS=6000
INIT_KL_COEF=0.2
ADAP_KL_CTRL=true
KL_PENALTY="full"
LOCAL_RANK=0

PROJECT_NAME="Text2Grad-SLF5K" # e.g., "Text2Grad-SLF5K"
DATA_FILE_PATH="<PATH_TO_TRAINING_DATA>" # e.g., "../data/train.json"
VALID_DATA_FILE_PATH="<PATH_TO_VALIDATION_DATA>" # e.g., "../data/valid.json"
TRACKER_KWARGS="{\"wandb\": {\"entity\": \"<YOUR_WANDB_ENTITY>\", \"name\": \"<YOUR_RUN_NAME>\"}}"
PROMPT_MAX_LENGTH=640
ANSWER_MAX_LENGTH=128
MASK_LOSS="" # Leave empty if not needed
STRATEGY="qa_strategy"

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run PPO training
echo "Starting PPO training..."
python train.py \
    --base_model_name "$BASE_MODEL_NAME" \
    --base_model_adapter_model "$BASE_MODEL_ADAPTER_MODEL" \
    --reward_model_name "$REWARD_MODEL_NAME" \
    --log_with "$LOG_WITH" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --output_max_length "$OUTPUT_MAX_LENGTH" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --ppo_epochs "$PPO_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --adafactor "$ADAFACTOR" \
    --early_stopping "$EARLY_STOPPING" \
    --target_kl "$TARGET_KL" \
    --kl_penalty "$KL_PENALTY" \
    --batched_gen "$BATCHED_GEN" \
    --save_freq "$SAVE_FREQ" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --train_epochs "$TRAIN_EPOCHS" \
    --steps "$STEPS" \
    --init_kl_coef "$INIT_KL_COEF" \
    --adap_kl_ctrl "$ADAP_KL_CTRL" \
    --local_rank "$LOCAL_RANK" \
    --project_name "$PROJECT_NAME" \
    --data_file_path "$DATA_FILE_PATH" \
    --valid_data_file_path "$VALID_DATA_FILE_PATH" \
    --tracker_kwargs "$TRACKER_KWARGS" \
    --prompt_max_length "$PROMPT_MAX_LENGTH" \
    --answer_max_length "$ANSWER_MAX_LENGTH" \
    --mask_loss "$MASK_LOSS" \
    --reward_mode "$REWARD_MODE" \
    --strategy "$STRATEGY"