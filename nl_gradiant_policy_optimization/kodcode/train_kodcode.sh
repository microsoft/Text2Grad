#!/bin/bash
# KodCode Training Script
# This script trains a language model using PPO with KodCode dataset

# ===== Model Configuration =====
# Reward model path
REWARD_MODEL_NAME="../nl_reward_model/kodcode/ckpt/text2grad_kodcode_RM/0_4400_merge"

# Base model configuration
BASE_MODEL_ADAPTER_MODEL=""
BASE_MODEL_NAME='meta-llama/Llama-3.2-3B-Instruct'

# ===== Training Parameters =====
# Output and logging
OUTPUT_DIR="ckpt/text2grad_kodcode"
LOG_WITH="wandb"  # Options: wandb, tensorboard, or none
PROJECT_NAME="kodcode"  # Replace with your project name
TRACKER_KWARGS="{\"wandb\": {\"entity\": \"your_wandb_entity\", \"name\": \"your_run_name\"}}"

# Learning parameters
LEARNING_RATE=5e-7
LR_SCHEDULER_TYPE="cosine"
TRAIN_EPOCHS=4
STEPS=6000
SEED=42

# Batch settings
MINI_BATCH_SIZE=1
BATCH_SIZE=8
PPO_EPOCHS=4
GRADIENT_ACCUMULATION_STEPS=8

# Generation settings
PROMPT_MAX_LENGTH=970
ANSWER_MAX_LENGTH=128
BATCHED_GEN=true

# PPO specific settings
INIT_KL_COEF=0.07
ADAP_KL_CTRL=true
KL_PENALTY="full"
TARGET_KL=1
EARLY_STOPPING=false
ADAFACTOR=false
REWARD_MODE=1
MASK_LOSS=""  # loss_v

# Checkpoint settings
SAVE_FREQ=20
LOCAL_RANK=0

# ===== Data Configuration =====
DATA_FILE_PATH='./data/kodcode_light_rl_10k_train.json'

# ===== Environment Configuration =====
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_NTHREADS=4
export WANDB_INIT_TIMEOUT=600

# ===== Training Function =====
# This function attempts to run training and automatically resumes from the latest checkpoint if it fails
run_training() {
    if ! accelerate launch \
        --config_file accelerate_config.yaml \
        train_kodcode.py \
        --base_model_name "$BASE_MODEL_NAME" \
        --base_model_adapter_model "$BASE_MODEL_ADAPTER_MODEL" \
        --reward_model_name "$REWARD_MODEL_NAME" \
        --log_with "$LOG_WITH" \
        --learning_rate "$LEARNING_RATE" \
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
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
        --tracker_kwargs "$TRACKER_KWARGS" \
        --prompt_max_length "$PROMPT_MAX_LENGTH" \
        --answer_max_length "$ANSWER_MAX_LENGTH" \
        --mask_loss "$MASK_LOSS"; then

        echo "Training failed, attempting to resume from latest checkpoint..."
        # Find the latest checkpoint
        LATEST_CKPT=$(ls -t "$OUTPUT_DIR"/epoch_*_step_* | head -n 1)
        if [ -n "$LATEST_CKPT" ]; then
            export BASE_MODEL_ADAPTER_MODEL="$LATEST_CKPT"
            echo "Resuming training from checkpoint: $LATEST_CKPT"
            # Restart training
            run_training
        else
            echo "No checkpoints found in $OUTPUT_DIR"
            exit 1
        fi
    fi
}

# Start the training process
run_training
