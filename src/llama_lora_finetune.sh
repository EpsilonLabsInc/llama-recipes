#!/bin/bash
set -x

# Set default values
GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# Model and dataset settings
DATASET_FILE="recipes/quickstart/finetuning/datasets/mimic2_dataset.py"
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
DIST_CHECKPOINT_ROOT_FOLDER="./finetuned_model"
DIST_CHECKPOINT_FOLDER="fine-tuned"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=29500  # Change port if needed

LR=1e-5
NUM_EPOCHS=3

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/mnt/data/ruian/llama3.2/${TIMESTAMP}_${LR}"

# Ensure the output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Run the torchrun command
torchrun \
  --nnodes=1 \
  --nproc_per_node=${GPUS} \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT} \
  recipes/quickstart/finetuning/finetuning.py \
  --enable_fsdp \
  --lr ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size_training ${PER_DEVICE_BATCH_SIZE} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --model_name ${MODEL_NAME} \
  --dist_checkpoint_root_folder ${DIST_CHECKPOINT_ROOT_FOLDER} \
  --dist_checkpoint_folder ${DIST_CHECKPOINT_FOLDER} \
  --use_fast_kernels \
  --dataset "custom_dataset" \
  --custom_dataset.test_split "test" \
  --custom_dataset.file ${DATASET_FILE} \
  --run_validation True \
  --batching_strategy "padding" \
  --use_peft \
  --peft_method "lora" \
  --output_dir ${OUTPUT_DIR} \
  --report_to "tensorboard" \
  --logging_steps 5 \
  --save_metrics True \
  --save_steps 100 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
