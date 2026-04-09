#!/bin/bash

# GPU settings (device numbers to use)
export CUDA_VISIBLE_DEVICES=0,1

# Receive the first argument when running the script (default: gpt2)
MODEL_TYPE=${1:-gpt2}

# Automatically map the config file based on the input model type
if [ "$MODEL_TYPE" == "gpt2" ]; then
    CONFIG_FILE="configs/gpt_config.yaml"
elif [ "$MODEL_TYPE" == "bert" ]; then
    CONFIG_FILE="configs/bert_config.yaml"
else
    echo "Error: Unsupported model type. Please enter 'gpt2' or 'bert'."
    echo "Usage: bash scripts/train.sh [gpt2|bert]"
    exit 1
fi

echo "========================================================"
echo "Training started: $MODEL_TYPE model"
echo "Config file: $CONFIG_FILE"
echo "========================================================"

# Run the main script
python main.py \
    --model_type "$MODEL_TYPE" \
    --config "$CONFIG_FILE"