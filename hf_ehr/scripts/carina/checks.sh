#!/bin/bash

# Sanity checks for script CLI args
MODEL_SIZE=$1
TOKENIZER=$2
CONTEXT_LENGTH=$3
DATALOADER_MODE=$4

# Model size
VALID_MODEL_SIZES=("tiny" "base" "medium" "large")
if [[ ! " ${VALID_MODEL_SIZES[*]} " == *" $MODEL_SIZE "* ]]; then
    echo "Error: The command line argument '$MODEL_SIZE' is not valid. It must be one of: ${VALID_MODEL_SIZES[*]}"
    exit 1
fi

# Tokenizer
VALID_TOKENIZERS=("femr" "clmbr" "desc")
if [[ ! " ${VALID_TOKENIZERS[*]} " == *" $TOKENIZER "* ]]; then
    echo "Error: The command line argument '$TOKENIZER' is not valid. It must be one of: ${VALID_TOKENIZERS[*]}"
    exit 1
fi

# Context length
if [[ ! "$CONTEXT_LENGTH" =~ ^[0-9]+$ ]]; then
    echo "Error: The command line argument '$CONTEXT_LENGTH' is not a positive integer."
    exit 1
fi

# Dataloader mode
VALID_DATALOADER_MODES=("approx" "batch")
if [[ ! " ${VALID_DATALOADER_MODES[*]} " == *" $DATALOADER_MODE "* ]]; then
    echo "Error: The command line argument '$DATALOADER_MODE' is not valid. It must be one of: ${VALID_DATALOADER_MODES[*]}"
    exit 1
fi