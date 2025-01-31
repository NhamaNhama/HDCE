#!/usr/bin/env bash

set -e

CONFIG_PATH="config/default.yaml"
CKPT_PATH="my_model_checkpoint.ckpt"
TEXT="This is a sample input for HDCE."
CONCEPT="apple"

python src/inference/infer.py \
  --config $CONFIG_PATH \
  --ckpt $CKPT_PATH \
  --text "$TEXT" \
  --concept "$CONCEPT" 