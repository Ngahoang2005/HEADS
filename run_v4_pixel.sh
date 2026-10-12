#!/usr/bin/env bash
set -euo pipefail

cd /home/s24gbn1/Documents/httn/unilm/layoutlmv3

PYTHON=python
MODEL=models/layoutlmv3-base
HEAD=segment_outputs/segment_head_v4_pixel.pt

TRAIN_IMAGES=datasets/funsd/dataset/training_data/images
TRAIN_ANN=datasets/funsd/dataset/training_data/annotations
TEST_IMAGES=datasets/funsd/dataset/testing_data/images
TEST_ANN=datasets/funsd/dataset/testing_data/annotations

mkdir -p segment_outputs

# 0. syntax check
$PYTHON -m py_compile segment_v4_pixel.py

# 1. train V4 pixel-aware relation head
$PYTHON segment_v4_pixel.py train \
  --model-path "$MODEL" \
  --device cuda \
  --train-images "$TRAIN_IMAGES" \
  --train-annotations "$TRAIN_ANN" \
  --epochs 15 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --negative-ratio 3 \
  --max-positive-per-page 4000 \
  --batch-size 1024 \
  --top-k 64 \
  --pixel-top-k 24 \
  --seed 1993 \
  --head-out "$HEAD"

# 2. diagnose raw pair classifier on the FUNSD test split
$PYTHON segment_v4_pixel.py diagnose \
  --model-path "$MODEL" \
  --device cuda \
  --head "$HEAD" \
  --images "$TEST_IMAGES" \
  --annotations "$TEST_ANN" \
  --top-k 64 \
  --pixel-top-k 24 \
  --output segment_outputs/diagnostic_v4_pixel

# 3. full test prediction/evaluation
$PYTHON segment_v4_pixel.py predict \
  --model-path "$MODEL" \
  --device cuda \
  --head "$HEAD" \
  --images "$TEST_IMAGES" \
  --annotations "$TEST_ANN" \
  --top-k 64 \
  --pixel-top-k 24 \
  --threshold 0.50 \
  --edge-top-k 16 \
  --pixel-gate 0.12 \
  --pixel-order-weight 0.10 \
  --evaluate \
  --output segment_outputs/v4_test

# 4. full training-set segment predictions for KIE training
$PYTHON segment_v4_pixel.py predict \
  --model-path "$MODEL" \
  --device cuda \
  --head "$HEAD" \
  --images "$TRAIN_IMAGES" \
  --annotations "$TRAIN_ANN" \
  --top-k 64 \
  --pixel-top-k 24 \
  --threshold 0.50 \
  --edge-top-k 16 \
  --pixel-gate 0.12 \
  --pixel-order-weight 0.10 \
  --output segment_outputs/v4_train

printf '\nDONE\n'
printf 'Test segments : %s\n' "$(realpath segment_outputs/v4_test)"
printf 'Train segments: %s\n' "$(realpath segment_outputs/v4_train)"
