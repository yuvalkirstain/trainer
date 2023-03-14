#!/bin/bash


for lr in 1e-5 2e-5 3e-5; do
  echo "Running with lr $lr"
  accelerate launch --num_processes 8 trainer/train.py +experiment=clip_h optimizer.lr=$lr
done