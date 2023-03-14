#!/bin/bash


for lr in 1e-7 1e-6 3e-6; do
  echo "Running with lr $lr"
  accelerate launch --num_processes 8 trainer/train.py +experiment=clip_h optimizer.lr=$lr
done