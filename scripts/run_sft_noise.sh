#!/bin/bash

for ratio in 0.6; do
  CUDA_VISIBLE_DEVICES=0 python main.py --supervised \
    --experiment_name sft_noise_${ratio} \
    --noise_ratio ${ratio}
done

