#!/bin/bash

for ratio in 0.4 0.5; do
  CUDA_VISIBLE_DEVICES=1 python main.py --supervised \
    --experiment_name sft_noise_${ratio} \
    --noise_ratio ${ratio}
done

