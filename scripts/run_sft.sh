#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --supervised --experiment_name sft_subset0.75 --subset_ratio 0.75
