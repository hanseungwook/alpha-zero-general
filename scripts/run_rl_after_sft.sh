#!/bin/bash

#SBATCH --job-name=alphazero_0.3
#SBATCH --output=logs/alphazero_0.3-%j.out
#SBATCH --error=logs/alphazero_0.3-%j.err
#SBATCH --partition=vision-pulkitag-v100
#SBATCH --qos=vision-pulkitag-main
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

python main.py --experiment_name rl_sft_noise0.3 --numEps 200 --updateThreshold 0.55 --numMCTSSims 200 --checkpoint_folder ./checkpoints/sft_noise_0.3/ --checkpoint_filename checkpoint_9.pth.tar
