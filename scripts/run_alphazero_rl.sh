#!/bin/bash

#SBATCH --job-name=alphazero
#SBATCH --output=logs/alphazero-%j.out
#SBATCH --error=logs/alphazero-%j.err
#SBATCH --partition=vision-pulkitag-h100
#SBATCH --qos=vision-pulkitag-main
#SBATCH --mem=400G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

python main.py --experiment_name alphazero_rl
