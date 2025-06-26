#!/bin/bash
#SBATCH -A NAISS2024-22-1394 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --job-name=paaa
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=4:30:00

# Qwen/Qwen3-0.6B
# meta-llama/Llama-3.1-8B-Instruct

python new.py


    