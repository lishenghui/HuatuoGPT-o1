#!/bin/bash
#SBATCH -A NAISS2024-22-1394 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=paaa
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=4:30:00

# Qwen/Qwen3-0.6B
# meta-llama/Llama-3.1-8B-Instruct

accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 4  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard sft.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --data_path medical_o1_sft_mix.json
    