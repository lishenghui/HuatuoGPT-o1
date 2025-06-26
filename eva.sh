#!/bin/bash
#SBATCH -A NAISS2024-22-1394 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=paaa
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=2:30:00

log_num=0
model_name="/mimer/NOBACKUP/groups/bloom/shenghui/HuatuoGPT-o1/ckpts/sft_stage1/checkpoint-441"
model_name="/mimer/NOBACKUP/groups/bloom/shenghui/HuatuoGPT-o1/trained_full_funing/checkpoint-3696"
model_name="unsloth/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.2-1B-Instruct"
# model_name="meta-llama/Llama-3.2-1B-Instruct"
# model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_name="ContactDoctor/Bio-Medical-Llama-3-8B"
# model_name="FreedomIntelligence/HuatuoGPT-o1-8B"
port=28${log_num}35

# 启动 sglang server
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path $model_name \
  --port $port \
  --mem-fraction-static 0.8 \
  --dp 4 --tp 1 > sglang${log_num}.log 2>&1 &

# 获取 server PID（可选）
sglang_pid=$!

# 使用 Python 等待端口最多 1200 秒
echo "Waiting for sglang server to be ready on port $port..."
echo "Evaluating model: $model_name"
python3 - <<EOF
import socket
import time
import sys

host = "localhost"
port = $port
timeout = 1200  # 最长等待时间（秒）
interval = 5   # 每次检测间隔（秒）

start = time.time()
while time.time() - start < timeout:
    try:
        with socket.create_connection((host, port), timeout=2):
            print("Server is up!")
            sys.exit(0)
    except Exception:
        time.sleep(interval)

print("Error: sglang server did not start within expected time.", file=sys.stderr)
sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Exiting due to server not ready."
    exit 1
fi

# Evaluate the model
python evaluation/eval.py \
  --model_name $model_name \
  --eval_file evaluation/data/eval_data.json \
  --port $port
