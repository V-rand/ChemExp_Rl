#!/bin/bash

# ================= 配置区 =================
# MODEL_PATH="Qwen/Qwen3-8B-Base"  # 替换为你的模型路径
MODEL_PATH="/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/sft_output/v2-20260314-214543/checkpoint-78-merged"  # 替换为你的模型路径
MODEL_NAME="chem_expert_v1"  # 给模型起个别名，评测脚本里用这个名字
PORT=8002
GPU_IDS="5"                  # 指定使用哪块显卡
MAX_MODEL_LEN=4096           # 最大上下文长度
GPU_MEMORY=0.9               # 显存占用率 (0.0 - 1.0)
# ==========================================

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "正在启动 vLLM 服务: $MODEL_NAME"
echo "模型路径: $MODEL_PATH"
echo "访问端口: $PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --trust-remote-code \
    --disable-log-requests   # 关闭请求日志，保持终端干净