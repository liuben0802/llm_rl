#!/usr/bin/env bash
# run_grpo.sh
# veRL GRPO 训练完整执行脚本
# 前提：
#   - SFT 完整权重已在 /data4/mirror/model/saves/sft_merged
#   - RM  完整权重已在 /data4/mirror/model/saves/rm_merged
#   - teacher_responses.jsonl 已在 /data4/datas/
#   - 宿主机已安装 Docker + nvidia-container-toolkit
set -euo pipefail

# ── 路径配置（按实际修改）──
HOST_MODELS=/data4/mirror/Qwen
HOST_SAVES=/data4/mirror/model/saves
HOST_DATA=/data4/datas
WORKSPACE=$(pwd)/workspace           # 本脚本所在目录的 workspace 子目录

VERL_IMAGE=verlai/verl:sgl059.latest
CONTAINER_NAME=rec_verl_grpo
GPU_IDS='"device=0,1,2,3"'          # 只用 0-3 号卡

divider() { echo ""; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }

# ════════════════════════════════════════════════════════════
# Step 0：生成 grpo_train.parquet
# （在宿主机运行，不需要 GPU）
# ════════════════════════════════════════════════════════════
step_data() {
  divider "Step 0 — 生成 grpo_train.parquet"
  pip install pandas pyarrow -q
  python scripts/prepare_grpo_data.py
  echo "  ✓ /data4/datas/grpo_train.parquet"
}

# ════════════════════════════════════════════════════════════
# Step 1：启动 veRL 容器（常驻）
# ════════════════════════════════════════════════════════════
step_start_container() {
  divider "Step 1 — 启动 veRL 容器"

  # 如果容器已存在则先删除
  docker rm -f $CONTAINER_NAME 2>/dev/null || true

  docker run -d \
    --name $CONTAINER_NAME \
    --gpus $GPU_IDS \
    --shm-size 32g \
    --ipc host \
    --pid host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e NCCL_DEBUG=WARN \
    -e NCCL_IB_DISABLE=1 \
    -e PYTHONPATH=/workspace \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -e RAY_ADDRESS=local \
    -v ${HOST_MODELS}:/models:ro \
    -v ${HOST_DATA}:/data \
    -v ${HOST_SAVES}:/saves \
    -v ${WORKSPACE}:/workspace \
    -p 6006:6006 \
    ${VERL_IMAGE} \
    tail -f /dev/null

  echo "  ✓ 容器已启动：$CONTAINER_NAME"

  # 等待容器就绪
  sleep 3
  docker exec $CONTAINER_NAME python -c "import torch; print(f'  GPUs: {torch.cuda.device_count()}')"
}

# ════════════════════════════════════════════════════════════
# Step 2：验证环境（确认 SGLang、veRL、模型路径均正常）
# ════════════════════════════════════════════════════════════
step_verify() {
  divider "Step 2 — 验证环境"

  docker exec $CONTAINER_NAME bash -c "
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo '── Python packages ──'
    python -c \"
    import torch, verl, sglang
    print(f'  torch:  {torch.__version__}')
    print(f'  verl:   {verl.__version__}')
    print(f'  sglang: {sglang.__version__}')
    print(f'  GPUs:   {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU{i}: {torch.cuda.get_device_name(i)} ({mem:.0f}GB)')
    \"
    echo ''
    echo '── 模型路径检查 ──'
    ls /saves/sft_merged/config.json   && echo '  ✓ sft_merged' || echo '  ✗ sft_merged 不存在'
    ls /saves/rm_merged/config.json    && echo '  ✓ rm_merged'  || echo '  ✗ rm_merged 不存在'
    ls /data/grpo_train.parquet        && echo '  ✓ grpo_train.parquet' || echo '  ✗ parquet 不存在'
    echo ''
    echo '── reward_fn 验证 ──'
    python -c \"
import sys; sys.path.insert(0,'/workspace')
from src.reward_fn import compute_score
score = compute_score('rec_system',
    '{\"1\":{\"name\":\"可口可乐500ml（1*24）\",\"score\":0.900,\"reason\":\"近期热销\"}}',
    '{\"products_90d\":[\"可口可乐500ml（1*24）\"],\"products_lastyear\":[],\"products_3d\":[]}')
print(f'  compute_score test: {score:.3f}  (期望 > 0.5)')
assert score > 0.5, 'reward_fn 验证失败'
print('  ✓ reward_fn 正常')
\"
  "
}

# ════════════════════════════════════════════════════════════
# Step 3：启动 GRPO 训练
# ════════════════════════════════════════════════════════════
step_train() {
  divider "Step 3 — GRPO 训练（veRL + SGLang）"
  echo "  预计时间：视数据量和 epoch 数而定"
  echo "  日志：docker logs -f $CONTAINER_NAME"
  echo "  TensorBoard：http://localhost:6006"
  echo ""

  docker exec -it $CONTAINER_NAME bash -c "
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # 启动 TensorBoard（后台）
    tensorboard --logdir /saves --host 0.0.0.0 --port 6006 &>/dev/null &

    # 启动 GRPO 训练
    python -m verl.trainer.main_ppo \
      --config-path /workspace/configs \
      --config-name verl_grpo
  "
  echo "  ✓ 训练完成，checkpoint → /data4/mirror/model/saves/grpo"
}

# ════════════════════════════════════════════════════════════
# Step 4：导出最终模型（合并 GRPO LoRA）
# ════════════════════════════════════════════════════════════
step_export() {
  divider "Step 4 — 导出最终模型"

  FINAL_ADAPTER=/saves/grpo/global_step_final

  docker exec $CONTAINER_NAME bash -c "
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    if [ -f '${FINAL_ADAPTER}/adapter_config.json' ]; then
      echo '  检测到 LoRA adapter，开始合并...'
      python - << 'PYEOF'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = '/saves/sft_merged'
adapter = '${FINAL_ADAPTER}'
output = '/saves/final'

print(f'  Loading base model from {base}')
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

print(f'  Loading adapter from {adapter}')
model = PeftModel.from_pretrained(model, adapter)

print('  Merging LoRA weights...')
model = model.merge_and_unload()

print(f'  Saving to {output}')
model.save_pretrained(output, safe_serialization=True)
tokenizer.save_pretrained(output)
print('  ✓ Done')
PYEOF
    else
      echo '  GRPO 输出已是完整权重，直接复制'
      cp -r ${FINAL_ADAPTER} /saves/final
    fi
  "
  echo "  ✓ 最终模型 → /data4/mirror/model/saves/final"
}

# ════════════════════════════════════════════════════════════
# Step 5：部署验证（SGLang 推理服务）
# ════════════════════════════════════════════════════════════
step_deploy() {
  divider "Step 5 — 启动 SGLang 推理服务（验证用）"

  # 停掉训练容器，用 GPU 0-1 起推理服务
  docker stop $CONTAINER_NAME 2>/dev/null || true

  docker run -d \
    --name rec_infer \
    --gpus '"device=0,1"' \
    --shm-size 16g \
    -p 8001:8001 \
    -v ${HOST_SAVES}/final:/model:ro \
    ${VERL_IMAGE} \
    python -m sglang.launch_server \
      --model-path /model \
      --tp 2 \
      --port 8001 \
      --host 0.0.0.0 \
      --trust-remote-code

  echo "  等待服务就绪..."
  until curl -sf http://localhost:8001/health &>/dev/null; do
    sleep 5; printf "."
  done
  echo -e "\n  ✓ SGLang 推理服务已启动：http://localhost:8001"

  # 快速冒烟测试
  curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "final",
      "messages": [{"role": "user", "content": "输出：{\"1\":{\"name\":\"test\",\"score\":0.9,\"reason\":\"test\"}}"}],
      "max_tokens": 100
    }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('  smoke test OK:', r['choices'][0]['message']['content'][:50])"
}

# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
STEP=${1:-all}

case "$STEP" in
  all)
    step_data
    step_start_container
    step_verify
    step_train
    step_export
    step_deploy
    ;;
  data)      step_data             ;;
  start)     step_start_container  ;;
  verify)    step_verify           ;;
  train)     step_train            ;;
  export)    step_export           ;;
  deploy)    step_deploy           ;;
  *)
    echo "用法：bash run_grpo.sh [all|data|start|verify|train|export|deploy]"
    exit 1
    ;;
esac

echo ""
echo "🎉 ${STEP} 完成！"
