#!/usr/bin/env bash
# run_all.sh —— 宿主机执行，通过 docker exec 向容器发送训练指令
# 用法：bash run_all.sh [stage]
#   stage: all | data | sft | export | rm | grpo   默认 all
set -euo pipefail

STAGE=${1:-all}

LF=rec_llamafactory       # LLaMA-Factory 容器名
VERL=rec_verl             # veRL 容器名
TEACHER=rec_teacher       # Teacher 容器名

# ── 公共执行函数 ──
# 容器内 CUDA_VISIBLE_DEVICES 已通过 docker-compose env 设为 0,1,2,3
# 这里再显式传一次，防止 docker exec 新建 shell 时环境变量丢失
GPU_ENV="CUDA_VISIBLE_DEVICES=0,1,2,3"
lf()   { docker exec -it "$LF"   bash -c "export ${GPU_ENV}; $*"; }
verl() { docker exec -it "$VERL" bash -c "export ${GPU_ENV}; $*"; }

divider() { echo ""; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }

# ────────────────────────────────────────────────────────────────
# 0. 启动常驻容器（不含 teacher）
# ────────────────────────────────────────────────────────────────
start_containers() {
  divider "启动 llamafactory / verl 容器"
  docker compose up -d llamafactory verl
  echo "  ✓ 容器已启动（后台常驻）"
}

# ────────────────────────────────────────────────────────────────
# 1. 数据生成（需要 teacher 模型）
# ────────────────────────────────────────────────────────────────
stage_data() {
  divider "Stage 0 — 数据生成"

  # 启动 teacher，等待健康检查通过
  echo "  启动 teacher 容器（Qwen2.5-72B）..."
  docker compose up -d teacher
  echo "  等待 teacher 服务就绪..."
  until docker exec "$TEACHER" curl -sf http://localhost:8000/health &>/dev/null; do
    sleep 10; printf "."
  done
  echo -e "\n  ✓ teacher 就绪"

  # 在 llamafactory 容器内运行数据生成脚本
  lf "cd /workspace && TEACHER_BASE=http://host.docker.internal:8000/v1 \
      python data/prepare_data.py --stage all"

  echo "  ✓ 数据生成完成"

  # 停止 teacher，释放 GPU 给后续训练
  echo "  停止 teacher 容器，释放 GPU..."
  docker compose stop teacher
  echo "  ✓ teacher 已停止"
}

# ────────────────────────────────────────────────────────────────
# 2. SFT（LLaMA-Factory + FSDP2）
# ────────────────────────────────────────────────────────────────
stage_sft() {
  divider "Stage 1 — SFT 微调（LLaMA-Factory + FSDP2）"
  lf "cd /workspace && accelerate launch \
      --config_file /workspace/configs/fsdp2_4gpu.yaml \
      \$(which llamafactory-cli) train /workspace/configs/sft_lf.yaml"
  echo "  ✓ SFT adapter → /workspace/checkpoints/sft"
}

# ────────────────────────────────────────────────────────────────
# 3. 导出 SFT 权重（合并 LoRA，veRL 需要完整权重）
# ────────────────────────────────────────────────────────────────
stage_export() {
  divider "Stage 1b — 导出 SFT 完整权重（llamafactory-cli export）"
  lf "llamafactory-cli export /workspace/configs/export_sft.yaml"
  echo "  ✓ 完整权重 → /workspace/checkpoints/sft_merged"
}

# ────────────────────────────────────────────────────────────────
# 4. Reward Model（LLaMA-Factory + FSDP2）
# ────────────────────────────────────────────────────────────────
stage_rm() {
  divider "Stage 2 — Reward Model 训练（LLaMA-Factory + FSDP2）"
  lf "cd /workspace && accelerate launch \
      --config_file /workspace/configs/fsdp2_4gpu.yaml \
      \$(which llamafactory-cli) train /workspace/configs/rm_lf.yaml"
  echo "  ✓ RM adapter → /workspace/checkpoints/rm"

  # RM 也需要导出完整权重，veRL reward_model 不接受 PEFT adapter
  divider "Stage 2b — 导出 RM 完整权重"
  lf "llamafactory-cli export /workspace/configs/export_rm.yaml"
  echo "  ✓ RM 完整权重 → /workspace/checkpoints/rm_merged"
}

# ────────────────────────────────────────────────────────────────
# 5. GRPO（veRL，hybrid engine）
# ────────────────────────────────────────────────────────────────
stage_grpo() {
  divider "Stage 3 — GRPO（veRL + hybrid engine）"
  verl "cd /workspace && python -m verl.trainer.main_ppo \
      --config-path /workspace/configs \
      --config-name verl_grpo"
  echo "  ✓ GRPO checkpoint → /workspace/checkpoints/grpo"
}

# ────────────────────────────────────────────────────────────────
# 6. 导出最终模型
# ────────────────────────────────────────────────────────────────
stage_final() {
  divider "最终模型导出"
  FINAL_ADAPTER="/workspace/checkpoints/grpo/global_step_final"

  # 检查是否有 LoRA adapter（veRL 用 LoRA 时会有 adapter_config.json）
  if docker exec "$LF" test -f "${FINAL_ADAPTER}/adapter_config.json"; then
    cat > /tmp/_export_final.yaml <<EOF
model_name_or_path: /workspace/checkpoints/sft_merged
adapter_name_or_path: ${FINAL_ADAPTER}
template: qwen3
finetuning_type: lora
trust_remote_code: true
export_dir: /workspace/checkpoints/final_model
export_size: 4
export_dtype: bfloat16
export_device: cuda
export_legacy_format: false
EOF
    docker cp /tmp/_export_final.yaml "${LF}:/tmp/export_final.yaml"
    lf "llamafactory-cli export /tmp/export_final.yaml"
  else
    echo "  GRPO 输出已是完整权重，无需合并"
    lf "cp -r ${FINAL_ADAPTER} /workspace/checkpoints/final_model"
  fi
  echo "  ✓ 最终模型 → /workspace/checkpoints/final_model"
}

# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────
start_containers

case "$STAGE" in
  all)
    stage_data
    stage_sft
    stage_export
    stage_rm
    stage_grpo
    stage_final
    ;;
  data)      stage_data      ;;
  sft)       stage_sft       ;;
  export)    stage_export    ;;
  rm)        stage_rm        ;;
  export-rm) # 单独导出 RM（rm 阶段已内含，此处供手动重跑）
    lf "llamafactory-cli export /workspace/configs/export_rm.yaml"
    ;;
  grpo)      stage_grpo      ;;
  final)     stage_final     ;;
  *)
    echo "用法：bash run_all.sh [all|data|sft|export|rm|grpo|final]"
    exit 1
    ;;
esac

echo ""
echo "🎉 ${STAGE} 完成！"
echo ""
echo "部署命令（宿主机）："
echo "  docker run --gpus all --rm -v \$(pwd)/workspace:/workspace \\"
echo "    vllm/vllm-openai:latest \\"
echo "    python -m vllm.entrypoints.openai.api_server \\"
echo "    --model /workspace/checkpoints/final_model \\"
echo "    --tensor-parallel-size 2 --port 8001"
