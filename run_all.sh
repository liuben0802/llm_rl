#!/usr/bin/env bash
# run_all.sh —— 宿主机执行，通过 docker exec 驱动容器训练
# 用法：bash run_all.sh [all|data|sft|export-sft|rm|grpo|final]
set -euo pipefail

STAGE=${1:-all}
LF=rec_llamafactory
VERL=rec_verl
GPU_ENV="CUDA_VISIBLE_DEVICES=0,1,2,3"

lf()   { docker exec -it "$LF"   bash -c "export ${GPU_ENV}; $*"; }
verl() { docker exec -it "$VERL" bash -c "export ${GPU_ENV}; $*"; }

divider() { echo ""; echo "══════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════"; }

start_containers() {
  divider "启动容器"
  docker compose up -d llamafactory verl
  echo "  ✓ 容器已启动"
}

stage_data() {
  divider "Stage 0 — 数据生成（读取本地 teacher response）"
  # teacher_responses.jsonl 已在 /data4/datas/，直接生成三份训练数据
  lf "python /workspace/data/prepare_data.py --stage all \
      --input /data/teacher_responses.jsonl"
  echo "  ✓ sft_train.jsonl / rm_train.jsonl / grpo_train.parquet → /data4/datas/"
}

stage_sft() {
  divider "Stage 1 — SFT（LLaMA-Factory + FSDP2）"
  lf "accelerate launch \
      --config_file /workspace/configs/fsdp2_4gpu.yaml \
      \$(which llamafactory-cli) train /workspace/configs/sft_lf.yaml"
  echo "  ✓ SFT adapter → /data4/mirror/model/saves/sft"
}

stage_export_sft() {
  divider "Stage 1b — 导出 SFT 完整权重"
  lf "llamafactory-cli export /workspace/configs/export_sft.yaml"
  echo "  ✓ SFT 完整权重 → /data4/mirror/model/saves/sft_merged"
}

stage_rm() {
  divider "Stage 2 — Reward Model（LLaMA-Factory + FSDP2）"
  lf "accelerate launch \
      --config_file /workspace/configs/fsdp2_4gpu.yaml \
      \$(which llamafactory-cli) train /workspace/configs/rm_lf.yaml"
  echo "  ✓ RM adapter → /data4/mirror/model/saves/rm"

  divider "Stage 2b — 导出 RM 完整权重"
  lf "llamafactory-cli export /workspace/configs/export_rm.yaml"
  echo "  ✓ RM 完整权重 → /data4/mirror/model/saves/rm_merged"
}

stage_grpo() {
  divider "Stage 3 — GRPO（veRL + hybrid engine）"
  verl "python -m verl.trainer.main_ppo \
      --config-path /workspace/configs \
      --config-name verl_grpo"
  echo "  ✓ GRPO checkpoint → /data4/mirror/model/saves/grpo"
}

stage_final() {
  divider "最终模型导出"
  FINAL_ADAPTER="/saves/grpo/global_step_final"
  if docker exec "$LF" test -f "${FINAL_ADAPTER}/adapter_config.json" 2>/dev/null; then
    cat > /tmp/_export_final.yaml << EOF
model_name_or_path: /saves/sft_merged
adapter_name_or_path: ${FINAL_ADAPTER}
template: qwen3
finetuning_type: lora
trust_remote_code: true
export_dir: /saves/final
export_size: 4
export_dtype: bfloat16
export_device: cuda
export_legacy_format: false
EOF
    docker cp /tmp/_export_final.yaml "${LF}:/tmp/export_final.yaml"
    lf "llamafactory-cli export /tmp/export_final.yaml"
  else
    lf "cp -r ${FINAL_ADAPTER} /saves/final"
  fi
  echo "  ✓ 最终模型 → /data4/mirror/model/saves/final"
}

start_containers

case "$STAGE" in
  all)        stage_data; stage_sft; stage_export_sft; stage_rm; stage_grpo; stage_final ;;
  data)       stage_data        ;;
  sft)        stage_sft         ;;
  export-sft) stage_export_sft  ;;
  rm)         stage_rm          ;;
  grpo)       stage_grpo        ;;
  final)      stage_final       ;;
  *) echo "用法：bash run_all.sh [all|data|sft|export-sft|rm|grpo|final]"; exit 1 ;;
esac

echo ""
echo "🎉 ${STAGE} 完成！"
