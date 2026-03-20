# Makefile —— 常用操作快捷方式
.PHONY: up down logs-lf logs-verl data sft export rm grpo all clean

# ── 容器管理 ──
up:
	docker compose up -d llamafactory verl

down:
	docker compose down

logs-lf:
	docker logs -f rec_llamafactory

logs-verl:
	docker logs -f rec_verl

# ── 各阶段入口（单独触发）──
data:
	bash run_all.sh data

sft:
	bash run_all.sh sft

export:
	bash run_all.sh export

rm:
	bash run_all.sh rm

grpo:
	bash run_all.sh grpo

final:
	bash run_all.sh final

# ── 一键全跑 ──
all:
	bash run_all.sh all

# ── TensorBoard 查看训练曲线 ──
board:
	docker run --rm -p 6006:6006 \
		-v $$(pwd)/workspace/logs:/logs \
		tensorflow/tensorflow tensorboard --logdir /logs --host 0.0.0.0

# ── 进入容器 shell（调试用）──
shell-lf:
	docker exec -it rec_llamafactory bash

shell-verl:
	docker exec -it rec_verl bash

# ── 清理（保留 checkpoints）──
clean:
	rm -rf workspace/logs/*
	docker compose down --remove-orphans
