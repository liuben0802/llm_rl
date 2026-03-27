.PHONY: data start verify train export deploy all logs board shell clean

data:    ; bash run_grpo.sh data
start:   ; bash run_grpo.sh start
verify:  ; bash run_grpo.sh verify
train:   ; bash run_grpo.sh train
export:  ; bash run_grpo.sh export
deploy:  ; bash run_grpo.sh deploy
all:     ; bash run_grpo.sh all

# 实时查看训练日志
logs:
	docker logs -f rec_verl_grpo

# TensorBoard（已在训练时后台启动，此处仅提示地址）
board:
	@echo "TensorBoard: http://localhost:6006"
	@echo "（已在训练容器内后台运行，如未启动请执行 make start-board）"

start-board:
	docker exec rec_verl_grpo \
	  tensorboard --logdir /saves --host 0.0.0.0 --port 6006 &

# 进入训练容器 shell（调试用）
shell:
	docker exec -it rec_verl_grpo bash

# 查看 GPU 使用情况
gpu:
	CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

clean:
	docker rm -f rec_verl_grpo rec_infer 2>/dev/null || true
