.PHONY: up down data sft export-sft rm grpo final all board shell-lf shell-verl clean

up:
	docker compose up -d llamafactory verl

down:
	docker compose down

data:        ; bash run_all.sh data
sft:         ; bash run_all.sh sft
export-sft:  ; bash run_all.sh export-sft
rm:          ; bash run_all.sh rm
grpo:        ; bash run_all.sh grpo
final:       ; bash run_all.sh final
all:         ; bash run_all.sh all

board:
	docker run --rm -p 6006:6006 \
		-v /data4/mirror/model/saves:/saves \
		-v $$(pwd)/workspace/logs:/logs \
		tensorflow/tensorflow \
		tensorboard --logdir /saves --host 0.0.0.0

shell-lf:
	docker exec -it rec_llamafactory bash

shell-verl:
	docker exec -it rec_verl bash

clean:
	rm -rf workspace/logs/*
	docker compose down --remove-orphans
