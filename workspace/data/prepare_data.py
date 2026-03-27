"""
data/prepare_data.py
从本地已保存的 teacher response 文件生成三份训练数据集。

输入文件格式（/data/teacher_responses.jsonl，每行一条）：
{
  "shop_name": "...",
  "categories": "...",
  "brands": "...",
  "products_90d":      [...],
  "products_lastyear": [...],
  "products_3d":       [...],
  "candidate_pool":    "...",
  "response":          "{\"1\":{...}}"   ← Qwen2.5-72B-Instruct-GPTQ-Int4 的输出
}

输出（到 /data/）：
  sft_train.jsonl       LLaMA-Factory sharegpt 格式
  rm_train.jsonl        LLaMA-Factory pairwise 格式
  grpo_train.parquet    veRL parquet 格式

运行（在 llamafactory 容器内）：
  python /workspace/data/prepare_data.py --stage all
"""

from __future__ import annotations
import json, random, re, sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/workspace")
from workspace.src import SYSTEM_PROMPT, build_user_prompt
from workspace.src import rule_reward

DATA_DIR = Path("/data")
# 本地已保存的 teacher response 文件路径
TEACHER_RESP_FILE = DATA_DIR / "teacher_responses.jsonl"


# ────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────

def load_teacher_responses(path: Path = TEACHER_RESP_FILE) -> list[dict]:
    """加载本地 teacher response 文件"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[Load] {len(samples)} records from {path}")
    return samples


def build_prompt(s: dict) -> tuple[str, str]:
    """返回 (system, user) 元组，全链路统一，不截断"""
    user = build_user_prompt(
        shop_name=s["shop_name"],
        categories=s["categories"],
        brands=s["brands"],
        products_90d=s["products_90d"],
        products_lastyear=s["products_lastyear"],
        products_3d=s.get("products_3d", []),
        candidate_pool=s.get("candidate_pool", ""),
    )
    return SYSTEM_PROMPT, user


def corrupt(output: str) -> str:
    """对 teacher response 施加随机扰动，生成 rejected 样本"""
    try:
        data = json.loads(re.search(r'\{[\s\S]*\}', output).group())
    except Exception:
        return output + "\n（格式错误：含额外文字）"
    strategy = random.choice(["hallucination", "score_oob", "brand_cluster", "reason_long"])
    if strategy == "hallucination":
        k = str(len(data) + 1)
        data[k] = {"name": "虚构商品特供版500ml（1*24）",
                   "score": round(random.uniform(0.7, 0.9), 3), "reason": "热销"}
    elif strategy == "score_oob":
        for k in list(data)[:3]:
            data[k]["score"] = round(random.uniform(1.1, 1.5), 3)
    elif strategy == "brand_cluster":
        vals = list(data.values()); random.shuffle(vals)
        data = {str(i + 1): v for i, v in enumerate(vals)}
    elif strategy == "reason_long":
        for v in data.values():
            v["reason"] = "这是一个超过二十汉字限制的非常冗长推荐理由属于明显违规输出"
    return json.dumps(data, ensure_ascii=False)


# ────────────────────────────────────────────────────────────────
# SFT 数据
# ────────────────────────────────────────────────────────────────

def make_sft(samples: list[dict], min_reward: float = 0.65):
    """
    直接用本地 teacher response 作为 assistant 输出。
    过滤掉规则 reward 低于阈值的样本（teacher 偶尔也会输出低质量结果）。
    """
    path = DATA_DIR / "sft_train.jsonl"
    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="SFT"):
            system, user = build_prompt(s)
            response = s["response"]

            # 过滤低质量 teacher 输出
            r = rule_reward(
                response,
                s["products_90d"],
                s["products_lastyear"],
                s.get("products_3d", []),
            )
            if r < min_reward:
                continue

            # LLaMA-Factory sharegpt 格式
            record = {"conversations": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": response},
            ]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[SFT] {kept}/{len(samples)} → {path}")


# ────────────────────────────────────────────────────────────────
# RM 数据
# ────────────────────────────────────────────────────────────────

def make_rm(samples: list[dict], min_gap: float = 0.15):
    """
    chosen  = 本地 teacher response（高质量）
    rejected = 对 chosen 施加随机扰动（低质量）
    """
    path = DATA_DIR / "rm_train.jsonl"
    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="RM"):
            system, user = build_prompt(s)
            chosen   = s["response"]
            rejected = corrupt(chosen)

            r_c = rule_reward(chosen,   s["products_90d"], s["products_lastyear"], s.get("products_3d", []))
            r_r = rule_reward(rejected, s["products_90d"], s["products_lastyear"], s.get("products_3d", []))

            # gap 不够大则跳过（避免噪声对）
            if r_c - r_r < min_gap:
                continue

            def msgs(resp: str) -> list[dict]:
                return [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": user},
                    {"role": "assistant", "content": resp},
                ]

            record = {"chosen": msgs(chosen), "rejected": msgs(rejected)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[RM]  {kept}/{len(samples)} → {path}")


# ────────────────────────────────────────────────────────────────
# GRPO 数据
# ────────────────────────────────────────────────────────────────

def make_grpo(samples: list[dict]):
    """
    GRPO 只需要 prompt（veRL 在线采样 response）。
    prompt 与线上推理完全相同：SYSTEM_PROMPT + 完整 user prompt。
    """
    rows = []
    for s in samples:
        system, user = build_prompt(s)
        rows.append({
            "prompt": json.dumps([
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]),
            "data_source": "rec_system",
            "ground_truth": json.dumps({
                "products_90d":      s["products_90d"],
                "products_lastyear": s["products_lastyear"],
                "products_3d":       s.get("products_3d", []),
            }),
        })

    path = DATA_DIR / "grpo_train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[GRPO] {len(rows)} → {path}")


# ────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage",  choices=["sft", "rm", "grpo", "all"], default="all")
    p.add_argument("--input",  default=str(TEACHER_RESP_FILE),
                   help="本地 teacher response 文件路径")
    p.add_argument("--min_reward", type=float, default=0.65,
                   help="SFT 样本最低 rule_reward 阈值")
    p.add_argument("--min_gap",    type=float, default=0.15,
                   help="RM 样本 chosen-rejected 最小 reward 差距")
    a = p.parse_args()

    samples = load_teacher_responses(Path(a.input))

    if a.stage in ("sft",  "all"): make_sft(samples,  min_reward=a.min_reward)
    if a.stage in ("rm",   "all"): make_rm(samples,   min_gap=a.min_gap)
    if a.stage in ("grpo", "all"): make_grpo(samples)
