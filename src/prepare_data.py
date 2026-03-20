"""
data/prepare_data.py
生成三份数据集：
  data/sft_train.jsonl      ← LLaMA-Factory sharegpt 格式（conversations 字段）
  data/rm_train.jsonl       ← LLaMA-Factory pairwise 格式（chosen/rejected 字段）
  data/grpo_train.parquet   ← veRL parquet 格式

运行：
  python data/prepare_data.py --stage all
"""

from __future__ import annotations
import json, random, re, sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompt_utils import SYSTEM_TRAIN, SYSTEM_INFER, build_user_prompt
from src.reward_fn import rule_reward

TEACHER_BASE  = "http://localhost:8000/v1"
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────
# 工具
# ────────────────────────────────────────────────────────────────

def _call(system: str, user: str, temperature: float = 0.2) -> Optional[str]:
    client = OpenAI(base_url=TEACHER_BASE, api_key="EMPTY")
    try:
        r = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role":"system","content":system},
                      {"role":"user",  "content":user}],
            temperature=temperature, max_tokens=4096,
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"  [call error] {e}"); return None


def _corrupt(output: str) -> str:
    try:
        data = json.loads(re.search(r'\{[\s\S]*\}', output).group())
    except Exception:
        return output + "\n（额外文字导致格式非法）"
    strategy = random.choice(["hallucination","score_oob","brand_cluster","reason_long"])
    if strategy == "hallucination":
        k = str(len(data)+1)
        data[k] = {"name":"虚构商品特供版500ml（1*24）","score":round(random.uniform(0.7,0.9),3),"reason":"热销"}
    elif strategy == "score_oob":
        for k in list(data)[:3]: data[k]["score"] = round(random.uniform(1.1,1.5),3)
    elif strategy == "brand_cluster":
        vals = list(data.values()); random.shuffle(vals)
        data = {str(i+1):v for i,v in enumerate(vals)}
    elif strategy == "reason_long":
        for v in data.values(): v["reason"] = "这是一个超过二十汉字限制的非常冗长推荐理由属于明显违规输出"
    return json.dumps(data, ensure_ascii=False)


def _user(s: dict) -> str:
    return build_user_prompt(
        shop_name=s["shop_name"], categories=s["categories"],
        brands=s["brands"], products_90d=s["products_90d"],
        products_lastyear=s["products_lastyear"],
        products_3d=s.get("products_3d",[]),
        candidate_pool=s.get("candidate_pool",""),
    )


# ────────────────────────────────────────────────────────────────
# Stage SFT
# ────────────────────────────────────────────────────────────────

def make_sft(samples: list[dict], min_reward: float = 0.65):
    path = OUT / "sft_train.jsonl"
    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="SFT"):
            user = _user(s)
            asst = _call(SYSTEM_TRAIN, user)
            if not asst: continue
            r = rule_reward(asst, s["products_90d"], s["products_lastyear"], s.get("products_3d",[]))
            if r < min_reward: continue
            # LLaMA-Factory sharegpt 格式
            record = {"conversations": [
                {"role":"system",    "content": SYSTEM_TRAIN},
                {"role":"user",      "content": user},
                {"role":"assistant", "content": asst},
            ]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[SFT] {kept}/{len(samples)} → {path}")


# ────────────────────────────────────────────────────────────────
# Stage RM
# ────────────────────────────────────────────────────────────────

def make_rm(samples: list[dict], min_gap: float = 0.15):
    path = OUT / "rm_train.jsonl"
    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="RM"):
            user   = _user(s)
            chosen = _call(SYSTEM_TRAIN, user, temperature=0.2)
            if not chosen: continue
            r_c = rule_reward(chosen,   s["products_90d"], s["products_lastyear"], s.get("products_3d",[]))
            rejected = _corrupt(chosen)
            r_r = rule_reward(rejected, s["products_90d"], s["products_lastyear"], s.get("products_3d",[]))
            if r_c - r_r < min_gap: continue

            def _msgs(response: str) -> list[dict]:
                return [{"role":"system","content":SYSTEM_TRAIN},
                        {"role":"user",  "content":user},
                        {"role":"assistant","content":response}]

            # LLaMA-Factory pairwise（ranking）格式
            record = {
                "chosen":   _msgs(chosen),
                "rejected": _msgs(rejected),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[RM]  {kept}/{len(samples)} → {path}")


# ────────────────────────────────────────────────────────────────
# Stage GRPO（veRL parquet）
# ────────────────────────────────────────────────────────────────

def make_grpo(samples: list[dict]):
    """
    GRPO prompt 必须与线上推理完全一致：
      - system prompt → SYSTEM_INFER（完整版，~1940 token）
      - user prompt   → 不截断（max_tokens=99999）
    这样 rollout 生成的 8 条候选和 reward 信号，
    与线上真实分布对齐，RL 优化方向才正确。

    注意：max_prompt_length 在 verl_grpo.yaml 中已改为 4600，
    可容纳 SYSTEM_INFER(1940) + user(2500) ≈ 4440 token。
    """
    rows = []
    for s in samples:
        # user prompt 推理时不截断
        user = build_user_prompt(
            shop_name=s["shop_name"],
            categories=s["categories"],
            brands=s["brands"],
            products_90d=s["products_90d"],
            products_lastyear=s["products_lastyear"],
            products_3d=s.get("products_3d", []),
            candidate_pool=s.get("candidate_pool", ""),
            max_tokens=99999,           # ← 不截断
        )
        rows.append({
            "prompt": json.dumps([
                {"role": "system", "content": SYSTEM_INFER},   # ← 完整版
                {"role": "user",   "content": user},
            ]),
            "data_source": "rec_system",
            "ground_truth": json.dumps({
                "products_90d":      s["products_90d"],
                "products_lastyear": s["products_lastyear"],
                "products_3d":       s.get("products_3d", []),
            }),
        })
    path = OUT / "grpo_train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[GRPO] {len(rows)} → {path}")


# ────────────────────────────────────────────────────────────────
# 示例数据（替换为真实超市数据集）
# ────────────────────────────────────────────────────────────────

SAMPLES = [
    {
        "shop_name": "尖垡村馒头房超市",
        "categories": "饮料、休食、副食、白酒、日化、啤酒",
        "brands": "牛栏山、可口可乐、乐事、娃哈哈、双汇、蒙牛、百岁山、伊利、达利园、名仁",
        "products_90d": [
            "蒙牛特仑苏纯牛奶250ml（1*6*12）","牛栏山二锅头桶装56度2L",
            "百岁山饮用天然矿泉水570ml（1*24）","娃哈哈AD钙奶220g（1*24）",
            "名仁苏打水饮料无糖无汽375ml（1*24）","伊利安慕希希腊风味酸奶205ml（1*8*12）",
            "可口可乐2L（1*6）","双汇鸡肉香肠东北风味58g（1*50）",
            "乐事无限薯片醇香原味90g","可口可乐500ml（1*24）",
            "牛栏山二锅头桶装42度2L","康师傅冰红茶500ml（1*15）",
        ],
        "products_lastyear": [
            "乐事薯片青柠味70g","燕京啤酒U8热爱罐8度500ml（1*12）",
            "牛栏山二锅头桶装42度2L","蒙牛特仑苏纯牛奶250ml（1*6*12）",
            "乐事薯片美国经典原味70g","百岁山饮用天然矿泉水570ml（1*24）",
        ],
        "products_3d": [],
        "candidate_pool": "- 饮料:康师傅冰糖雪梨500ml（1*15）、雪碧2L（1*6）\n- 休食:洽洽香瓜子奶香味285g",
    },
]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft","rm","grpo","all"], default="all")
    a = p.parse_args()
    if a.stage in ("sft","all"):  make_sft(SAMPLES)
    if a.stage in ("rm","all"):   make_rm(SAMPLES)
    if a.stage in ("grpo","all"): make_grpo(SAMPLES)
