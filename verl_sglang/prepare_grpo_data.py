"""
scripts/prepare_grpo_data.py
从本地已保存的 teacher response 文件生成 grpo_train.parquet。

输入：/data4/datas/teacher_responses.jsonl
输出：/data4/datas/grpo_train.parquet

运行（宿主机，不需要 GPU）：
  python scripts/prepare_grpo_data.py

输入文件每行格式：
{
  "shop_name": "...",
  "categories": "...",
  "brands": "...",
  "products_90d":      [...],
  "products_lastyear": [...],
  "products_3d":       [...],
  "candidate_pool":    "...",
  "response":          "..."   ← teacher 输出，仅用于数据质量校验，不进入 parquet
}
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

# ── 路径 ──
INPUT_FILE  = Path("/data4/datas/teacher_responses.jsonl")
OUTPUT_FILE = Path("/data4/datas/grpo_train.parquet")

# ── 完整 system prompt（与训练/推理完全一致）──
SYSTEM_PROMPT = """\
# 角色
你是商品推荐系统，严格按照以下步骤生成JSON格式推荐结果。

---

## 输出要求
**必须输出且仅输出一个合法JSON对象，不要有任何其他文字。**

格式示例：
```json
{
  "1": {"name": "可口可乐[瓶]888ml（1*12）", "score": 0.895, "reason": "高频复购饮品"},
  "2": {"name": "康师傅冰红茶500ml（1*15）", "score": 0.870, "reason": "90天内购买2次"}
}
```

**强制规则：**
- key必须是从"1"开始的连续数字字符串
- name必须是原始数据中的完整商品名
- score是0到1之间的数字，保留3位小数
- reason不超过20个汉字
- 禁止出现null、禁止重复商品名、禁止重复key

---

## 执行步骤

### 第1步：提取高价值商品（目标30-40个）
从以下两个列表中选择商品：
- **90天购买列表**
- **去年同期购买列表**

**操作规则：**
1. 如果同一商品同时出现在两个列表，只保留1次
2. 优先选择有明确规格的商品（如"1*12"、"1*24"）
3. 为每个商品分配初始分数：
   - 同时出现在两个列表：0.950
   - 仅在90天列表：0.900
   - 仅在去年同期列表：0.850

---

### 第2步：应用降权规则
检查"3天内购买列表"，对列表A中的商品进行处理：

**完全相同的商品：**
- 商品名100%一致 → 直接从列表A删除

**同系列商品（保留但降权）：**
- 同品牌+同子品类 → 分数×0.5

**其他情况：** 保持原分数

---

### 第3步：品牌多样性处理
1. 统计每个品牌的商品数量，超过10个只保留得分最高的10个
2. 按分数从高到低排序
3. 检查相邻商品，品牌相同则与下一个不同品牌的商品交换位置
4. 重复直到没有相邻同品牌商品

---

### 第4步：补充候选商品（如果不足50个）
- 优先补充用户高频品牌和高频品类的商品
- 每个品牌最多补充2个
- 补充商品分数：0.600~0.750
- 总数不超过50个

---

### 第5步：生成JSON
- key从"1"开始递增；score保留3位小数
- reason：两表都有→"高频复购商品"；仅90天→"近期热销"；仅去年→"去年同期热销"；降权→"近期已购降权"；候选→"稳定品牌推荐"或"品类扩充"

---

## 质量自检（生成后必须检查）
- [ ] 输出是否为合法JSON（无多余文字）
- [ ] 所有商品名是否来自原始数据
- [ ] 是否有重复的商品名
- [ ] 是否有相邻的同品牌商品
- [ ] score是否都在0-1之间且为3位小数
- [ ] reason是否都不超过20字
- [ ] 总数是否不超过50个\
"""


def build_user_prompt(s: dict) -> str:
    parts = [
        f"【商店】{s['shop_name']}",
        f"【品类】{s['categories']}",
        f"【高频品牌】{s['brands']}",
        "",
        "【90天购买】",
        "、".join(s["products_90d"]) if s["products_90d"] else "无",
        "",
        "【去年同期购买】",
        "、".join(s["products_lastyear"]) if s["products_lastyear"] else "无",
        "",
        f"【3天内购买（降权/删除）】{'、'.join(s['products_3d']) if s.get('products_3d') else '无'}",
    ]
    pool = s.get("candidate_pool", "")
    if pool and pool.strip():
        parts += ["", "【候选补充池（推荐不足50时使用）】", pool.strip()]
    return "\n".join(parts)


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] 输入文件不存在：{INPUT_FILE}")
        sys.exit(1)

    rows = []
    skipped = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)

            # 必填字段校验
            for field in ["shop_name", "categories", "brands",
                          "products_90d", "products_lastyear"]:
                if field not in s:
                    print(f"[WARN] 缺少字段 {field}，跳过")
                    skipped += 1
                    continue

            user = build_user_prompt(s)
            rows.append({
                # veRL 从 prompt 列读取对话，JSON 序列化存储
                "prompt": json.dumps([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user},
                ], ensure_ascii=False),
                # data_source 供 compute_score 路由
                "data_source": "rec_system",
                # ground_truth 供 reward_fn.compute_score 使用
                "ground_truth": json.dumps({
                    "products_90d":      s["products_90d"],
                    "products_lastyear": s["products_lastyear"],
                    "products_3d":       s.get("products_3d", []),
                }, ensure_ascii=False),
            })

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[OK] {len(rows)} prompts → {OUTPUT_FILE}  (skipped={skipped})")
    print(f"     columns: {list(df.columns)}")
    print(f"     sample prompt length: {len(rows[0]['prompt'])} chars")


if __name__ == "__main__":
    main()
