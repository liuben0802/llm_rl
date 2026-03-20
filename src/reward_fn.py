"""
src/reward_fn.py  ——  规则 reward + veRL compute_score 接口
"""
from __future__ import annotations
import json, re
import numpy as np

sku_brand_dict_path = "/app/data/sku_brand_dict.npy"
sku_brand_dict, _ = np.load(sku_brand_dict_path, allow_pickle=True)

def rule_reward(
    output: str,
    products_90d: list[str],
    products_lastyear: list[str],
    products_3d: list[str],
) -> float:
    score = 0.0
    try:
        m = re.search(r'\{[\s\S]*\}', output)
        if not m:
            return 0.0
        data: dict = json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return 0.02

    items = list(data.items())

    # 维度1：格式（20分）
    fmt = 20.0
    if [k for k, _ in items] != [str(i + 1) for i in range(len(items))]:
        fmt -= 6.0
    seen: set[str] = set()
    for _, v in items:
        if not isinstance(v, dict):
            fmt -= 2; continue
        s = v.get("score")
        if s is None or not (0 <= float(s) <= 1):
            fmt -= 1
        if len(v.get("reason", "")) > 20:
            fmt -= 1
        if None in v.values():
            fmt -= 2
        n = v.get("name", "")
        if n in seen:
            fmt -= 3
        seen.add(n)
    score += max(0.0, fmt)

    # 维度2：来源（20分）
    valid = set(products_90d) | set(products_lastyear)
    halluc = sum(
        1 for _, v in items
        if isinstance(v, dict) and v.get("name") and v["name"] not in valid
    )
    score += max(0.0, 20.0 - halluc * 5)

    # 维度3：业务规则（30分）
    biz = 30.0
    high = set(products_90d) | set(products_lastyear)
    rec  = {v.get("name", "") for _, v in items if isinstance(v, dict)}
    cov  = len(high & rec) / max(len(high), 1)
    biz  = biz - 10 + cov * 10
    p3d_set = set(products_3d)
    biz -= min(
        sum(2 for _, v in items
            if isinstance(v, dict)
            and v.get("name") in p3d_set
            and (v.get("score") or 0) > 0.6),
        10
    )
    def _brand(name: str) -> str:
        if name in sku_brand_dict:
            return sku_brand_dict[name]
        return name[:3]
    brands = [_brand(v.get("name", "")) for _, v in items if isinstance(v, dict)]
    biz -= min(
        sum(1 for i in range(len(brands) - 1) if brands[i] == brands[i + 1]) * 2,
        10
    )
    score += max(0.0, biz)

    # 维度4：质量（20分）
    qual = 20.0
    both   = set(products_90d) & set(products_lastyear)
    only90 = set(products_90d) - set(products_lastyear)
    for _, v in items:
        if not isinstance(v, dict): continue
        nm, s = v.get("name", ""), v.get("score") or 0
        if nm in both and s < 0.85: qual -= 1.5
        elif nm in only90 and not (0.80 <= s <= 0.95): qual -= 1
        if not v.get("reason") or len(v.get("reason", "")) < 2: qual -= 1
    n = len(items)
    if n < 10: qual -= 5
    elif n > 50: qual -= 3
    score += max(0.0, qual)

    # 维度5：覆盖度（10分）
    score += cov * 10
    return round(min(score / 100.0, 1.0), 4)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict | None = None,
) -> float:
    """veRL GRPO reward 接口"""
    if not isinstance(ground_truth, dict):
        return 0.0
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    return rule_reward(
        output=solution_str,
        products_90d=ground_truth.get("products_90d", []),
        products_lastyear=ground_truth.get("products_lastyear", []),
        products_3d=ground_truth.get("products_3d", []),
    )
