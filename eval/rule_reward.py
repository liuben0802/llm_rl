import numpy as np

sku_brand_dict_path = "/app/data/sku_brand_dict.npy"
sku_brand_dict, _ = np.load(sku_brand_dict_path, allow_pickle=True)

def rule_based_reward(
    model_output: str,
    products_90d: list,
    products_lastyear: list,
    products_3d: list,
    candidate_pool: list,
) -> float:
    """
    纯规则 Reward 函数，返回 [0, 1] 的 float。
    速度快，适合在 GRPO rollout 时批量计算。
    """
    import json, re

    score = 0.0
    max_score = 100.0

    # ── 1. 格式合规性 (20分) ──
    fmt_score = 20.0
    try:
        # 尝试从输出中提取 JSON（兼容带 markdown 代码块的情况）
        json_match = re.search(r'\{[\s\S]*\}', model_output)
        if not json_match:
            return 0.0  # 完全无 JSON，直接返回0
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return 0.05  # JSON 解析失败，最低分

    # 检查 key 连续性
    keys = list(data.keys())
    expected_keys = [str(i) for i in range(1, len(keys) + 1)]
    if keys != expected_keys:
        fmt_score -= min(len(keys) * 2, 10)

    names_seen = set()
    for k, v in data.items():
        if not isinstance(v, dict):
            fmt_score -= 2
            continue
        # score 检查
        s = v.get("score", None)
        if s is None or not (0 <= s <= 1):
            fmt_score -= 1
        # reason 长度
        reason = v.get("reason", "")
        if len(reason) > 20:
            fmt_score -= 1
        # null 检查
        if None in v.values():
            fmt_score -= 2
        # 重复商品名
        print(v)
        name = v.get("name", "")
        if name in names_seen:
            fmt_score -= 3
        names_seen.add(name)

    fmt_score = max(0, fmt_score)
    score += fmt_score

    # ── 2. 商品来源合规性 (20分) ──
    src_score = 20.0
    all_valid_products = set(products_90d + products_lastyear + candidate_pool)
    hallucination_count = 0
    for k, v in data.items():
        name = v.get("name", "")
        if name and name not in all_valid_products:
            hallucination_count += 1
    src_score -= min(hallucination_count * 5, 20)
    src_score = max(0, src_score)
    score += src_score

    # ── 3. 业务规则遵守 (30分) ──
    biz_score = 30.0

    # 3a. 高价值商品优先推荐 (10分)
    high_value = set(products_90d) | set(products_lastyear)
    rec_names = set(v.get("name", "") for v in data.values())
    high_value_covered = len(high_value & rec_names)
    high_value_ratio = min(high_value_covered / max(len(high_value), 1), 1.0)
    biz_score_hv = high_value_ratio * 10
    biz_score = biz_score - 10 + biz_score_hv  # 替换10分部分

    # 3b. 降权规则 (10分)
    products_3d_set = set(products_3d)
    deduction_3d = 0
    for k, v in data.items():
        name = v.get("name", "")
        s = v.get("score", 0)
        if name in products_3d_set and s > 0.6:
            # 3天内购买商品没被降权（分数应该 < 0.6）
            deduction_3d += 2
    biz_score -= min(deduction_3d, 10)

    # 3c. 品牌多样性 (10分)
    items_list = list(data.values())
    brand_adjacent_violations = 0
    def get_brand(name):
        if name in sku_brand_dict:
            return sku_brand_dict[name]
        return name[:3]  # fallback

    for i in range(len(items_list) - 1):
        b1 = get_brand(items_list[i].get("name", ""))
        b2 = get_brand(items_list[i+1].get("name", ""))
        if b1 == b2:
            brand_adjacent_violations += 1
    biz_score -= min(brand_adjacent_violations * 2, 10)
    biz_score = max(0, biz_score)
    score += biz_score

    # ── 4. 推荐质量 (20分) ──
    qual_score = 20.0
    both_lists = set(products_90d) & set(products_lastyear)
    only_90d = set(products_90d) - set(products_lastyear)
    only_ly = set(products_lastyear) - set(products_90d)

    score_violations = 0
    reason_violations = 0
    for k, v in data.items():
        name = v.get("name", "")
        s = v.get("score", 0)
        reason = v.get("reason", "")
        # 分数合理性
        if name in both_lists and s < 0.85:
            score_violations += 1
        elif name in only_90d and not (0.8 <= s <= 0.95):
            score_violations += 1
        elif name in only_ly and not (0.75 <= s <= 0.9):
            score_violations += 1
        # reason 质量
        if not reason or len(reason) < 2:
            reason_violations += 1

    qual_score -= min(score_violations * 1.5, 10)
    qual_score -= min(reason_violations * 1, 5)

    # 数量合理性
    n_items = len(data)
    if n_items < 10:
        qual_score -= 5
    elif n_items > 50:
        qual_score -= 3
    qual_score = max(0, qual_score)
    score += qual_score

    # ── 5. 整体覆盖度 (10分) ──
    cov_score = 10.0
    if len(high_value) > 0:
        coverage = min(high_value_covered / min(len(high_value), 30), 1.0)
        cov_score = coverage * 10
    score += cov_score

    normalized = round(score / max_score, 4)
    return float(normalized)