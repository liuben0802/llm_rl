import numpy as np
import re
import json
# sku_brand_dict_path = "/app/data/sku_brand_dict.npy"
# sku_brand_dict, _ = np.load(sku_brand_dict_path, allow_pickle=True)

sku_brand_dict = {}

def parse_products_from_user_input(user_input: str) -> dict:
    """从 user prompt 中解析商品列表（用于 reward 计算）"""
    shop_name = user_input.split("【商店信息】\n店名：")[-1].split("【用户长期消费特征】")[0].strip()
    categories = user_input.split("【用户长期消费特征】\n-")[-1].split("【稳定高频品牌（安全推荐池）")[0].strip()
    brands = user_input.split("【稳定高频品牌（安全推荐池）】\n- ")[-1].split("【高优先级商品来源规则")[0].strip()

    products_90d, products_lastyear, products_3d, candidate_pool = [], [], [], []

    # 解析90天购买
    m = re.search(r'用户最近90天购买商品：(.*?)(?:\n|3\.)', user_input, re.DOTALL)
    if m:
        products_90d = [p.strip() for p in m.group(1).split('、') if p.strip()]

    # 解析去年同期
    m = re.search(r'用户去年同期购买商品：(.*?)(?:\n→|\Z)', user_input, re.DOTALL)
    if m:
        products_lastyear = [p.strip() for p in m.group(1).split('、') if p.strip()]

    # 解析3天购买
    m = re.search(r'近3天购买商品：(.*?)(?:\n\n|\Z)', user_input, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        if raw and raw != '无':
            products_3d = [p.strip() for p in raw.split('、') if p.strip()]

    # 解析候选池
    """
    m = re.search(r'【候选商品.*?】\n(.*)', user_input, re.DOTALL)
    if m:
        pool_text = m.group(1)
        # 从各品类提取商品名
        items = re.findall(r'[^\n:：、,，]+(?:ml|g|kg|L|片|盒|包|瓶|袋|罐|听|桶)\S*', pool_text)
        candidate_pool = [i.strip() for i in items if len(i.strip()) > 3]
    """
    candidate_pool = user_input.split("【候选商品（仅在推荐不足时使用）】")[-1].strip()
    return {
        "shop_name": shop_name,
        "categories": categories,
        "brands": brands,
        "products_90d": products_90d,
        "products_lastyear": products_lastyear,
        "products_3d": products_3d,
        "candidate_pool": candidate_pool,
    }


def rule_reward(
    output: str,
    products_90d: list[str],
    products_lastyear: list[str],
    products_3d: list[str],
    products_pool: list[str]
) -> float:
    # 整体格式
    try:
        m = re.search(r'\{[\s\S]*\}', output)
        if not m:
            return -3.0
        data: dict = json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return -2.5

    items = list(data.items())
    if len(items) == 0:
        return -2.0

    all_names = []
    for _, v in items:
        if isinstance(v, dict) and type(v.get("name")) is str:
            all_names.append(v.get("name", ""))

    unique_names = set(n for n in all_names if n)
    total_named = len([n for n in all_names if n])
    dup_count = total_named - len(unique_names)

    # ★ Fix Bug1 核心：有任何重复就提前重惩罚，不走正常评分流程
    # 重复率 > 30% 直接截断，避免其他维度的正分掩盖重复行为
    if total_named > 0 and dup_count / total_named > 0.3:
        return max(-3.0, -1.0 - (dup_count / total_named) * 1.5)

    score = 0.0
    # 推荐长度不超过50个
    if len(items) > 50:
        score = -30

    # 维度1：格式（20分）
    fmt = 20.0
    if [k for k, _ in items] != [str(i + 1) for i in range(len(items))]:
        fmt -= 6.0
    seen: set[str] = set()
    def checkItemDict(item):
        result = True
        if result and ("name" not in item or type(item["name"]) != str):
            result = False
        if result and ("reason" not in item or type(item["reason"]) != str):
            result = False
        if result and ("score" not in item or type(item["score"]) != float):
            result = False
        return result

    for _, v in items:
        if not isinstance(v, dict) or not checkItemDict(v):
            fmt -= 2 # 每个item格式不对
            continue
        s = v.get("score", 0.0)
        if type(s) == str:
            fmt -= 1
            s = float(s) # score 类型不对
        if s is None or not (0 <= s <= 1):
            fmt -= 1
        if len(v.get("reason", "")) > 20 or len(v.get("reason", "")) < 5:
            fmt -= 1
        if None in v.values():
            fmt -= 2
        n = v.get("name", "")
        if type(n) != str:
            n = ""
        if n == "":
            fmt -= 10  # 没有名称，重惩罚
        if n in seen:
            fmt -= 20
        seen.add(n)
    # score += max(0.0, fmt)
    score += fmt

    # 维度2：来源（30分）
    valid = set(products_90d) | set(products_lastyear) | set(products_pool)
    halluc = 0
    for _, v in items:
        if isinstance(v, dict) and "name" in v and type(v["name"]) == str and (v["name"] not in valid):
            halluc += 1

    # 一个不在来源的直接扣10分
    # score += max(0.0, 30.0 - halluc * 10)
    score += 30.0 - halluc * 10

    # 维度3：业务规则（30分）
    biz = 30.0
    high = set(products_90d) | set(products_lastyear)
    # ★ Fix Bug3：coverage 分母改为 min(推荐数, high总量)，反映实际可覆盖的上限
    rec = unique_names  # 用去重后的集合
    rec_in_high = high & rec
    # 分母取"模型推荐了多少个 unique 商品"与"high 商品总量"的较小值
    # 这样既不奖励推很多无关商品，也不因 high 基数大而永远低 cov
    denom = min(len(rec), max(len(high), 1))
    cov = len(rec_in_high) / denom
    biz = biz - 10 + cov * 10

    # ★ Fix Bug4：3d降权的 float 判断修复
    p3d = set(products_3d)
    both = set(products_90d) & set(products_lastyear)

    def _safe_float(x):
        try:
            return float(x) if x is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    for _, v in items:
        if not isinstance(v, dict):
            continue
        nm = v.get("name", "")
        if type(nm) is not str:
            nm = ""
        s = _safe_float(v.get("score"))

        if nm in p3d:
            # 3d 商品：无论是否在 both，期望分数压低到 [0.4, 0.5]
            # 超出上界说明模型在强推近期购买，扣分
            if s > 0.5:
                biz -= 2
            # 低于下界说明模型认为完全不值得推，也轻扣（0.3~0.6 是合理的"降权但保留"区间）
            elif s < 0.4:
                biz -= 0.5
    # 相邻品牌重复次数，10分
    def _brand(name: str) -> str:
        if name in sku_brand_dict:
            return sku_brand_dict[name]
        return name[:3]
    brand_list = [_brand(v.get("name", "")) for _, v in items if isinstance(v, dict) and type(v.get("name")) is str]
    from collections import Counter
    brand_counter = Counter(brand_list)
    # 任意品牌出现次数超过3次，每多一次扣2分，最多扣10
    brand_overload = sum(max(0, cnt - 3) * 2 for cnt in brand_counter.values())
    biz -= min(brand_overload, 10)

    score += biz

    # 维度4：质量（20分）
    qual = 20.0
    both = set(products_90d) & set(products_lastyear)
    only90 = set(products_90d) - set(products_lastyear)

    for _, v in items:
        if not isinstance(v, dict):
            continue
        nm = v.get("name", "")
        if type(nm) is not str:
            nm = ""
        s = _safe_float(v.get("score"))

        # ★ 关键修复：3d 商品已经在维度3 处理过期望分数，维度4 不再重复判断
        if nm in p3d:
            continue

        if nm in both and s < 0.85:
            qual -= 1.5
        elif nm in only90 and not (0.80 <= s <= 0.95):
            qual -= 1.0

        # ★ Fix Bug2 联动：reason 要求 5-20 字（有意义的简短理由）
        reason = v.get("reason", "")
        if not reason or len(reason) < 5:
            qual -= 1.0
    unique_n = len(unique_names)
    if unique_n < 10:
        qual -= 5
    elif unique_n > 50:
        qual -= 3
    score += qual

    result = round(min(score / 100.0, 1.0), 4)
    result = max(-3, result)
    return result


