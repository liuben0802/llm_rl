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
    score = 0.0
    # 整体格式
    try:
        m = re.search(r'\{[\s\S]*\}', output)
        if not m:
            return -3.0
        data: dict = json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return -2.0

    items = list(data.items())
    # 维度1：格式（20分）
    fmt = 20.0
    if [k for k, _ in items] != [str(i + 1) for i in range(len(items))]:
        fmt -= 6.0
    seen: set[str] = set()
    for _, v in items:
        if not isinstance(v, dict):
            fmt -= 2
            continue
        s = v.get("score")
        if type(s) == str:
            fmt -= 1
            s = float(s)
        if s is None or not (0 <= s <= 1):
            fmt -= 1
        if len(v.get("reason", "")) > 20:
            fmt -= 1
        if None in v.values():
            fmt -= 2
        n = v.get("name", "")
        if type(n) != str:
            n = ""
        if n == "":
            fmt -= 4  # 没有名称，重惩罚
        if n in seen:
            fmt -= 5
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
    rec = {v.get("name", "") for _, v in items if isinstance(v, dict) and type(v["name"]) is str}
    cov = len(high & rec) / max(len(high), 1)
    biz = biz - 10 + cov * 10
    p3d = set(products_3d)
    # 3 天降权，最多减10分
    biz -= min(
        sum(2 for _, v in items
            if isinstance(v, dict)
            and type(v.get("name")) is str and v.get("name") in p3d
            and (float(v.get("score")) or 0) > 0.5),
        10,
    )
    # 相邻品牌重复次数，10分
    def _brand(name: str) -> str:
        if type(name) != str:
            name = name["name"]
        if name in sku_brand_dict:
            return sku_brand_dict[name]
        return name[:3]
    brands = [_brand(v.get("name","")) for _, v in items if isinstance(v, dict)]
    biz -= min(
        sum(1 for i in range(len(brands)-1) if brands[i] == brands[i+1]) * 2, 10
    )
    # score += max(0.0, biz)
    score += biz

    # 维度4：质量（20分）
    qual = 20.0
    both = set(products_90d) & set(products_lastyear)
    only90 = set(products_90d) - set(products_lastyear)
    for _, v in items:
        if not isinstance(v, dict): continue
        nm, s = v.get("name",""), v.get("score") or 0
        if type(nm) != str:
            nm = ""
        if nm in both and s < 0.85: qual -= 1.5
        elif nm in only90 and not (0.80 <= float(s) <= 0.95): qual -= 1.0
        if not v.get("reason") or len(v.get("reason","")) < 2: qual -= 1.0
    n = len(items)
    if n < 10: qual -= 5
    elif n > 50: qual -= 3
    # score += max(0.0, qual)
    score += qual
    # if score < 0:
    #     score = 0.001
    result = round(min(score / 100.0, 1.0), 4)
    result = max(-3, result)
    return result


