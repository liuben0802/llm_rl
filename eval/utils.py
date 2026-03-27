import json
import re

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