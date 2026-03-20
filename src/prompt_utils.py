"""
src/prompt_utils.py  ——  训练用压缩 prompt + 智能截断
（逻辑与 v2 相同，此处直接复用）
"""

# 训练用压缩 system prompt（~500 token）
# 推理时仍使用完整版（~1940 token），两者解耦
SYSTEM_TRAIN = """\
你是商品推荐系统。输出且仅输出一个合法JSON对象，不含任何其他文字。

格式：{"1":{"name":"完整商品名","score":0.900,"reason":"≤20汉字"},...}

评分规则：
- 90天+去年同期都有 → 0.950；仅90天 → 0.900；仅去年同期 → 0.850
- 候选补充 → 0.600~0.750
- 3天内购买：完全相同 → 删除；同品牌同子品类 → 分数×0.5

输出规则：
- key从"1"连续递增；score保留3位小数；禁止null/重复商品名
- 相邻商品不同品牌；单品牌≤10个；总数≤50
- 所有name必须来自输入原始数据，禁止虚构

reason：两表都有→"高频复购商品"；仅90天→"近期热销"；仅去年→"去年同期热销"；降权→"近期已购降权"；候选→"稳定品牌推荐"或"品类扩充"\
"""

RM_SYSTEM_TRAIN = """\
你是推荐系统评估专家。对推荐结果打分，输出且仅输出JSON。

维度（共100分）：
- format_score(0-20): JSON合法/key连续/score范围/reason≤20字/无null
- source_score(0-20): 无虚构商品/无重复商品名
- business_score(0-30): 高价值商品优先/降权规则/品牌多样性
- quality_score(0-20): 分数合理/reason准确/数量合理(10~50)
- coverage_score(0-10): 高价值商品覆盖率

输出：{"format_score":N,"source_score":N,"business_score":N,"quality_score":N,"coverage_score":N,"total_score":N,"normalized_reward":0.XX}\
"""


def build_user_prompt(
    shop_name: str,
    categories: str,
    brands: str,
    products_90d: list[str],
    products_lastyear: list[str],
    products_3d: list[str],
    candidate_pool: str = "",
    *,
    tokenizer=None,
    max_tokens: int = 2200,
) -> str:
    """
    分级截断：永不裁 products_3d；优先裁 candidate_pool → lastyear → 90d
    1 汉字 ≈ 1.5 token（无 tokenizer 时的估算）
    """
    def _tok(t: str) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(t, add_special_tokens=False))
        return int(len(t) * 1.5)

    def _render(p90, ply, p3d, pool) -> str:
        parts = [
            f"【商店】{shop_name}",
            f"【品类】{categories}",
            f"【高频品牌】{brands}",
            "",
            "【90天购买】",
            "、".join(p90) if p90 else "无",
            "",
            "【去年同期购买】",
            "、".join(ply) if ply else "无",
            "",
            f"【3天内购买（降权/删除）】{'、'.join(p3d) if p3d else '无'}",
        ]
        if pool and pool.strip():
            parts += ["", "【候选补充池（推荐不足50时使用）】", pool.strip()]
        return "\n".join(parts)

    text = _render(products_90d, products_lastyear, products_3d, candidate_pool)
    if _tok(text) <= max_tokens:
        return text
    text = _render(products_90d, products_lastyear, products_3d, "")
    if _tok(text) <= max_tokens:
        return text
    ply = list(products_lastyear)[:20]
    text = _render(products_90d, ply, products_3d, "")
    if _tok(text) <= max_tokens:
        return text
    p90 = list(products_90d)[:20]
    return _render(p90, ply, products_3d, "")
