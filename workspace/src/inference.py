"""
src/inference.py  ——  线上推理入口
prompt 与训练完全一致，不存在分布偏移。
"""
from __future__ import annotations
from openai import OpenAI
from workspace.src import SYSTEM_PROMPT, build_user_prompt

INFER_BASE  = "http://localhost:8001/v1"
INFER_MODEL = "rec-qwen3-14b"


def recommend(
    shop_name: str,
    categories: str,
    brands: str,
    products_90d: list[str],
    products_lastyear: list[str],
    products_3d: list[str],
    candidate_pool: str = "",
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_base: str = INFER_BASE,
    model: str = INFER_MODEL,
) -> str:
    user = build_user_prompt(
        shop_name=shop_name, categories=categories, brands=brands,
        products_90d=products_90d, products_lastyear=products_lastyear,
        products_3d=products_3d, candidate_pool=candidate_pool,
    )
    client = OpenAI(base_url=api_base, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Rec Qwen3-14B")

    class RecRequest(BaseModel):
        shop_name: str
        categories: str
        brands: str
        products_90d: list[str]
        products_lastyear: list[str]
        products_3d: list[str] = []
        candidate_pool: str = ""

    @app.post("/recommend")
    def api_recommend(req: RecRequest) -> dict:
        return {"result": recommend(**req.model_dump())}

except ImportError:
    app = None
