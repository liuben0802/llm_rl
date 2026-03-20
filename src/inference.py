"""
src/inference.py
线上推理入口 —— 使用完整版 SYSTEM_INFER（~1940 token）
训练时用的 SYSTEM_TRAIN（~500 token）不在此处出现

使用方式：
  # 直接调用
  from src.inference import recommend
  result = recommend(shop_name=..., products_90d=[...], ...)

  # 或启动 FastAPI 服务
  uvicorn src.inference:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations
from openai import OpenAI
from src.prompt_utils import SYSTEM_INFER, build_user_prompt   # ← 推理用完整版


# ── vLLM 服务地址（部署后修改）──
INFER_BASE  = "http://localhost:8001/v1"
INFER_MODEL = "rec-qwen3-14b"  # vllm serve --served-model-name          # vllm serve 时的 --served-model-name


def recommend(
    shop_name: str,
    categories: str,
    brands: str,
    products_90d: list[str],
    products_lastyear: list[str],
    products_3d: list[str],
    candidate_pool: str = "",
    *,
    temperature: float = 0.1,          # 推理用低温，保证稳定性
    max_tokens: int = 4096,
    api_base: str = INFER_BASE,
    model: str = INFER_MODEL,
) -> str:
    """
    调用部署好的 Qwen3-14B-Rec 模型，返回 JSON 字符串。
    system prompt 使用完整版 SYSTEM_INFER（不是训练时的压缩版）。
    user prompt 同样不做截断（推理时 token 预算充足）。
    """
    user = build_user_prompt(
        shop_name=shop_name,
        categories=categories,
        brands=brands,
        products_90d=products_90d,
        products_lastyear=products_lastyear,
        products_3d=products_3d,
        candidate_pool=candidate_pool,
        max_tokens=99999,              # 推理时不截断
    )

    client = OpenAI(base_url=api_base, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INFER},   # ← 完整版
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ── FastAPI 服务（可选）──
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
        result = recommend(**req.model_dump())
        return {"result": result}

except ImportError:
    app = None   # FastAPI 未安装时跳过，直接函数调用仍可用
