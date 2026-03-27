"""
workspace/src/reward_fn.py

veRL 0.7 custom_reward_function 实现
= 规则 reward (70%) + SGLang RM Server HTTP /classify (30%)

SGLang RM Server 在训练启动前预先运行（run_grpo.sh 负责启动），
compute_score 通过 HTTP 调用 /classify 端点获取神经 RM 分数。

Tokenizer 和 HTTP session 使用模块级单例，只初始化一次。
"""

from __future__ import annotations
import json
import os
import re
import threading
from functools import lru_cache
from typing import Optional

import requests
import numpy as np

sku_brand_dict_path = "/data/sku_brand_dict.npy"
sku_brand_dict, _ = np.load(sku_brand_dict_path, allow_pickle=True)

# ── RM Server 配置（可通过环境变量覆盖）──
RM_SERVER_URL = os.getenv("RM_SERVER_URL", "http://localhost:8002")
RM_MODEL_NAME = os.getenv("RM_MODEL_NAME", "rm_merged")
RM_MODEL_PATH = os.getenv("RM_MODEL_PATH", "/saves/rm_merged")
RM_TIMEOUT    = int(os.getenv("RM_TIMEOUT", "30"))

# RM 分数权重
RULE_WEIGHT = 0.7
RM_WEIGHT   = 0.3

# 模块级单例锁（compute_score 会被多线程并发调用）
_lock = threading.Lock()
_tokenizer = None
_session   = None
_rm_available: Optional[bool] = None   # None=未检测，True/False=已检测


# ────────────────────────────────────────────────────────────────
# 单例初始化
# ────────────────────────────────────────────────────────────────

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(
                    RM_MODEL_PATH, trust_remote_code=True
                )
    return _tokenizer


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                _session = requests.Session()
                _session.headers.update({"Content-Type": "application/json"})
    return _session


def _check_rm_available() -> bool:
    """检测 RM Server 是否可用（只检测一次，结果缓存）"""
    global _rm_available
    if _rm_available is not None:
        return _rm_available
    with _lock:
        if _rm_available is not None:
            return _rm_available
        try:
            r = requests.get(f"{RM_SERVER_URL}/health", timeout=3)
            _rm_available = (r.status_code == 200)
        except Exception:
            _rm_available = False
        if _rm_available:
            print(f"[reward_fn] RM Server 可用：{RM_SERVER_URL}")
        else:
            print(f"[reward_fn] RM Server 不可用，仅使用规则 reward")
    return _rm_available


# ────────────────────────────────────────────────────────────────
# 规则 reward
# ────────────────────────────────────────────────────────────────

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
    p3d  = set(products_3d)
    biz -= min(
        sum(2 for _, v in items
            if isinstance(v, dict)
            and v.get("name") in p3d
            and (v.get("score") or 0) > 0.6),
        10,
    )
    def _brand(name: str) -> str:
        if name in sku_brand_dict:
            return sku_brand_dict[name]
        return name[:3]
    brands = [_brand(v.get("name","")) for _, v in items if isinstance(v, dict)]
    biz -= min(
        sum(1 for i in range(len(brands)-1) if brands[i] == brands[i+1]) * 2, 10
    )
    score += max(0.0, biz)

    # 维度4：质量（20分）
    qual   = 20.0
    both   = set(products_90d) & set(products_lastyear)
    only90 = set(products_90d) - set(products_lastyear)
    for _, v in items:
        if not isinstance(v, dict): continue
        nm, s = v.get("name",""), v.get("score") or 0
        if nm in both and s < 0.85: qual -= 1.5
        elif nm in only90 and not (0.80 <= s <= 0.95): qual -= 1.0
        if not v.get("reason") or len(v.get("reason","")) < 2: qual -= 1.0
    n = len(items)
    if n < 10: qual -= 5
    elif n > 50: qual -= 3
    score += max(0.0, qual)

    # 维度5：覆盖度（10分）
    score += cov * 10
    return round(min(score / 100.0, 1.0), 4)


# ────────────────────────────────────────────────────────────────
# 神经 RM：调用 SGLang /classify
# ────────────────────────────────────────────────────────────────

def _build_text(system: str, user: str, response: str) -> str:
    """apply_chat_template，与训练格式一致"""
    tok = _get_tokenizer()
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": response},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _rm_score(system: str, user: str, response: str) -> float:
    """
    调用 SGLang RM Server /classify 端点，返回 [0,1] 分数。
    失败时返回 None（调用方降级为纯规则 reward）。
    """
    try:
        text = _build_text(system, user, response)
        payload = {"model": RM_MODEL_NAME, "text": text}
        resp = _get_session().post(
            f"{RM_SERVER_URL}/classify",
            json=payload,
            timeout=RM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # 响应格式：[{"embedding": [value]}]
        raw = float(data[0]["embedding"][0])
        raw = (raw - 1.02029) / (1e-6 + 0.11817) # z-score
        # sigmoid 归一化（RM 输出原始 logit）
        import math
        score = 1.0 / (1.0 + math.exp(-raw))
        return round(score, 4)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────
# veRL 0.7 compute_score 接口
# ────────────────────────────────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
) -> float:
    """
    veRL 0.7 custom_reward_function 接口。

    final_reward = rule_reward × 0.7 + rm_score × 0.3
    若 RM Server 不可用，自动降级为纯规则 reward（权重 1.0）。

    extra_info 中包含完整 prompt，用于构建 RM 请求的 system/user 部分。
    """
    # 解析 ground_truth
    if not isinstance(ground_truth, dict):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            return 0.0

    products_90d = ground_truth.get("products_90d", [])
    products_lastyear = ground_truth.get("products_lastyear", [])
    products_3d = ground_truth.get("products_3d", [])

    # 规则 reward（主要信号）
    try:
        r_rule = rule_reward(solution_str, products_90d, products_lastyear, products_3d)
    except:
        print("rule reward 数据有问题\n")
        r_rule = 0.05

    # 神经 RM reward（辅助信号，需要 RM Server 可用）
    if not _check_rm_available():
        return r_rule  # 降级：纯规则 reward

    # 从 extra_info 提取 system/user（veRL 会把 prompt messages 传入）
    system, user = "", ""
    if extra_info and "prompt" in extra_info:
        try:
            messages = extra_info["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                elif msg.get("role") == "user":
                    user = msg.get("content", "")
        except Exception:
            pass

    if not system or not user:
        return r_rule  # 没有 prompt 信息则降级

    r_rm = 0 #_rm_score(system, user, solution_str)
    if r_rm is None:
        return r_rule  # HTTP 调用失败则降级

    print(f"score: rule: {r_rule}, rm: {r_rm}")
    return round(RULE_WEIGHT * r_rule + RM_WEIGHT * r_rm, 4)

