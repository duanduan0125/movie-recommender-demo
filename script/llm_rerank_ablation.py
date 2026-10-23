#!/usr/bin/env python
"""
阶段六消融实验：LLM 语义重排是否真的有效（论文的核心增量）

★ 为什么必须固定候选池（设计文档_修订版 §6.1）
  「NeuMF 直接取前 10」与「LLM 重排后取前 10」必须在**同一个候选池、同一批用户**
  上比较，只允许"顺序"不同。否则两者的候选集合都不一样，比较在方法上不成立：
    · NeuMF 的 HR@10 来自「1 正 + 99 随机负」的 100 候选口径；
    · 若 LLM 只在「NeuMF 前 20」里挑 10，候选池缩到 1/5，两个数字不可比。
  本脚本统一为：候选池 = 召回模型的前 P 个未交互影片（默认 P=50），
  然后分别计算「直接截取前 k」与「LLM 重排后截取前 k」的指标。

★ 主指标是 NDCG 而不是 HR
  重排只改顺序、不改集合，所以 HR@10 只有在正样本原本落在第 11~50 位时
  才可能被救回（上限 = 召回率 − HR@10）；而 NDCG@10 对顺序敏感，
  正样本从第 30 位提到第 3 位就产生收益，才反映重排的真实价值。
  两个指标都报，并做配对检验（不要只看均值差，那可能是采样噪声）。

★ LLM 输出的白名单校验（防幻觉，设计文档 §5.1/§5.4）
  大模型可能返回候选之外的 movie_id、编造片名或重复项 —— 一律判为无效，
  用召回原序补齐，并统计无效比例。这个比例本身就是论文里的一个数据点。

口径（与 ncf_train.py / eval_topn.py 共用 common.py，三处完全一致）：
    划分  三路：test = 每人最后一条（只用于报数）；val = 倒数第二条；train = 其余
    召回  召回模型只用 train 训练；候选池排除「train ∪ val」，但**不排除 test**
          （测试正样本必须允许出现在候选池里，否则评测无意义）
    指标  HR@5 / NDCG@5 / HR@10 / NDCG@10，基线 = 直接截取前 k

用法：
    export LLM_API_KEY=sk-xxx                      # 或 --api-key
    python llm_rerank_ablation.py --dry-run        # 先看候选池与 prompt，不调用 API
    python llm_rerank_ablation.py --users 500      # 正式跑，约 10 分钟（并发 6）

产出：
    llm_rerank_result.csv   汇总表（基线 / LLM 重排 / 配对检验 / 校验统计）
    llm_rerank_detail.csv   每人一行，供复核与画图
    .llm_rerank_cache.jsonl 调用缓存（可重跑、可断点续跑，已加入 .gitignore）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    build_eval_candidates,
    build_user_items,
    check_dataset,
    leave_one_out_with_val,
    load_movie_meta,
    load_ratings,
    reindex,
)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
# 2026-09 核对：旧的 deepseek-chat（V3）已下线，官方定价页当前只有
# deepseek-flash（DeepSeek-V4.1-Flash，本脚本默认）与 deepseek-v4-pro。
DEFAULT_MODEL = "deepseek-flash"

# DeepSeek 官方定价，单位：美元 / 百万 tokens
# 来源：https://api-docs.deepseek.com/quick_start/pricing （2026-09 核对）
# 低谷时段费率为高峰的 5 折；高峰 = UTC 周一至周五 01:00–04:00 与 06:00–10:00。
PRICES_USD_PER_M = {
    "deepseek-flash":   {"in_hit": 0.006,  "in_miss": 0.30, "out": 1.20},
    "deepseek-v4-pro":  {"in_hit": 0.044,  "in_miss": 1.32, "out": 3.96},
}


def estimate_cost_usd(model: str, tokens_in_miss: int, tokens_in_hit: int,
                      tokens_out: int, peak: bool = True) -> float:
    """
    按官方价算成本。peak=False 表示低谷时段（5 折）。
    模型不在价格表里时按 0 返回并提示 —— 宁可报 0 也不要给一个编出来的数字。
    """
    p = PRICES_USD_PER_M.get(model)
    if not p:
        return 0.0
    factor = 1.0 if peak else 0.5
    return (tokens_in_miss / 1e6 * p["in_miss"]
            + tokens_in_hit / 1e6 * p["in_hit"]
            + tokens_out / 1e6 * p["out"]) * factor

SYSTEM_PROMPT = (
    "你是电影推荐系统的排序模块。根据用户的历史偏好，对给定候选影片重新排序。"
    "只输出 JSON，不要输出任何解释性文字。"
)


API_KEY_FILE = ".llm_api_key"


def resolve_api_key(cli_key: str, path: str | None = None) -> str:
    """
    取 API Key：命令行/环境变量优先，其次读本脚本同目录的 .llm_api_key 文件。

    ★ 为什么要留文件兜底：定时任务（或已启动的 IDE 进程）拿不到"后来才设置的"
      环境变量 —— 进程的环境在启动时就固定了。密钥写进本地文件后，任何进程都能读；
      该文件已加入 .gitignore，不会入库。
    """
    if cli_key:
        return cli_key
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), API_KEY_FILE)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def count_tokens(prompts: list[str], system_prompt: str = SYSTEM_PROMPT) -> int | None:
    """
    估算这些 prompt 的总输入 token 数（含系统提示）。

    ★ 这只是**代理估计**：DeepSeek 未公开与官方等价的 tokenizer，
      这里用 tiktoken 的 o200k_base 编码近似（中文场景通常相差 ±15%）。
      精确账单只有两个来源：官方返回的 usage 字段（本脚本会统计），
      或控制台的用量明细。估算是为了"事前心里有数"，不是计费依据。
    """
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("o200k_base")
    n_sys = len(enc.encode(system_prompt))
    return sum(len(enc.encode(p)) + n_sys for p in prompts)


def print_cost_estimate(model: str, n_calls: int, tokens_in: int,
                        est_out_per_call: int = 400) -> None:
    """打印事前成本估算：给定输入 token 总量，输出按每次 est_out_per_call 估。"""
    tokens_out = n_calls * est_out_per_call
    for peak, label in ((True, "高峰"), (False, "低谷(5折)")):
        c = estimate_cost_usd(model, tokens_in, 0, tokens_out, peak=peak)
        print(f"        按 {model} {label}价：${c:.4f}（约 ¥{c * 7.1:.2f}）")


# --------------------------------------------------------------------------- #
# Prompt 构造
# --------------------------------------------------------------------------- #

def build_prompt(history: list[str], candidates: list[tuple[int, str, str]],
                 top_n: int) -> str:
    """
    history    : ["Shawshank Redemption (1994) 5分", ...]，只取训练集里的高分影片
    candidates : [(movie_id, "片名 (年份)", "类型1/类型2"), ...]，按召回分数降序

    ★ 用原始 movie_id 而不是内部索引：原始 id 与数据库、与前端展示一致，
      模型也更容易"照着抄"；内部索引一旦拼错就是静默串号。
    """
    lines = [
        f"请从下面的候选影片里，选出最符合该用户口味的 {top_n} 部，"
        "按推荐程度从高到低排序。",
        "",
        "用户历史（他喜欢过的影片）：",
        *[f"- {h}" for h in history],
        "",
        f"候选影片（共 {len(candidates)} 部；格式：movie_id | 片名 | 类型）：",
        *[f"- {mid} | {title} | {genres}" for mid, title, genres in candidates],
        "",
        "输出要求：",
        '1. 只输出 JSON，形如 {"ranking": [{"movie_id": 123, "reason": "不超过30字"}, ...]}；',
        f"2. ranking 恰好 {top_n} 项，不得重复；",
        "3. movie_id 必须严格取自上面的候选列表，禁止编造或使用列表之外的 id；",
        "4. 理由要具体，说明与用户历史偏好的关联，不要写空话。",
    ]
    return "\n".join(lines)


def format_candidate(mid: int, meta: dict) -> tuple[int, str, str]:
    """把影片元数据格式化成 prompt 里的一行"""
    title = meta["title"] if meta else f"movie_{mid}"
    if meta and meta.get("year"):
        title = f"{title} ({meta['year']})"
    genres = "/".join(meta["genres"]) if meta else "未知"
    return mid, title, genres


# --------------------------------------------------------------------------- #
# LLM 输出解析与校验
# --------------------------------------------------------------------------- #

def extract_json_object(text: str) -> dict | None:
    """
    从模型输出里抽出 JSON 对象。三种常见情况都要能处理：
      · 干净的 JSON；
      · 包在 ```json ... ``` 代码块里；
      · 前后夹了寒暄文字。
    """
    if not text:
        return None
    t = text.strip()
    for candidate in (t, _strip_code_fence(t), _slice_braces(t)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _strip_code_fence(t: str) -> str | None:
    if "```" not in t:
        return None
    body = t.split("```", 2)[1] if len(t.split("```")) >= 3 else t
    if body.lstrip().lower().startswith("json"):
        body = body.lstrip()[4:]
    return body.strip()


def _slice_braces(t: str) -> str | None:
    a, b = t.find("{"), t.rfind("}")
    return t[a:b + 1] if a >= 0 and b > a else None


def parse_ranking(text: str, whitelist: set[int], need: int) -> tuple[list[int], dict]:
    """
    解析并做白名单校验，返回 (有效 id 列表, 统计)。

    校验项（任一不通过就丢弃该条，而不是整体放弃）：
      · ranking 字段存在且是数组；
      · 每项的 movie_id 能转成整数（容忍 "123" 这种字符串）；
      · movie_id 落在本次候选白名单内 —— 这是防幻觉的关键，模型自编的 id 一定不在；
      · 不重复。
    """
    stats = {"返回条数": 0, "无效JSON": 0, "非白名单": 0, "重复": 0, "非法条目": 0}
    obj = extract_json_object(text)
    if obj is None:
        stats["无效JSON"] = 1
        return [], stats
    items = obj.get("ranking")
    if not isinstance(items, list):
        stats["无效JSON"] = 1
        return [], stats
    stats["返回条数"] = len(items)

    out: list[int] = []
    seen: set[int] = set()
    for it in items:
        mid = it.get("movie_id") if isinstance(it, dict) else None
        if isinstance(mid, str) and mid.strip().lstrip("-").isdigit():
            mid = int(mid.strip())
        if isinstance(mid, bool) or not isinstance(mid, (int, float)):
            stats["非法条目"] += 1
            continue
        mid = int(mid)
        if mid not in whitelist:
            stats["非白名单"] += 1
            continue
        if mid in seen:
            stats["重复"] += 1
            continue
        seen.add(mid)
        out.append(mid)
        if len(out) >= need:
            break
    return out, stats


def pad_ranking(llm_ids: list[int], pool_ids: list[int], need: int) -> list[int]:
    """
    第 3 层降级（设计文档 §5.4）：LLM 有效结果不足 need 条时，
    用召回原序补齐 —— 只补缺、不改动 LLM 已给出的相对顺序。
    """
    out = list(llm_ids)
    for mid in pool_ids:
        if len(out) >= need:
            break
        if mid not in out:
            out.append(mid)
    return out[:need]


# --------------------------------------------------------------------------- #
# 指标与配对检验
# --------------------------------------------------------------------------- #

def positive_rank(order: np.ndarray, positive_col: int = 0) -> int:
    """
    正样本在排序结果里的名次（0 起）。候选矩阵的第 positive_col 列约定为正样本。
    找不到时返回 -1。

    ★ 别写成 `order[0] == positive_col` —— 那只判断了"正样本是否排第一"，
      算出来的是 HR@1。这个错误实测会让基线 HR@10 从 0.6680 变成 0.1669，
      程序不报错，数字看起来还挺"合理"，属于最难发现的一类错。
    """
    hit = np.flatnonzero(order == positive_col)
    return int(hit[0]) if len(hit) else -1


def ranking_metrics(ranked: list[int], positive: int, k: int) -> tuple[float, float]:
    """
    在**给定候选池**上的指标。ranked 已按推荐度降序排列。
      正样本落在前 k 位 -> HR=1，NDCG=1/log2(rank+2)
      否则 -> 两者都是 0
    注意这与 common.evaluate_ranking 的口径不同：那个是「1 正 + 99 负」的固定
    候选集比较，这里是「前 k 位里有没有正样本」，HR@k 的含义一致，所以两者可比
    （本脚本开头会打印基线在抽样用户上的 HR@10，用来核对与全量结果是否一致）。
    """
    top = ranked[:k]
    if positive not in top:
        return 0.0, 0.0
    rank = top.index(positive)
    return 1.0, 1.0 / math.log2(rank + 2.0)


def mcnemar_p(baseline_hit: np.ndarray, treat_hit: np.ndarray) -> tuple[int, int, float]:
    """
    HR 的配对检验（McNemar 精确检验）。
    b = 基线命中而实验未命中；c = 基线未命中而实验命中。
    只看这两个反向不一致的格子；p 值大意味着"变化可能只是噪声"。
    """
    b = int(((baseline_hit == 1) & (treat_hit == 0)).sum())
    c = int(((baseline_hit == 0) & (treat_hit == 1)).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    from scipy.stats import binomtest
    return b, c, float(binomtest(c, n, 0.5, alternative="two-sided").pvalue)


def wilcoxon_p(baseline_ndcg: np.ndarray, treat_ndcg: np.ndarray) -> tuple[float, float]:
    """NDCG 的配对检验（Wilcoxon 符号秩），返回 (统计量, p 值)"""
    diff = treat_ndcg - baseline_ndcg
    if not np.any(diff != 0):
        return 0.0, 1.0
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(treat_ndcg, baseline_ndcg)
    return float(stat), float(p)


def build_request_body(model: str, user_prompt: str, thinking: str = "disabled",
                       max_tokens: int = 2000, json_mode: bool = True) -> dict:
    """
    构造 /chat/completions 请求体。

    ★ thinking 默认必须是 disabled：官方默认 reasoning_effort=high，实测每次调用
      多出约 5,700 个不可见的推理 tokens —— 输出成本约为关闭时的 12 倍，
      单次耗时从 2 秒涨到 36 秒，且远超设计文档要求的线上 5 秒超时。
      这是本脚本最影响成本的一行配置，因此单独抽出来由测试锁住。
    """
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": max_tokens,
    }
    if json_mode:
        # 官方要求同时用消息指示模型输出 JSON，否则可能一直输出空白到上限
        body["response_format"] = {"type": "json_object"}
    if thinking == "disabled":
        body["thinking"] = {"type": "disabled"}
        # 思考模式下 temperature 不生效，关闭后才起作用，用于降低随机性
        body["temperature"] = 0.2
    else:
        body["thinking"] = {"type": "enabled", "reasoning_effort": thinking}
    return body


# --------------------------------------------------------------------------- #
# LLM 客户端：超时重试 + 熔断（设计文档 §5.4 第 1 层）
# --------------------------------------------------------------------------- #

class LlmClient:
    """
    仅用标准库 urllib，不额外引入 requests —— 少一个依赖少一处安装失败。
    重试策略：超时与 5xx 重试 1 次；4xx 不重试（密钥错、参数错，重试也没用）。
    熔断：连续失败达到 max_fail 后，本批后续请求直接返回失败，不再逐条等超时。
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, retries: int = 1, max_fail: int = 5,
                 thinking: str = "disabled", max_tokens: int = 2000,
                 json_mode: bool = True, max_cost_usd: float = 0.0,
                 billing_peak: bool = True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_fail = max_fail
        # ★ 思考模式默认关闭。官方默认是 reasoning_effort=high，实测每次调用会多出
        #   约 5,700 个不可见的推理 tokens：成本约为关闭时的 12 倍，且单次耗时 36 秒，
        #   远超设计文档要求的线上 5 秒超时。本任务是"重排 50~100 个候选并给理由"，
        #   不需要思维链。
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        # 预算硬闸：只剩很少余额时必须能在跑到一半时自动刹车。
        # 按累计 tokens 实时折算，超限立即停止调用（已完成的部分仍写入缓存，重跑可续）。
        self.max_cost_usd = max_cost_usd
        self.billing_peak = billing_peak
        self.budget_stopped = False
        self._lock = threading.Lock()
        self._fail_streak = 0
        self.circuit_open = False
        self.n_calls = 0
        self.n_fail = 0
        # 真实用量统计：跑完直接给精确账单，不靠估算
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.reasoning_tokens = 0

    def billed_cost_usd(self) -> float:
        """按累计 tokens 折算的已花费金额（线程安全由调用方保证或可容忍微小偏差）"""
        return estimate_cost_usd(self.model, self.cache_miss_tokens,
                                 self.cache_hit_tokens, self.completion_tokens,
                                 peak=self.billing_peak)

    def chat(self, user_prompt: str) -> tuple[str | None, str | None]:
        with self._lock:
            if self.circuit_open:
                return None, "已熔断或已停止"
            # 预算闸：在发起请求前检查，超限就不再产生任何新费用
            if self.max_cost_usd > 0 and self.billed_cost_usd() >= self.max_cost_usd:
                self.budget_stopped = True
                self.circuit_open = True
                return None, "预算已达上限"
            self.n_calls += 1

        body = build_request_body(self.model, user_prompt, self.thinking,
                                  self.max_tokens, self.json_mode)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})

        err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self._record_usage(body.get("usage") or {})
                self._on_success()
                return body["choices"][0]["message"]["content"], None
            except urllib.error.HTTPError as e:
                detail = e.read()[:200].decode("utf-8", "ignore")
                err = f"HTTP {e.code}: {detail}"
                if not (500 <= e.code < 600) or attempt >= self.retries:
                    break
            except Exception as e:                       # 超时、连接失败、解析失败
                err = f"{type(e).__name__}: {e}"
                if attempt >= self.retries:
                    break
            time.sleep(2.0)
        self._on_failure()
        return None, err

    def _record_usage(self, usage: dict) -> None:
        """累计官方返回的 usage：这是唯一精确的计费依据（不靠字符数估算）"""
        with self._lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            hit = int(usage.get("prompt_cache_hit_tokens") or 0)
            miss = int(usage.get("prompt_cache_miss_tokens") or 0)
            self.cache_hit_tokens += hit
            self.cache_miss_tokens += miss
            # 少数兼容实现不返回缓存分项，按全部未命中计（宁可高估成本）
            if hit + miss == 0:
                self.cache_miss_tokens += int(usage.get("prompt_tokens") or 0)
            det = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += int(det.get("reasoning_tokens") or 0)

    def _on_success(self) -> None:
        with self._lock:
            self._fail_streak = 0

    def _on_failure(self) -> None:
        with self._lock:
            self.n_fail += 1
            self._fail_streak += 1
            if self._fail_streak >= self.max_fail and not self.circuit_open:
                self.circuit_open = True
                print(f"[熔断] 连续失败 {self._fail_streak} 次，本批后续请求直接降级"
                      "（不再逐条等待超时）")


# --------------------------------------------------------------------------- #
# 缓存：可重跑、可断点续跑
# --------------------------------------------------------------------------- #

def cache_key(recall: str, uidx: int, pool: list[int]) -> str:
    h = hashlib.sha1(",".join(str(x) for x in pool).encode()).hexdigest()[:12]
    return f"{recall}|{uidx}|{h}"


def load_cache(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in rec and "raw" in rec:
                out[rec["key"]] = rec["raw"]
    return out


def append_cache(path: str, key: str, raw: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def build_recall_scorer(args, train, n_users: int, n_items: int, users_orig, items_orig):
    """复用 eval_topn.py 里已有的打分器，不重复实现召回模型"""
    from eval_topn import make_itemknn_scorer, make_mostpop_scorer, make_neumf_scorer

    if args.recall == "neumf":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return make_neumf_scorer(args.ckpt, n_users, n_items, args.dim, device)
    if args.recall == "itemknn":
        return make_itemknn_scorer(train, n_users, n_items, topk_sim=args.knn_k)
    return make_mostpop_scorer(train, n_users, n_items)


def top_pool(score, uidx: int, exclude: set[int], n_items: int,
             pool: int) -> list[int]:
    """
    取该用户召回分数的前 pool 名（排除已知交互）。

    ★ exclude 只含「train ∪ val」，不含 test：
      测试正样本必须允许出现在候选池里，否则重排实验无从评测。
      这与 eval_topn.py 的负采样口径（排除全部交互）不同，是刻意为之 ——
      那边是在比较固定候选集里的分数，这边是在模拟线上"从未看过的东西里挑"。
    """
    cand = np.arange(n_items, dtype=np.int64)
    scores = score(np.full(n_items, uidx, dtype=np.int64), cand)
    if exclude:
        scores[np.fromiter(exclude, dtype=np.int64, count=len(exclude))] = -np.inf
    k = min(pool, n_items)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def full_ranking_diagnostic(score, pos_of: dict[int, int], known, n_items: int,
                            pool: int, k: int = 10) -> tuple[float, float, float]:
    """
    全量排序口径的诊断：把所有未交互影片一起排序，看正样本的真实名次。
    返回 (Recall@pool, HR@k, NDCG@k) 的用户平均值。

    ★ 为什么必须测这个：实验二用的是「1 正 + 99 随机负」的**采样指标**，
      它显著宽松于全量排序 —— 前者只要打败 99 个随机负样本，后者要在几千部
      未看过的影片里挤进前 k（Krichene & Rendle 2020 专门论证过两者不一致）。
      不给这个数，论文里的 HR@10 = 0.6680 会被误读成
      "66.8% 的用户能在全站排序里看到想要的影片"，那是错的。
    """
    rec_hit = hr_hit = 0
    ndcg = 0.0
    all_items = np.arange(n_items, dtype=np.int64)
    for u, pos in pos_of.items():
        s = score(np.full(n_items, u, dtype=np.int64), all_items)
        # 已知交互（train ∪ val）排除；正样本（test）不在其中，照常参与排名
        s[np.fromiter(known[u], dtype=np.int64, count=len(known[u]))] = -np.inf
        rank = int((s > s[pos]).sum())
        if rank < pool:
            rec_hit += 1
        if rank < k:
            hr_hit += 1
            ndcg += 1.0 / math.log2(rank + 2.0)
    n = max(1, len(pos_of))
    return rec_hit / n, hr_hit / n, ndcg / n


def build_user_history(train_df, uidx: int, items_orig, meta: dict,
                       n: int, min_rating: int) -> list[str]:
    """取该用户在**训练集**里的高分影片作为 prompt 里的历史（不含 val / test）"""
    sub = train_df[(train_df["uidx"] == uidx) & (train_df["rating"] >= min_rating)]
    sub = sub.sort_values(["rating", "timestamp"], ascending=False).head(n)
    out = []
    for _, r in sub.iterrows():
        mid = int(items_orig[int(r["iidx"])])
        m = meta.get(mid)
        title = m["title"] if m else f"movie_{mid}"
        if m and m.get("year"):
            title = f"{title} ({m['year']})"
        out.append(f"{title}　{int(r['rating'])}分")
    return out


def mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def pool_size_of(pools: dict[int, list[int]]) -> int:
    """
    候选池的**真实**大小。

    sampled 模式的池子是 1 个正样本 + n_neg 个随机负样本（默认 100），
    与命令行里的 --pool 无关（--pool 只在 full 模式下生效）。
    统计输出必须用真实大小，否则两种候选池的产物在结果表里无法区分。
    """
    return len(next(iter(pools.values())))


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 语义重排消融实验")
    ap.add_argument("--data-dir", default="./ml-1m")
    ap.add_argument("--recall", choices=["neumf", "itemknn", "mostpop"],
                    default="neumf", help="候选池的召回模型")
    ap.add_argument("--ckpt", default="ncf_model.pt")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--knn-k", type=int, default=100, help="--recall itemknn 时的邻居数")
    ap.add_argument("--pool-mode", choices=["sampled", "full"], default="sampled",
                    # 注意：argparse 会对 help 做 % 格式化，百分号必须写成 %%
                    help="sampled（默认）= 用与实验二主表完全相同的 1 正 + 99 随机负"
                         "候选集，覆盖率 100%% 且基线与主表可逐人对齐；"
                         "full = 用全量排序的前 P 名，对应线上真实召回路径"
                         "（实测覆盖率约 24%%，需配合 --require-pool-hit）")
    ap.add_argument("--pool", type=int, default=50,
                    help="--pool-mode full 时的候选池大小 P")
    ap.add_argument("--require-pool-hit", action="store_true",
                    help="--pool-mode full 时只抽「正样本在候选池内」的用户，"
                         "否则多数用户无从评测；属条件抽样，输出里会标注")
    ap.add_argument("--n-neg", type=int, default=99,
                    help="--pool-mode sampled 时的随机负样本数")
    ap.add_argument("--report-full-recall", action="store_true",
                    help="额外测量全量排序口径下的召回率，量化采样指标的宽松程度"
                         "（论文里讨论口径差异时用，会多花几十秒）")
    ap.add_argument("--top-n", type=int, default=10, help="LLM 重排后保留的条数")
    ap.add_argument("--users", type=int, default=500, help="抽样的测试用户数")
    ap.add_argument("--history", type=int, default=15, help="prompt 里给出的历史条数")
    ap.add_argument("--history-min-rating", type=int, default=4)
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-fail", type=int, default=5)
    ap.add_argument("--cache", default=".llm_rerank_cache.jsonl")
    ap.add_argument("--out", default="llm_rerank_result.csv")
    ap.add_argument("--detail", default="llm_rerank_detail.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="只构造候选池与 prompt 并打印样例，不调用 API")
    ap.add_argument("--count-tokens", action="store_true",
                    help="不调用 API，只估算 prompt 的输入 tokens 与成本"
                         "（用 tiktoken 作代理，需 pip install tiktoken）")
    ap.add_argument("--est-out-tokens", type=int, default=500,
                    help="估算成本时假设的每次输出 tokens（默认 500，"
                         "对应 10 条带理由的 JSON；输入 tokens 是实测的，输出只能假设）"
                         "注意：若开启思考模式，实际输出会包含大量推理 tokens（实测约 6200）")
    ap.add_argument("--thinking", choices=["disabled", "none", "low", "high", "max"],
                    default="disabled",
                    help="思考模式。默认 disabled：官方默认是 high，实测每次多出约 5700 个"
                         "推理 tokens（成本约 12 倍、单次耗时 36 秒）")
    ap.add_argument("--max-tokens", type=int, default=2000,
                    help="单次生成上限，防止异常情况下无限输出")
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="累计花费上限（美元），达到即停止调用；0 表示不限。"
                         "按 --billing 的费率实时折算，超限会保留已完成部分到缓存")
    ap.add_argument("--billing", choices=["peak", "off-peak"], default="peak",
                    help="计算预算与成本时按哪档费率：peak（默认，保守）或 off-peak（5 折）")
    args = ap.parse_args()

    df = load_ratings(args.data_dir)
    check_dataset(df)
    df, users_orig, items_orig = reindex(df)
    n_users, n_items = len(users_orig), len(items_orig)
    meta = load_movie_meta(args.data_dir)
    train, val, test = leave_one_out_with_val(df)
    print(f"[划分] 三路：train {len(train)} / val {len(val)} / test {len(test)}")

    # 召回模型只用 train 训练（与实验二口径一致）
    score = build_recall_scorer(args, train, n_users, n_items, users_orig, items_orig)

    known = {u: set() for u in range(n_users)}
    for d in (train, val):
        for u, i in zip(d["uidx"].values, d["iidx"].values):
            known[int(u)].add(int(i))

    rng = np.random.default_rng(args.seed)
    all_pos = dict(zip(test["uidx"].astype(int).tolist(),
                       test["iidx"].astype(int).tolist()))
    print(f"[测试集] 共 {len(all_pos)} 位用户（seed={args.seed}）")

    # ---------------- 候选池 ----------------
    t0 = time.time()
    pools: dict[int, list[int]] = {}
    if args.pool_mode == "sampled":
        # 候选池 = 与 eval_topn.py 完全相同的 1 正 + 99 随机负（seed 也相同），
        # 只是按召回分数重排。这样基线 HR@10 能与实验二主表逐人对齐，
        # 且正样本必然在池内（覆盖率 100%），配对检验用得上全部抽样用户。
        #
        # ★ 为什么还要有 full 模式：这个候选集是**采样指标**，比真实排序宽松得多。
        #   实测同一模型全量排序的 HR@10 只有 0.0728，而采样口径 0.6680。
        #   两种都跑，论文里才能既与主表对齐、又反映线上难度。
        all_items = build_user_items(df, n_users)
        u_all, cand_all = build_eval_candidates(test, all_items, n_items,
                                                n_neg=args.n_neg, seed=args.seed)
        hit10 = 0
        ordered = {}
        for r in range(len(u_all)):
            u = int(u_all[r])
            s = score(np.full(cand_all.shape[1], u, dtype=np.int64), cand_all[r])
            order = np.argsort(-s, kind="stable")
            # 第 0 列是正样本，它的名次 = 正样本在排序里的位置（0 起）
            if positive_rank(order) < 10:
                hit10 += 1
            ordered[u] = [int(items_orig[cand_all[r][j]]) for j in order]
        print(f"[口径校验] 全量 {len(u_all)} 户的基线 HR@10 = {hit10 / len(u_all):.4f}"
              f"（实验二主表 {args.recall} 的值应与之一致）")

        n_sample = min(args.users, len(all_pos))
        sel = np.sort(rng.choice(len(all_pos), size=n_sample, replace=False))
        chosen = [list(all_pos)[i] for i in sel]
        print(f"[抽样] 从 {len(all_pos)} 位测试用户中随机抽 {n_sample} 位")
        pos_of = {u: all_pos[u] for u in chosen}
        pools = {u: ordered[u] for u in chosen}
    else:
        # full：全量排序的前 P 名，对应线上「CF 召回 50 → LLM 重排」的真实路径。
        # 先对全部测试用户算一遍（实测约 10 秒），再按需抽样。
        by_user = {}
        hits = []
        for u, pos in all_pos.items():
            top = top_pool(score, u, known[u], n_items, args.pool)
            by_user[u] = [int(items_orig[i]) for i in top]
            if int(items_orig[pos]) in by_user[u]:
                hits.append(u)
        print(f"[全量排序] 前 {args.pool} 名的覆盖率 = {len(hits)}/{len(all_pos)} "
              f"= {len(hits) / len(all_pos):.4f}（正样本落在池内的用户比例）")

        if args.require_pool_hit:
            # 条件抽样：只从正样本在池内的用户里抽。
            # 绝对指标只在这个子集上成立（本来也只该在这个子集上谈），
            # 但两组（直接取前 n / LLM 重排）看到的是同一个池，对比是无偏的。
            pool_src = hits
            print(f"        ★ 条件抽样：只在正样本入池的 {len(hits)} 位用户中抽"
                  f"（否则重排无从评测）；绝对指标限于该子集，组间对比无偏")
        else:
            pool_src = list(all_pos)
        n_sample = min(args.users, len(pool_src))
        sel = np.sort(rng.choice(len(pool_src), size=n_sample, replace=False))
        chosen = [pool_src[i] for i in sel]
        print(f"[抽样] 从 {len(pool_src)} 位候选中随机抽 {n_sample} 位")
        pos_of = {u: all_pos[u] for u in chosen}
        pools = {u: by_user[u] for u in chosen}

    in_pool = sum(1 for u, p in pos_of.items() if int(items_orig[p]) in pools[u])
    print(f"[召回复盖] 测试正样本落在候选池内的用户：{in_pool}/{len(pos_of)} "
          f"= {in_pool / len(pos_of):.3f}  ← HR@10 的理论上限就是它")

    # 候选池的真实大小：sampled 模式是 1 + n_neg，full 模式才是 --pool。
    # 结果表里必须写真实大小，否则两种池的产物无法区分（曾误写 --pool 导致 A 组标成 50）。
    pool_size = pool_size_of(pools)

    if args.report_full_recall:
        rec, hr, nd = full_ranking_diagnostic(score, pos_of, known, n_items, args.pool)

    if args.report_full_recall:
        rec, hr, nd = full_ranking_diagnostic(score, pos_of, known, n_items, args.pool)
        print(f"[诊断] 全量排序口径（同一批用户，所有未看过的影片一起排，耗时 "
              f"{time.time() - t0:.0f}s）：")
        print(f"        Recall@{args.pool} = {rec:.4f}   HR@10 = {hr:.4f}   "
              f"NDCG@10 = {nd:.4f}")
        print(f"        ← 采样口径（1 正 + 99 随机负）下同一模型的 HR@10 远高于此，"
              f"这就是采样指标的宽松程度，写论文时必须说明口径")

    prompts: dict[int, str] = {}
    hist_lens = []
    for u in pos_of:
        hist = build_user_history(train, u, items_orig, meta,
                                  args.history, args.history_min_rating)
        hist_lens.append(len(hist))
        cands = [format_candidate(mid, meta.get(mid)) for mid in pools[u]]
        prompts[u] = build_prompt(hist, cands, args.top_n)
    print(f"[候选池] 每池 {pool_size} 部，"
          f"prompt 历史条数 均值 {np.mean(hist_lens):.1f}")
    avg_prompt = np.mean([len(p) for p in prompts.values()])
    print(f"[prompt] 平均 {avg_prompt:.0f} 字符（约 {avg_prompt / 1.5:.0f} tokens）")

    if args.count_tokens:
        tok = count_tokens(list(prompts.values()))
        n = len(prompts)
        if tok is None:
            print("[token] 未安装 tiktoken，字符数均值 "
                  f"{avg_prompt:.0f}；可 pip install tiktoken 得到代理估计")
        else:
            print(f"[token] 用 o200k_base 代理估计：{n} 次调用共 {tok} 输入 tokens"
                  f"（均值 {tok / n:.0f}/次，含系统提示）")
            print_cost_estimate(args.model, n, tok, args.est_out_tokens)
            print("        ★ 仅为代理估计；脚本跑完后会打印官方 usage 的精确用量与成本")
        return 0

    if args.dry_run:
        u = next(iter(prompts))
        print("\n---------------- prompt 样例 ----------------")
        print(prompts[u][:1200] + ("\n...(略)" if len(prompts[u]) > 1200 else ""))
        print("---------------------------------------------")
        print("[dry-run] 未调用 API，退出")
        return 0

    args.api_key = resolve_api_key(args.api_key)
    if not args.api_key:
        raise SystemExit("[中止] 缺少 API Key：设置环境变量 LLM_API_KEY、"
                         f"传 --api-key，或写入 script/{API_KEY_FILE}（该文件不入库）")

    # ---------------- 调用 LLM ----------------
    cache = load_cache(args.cache)
    client = LlmClient(args.base_url, args.api_key, args.model,
                       timeout=args.timeout, max_fail=args.max_fail,
                       thinking=args.thinking, max_tokens=args.max_tokens,
                       max_cost_usd=args.max_cost_usd,
                       billing_peak=(args.billing == "peak"))
    if args.max_cost_usd > 0:
        print(f"[预算] 上限 ${args.max_cost_usd:.3f}"
              f"（约 ¥{args.max_cost_usd * 7.1:.2f}，按{args.billing}费率折算），"
              f"达到即停止调用")
    todo = [u for u in pools if cache_key(args.recall, u, pools[u]) not in cache]
    print(f"[调用] 缓存命中 {len(pools) - len(todo)}，待调用 {len(todo)}，"
          f"并发 {args.workers}，模型 {args.model}")

    raw: dict[int, str] = {}
    for u in pools:
        k = cache_key(args.recall, u, pools[u])
        if k in cache:
            raw[u] = cache[k]

    if todo:
        counter = {"done": 0}
        lock = threading.Lock()
        t_call = time.time()

        def work(u: int):
            text, err = client.chat(prompts[u])
            with lock:
                counter["done"] += 1
                if text is not None:
                    raw[u] = text
                    append_cache(args.cache, cache_key(args.recall, u, pools[u]), text)
                if counter["done"] % 25 == 0 or counter["done"] == len(todo):
                    el = time.time() - t_call
                    print(f"  进度 {counter['done']}/{len(todo)}  耗时 {el:.0f}s  "
                          f"失败 {client.n_fail}")
                if text is None:
                    print(f"  [失败] uidx={u}: {err}")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        print(f"[调用] 完成，成功 {len(raw)}/{len(pools)}，"
              f"失败 {client.n_fail}，耗时 {time.time() - t_call:.0f}s")

    if client.budget_stopped:
        # 结果不完整时坚决不写文件：未调用到的那部分用户会回落到召回原序，
        # 把两组指标强行拉平，写出去就是误导性数据。
        # 已完成的调用都在缓存里，提高上限或减少 --users 后重跑即可续上。
        print(f"\n[中止] 累计花费已达上限 "
              f"${client.billed_cost_usd():.4f}（{args.billing} 费率），"
              f"已完成 {len(raw)}/{len(pools)} 位用户。")
        print("       未写入结果文件（避免部分降级导致两组指标被拉平）。")
        print("       已完成的调用已缓存，重跑会从断点继续，不会重复计费。")
        return 2

    cost_peak = 0.0
    if client.n_calls:
        cost_peak = estimate_cost_usd(args.model, client.cache_miss_tokens,
                                      client.cache_hit_tokens,
                                      client.completion_tokens, peak=True)
        print(f"[用量] 官方 usage 累计：调用 {client.n_calls} 次，"
              f"输入 {client.prompt_tokens} tokens"
              f"（缓存命中 {client.cache_hit_tokens} / 未命中 {client.cache_miss_tokens}），"
              f"输出 {client.completion_tokens} tokens"
              f"（其中推理 {client.reasoning_tokens}，思考模式={args.thinking}）")
        print(f"[成本] 按 {args.model}：高峰 ${cost_peak:.4f}（约 ¥{cost_peak * 7.1:.2f}）"
              f" / 低谷 ${cost_peak * 0.5:.4f}（约 ¥{cost_peak * 0.5 * 7.1:.2f}）"
              f"　← 以官方 usage 计，非估算")

    # ---------------- 校验 + 指标 ----------------
    agg = {name: [] for name in
           ["base_hr5", "treat_hr5", "base_ndcg5", "treat_ndcg5",
            "base_hr10", "treat_hr10", "base_ndcg10", "treat_ndcg10"]}
    n_pad_total = n_llm_only = 0
    stat_names = ["返回条数", "无效JSON", "非白名单", "重复", "非法条目"]
    stats_sum = {s: 0 for s in stat_names}
    n_fail = 0
    detail_rows = []

    for u in sorted(pools):
        pos = int(items_orig[pos_of[u]])
        pool = pools[u]
        base = pool[:args.top_n]
        b_hr10, b_ndcg10 = ranking_metrics(base, pos, 10)
        b_hr5, b_ndcg5 = ranking_metrics(base, pos, 5)

        text = raw.get(u)
        if text is None:
            n_fail += 1
            # 第 1 层降级：调用失败 -> 完全回落到召回原序，两列指标相同
            rank_ids, n_llm = list(base), 0
            lstats = {s: 0 for s in stat_names}
        else:
            valid, lstats = parse_ranking(text, set(pool), args.top_n)
            rank_ids = pad_ranking(valid, pool, args.top_n)
            n_llm = len(valid)
        for s in stat_names:
            stats_sum[s] += lstats[s]
        n_llm_only += n_llm
        n_pad_total += max(0, args.top_n - n_llm)

        t_hr10, t_ndcg10 = ranking_metrics(rank_ids, pos, 10)
        t_hr5, t_ndcg5 = ranking_metrics(rank_ids, pos, 5)
        for name, val in (("base_hr5", b_hr5), ("treat_hr5", t_hr5),
                          ("base_ndcg5", b_ndcg5), ("treat_ndcg5", t_ndcg5),
                          ("base_hr10", b_hr10), ("treat_hr10", t_hr10),
                          ("base_ndcg10", b_ndcg10), ("treat_ndcg10", t_ndcg10)):
            agg[name].append(val)

        detail_rows.append({
            "uidx": u, "user_id": int(users_orig[u]), "movie_id": pos,
            "正样本在池内": int(pos in pool),
            "基线排名": (base.index(pos) + 1) if pos in base else 0,
            "重排后排名": (rank_ids.index(pos) + 1) if pos in rank_ids else 0,
            "基线HR@10": int(b_hr10), "重排HR@10": int(t_hr10),
            "基线NDCG@10": round(b_ndcg10, 6), "重排NDCG@10": round(t_ndcg10, 6),
            "LLM有效条数": n_llm, "补齐条数": max(0, args.top_n - n_llm),
            "调用失败": int(text is None),
            **{f"校验_{s}": lstats[s] for s in stat_names},
        })

    A = {k: np.asarray(v) for k, v in agg.items()}
    b10, c10, p_hr10 = mcnemar_p(A["base_hr10"], A["treat_hr10"])
    b5, c5, p_hr5 = mcnemar_p(A["base_hr5"], A["treat_hr5"])
    _, p_ndcg10 = wilcoxon_p(A["base_ndcg10"], A["treat_ndcg10"])
    _, p_ndcg5 = wilcoxon_p(A["base_ndcg5"], A["treat_ndcg5"])

    print("\n================ LLM 重排消融结果 ================")
    print(f"抽样用户 {len(A['base_hr10'])} | 召回模型 {args.recall} | "
          f"候选池 {pool_size}（{args.pool_mode}）| "
          f"重排后保留 {args.top_n} | 思考模式={args.thinking}")
    print(f"  召回复盖率（正样本在池内）= {in_pool / len(pos_of):.4f}"
          f"  <- HR@10 的理论上限")
    print(f"  {'指标':<14}{'基线(直接取前n)':>16}{'LLM重排后':>12}{'差值':>10}{'p值':>12}")
    for k in (5, 10):
        print(f"  {'HR@' + str(k):<14}{mean(A[f'base_hr{k}']):>16.4f}"
              f"{mean(A[f'treat_hr{k}']):>12.4f}"
              f"{mean(A[f'treat_hr{k}']) - mean(A[f'base_hr{k}']):>+10.4f}"
              f"{(p_hr5 if k == 5 else p_hr10):>12.4f}")
        print(f"  {'NDCG@' + str(k):<14}{mean(A[f'base_ndcg{k}']):>16.4f}"
              f"{mean(A[f'treat_ndcg{k}']):>12.4f}"
              f"{mean(A[f'treat_ndcg{k}']) - mean(A[f'base_ndcg{k}']):>+10.4f}"
              f"{(p_ndcg5 if k == 5 else p_ndcg10):>12.4f}")
    print(f"  McNemar HR@10: 基线命中/重排未命中 b={b10}，反向 c={c10}，"
          f"p={p_hr10:.4f}{'（显著）' if p_hr10 < 0.05 else '（不显著，差异可能是噪声）'}")
    print(f"  McNemar HR@5 : b={b5}, c={c5}, p={p_hr5:.4f}")
    print(f"  Wilcoxon NDCG@10 p={p_ndcg10:.4f}，NDCG@5 p={p_ndcg5:.4f}")
    print("  ---- LLM 输出校验（防幻觉）----")
    print(f"  调用失败 {n_fail}/{len(pools)}"
          f"{'（已熔断或网络异常，该用户完全回落召回原序）' if n_fail else ''}")
    print(f"  有效条数合计 {n_llm_only}，补齐 {n_pad_total}"
          f"（补齐率 {n_pad_total / max(1, n_llm_only + n_pad_total):.3f}）")
    print(f"  校验明细：{stats_sum}")
    print("=================================================")

    write_outputs(args, A, detail_rows, summary_metadata(
        args, client, pool_size=pool_size, n_users=len(A["base_hr10"]),
        in_pool=in_pool, n_fail=n_fail, n_pad_total=n_pad_total,
        cost_peak=cost_peak, b10=b10, c10=c10, p_hr10=p_hr10,
        p_ndcg10=p_ndcg10, p_hr5=p_hr5, p_ndcg5=p_ndcg5), stats_sum)
    return 0


def summary_metadata(args, client, *, pool_size: int, n_users: int, in_pool: int,
                     n_fail: int, n_pad_total: int, cost_peak: float,
                     b10: int, c10: int, p_hr10: float, p_ndcg10: float,
                     p_hr5: float, p_ndcg5: float) -> dict:
    """
    结果表第二段的「口径元数据」。

    单独抽成函数是为了让「结果表里到底写了什么口径」可以被单元测试直接断言：
    **候选池大小必须用真实值**（sampled 模式是 1 + n_neg），
    这里曾写成 args.pool，导致 sampled 组（真实池 100）在结果表里被标成 50 —— 
    一旦写错，A/B 两种候选池的产物就再也分不开了。
    """
    return {
        "召回模型": args.recall, "候选池": pool_size, "重排保留": args.top_n,
        "抽样用户": n_users, "模型": args.model, "seed": args.seed,
        "召回复盖率": round(in_pool / n_users, 4),
        "调用失败": n_fail, "补齐条数": n_pad_total,
        "调用次数": client.n_calls,
        "输入tokens": client.prompt_tokens,
        "缓存命中tokens": client.cache_hit_tokens,
        "缓存未命中tokens": client.cache_miss_tokens,
        "输出tokens": client.completion_tokens,
        "成本USD_高峰": round(cost_peak, 6),
        "成本USD_低谷": round(cost_peak * 0.5, 6),
        "McNemar_b_HR10": b10, "McNemar_c_HR10": c10, "p_HR10": round(p_hr10, 6),
        "p_NDCG10": round(p_ndcg10, 6), "p_HR5": round(p_hr5, 6),
        "p_NDCG5": round(p_ndcg5, 6),
    }


def write_outputs(args, A, detail_rows, summary_extra, stats_sum) -> None:
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["策略", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "用户数"])
        w.writerow(["基线：召回直接取前n", f"{mean(A['base_hr5']):.4f}",
                    f"{mean(A['base_ndcg5']):.4f}", f"{mean(A['base_hr10']):.4f}",
                    f"{mean(A['base_ndcg10']):.4f}", len(A["base_hr10"])])
        w.writerow(["LLM 语义重排后取前n", f"{mean(A['treat_hr5']):.4f}",
                    f"{mean(A['treat_ndcg5']):.4f}", f"{mean(A['treat_hr10']):.4f}",
                    f"{mean(A['treat_ndcg10']):.4f}", len(A["base_hr10"])])
        w.writerow([])
        w.writerow(["配对检验 / 校验统计", "值"])
        for k, v in summary_extra.items():
            w.writerow([k, v])
        for k, v in stats_sum.items():
            w.writerow([f"校验_{k}", v])
    print(f"[保存] {args.out}")

    with open(args.detail, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    print(f"[保存] {args.detail}（每人一行，可复核、可画图）")


if __name__ == "__main__":
    raise SystemExit(main())
