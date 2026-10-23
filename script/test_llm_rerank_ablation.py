#!/usr/bin/env python
"""
llm_rerank_ablation.py 的单元测试（不联网、不调用大模型）

重点覆盖三处最容易出错的地方：
  1. 输出解析与白名单校验 —— 模型会返回候选外 id、重复项、代码块包裹的 JSON，
     校验不严就会把不存在的影片推给用户（幻觉进推荐结果）；
  2. 补齐逻辑 —— 有效结果不足时用召回原序补齐，但不能打乱 LLM 已给出的相对顺序；
  3. 配对检验 —— 用固定的小样本验证 McNemar 的 b/c 计数方向没搞反。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_rerank_ablation import (  # noqa: E402
    DEFAULT_MODEL,
    PRICES_USD_PER_M,
    LlmClient,
    append_cache,
    build_prompt,
    build_request_body,
    cache_key,
    estimate_cost_usd,
    extract_json_object,
    format_candidate,
    load_cache,
    mcnemar_p,
    pad_ranking,
    parse_ranking,
    pool_size_of,
    positive_rank,
    ranking_metrics,
    resolve_api_key,
    summary_metadata,
    wilcoxon_p,
)

DATA_DIR = "./ml-1m"


# --------------------------------------------------------------------------- #
# prompt 构造
# --------------------------------------------------------------------------- #

def test_build_prompt_contains_history_candidates_and_rules():
    history = ["Shawshank Redemption (1994)　5分", "Godfather, The (1972)　4分"]
    cands = [(318, "Shawshank Redemption (1994)", "Crime/Drama"),
             (858, "Godfather, The (1972)", "Action/Crime")]
    p = build_prompt(history, cands, top_n=10)

    for h in history:
        assert h in p, "历史影片必须出现在 prompt 里"
    assert "318 | Shawshank Redemption (1994) | Crime/Drama" in p
    assert "858 | Godfather, The (1972) | Action/Crime" in p
    assert "恰好 10 项" in p
    assert "禁止编造" in p, "必须显式禁止模型使用候选之外的 id"
    assert "ranking" in p


def test_build_prompt_uses_original_movie_id_not_internal_index():
    """prompt 里必须是 MovieLens 原始 id：内部索引一旦拼错就是静默串号"""
    p = build_prompt(["A (1990)　5分"], [(3952, "X (2000)", "Drama")], 5)
    assert "- 3952 | X (2000) | Drama" in p


def test_format_candidate_handles_missing_meta():
    assert format_candidate(1, {"title": "Toy Story", "year": 1995,
                                "genres": ["Animation"]}) == (
        1, "Toy Story (1995)", "Animation")
    mid, title, genres = format_candidate(999, {})
    assert mid == 999 and title == "movie_999" and genres == "未知"
    # 年份缺失时不要拼出 "Title (None)"
    _, title2, _ = format_candidate(2, {"title": "No Year", "year": None,
                                        "genres": []})
    assert title2 == "No Year"


# --------------------------------------------------------------------------- #
# JSON 抽取：模型输出的三种常见形态
# --------------------------------------------------------------------------- #

def test_extract_json_clean():
    assert extract_json_object('{"ranking": []}') == {"ranking": []}


def test_extract_json_from_code_fence():
    text = '```json\n{"ranking": [{"movie_id": 1}]}\n```'
    assert extract_json_object(text) == {"ranking": [{"movie_id": 1}]}


def test_extract_json_with_small_talk():
    text = '好的，以下是结果：\n{"ranking": [{"movie_id": 7}]}\n希望有帮助！'
    assert extract_json_object(text) == {"ranking": [{"movie_id": 7}]}


def test_extract_json_rejects_garbage():
    assert extract_json_object("抱歉，我无法完成这个请求。") is None
    assert extract_json_object("") is None
    assert extract_json_object("") is None
    assert extract_json_object('{"ranking": [1, 2') is None


# --------------------------------------------------------------------------- #
# 白名单校验：防幻觉的核心
# --------------------------------------------------------------------------- #

def test_parse_ranking_accepts_valid_and_preserves_order():
    text = json.dumps({"ranking": [{"movie_id": 30, "reason": "x"},
                                   {"movie_id": 10, "reason": "y"}]})
    ids, stats = parse_ranking(text, whitelist={10, 20, 30}, need=10)
    assert ids == [30, 10], "顺序必须保持模型给出的降序"
    assert stats["非白名单"] == 0
    assert stats["返回条数"] == 2


def test_parse_ranking_drops_hallucinated_ids():
    """候选之外的 id 必须被剔除 —— 否则会推荐不存在的影片"""
    text = json.dumps({"ranking": [{"movie_id": 30}, {"movie_id": 999},
                                   {"movie_id": 12345}]})
    ids, stats = parse_ranking(text, whitelist={10, 20, 30}, need=10)
    assert ids == [30]
    assert stats["非白名单"] == 2


def test_parse_ranking_dedupes():
    text = json.dumps({"ranking": [{"movie_id": 10}, {"movie_id": 10},
                                   {"movie_id": 20}]})
    ids, stats = parse_ranking(text, whitelist={10, 20}, need=10)
    assert ids == [10, 20]
    assert stats["重复"] == 1


def test_parse_ranking_tolerates_string_and_float_ids():
    text = json.dumps({"ranking": [{"movie_id": "10"}, {"movie_id": 20.0}]})
    ids, _ = parse_ranking(text, whitelist={10, 20}, need=10)
    assert ids == [10, 20]


def test_parse_ranking_rejects_malformed_items():
    text = json.dumps({"ranking": [{"movie_id": 10}, {"id": 20}, "30",
                                   {"movie_id": None}, {"movie_id": True}]})
    ids, stats = parse_ranking(text, whitelist={10, 20, 30}, need=10)
    assert ids == [10], "只有格式正确的条目可用"
    assert stats["非法条目"] == 4


def test_parse_ranking_truncates_to_need():
    text = json.dumps({"ranking": [{"movie_id": i} for i in range(10, 40)]})
    ids, _ = parse_ranking(text, whitelist=set(range(10, 40)), need=10)
    assert len(ids) == 10 and ids == list(range(10, 20))


def test_parse_ranking_flags_invalid_json():
    ids, stats = parse_ranking("模型返回了一段散文", whitelist={1}, need=10)
    assert ids == [] and stats["无效JSON"] == 1


def test_parse_ranking_flags_missing_ranking_field():
    ids, stats = parse_ranking('{"result": [1, 2]}', whitelist={1}, need=10)
    assert ids == [] and stats["无效JSON"] == 1


# --------------------------------------------------------------------------- #
# 补齐降级
# --------------------------------------------------------------------------- #

def test_pad_ranking_keeps_llm_order_then_appends_pool_order():
    ids = pad_ranking([5, 6], [1, 2, 3, 4, 5, 6], need=5)
    assert ids == [5, 6, 1, 2, 3], "先保留模型给出的顺序，再按召回原序补缺"


def test_pad_ranking_no_duplicates_and_exact_length():
    ids = pad_ranking([2, 4], [1, 2, 3, 4, 5], need=4)
    assert ids == [2, 4, 1, 3]
    assert len(ids) == len(set(ids)) == 4


def test_pad_ranking_pool_smaller_than_need():
    """池子比 need 还小：能补多少补多少，不能凭空造"""
    assert pad_ranking([1], [1, 2], need=10) == [1, 2]


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def test_ranking_metrics_hit_and_position():
    hr, ndcg = ranking_metrics([1, 2, 3], positive=1, k=10)
    assert (hr, ndcg) == (1.0, 1.0)
    hr, ndcg = ranking_metrics([1, 2, 3], positive=2, k=10)
    assert hr == 1.0
    assert ndcg == pytest.approx(1 / np.log2(3), abs=1e-9)


def test_ranking_metrics_miss():
    assert ranking_metrics([1, 2, 3], positive=99, k=10) == (0.0, 0.0)


def test_ranking_metrics_respects_k_boundary():
    """正样本排第 3 位：k=3 命中，k=2 不命中 —— 边界不能差一位"""
    assert ranking_metrics([1, 2, 3], positive=3, k=3)[0] == 1.0
    assert ranking_metrics([1, 2, 3], positive=3, k=2)[0] == 0.0


def test_ranking_metrics_reordering_improves_ndcg():
    """重排的价值：正样本从第 8 位提到第 1 位，HR 不变但 NDCG 大幅提升"""
    before = ranking_metrics(list(range(1, 11)), positive=8, k=10)
    after = ranking_metrics([8] + list(range(1, 8)) + [9, 10], positive=8, k=10)
    assert before[0] == after[0] == 1.0
    assert after[1] > before[1]


# --------------------------------------------------------------------------- #
# 正样本名次：这里出过一个"不报错但数字全错"的 bug
# --------------------------------------------------------------------------- #

def test_positive_rank_is_zero_based_position():
    assert positive_rank(np.array([0, 3, 1, 2])) == 0
    assert positive_rank(np.array([3, 0, 1, 2])) == 1
    assert positive_rank(np.array([3, 1, 2, 0])) == 3


def test_positive_rank_is_not_only_about_being_first():
    """
    回归测试：早期把「正样本是否排第一」错当成 HR@10 的判定，
    基线 HR@10 从 0.6680 变成 0.1669（实际算的是 HR@1），且程序不报错。
    """
    order = np.array([5, 7, 0, 9])          # 正样本（第 0 列）排第 3 位
    assert order[0] != 0, "确实不是排第一"
    assert positive_rank(order) == 2, "名次应是 2（0 起），不是 -1 也不是 999"
    assert positive_rank(order) < 10, "该用户应计入 HR@10"
    assert not (order[0] == 0), "老写法会把它判成未命中，两者必须区分开"


def test_positive_rank_missing_returns_minus_one():
    assert positive_rank(np.array([1, 2, 3])) == -1


# --------------------------------------------------------------------------- #
# 配对检验
# --------------------------------------------------------------------------- #

def test_mcnemar_counts_direction():
    base = np.array([1, 1, 0, 0])
    treat = np.array([1, 0, 0, 1])
    b, c, p = mcnemar_p(base, treat)
    assert (b, c) == (1, 1), "b=基线命中策略未命中，c=反向，方向不能反"
    assert p == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs():
    base = np.array([1, 0, 1])
    b, c, p = mcnemar_p(base, base.copy())
    assert (b, c, p) == (0, 0, 1.0)


def test_mcnemar_detects_consistent_improvement():
    """重排稳定救回 10 个用户、只坏 1 个 —— 应判为显著"""
    base = np.array([0] * 10 + [1])
    treat = np.array([1] * 10 + [0])
    b, c, p = mcnemar_p(base, treat)
    assert (b, c) == (1, 10)
    assert p < 0.05


def test_wilcoxon_identical_is_not_significant():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert wilcoxon_p(x, x.copy())[1] == 1.0


def test_wilcoxon_detects_shift():
    base = np.zeros(20)
    treat = np.full(20, 0.5)
    assert wilcoxon_p(base, treat)[1] < 0.05


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #

def test_cache_key_depends_on_pool_content():
    a = cache_key("neumf", 1, [1, 2, 3])
    assert a == cache_key("neumf", 1, [1, 2, 3])
    assert a != cache_key("neumf", 1, [3, 2, 1]), "候选池内容不同必须视为不同请求"
    assert a != cache_key("itemknn", 1, [1, 2, 3]), "召回模型不同也必须区分"


def test_cache_roundtrip(tmp_path):
    path = str(tmp_path / "c.jsonl")
    assert load_cache(path) == {}
    append_cache(path, "k1", '{"ranking": []}')
    append_cache(path, "k2", 'raw text')
    got = load_cache(path)
    assert got == {"k1": '{"ranking": []}', "k2": "raw text"}


def test_cache_ignores_broken_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"key":"k1","raw":"a"}\n这不是JSON\n\n', encoding="utf-8")
    assert load_cache(str(path)) == {"k1": "a"}


# --------------------------------------------------------------------------- #
# 成本计算：写论文时会被问到"这个实验花了多少钱"，必须算得清
# --------------------------------------------------------------------------- #

def test_estimate_cost_usd_matches_official_rates():
    """
    deepseek-flash 官方价（2026-09 核对）：输入未命中 $0.30/M、缓存命中 $0.006/M、
    输出 $1.20/M。这里各取 1M tokens 验证加总。
    """
    assert estimate_cost_usd("deepseek-flash", 1_000_000, 0, 0) == pytest.approx(0.30)
    assert estimate_cost_usd("deepseek-flash", 0, 1_000_000, 0) == pytest.approx(0.006)
    assert estimate_cost_usd("deepseek-flash", 0, 0, 1_000_000) == pytest.approx(1.20)
    assert estimate_cost_usd("deepseek-flash", 1_000_000, 0, 1_000_000) == \
        pytest.approx(1.50)


def test_estimate_cost_off_peak_is_half():
    """低谷时段费率为高峰的 5 折 —— 别把这个折扣记反或漏掉"""
    peak = estimate_cost_usd("deepseek-flash", 500_000, 0, 500_000, peak=True)
    off = estimate_cost_usd("deepseek-flash", 500_000, 0, 500_000, peak=False)
    assert off == pytest.approx(peak * 0.5)


def test_estimate_cost_cache_hit_is_much_cheaper():
    hit = estimate_cost_usd("deepseek-flash", 0, 1_000_000, 0)
    miss = estimate_cost_usd("deepseek-flash", 1_000_000, 0, 0)
    assert hit < miss / 10


def test_estimate_cost_unknown_model_returns_zero():
    """未知模型宁可返回 0 并提示，也不要编一个数字出来"""
    assert estimate_cost_usd("gpt-9-ultra", 1_000_000, 0, 1_000_000) == 0.0


def test_estimate_cost_scales_linearly():
    a = estimate_cost_usd("deepseek-flash", 100_000, 0, 50_000)
    b = estimate_cost_usd("deepseek-flash", 200_000, 0, 100_000)
    assert b == pytest.approx(a * 2)


def test_price_table_covers_default_model():
    """默认模型必须在价格表里，否则成本会静默算成 0"""
    assert DEFAULT_MODEL in PRICES_USD_PER_M


# --------------------------------------------------------------------------- #
# 请求体构造：thinking 默认关闭是本脚本最影响成本的一行配置
# --------------------------------------------------------------------------- #

def test_request_body_disables_thinking_by_default():
    """
    回归测试：官方默认 reasoning_effort=high，实测每次多出约 5,700 个推理 tokens，
    输出成本约 12 倍、单次耗时 36 秒。默认必须显式关闭。
    """
    body = build_request_body("deepseek-flash", "给我重排")
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in json.dumps(body)
    assert body["temperature"] == 0.2, "关闭思考后 temperature 才生效，用于降随机性"


def test_request_body_enables_thinking_when_asked():
    body = build_request_body("deepseek-flash", "x", thinking="high")
    assert body["thinking"] == {"type": "enabled", "reasoning_effort": "high"}
    # 思考模式下 temperature 不生效，不应传（避免造成"设了但没用"的误解）
    assert "temperature" not in body


def test_request_body_sets_json_mode_and_max_tokens():
    body = build_request_body("deepseek-flash", "x", max_tokens=1234, json_mode=True)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 1234
    body2 = build_request_body("deepseek-flash", "x", json_mode=False)
    assert "response_format" not in body2


def test_request_body_carries_model_and_prompt():
    body = build_request_body("deepseek-flash", "候选列表如下")
    assert body["model"] == "deepseek-flash"
    assert body["messages"][0]["role"] == "system"
    assert "JSON" in body["messages"][0]["content"], "系统提示必须要求输出 JSON"
    assert body["messages"][1]["content"] == "候选列表如下"
    assert body["stream"] is False


# --------------------------------------------------------------------------- #
# 预算硬闸：余额很少时必须在跑到一半时自动刹车
#
# 注意：这些用例都不会真的发出网络请求 —— 预算超限的分支在发起请求之前就返回了。
# --------------------------------------------------------------------------- #

def _client(**kw) -> LlmClient:
    return LlmClient("https://example.invalid/v1", "sk-test", "deepseek-flash", **kw)


def test_budget_stops_before_any_further_call():
    c = _client(max_cost_usd=0.001, billing_peak=True)
    c.cache_miss_tokens = 10_000            # 10k × $0.30/M = $0.003 > $0.001
    text, err = c.chat("随便")
    assert text is None and "预算" in err
    assert c.budget_stopped is True
    assert c.circuit_open is True
    assert c.n_calls == 0, "拦在请求之前，所以不应产生调用计数（也就没有费用）"


def test_budget_uses_billing_flag():
    """低谷 5 折：同一批 tokens 折算出的金额应为高峰的一半"""
    peak, off = _client(billing_peak=True), _client(billing_peak=False)
    for c in (peak, off):
        c.cache_miss_tokens = 1_000_000
        c.completion_tokens = 1_000_000
    assert peak.billed_cost_usd() == pytest.approx(1.50)
    assert off.billed_cost_usd() == pytest.approx(0.75)


def test_budget_disabled_by_zero():
    c = _client(max_cost_usd=0.0)
    c.cache_miss_tokens = 10_000_000
    assert c.budget_stopped is False
    assert c.max_cost_usd == 0.0, "0 表示不限，不应被当成“上限为 0 立即停止”"


def test_circuit_open_short_circuits_later_calls():
    c = _client()
    c.circuit_open = True
    text, err = c.chat("随便")
    assert text is None
    assert c.n_calls == 0


# --------------------------------------------------------------------------- #
# API Key 读取：定时任务拿不到后面才设的环境变量，必须有文件兜底
# --------------------------------------------------------------------------- #

def test_resolve_api_key_prefers_explicit_value(tmp_path):
    f = tmp_path / ".llm_api_key"
    f.write_text("from-file", encoding="utf-8")
    assert resolve_api_key("from-arg", str(f)) == "from-arg"


def test_resolve_api_key_falls_back_to_file(tmp_path):
    f = tmp_path / ".llm_api_key"
    f.write_text("  sk-from-file\n", encoding="utf-8")
    assert resolve_api_key("", str(f)) == "sk-from-file", "要去掉首尾空白"


def test_resolve_api_key_returns_empty_when_nothing_available(tmp_path):
    assert resolve_api_key("", str(tmp_path / "不存在")) == ""


# --------------------------------------------------------------------------- #
# 结果表口径：候选池大小必须是真实值，不能照抄 --pool
# --------------------------------------------------------------------------- #

def test_pool_size_of_sampled_pool_is_one_plus_n_neg():
    """sampled 模式的池子 = 1 正 + n_neg 负（默认 100），与 --pool 无关"""
    pools = {1: list(range(100)), 2: list(range(100))}
    assert pool_size_of(pools) == 100


def test_pool_size_of_full_pool():
    pools = {7: list(range(50))}
    assert pool_size_of(pools) == 50


def _fake_args(**kw):
    import argparse
    base = dict(recall="neumf", pool=50, pool_mode="sampled", top_n=10,
                model="deepseek-flash", seed=42)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_client():
    return LlmClient(model="deepseek-flash", api_key="x", base_url="http://x")


def test_summary_metadata_reports_real_pool_size_not_pool_flag():
    """
    回归测试：sampled 组真实池是 100，但命令行 --pool 是 50。
    结果表曾照抄 --pool 写成 50，使 A 组（采样）与 B 组（全量前 50）的产物无法区分。
    """
    md = summary_metadata(_fake_args(pool_mode="sampled", pool=50), _fake_client(),
                          pool_size=100, n_users=500, in_pool=500, n_fail=0,
                          n_pad_total=1, cost_peak=0.245745, b10=156, c10=40,
                          p_hr10=0.0, p_ndcg10=0.0, p_hr5=0.0, p_ndcg5=1e-5)
    assert md["候选池"] == 100, "sampled 模式必须写真实池大小（1+n_neg）"
    assert md["重排保留"] == 10 and md["抽样用户"] == 500


def test_summary_metadata_full_mode_pool_equals_pool_flag():
    md = summary_metadata(_fake_args(pool_mode="full", pool=50), _fake_client(),
                          pool_size=50, n_users=500, in_pool=500, n_fail=0,
                          n_pad_total=0, cost_peak=0.180573, b10=90, c10=87,
                          p_hr10=0.88056, p_ndcg10=0.813156, p_hr5=0.459691,
                          p_ndcg5=0.253187)
    assert md["候选池"] == 50


# --------------------------------------------------------------------------- #
# 真实数据（缺数据集时自动跳过）
# --------------------------------------------------------------------------- #

skip_no_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "movies.dat")),
    reason="未找到 ml-1m 数据集")


@skip_no_data
def test_load_movie_meta_on_real_data():
    from common import load_movie_meta
    meta = load_movie_meta(DATA_DIR)
    assert len(meta) == 3883
    toy = meta[1]
    assert toy["title"] == "Toy Story" and toy["year"] == 1995
    assert toy["genres"] == ["Animation", "Children's", "Comedy"]
    # 带逗号/重音/括号的片名不能被年份解析带偏
    assert meta[2]["title"] == "Jumanji" and meta[2]["year"] == 1995
    # 实测：ml-1m 里没有缺年份的影片
    assert all(m["year"] is not None for m in meta.values())
    assert all(m["title"] for m in meta.values())


@skip_no_data
def test_load_movie_meta_keeps_parentheses_inside_titles():
    """
    回归测试：ml-1m 里有 269 部影片的片名本身带括号（括号里是别名），
    原始行形如 "30::Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)::Drama"。
    必须取**最后一个** (YYYY) 当年份，否则这些片名会被截断
    （常见错法是 title.split("(")[0]，会把片名切成 "Shanghai Triad "）。
    """
    from common import load_movie_meta
    meta = load_movie_meta(DATA_DIR)

    assert meta[30]["title"] == "Shanghai Triad (Yao a yao yao dao waipo qiao)"
    assert meta[30]["year"] == 1995
    assert meta[47]["title"] == "Seven (Se7en)"
    assert meta[47]["year"] == 1995
    n_with_parens = sum(1 for m in meta.values() if m["title"].endswith(")"))
    assert n_with_parens == 269, f"带括号片名数变了：{n_with_parens} != 269"
