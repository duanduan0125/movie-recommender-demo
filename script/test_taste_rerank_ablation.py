"""taste_rerank_ablation.py 的单元测试（**不连库**）。

为什么要测：这份脚本是"默认值要不要改"的**证据来源** —— 证据本身算错，
结论就会跟着错。所以把它里面所有**会算数的纯函数**钉住：
归一化（min-max 的口径）、MMR 贪心（顺序与惩罚）、类型多样性、符号检验。

运行：python -m pytest script/test_taste_rerank_ablation.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taste_rerank_ablation import (  # noqa: E402
    ild,
    mmr_order,
    norm_scores,
    overlap,
    sign_test_p,
    taste_ratio,
)


def test_norm_scores_is_min_max_not_divide_by_max():
    """★ 归一化必须是 min-max。

    这条守的是一个**实测踩过的坑**：rec_result.score 是 NeuMF 的 logit
    （演示账号榜首 3.2356），池里出现负值时"除以最大值"会算出负的模型分，
    与 0~1 的口味分相加等于把口味权重凭空放大甚至反向排序。
    """
    pool = [(1, 3.0), (2, 1.0), (3, -1.0)]
    norm = norm_scores(pool)
    assert norm[1] == pytest.approx(1.0)
    assert norm[2] == pytest.approx(0.5)
    assert norm[3] == pytest.approx(0.0)


def test_norm_scores_with_all_equal_scores():
    """全部分数相同时 span 退化，不该出现除零或 NaN。"""
    norm = norm_scores([(1, 2.0), (2, 2.0)])
    assert all(v == 0.0 for v in norm.values())


def test_overlap_is_jaccard():
    genres = {1: {12, 7}, 2: {12}, 3: {5}}
    assert overlap(1, [2], genres) == pytest.approx(0.5)   # {12,7} ∩ {12} → 1/2
    assert overlap(1, [3], genres) == pytest.approx(0.0)   # 完全不相交
    assert overlap(1, [], genres) == pytest.approx(0.0)    # 还没有已选项
    assert overlap(4, [1], genres) == pytest.approx(0.0)   # 缺类型信息时不惩罚


def test_ild_counts_disjoint_pairs():
    genres = {1: {12}, 2: {12}, 3: {7}}
    # 两两组合：{1,2} 相交、{1,3} 不相交、{2,3} 不相交 → 2/3
    assert ild([1, 2, 3], genres) == pytest.approx(2 / 3)
    assert ild([1, 2], genres) == pytest.approx(0.0)


def test_taste_ratio_is_normalized_by_total_weight():
    genres = {1: {12}, 2: {7}}
    assert taste_ratio(1, {12: 3, 7: 1}, genres, 4) == pytest.approx(0.75)
    assert taste_ratio(2, {12: 3, 7: 1}, genres, 4) == pytest.approx(0.25)
    assert taste_ratio(1, {}, genres, 0) == pytest.approx(0.0)   # 没有口味信号


def test_mmr_pushes_down_a_repeat():
    """λ>0 时，与已选项完全重合的候选要让位给类型不同的那一个。"""
    pool = [(101, 0.90), (102, 0.50), (103, 0.10)]
    norm = norm_scores(pool)
    taste = {12: 1}
    genres = {900: {12}, 101: {12}, 102: {12}, 103: {7}}

    # λ=0：纯相关性 —— 102 命中口味，排在 103 前面
    assert mmr_order(pool, norm, taste, genres, 1, 0.25, 0.0) == [101, 102, 103]
    # λ=1：102 与已选的 101 是**同一类型**（Jaccard=1），扣分后让位给类型不同的 103。
    # （这里刻意取一个明显大于线上默认 0.30 的 λ：本用例只证明**惩罚真的参与了排序**，
    #   线上那组参数的具体名次由 EXP-04 的扫描负责。）
    assert mmr_order(pool, norm, taste, genres, 1, 0.25, 1.0) == [101, 103, 102]


def test_mmr_is_deterministic_on_ties():
    """平价时按召回名次打破 —— 顺序不确定的话，首页每次请求都会微抖。"""
    pool = [(1, 1.0), (2, 1.0), (3, 1.0)]
    norm = norm_scores(pool)
    for _ in range(3):
        assert mmr_order(pool, norm, {}, {}, 0, 0.25, 0.3) == [1, 2, 3]


def test_sign_test_p_matches_known_values():
    """配对符号检验（双侧精确二项）与手算一致：8 升 0 降 → 2×(1/2^8)。"""
    assert sign_test_p(8, 0) == pytest.approx(2 / 256)
    assert sign_test_p(0, 0) == pytest.approx(1.0)
    # 方向越一致 p 越小
    assert sign_test_p(50, 5) < sign_test_p(30, 25)
