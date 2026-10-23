"""
eval_topn.py 的单元测试

运行：
    python -m pytest script/test_eval_topn.py -v

只测不依赖数据集的部分：结果表（topn_result.csv）的写入逻辑。
这段逻辑看着简单，但出错会直接污染实验记录 ——
比如同一算法重复跑出两行、或把别的算法的行覆盖掉，
而实验表是论文里要直接引用的东西。
打分与指标口径的测试在 test_common.py 里。
"""

from __future__ import annotations

import csv
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from eval_topn import (  # noqa: E402
    build_binary_matrix,
    build_parser,
    item_item_cosine,
    itemknn_scores,
    make_matrix_scorer,
    mostpop_scores,
    truncate_topk_similarity,
    write_row,
)


# --------------------------------------------------------------------------- #
# 命令行：设计文档 §3.2 的用法必须照着敲就能跑
# --------------------------------------------------------------------------- #

def test_parser_accepts_documented_underscore_data_dir():
    """文档 §3.2 的用法写的是 --data_dir；照文档抄的命令不能报错"""
    args = build_parser().parse_args(["--data_dir", "./ml-1m", "--algo", "itemcf"])
    assert args.data_dir == "./ml-1m"


def test_parser_accepts_hyphen_data_dir():
    """两种写法都收，且落到同一个 dest，不会出现"传了但没生效\""""
    args = build_parser().parse_args(["--data-dir", "./ml-1m", "--algo", "itemcf"])
    assert args.data_dir == "./ml-1m"


def test_parser_defaults_match_documented_candidate_set():
    """§3.2：候选集 1 正 + 99 负、seed=42、指标 HR@10 / NDCG@10、输出 topn_result.csv"""
    args = build_parser().parse_args(["--algo", "neumf"])
    assert args.n_neg == 99
    assert args.seed == 42
    assert args.eval_k == 10
    assert args.out == "topn_result.csv"


def test_parser_default_split_is_two_way():
    """
    默认仍是二路口径（与既有命令行为一致），但实验二主表必须显式加 --three-way；
    不加时脚本会打印警示（见 main 里的 [划分] 提示），不会静默给出不对称的结果。
    """
    args = build_parser().parse_args(["--algo", "itemcf"])
    assert args.three_way is False


def args_stub(k=10, n_neg=99):
    return SimpleNamespace(eval_k=k, n_neg=n_neg)


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def test_creates_file_with_header(tmp_path):
    path = str(tmp_path / "topn_result.csv")
    write_row(path, "svd", 0.1234, 0.0567, 6040, 42.0, args_stub())

    rows = read_rows(path)
    assert rows[0] == ["algo", "HR@k", "NDCG@k", "users", "k", "n_neg", "耗时(s)"]
    assert rows[1][0] == "svd"
    assert rows[1][1] == "0.1234"


def test_appends_different_algos(tmp_path):
    path = str(tmp_path / "topn_result.csv")
    write_row(path, "itemcf", 0.60, 0.34, 6040, 100.0, args_stub())
    write_row(path, "svd", 0.62, 0.35, 6040, 50.0, args_stub())
    write_row(path, "neumf", 0.70, 0.42, 6040, 30.0, args_stub())

    rows = read_rows(path)
    assert len(rows) == 4, "表头 + 3 行"
    assert [r[0] for r in rows[1:]] == ["itemcf", "svd", "neumf"]


def test_rerun_same_algo_overwrites_instead_of_duplicating(tmp_path):
    """同一算法重跑（比如调参后）应当覆盖旧行，否则表里会出现两行同算法"""
    path = str(tmp_path / "topn_result.csv")
    write_row(path, "itemcf", 0.60, 0.34, 6040, 100.0, args_stub())
    write_row(path, "itemcf", 0.63, 0.36, 6040, 90.0, args_stub())

    rows = read_rows(path)
    assert len(rows) == 2, "只能有一行 itemcf"
    assert rows[1][1] == "0.6300", "应当是后一次的结果"
    assert rows[1][6] == "90"


def test_overwrite_keeps_other_rows(tmp_path):
    path = str(tmp_path / "topn_result.csv")
    write_row(path, "itemcf", 0.60, 0.34, 6040, 100.0, args_stub())
    write_row(path, "svd", 0.62, 0.35, 6040, 50.0, args_stub())
    write_row(path, "itemcf", 0.61, 0.35, 6040, 95.0, args_stub())

    rows = read_rows(path)
    assert [r[0] for r in rows[1:]] == ["svd", "itemcf"], "svd 不能被丢掉"
    assert rows[1][1] == "0.6200"


def test_records_run_config(tmp_path):
    """k / n_neg / 用户数要落表 —— 论文里必须能说清指标是在什么口径下算的"""
    path = str(tmp_path / "topn_result.csv")
    write_row(path, "neumf", 0.70, 0.42, 6040, 30.0, args_stub(k=10, n_neg=99))

    row = read_rows(path)[1]
    assert row[3] == "6040"      # users
    assert row[4] == "10"        # k
    assert row[5] == "99"        # n_neg


# --------------------------------------------------------------------------- #
# 排序式基线：ItemKNN / MostPop
#
#   手工验算用的小例子（3 用户 × 3 物品）：
#       u0 看过 i0, i1      u1 看过 i0      u2 看过 i1, i2
#   物品的用户集合：U(i0)={u0,u1}  U(i1)={u0,u2}  U(i2)={u2}
#   余弦相似度 sim(i,j)=|U_i∩U_j| / sqrt(|U_i|·|U_j|)：
#       sim(i0,i1)=|{u0}|/sqrt(2·2)=0.5
#       sim(i0,i2)=|∅|/sqrt(2·1)=0
#       sim(i1,i2)=|{u2}|/sqrt(2·1)=1/√2≈0.7071   ← u2 同时看过 i1 和 i2
# --------------------------------------------------------------------------- #

TINY_U = np.array([0, 0, 1, 2, 2])
TINY_I = np.array([0, 1, 0, 1, 2])
N_USERS, N_ITEMS = 3, 3


def tiny_binary():
    return build_binary_matrix(TINY_U, TINY_I, N_USERS, N_ITEMS)


def test_build_binary_matrix_shape_and_values():
    X = tiny_binary()
    assert X.shape == (N_USERS, N_ITEMS)
    assert X.nnz == 5, "5 条交互，二值化后不重复计数"
    # 二值：忽略评分值，出现即 1
    assert X.toarray().tolist() == [[1.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0]]


def test_item_item_cosine_is_symmetric_with_unit_diagonal():
    S = item_item_cosine(tiny_binary())
    assert S.shape == (N_ITEMS, N_ITEMS)
    assert np.allclose(np.diag(S), 1.0), "物品与自身相似度为 1"
    assert np.allclose(S, S.T), "余弦相似度必须对称"


def test_item_item_cosine_known_values():
    S = item_item_cosine(tiny_binary())
    assert S[0, 1] == pytest.approx(0.5)
    assert S[0, 2] == pytest.approx(0.0)
    # u2 同时看过 i1 与 i2，交集非空
    assert S[1, 2] == pytest.approx(1 / np.sqrt(2))


def test_item_item_cosine_survives_item_without_interactions():
    """存在从未被交互的物品时不能除零（冷门物品在真实数据里很常见）"""
    X = build_binary_matrix(np.array([0]), np.array([0]), n_users=2, n_items=3)
    S = item_item_cosine(X)
    assert np.isfinite(S).all()


def test_itemknn_scores_sum_similarity_over_history():
    """
    u0 历史 {i0,i1}：
        score(i0) = sim(i0,i0) + sim(i0,i1) = 1 + 0.5      = 1.5
        score(i1) = sim(i1,i0) + sim(i1,i1) = 0.5 + 1      = 1.5
        score(i2) = sim(i2,i0) + sim(i2,i1) = 0 + 1/√2     ≈ 0.7071
    u2 历史 {i1,i2}：
        score(i2) = sim(i2,i1) + sim(i2,i2) = 1/√2 + 1     ≈ 1.7071
    """
    X = tiny_binary()
    full = itemknn_scores(item_item_cosine(X), X)

    assert full.shape == (N_ITEMS, N_USERS)
    assert full[0, 0] == pytest.approx(1.5)
    assert full[1, 0] == pytest.approx(1.5)
    assert full[2, 0] == pytest.approx(1 / np.sqrt(2))
    assert full[2, 2] == pytest.approx(1 + 1 / np.sqrt(2))


def test_itemknn_ranks_unseen_items_above_seen_ones_for_same_item():
    """同一物品对"有相关历史"的用户分数更高 —— 这是 ItemKNN 能工作的前提"""
    X = tiny_binary()
    full = itemknn_scores(item_item_cosine(X), X)
    # i0 与 u0 的历史(i0,i1)相关，与 u2 的历史(i1,i2)几乎无关
    assert full[0, 0] > full[0, 2]


def test_mostpop_scores_are_interaction_counts():
    assert mostpop_scores(tiny_binary()).tolist() == [2.0, 2.0, 1.0]


def test_matrix_scorer_indexes_item_then_user():
    """打分矩阵是 (n_items, n_users)，取值时不能把两个维度写反"""
    full = np.array([[10.0, 11.0],
                     [20.0, 21.0]])
    score = make_matrix_scorer(full)
    got = score(uidx=np.array([0, 1]), iidx=np.array([1, 0]))
    assert got.tolist() == [20.0, 11.0], "行是物品、列是用户"


# --------------------------------------------------------------------------- #
# 相似度 top-K 截断
# --------------------------------------------------------------------------- #

S4 = np.array([
    [1.00, 0.90, 0.10, 0.20],
    [0.90, 1.00, 0.30, 0.05],
    [0.10, 0.30, 1.00, 0.80],
    [0.20, 0.05, 0.80, 1.00],
], dtype=np.float32)


def test_topk_keeps_exactly_k_neighbours_per_row():
    out = truncate_topk_similarity(S4, 1)
    assert (out > 0).sum(axis=1).tolist() == [1, 1, 1, 1]
    out2 = truncate_topk_similarity(S4, 2)
    assert (out2 > 0).sum(axis=1).tolist() == [2, 2, 2, 2]


def test_topk_keeps_the_largest_values():
    # 用容差比较：float32 下 0.9 实际存为 0.89999997
    out = truncate_topk_similarity(S4, 1)
    assert np.allclose(out[0], [0.0, 0.9, 0.0, 0.0])     # 0.9 > 0.2 > 0.1
    assert np.allclose(out[1], [0.9, 0.0, 0.0, 0.0])
    assert np.allclose(out[2], [0.0, 0.0, 0.0, 0.8])     # 0.8 > 0.3 > 0.1
    assert np.allclose(out[3], [0.0, 0.0, 0.8, 0.0])


def test_topk_drops_diagonal_so_self_never_takes_a_slot():
    """对角线是自己，不该占用邻居名额 —— 否则 k=1 时每行只剩自己，退化成空"""
    out = truncate_topk_similarity(S4, 1)
    assert np.allclose(np.diag(out), 0.0)
    out3 = truncate_topk_similarity(S4, 3)
    assert np.allclose(np.diag(out3), 0.0), "k=3（全部邻居）时对角线同样是 0"


def test_topk_zero_or_negative_means_no_truncation():
    for k in (0, -1):
        out = truncate_topk_similarity(S4, k)
        assert np.array_equal(out, S4), "k<=0 应当原样返回"


def test_topk_larger_than_size_is_clamped():
    out = truncate_topk_similarity(S4, 999)
    expected = S4.copy()
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(out, expected)


def test_topk_does_not_mutate_input():
    """必须复制后再截断：直接改原矩阵会污染后续使用同一相似度矩阵的代码"""
    before = S4.copy()
    truncate_topk_similarity(S4, 1)
    assert np.array_equal(S4, before)
