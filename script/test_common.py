"""
common.py 的单元测试

运行：
    cd d:/a/gradu-prj/movie-recommend
    python -m pytest script/test_common.py -v

重点覆盖三类容易出错但不会报错的地方：
  1. 留一法的划分是否真的"每人最后一条"，错了指标会整体偏低；
  2. 负采样是否泄漏了测试正样本（原实现就漏了这一条，实测约 2.7% 的用户命中）；
  3. HR/NDCG 的计算口径，正样本排第 1 时 NDCG 必须等于 1。
需要数据集的用例会自动跳过（数据集不在仓库内）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import (  # noqa: E402
    build_eval_candidates,
    build_user_items,
    check_dataset,
    dataset_stats,
    evaluate_ranking,
    leave_one_out,
    leave_one_out_with_val,
    load_ratings,
    rank_of_positive,
    reindex,
    sample_negatives,
)

DATA_DIR = os.environ.get("ML1M_DIR", "./ml-1m")
_data_ready = os.path.exists(os.path.join(DATA_DIR, "ratings.dat"))
skip_no_data = pytest.mark.skipif(not _data_ready, reason=f"数据集不存在：{DATA_DIR}")


# --------------------------------------------------------------------------- #
# 夹具：一个能手工验算的小数据集
#
#   原始影片 id {10,20,30,50} -> iidx {0,1,2,3}
#   原始用户 id {1,2,3}       -> uidx {0,1,2}
#
#   u0: item10(ts100) -> item30(ts200) -> item20(ts300)   测试 = iidx 1
#   u1: item10(ts110) -> item50(ts220)                    测试 = iidx 3
#   u2: item10(ts120) -> item30(ts210) -> item50(ts330)   测试 = iidx 3
#   训练集已交互集合：u0={0,2}  u1={0}  u2={0,2}
# --------------------------------------------------------------------------- #

TINY_ITEMS = 4


def tiny_df() -> pd.DataFrame:
    rows = [
        (1, 10, 5, 100), (1, 30, 4, 200), (1, 20, 3, 300),
        (2, 10, 4, 110), (2, 50, 5, 220),
        (3, 10, 3, 120), (3, 30, 5, 210), (3, 50, 4, 330),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    out, _, _ = reindex(df)
    return out


# --------------------------------------------------------------------------- #
# reindex
# --------------------------------------------------------------------------- #

def test_reindex_is_continuous_and_preserves_originals():
    out = tiny_df()
    assert sorted(out["uidx"].unique()) == [0, 1, 2]
    assert sorted(out["iidx"].unique()) == [0, 1, 2, 3]      # 连续无空洞
    # uidx/iidx 只是模型要的连续索引，原始 id 必须保留（导出时要还原，见设计文档 R1）
    assert out["user_id"].min() == 1
    assert out["item_id"].max() == 50


# --------------------------------------------------------------------------- #
# 留一法
# --------------------------------------------------------------------------- #

def test_leave_one_out_takes_last_by_timestamp():
    train, test = leave_one_out(tiny_df())

    assert len(test) == 3, "每位用户恰好 1 条测试"
    got = dict(zip(test["uidx"].tolist(), test["iidx"].tolist()))
    assert got == {0: 1, 1: 3, 2: 3}, "测试集必须是每人时间戳最大的那条"
    assert len(train) == 5


def test_leave_one_out_disjoint_and_complete():
    out = tiny_df()
    train, test = leave_one_out(out)

    train_pairs = set(zip(train["uidx"], train["iidx"]))
    test_pairs = set(zip(test["uidx"], test["iidx"]))
    assert not (train_pairs & test_pairs), "训练集与测试集不能重叠"
    assert len(train) + len(test) == len(out), "不能丢样本"
    # 每人至少留 1 条训练数据，否则该用户无法被建模
    assert train.groupby("uidx").size().min() >= 1


def test_leave_one_out_single_interaction_user():
    """只有 1 条交互的用户：测试集拿到它，训练集为空 —— 不能崩，也不能丢人"""
    df = pd.DataFrame([(1, 10, 5, 100)],
                      columns=["user_id", "item_id", "rating", "timestamp"])
    out, _, _ = reindex(df)
    train, test = leave_one_out(out)
    assert len(test) == 1 and len(train) == 0


# --------------------------------------------------------------------------- #
# 三路划分：test（最后一条）+ val（倒数第二条）+ train（其余）
#
# 针对 tiny_df：
#   u0: iidx 0 -> 2 -> 1   => test=1  val=2  train={0}
#   u1: iidx 0 -> 3        => test=3  val=0  train={} （只有 2 条，无训练数据）
#   u2: iidx 0 -> 2 -> 3   => test=3  val=2  train={0}
# --------------------------------------------------------------------------- #

def test_three_way_splits_last_two_in_order():
    train, val, test = leave_one_out_with_val(tiny_df())

    assert dict(zip(test["uidx"].tolist(), test["iidx"].tolist())) == {0: 1, 1: 3, 2: 3}
    assert dict(zip(val["uidx"].tolist(), val["iidx"].tolist())) == {0: 2, 1: 0, 2: 2}
    assert (len(train), len(val), len(test)) == (2, 3, 3)


def test_three_way_disjoint_and_complete():
    out = tiny_df()
    train, val, test = leave_one_out_with_val(out)

    parts = [set(zip(d["uidx"], d["iidx"])) for d in (train, val, test)]
    assert not (parts[0] & parts[1]), "train 与 val 不能重叠"
    assert not (parts[0] & parts[2]), "train 与 test 不能重叠"
    assert not (parts[1] & parts[2]), "val 与 test 不能重叠 —— 否则等于在测试集上选参"
    assert len(train) + len(val) + len(test) == len(out), "不能丢样本"


def test_three_way_single_interaction_user_has_no_val():
    """1 条交互的用户只进 test：没有倒数第二条，不该凭空造一条验证样本"""
    df = pd.DataFrame([(1, 10, 5, 100)],
                      columns=["user_id", "item_id", "rating", "timestamp"])
    out, _, _ = reindex(df)
    train, val, test = leave_one_out_with_val(out)

    assert len(test) == 1
    assert len(val) == 0
    assert len(train) == 0


def test_three_way_two_interaction_user_has_empty_train():
    df = pd.DataFrame([(1, 10, 5, 100), (1, 20, 4, 200)],
                      columns=["user_id", "item_id", "rating", "timestamp"])
    out, _, _ = reindex(df)
    train, val, test = leave_one_out_with_val(out)

    assert len(train) == 0, "两条交互切完没有剩余，就是没有训练数据"
    assert len(val) == 1 and len(test) == 1
    assert int(test["iidx"].iloc[0]) == 1, "test 是时间戳最大的那条"


# --------------------------------------------------------------------------- #
# build_user_items
# --------------------------------------------------------------------------- #

def test_build_user_items():
    train, test = leave_one_out(tiny_df())
    ui = build_user_items(train, n_users=3)

    assert set(ui.keys()) == {0, 1, 2}
    assert ui[0] == {0, 2}
    assert ui[1] == {0}
    # 测试集那条不能混进"已交互集合"，否则评测时会把答案当已知信息
    assert 1 not in ui[0]


# --------------------------------------------------------------------------- #
# 负采样
# --------------------------------------------------------------------------- #

def test_sample_negatives_excludes_and_dedupes():
    got = sample_negatives(np.random.default_rng(0), exclude={0, 1, 2, 3, 4, 5},
                           n_items=20, n=10)

    assert len(got) == 10
    assert len(set(got)) == 10, "负样本不能重复"
    assert not (set(got) & {0, 1, 2, 3, 4, 5}), "不能采到已交互物品"


def test_sample_negatives_reproducible():
    a = sample_negatives(np.random.default_rng(42), {0}, 50, 10)
    b = sample_negatives(np.random.default_rng(42), {0}, 50, 10)
    assert a == b, "同一种子必须复现，否则不同算法之间无法公平比较"


# --------------------------------------------------------------------------- #
# 候选集构造：最关键的回归 —— 正样本不得泄漏成负样本
# --------------------------------------------------------------------------- #

def test_candidates_put_positive_first():
    out = tiny_df()
    train, test = leave_one_out(out)
    all_items = build_user_items(out, n_users=3)          # 训练集 ∪ 测试集

    users, cand = build_eval_candidates(test, all_items, n_items=TINY_ITEMS,
                                        n_neg=2, seed=1)

    assert users.tolist() == [0, 1, 2]
    assert cand.shape == (3, 3), "1 个正样本 + 2 个负样本"
    assert cand[:, 0].tolist() == [1, 3, 3], "第 0 列必须是测试正样本"


def test_positive_never_leaks_into_negatives():
    """
    回归测试：原实现用「训练集」做排除集合，测试正样本可能被采成负样本，
    候选集里出现重复正样本（实测约 2.7% 的用户命中）。

    这里把物品空间压到很小，若排除集合漏了测试正样本，
    在如此小的空间里几乎必然被采中，因此本用例能有效拦住该缺陷。
    """
    n_users = 200
    rows = []
    for u in range(n_users):
        for k in range(9):
            rows.append((u + 1, 100 + (u * 9 + k) % 10, 4, 100 + k))
    df = (pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
            .drop_duplicates(subset=["user_id", "item_id"]))
    out, _, _ = reindex(df)
    n_items = int(out["iidx"].max()) + 1
    train, test = leave_one_out(out)
    all_items = build_user_items(out, n_users=n_users)     # 含测试集

    _, cand = build_eval_candidates(test, all_items, n_items=n_items,
                                    n_neg=3, seed=7)

    assert n_items <= 10, "前提：物品空间要足够小，否则拦不住这个缺陷"
    for row in cand:
        assert row[0] not in set(row[1:].tolist()), (
            f"正样本 {row[0]} 出现在负样本里：候选集被污染，指标会被人为抬高")


def test_candidates_reproducible_across_runs():
    out = tiny_df()
    _, test = leave_one_out(out)
    all_items = build_user_items(out, n_users=3)

    _, c1 = build_eval_candidates(test, all_items, n_items=TINY_ITEMS, n_neg=2, seed=42)
    _, c2 = build_eval_candidates(test, all_items, n_items=TINY_ITEMS, n_neg=2, seed=42)
    assert np.array_equal(c1, c2), "多算法对比必须用同一批负样本"


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def test_rank_of_positive():
    """
    约定 scores[0] 是正样本，rank 的含义是"比正样本分高的候选有几个"。
      [0.9, 0.1, 0.2] 正样本最高 -> 0 个比它高 -> rank 0
      [0.3, 0.5, 0.1] 只有 0.5 比它高      -> rank 1
      [0.1, 0.9, 0.2] 0.9 与 0.2 都比它高  -> rank 2
    """
    assert rank_of_positive(np.array([0.9, 0.1, 0.2])) == 0
    assert rank_of_positive(np.array([0.3, 0.5, 0.1])) == 1
    assert rank_of_positive(np.array([0.1, 0.9, 0.2])) == 2
    # 并列不计入（严格大于），避免把正样本自己算成被别的候选压过
    assert rank_of_positive(np.array([0.5, 0.5, 0.1])) == 0


def test_evaluate_ranking_perfect_and_worst():
    perfect = np.array([[3.0, 1.0, 2.0], [5.0, 0.0, 1.0]])
    assert evaluate_ranking(perfect, k=10) == (1.0, 1.0), "排第 1 时 NDCG 必须是 1"

    # 正样本排在最后一名，且候选足够多（100 个）使 rank=99 -> HR@10 落空
    worst = np.array([[0.0] + [1.0] * 99])
    assert evaluate_ranking(worst, k=10) == (0.0, 0.0)


def test_hr_is_trivially_one_when_candidates_fewer_than_k():
    """
    记录一个容易踩的坑：候选集比 k 还小时，HR@k 恒等于 1，指标失去意义。
    真实评测必须用 1 + 99 个候选（本项目的约定），否则算出来的数字没法用。
    """
    scores = np.array([[0.0, 1.0, 2.0]])            # 只有 3 个候选
    hr, _ = evaluate_ranking(scores, k=10)
    assert hr == 1.0, "这正是为什么不能用小候选集评测"

    big = np.array([[0.0] + [1.0] * 99])            # 100 个候选
    assert evaluate_ranking(big, k=10)[0] == 0.0


def test_evaluate_ranking_k_boundary():
    # 正样本排第 10（0 起 rank=9）命中；排第 11（rank=10）落空
    row_hit = np.array([[0.0] + [1.0] * 9 + [0.0] * 90])
    row_miss = np.array([[0.0] + [1.0] * 10 + [0.0] * 89])
    assert evaluate_ranking(row_hit, k=10)[0] == 1.0
    assert evaluate_ranking(row_miss, k=10)[0] == 0.0


def test_evaluate_ranking_ndcg_matches_formula():
    # 正样本排第 3（0 起 rank=2）-> NDCG = 1/log2(4) = 0.5
    assert evaluate_ranking(np.array([[0.0, 1.0, 1.0, 0.0]]), k=10)[1] == pytest.approx(0.5)


def test_evaluate_ranking_rejects_1d():
    with pytest.raises(ValueError):
        evaluate_ranking(np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# 数据集自检
# --------------------------------------------------------------------------- #

def test_check_dataset_rejects_wrong_size():
    """规模不符时必须直接中止 —— 否则可能拿 ml-100k 的指标去写论文"""
    df = pd.DataFrame([(1, 1, 5, 1)],
                      columns=["user_id", "item_id", "rating", "timestamp"])
    with pytest.raises(SystemExit):
        check_dataset(df)
    check_dataset(df, strict=False)      # 放宽时只打印不抛


def test_dataset_stats():
    s = dataset_stats(tiny_df())
    assert (s["ratings"], s["users"], s["items"]) == (8, 3, 4)


# --------------------------------------------------------------------------- #
# 真实数据（存在数据集时才跑）
# --------------------------------------------------------------------------- #

@skip_no_data
def test_real_dataset_scale_and_partition():
    df = load_ratings(DATA_DIR)
    check_dataset(df)

    out, users, items = reindex(df)
    train, test = leave_one_out(out)

    assert len(users) == common.EXPECT_USERS
    assert len(items) == common.EXPECT_ITEMS
    assert len(test) == common.EXPECT_USERS, "留一法应给每位用户一条测试样本"
    assert len(train) + len(test) == len(out)
    assert out.groupby("uidx").size().min() >= 2, "每人至少 2 条评分才能留一法"


@skip_no_data
def test_real_candidates_have_no_leak():
    """真实数据上也确认一遍：正样本与已交互物品都不得出现在负样本里"""
    df = load_ratings(DATA_DIR)
    out, users, items = reindex(df)
    train, test = leave_one_out(out)
    all_items = build_user_items(out, n_users=len(users))

    users_idx, cand = build_eval_candidates(test, all_items, n_items=len(items),
                                            n_neg=99, seed=42)

    for u_idx, row in list(zip(users_idx, cand))[:200]:     # 抽查 200 个用户
        negs = set(row[1:].tolist())
        assert row[0] not in negs
        assert not (negs & all_items[int(u_idx)]), "采到了用户已交互过的物品"
