"""
baseline_surprise.py 的单元测试（不联网、不跑交叉验证）

运行：
    python -m pytest script/test_baseline_surprise.py -v

只测"基线的构成"，不测指标（那要跑十几分钟，属于实验而非单测）。
这里的断言把「实验一的基线就是文档里那五项」这条约定固定下来：
算法清单、UserCF/ItemCF 的方向、以及 SVD/NMF 的随机种子。
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_surprise import build_algos, build_parser  # noqa: E402

EXPECTED = ["BaselineOnly", "UserCF", "ItemCF", "SVD", "NMF"]


def test_baseline_set_matches_documented_scope():
    """实验一的表格有几行，这里就该有几个算法 —— 多一个少一个都会与文档脱节"""
    algos = build_algos()
    assert list(algos) == EXPECTED


def test_user_item_cf_directions_are_opposite():
    """UserCF / ItemCF 的 user_based 不能设反，否则两个名字对调、表格也跟着错"""
    algos = build_algos()
    assert algos["UserCF"].sim_options["user_based"] is True
    assert algos["ItemCF"].sim_options["user_based"] is False


def test_knn_uses_with_means_not_basic():
    """
    必须是 KNNWithMeans：KNNBasic 在 ml-1m 上 RMSE 通常 0.97~0.98，
    与文档锚点（≈0.92）对不上，会让人误判"自己跑错了"。
    """
    for name in ("UserCF", "ItemCF"):
        assert type(build_algos()[name]).__name__ == "KNNWithMeans"


@pytest.mark.parametrize("name", ["SVD", "NMF"])
def test_random_seed_is_fixed(name):
    """SVD/NMF 的随机初始化若不固定，每次跑出的数字都不同，论文数据不可复现"""
    algo = build_algos()[name]
    assert algo.random_state == 42


def test_baseline_only_needs_no_similarity():
    assert type(build_algos()["BaselineOnly"]).__name__ == "BaselineOnly"


# --------------------------------------------------------------------------- #
# 命令行：设计文档 §3.4 的约定（折数 3、数据目录两种写法、进度打印）
# --------------------------------------------------------------------------- #

def test_parser_accepts_both_data_dir_spellings():
    assert build_parser().parse_args(
        ["--data-dir", "./ml-1m"]).data_dir == "./ml-1m"
    assert build_parser().parse_args(
        ["--data_dir", "./ml-1m"]).data_dir == "./ml-1m"


def test_parser_default_cv_is_three():
    """§3.4：折数默认从 5 降为 3（KNN/NMF 在 100 万评分上 5 折非常慢）"""
    assert build_parser().parse_args([]).cv == 3


def test_progress_is_printed_by_default():
    """
    §3.4 第 5 条要求打印进度：UserCF 单折要几分钟，
    cross_validate 的逐折输出默认必须打开，--quiet 才能关。
    """
    assert build_parser().parse_args([]).quiet is False
    assert build_parser().parse_args(["--quiet"]).quiet is True


def test_parser_default_seed_is_42():
    """§3.4：random_state=42，否则每次跑出的数字都不同，论文数据不可复现"""
    assert build_parser().parse_args([]).seed == 42
