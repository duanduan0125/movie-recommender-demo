#!/usr/bin/env python
"""
算法层共用模块：数据读取、留一法划分、负采样、Top-N 指标

★ 为什么必须抽出来（设计文档_修订版 §0 原则 6）：
   实验二要求「ItemCF / SVD / NeuMF 在同一划分、同一负样本下对比」。
   原先三个脚本各写一套，靠人工保证划分一致是不可靠的 —— 一旦某处用训练集
   做负样本排除、另一处用了训练集∪测试集，指标就不可比，而且没有报错。
   口径一致只能靠共享代码来保证。

被以下脚本共用：
    ncf_train.py       NeuMF 训练 + 评估 + 导出 Top-50
    eval_topn.py       统一口径的 Top-N 评测（ItemCF / SVD / NeuMF）
    baseline_surprise.py 评分预测基线（RMSE / MAE，只用这里的读取与划分）

用法：
    from common import load_ratings, reindex, leave_one_out, ...
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

# ml-1m 官方规模，用于早期发现数据读错（如分隔符或编码问题）
EXPECT_RATINGS = 1_000_209
EXPECT_USERS = 6_040
EXPECT_ITEMS = 3_706          # 出现在 ratings.dat 中的影片数（movies.dat 是 3883）


# --------------------------------------------------------------------------- #
# 数据读取与划分
# --------------------------------------------------------------------------- #

def load_ratings(data_dir: str) -> pd.DataFrame:
    """
    读取 ratings.dat，返回原始 id 的 DataFrame。

    注意：这里保留 MovieLens 原始 id，索引化只在 reindex() 里做，
    导出推荐结果时再还原成原始 id（库内一律用原始 id，见设计文档 R1）。
    """
    path = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(path):
        raise SystemExit(f"[中止] 找不到 {path}")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df


def load_movie_meta(data_dir: str) -> dict[int, dict]:
    """
    读取 movies.dat -> {原始 movie_id: {"title":..., "year":..., "genres":[...]}}。

    三个容易踩的坑（db/import_ml1m.py 里同样的处理）：
      · 文件是 latin-1 编码，按 utf-8 读会直接抛 UnicodeDecodeError；
      · 片名形如 "Toy Story (1995)"，年份要从末尾括号里取，且片名里可能还有括号；
      · 类型用 "|" 分隔。
    """
    path = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(path):
        raise SystemExit(f"[中止] 找不到 {path}")
    out: dict[int, dict] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            mid, title, genres = parts
            year = None
            m = re.search(r"\((\d{4})\)\s*$", title)
            if m:
                year = int(m.group(1))
                title = title[:m.start()].strip()
            out[int(mid)] = {
                "title": title,
                "year": year,
                "genres": [g for g in genres.split("|") if g],
            }
    return out


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    原始 id -> 0..n-1 连续索引（模型嵌入层要用连续索引）。
    返回 (带 uidx/iidx 的 DataFrame, 原始用户 id 数组, 原始影片 id 数组)。
    """
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {v: j for j, v in enumerate(items)}

    out = df.copy()
    out["uidx"] = out["user_id"].map(u_map).astype(int)
    out["iidx"] = out["item_id"].map(i_map).astype(int)
    return out, users, items


def leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    留一法：每位用户按时间戳排序后，最后一次交互作测试集，其余作训练集。

    这是 Top-N 推荐的标准设定（用历史行为预测"下一部会看的电影"），
    比随机划分更贴近真实场景，也是与已发表结果可比的常用口径。
    """
    ordered = df.sort_values(["uidx", "timestamp"]).reset_index(drop=True)
    rev = ordered.groupby("uidx").cumcount(ascending=False)   # 每人最后一条 = 0
    test = ordered[rev == 0].reset_index(drop=True)
    train = ordered[rev > 0].reset_index(drop=True)
    return train, test


def leave_one_out_with_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    三路划分：在留一法的基础上再切出一份验证集。

        test  = 每人时间戳最后一条（只用于最终报数）
        val   = 每人时间戳倒数第二条（用于选超参，如 ItemKNN 的邻居数 K）
        train = 其余

    ★ 为什么必须再切一份验证集：
      超参在测试集上挑（先扫一遍 K 再报最好的那个）等于在测试集上调参，
      报出来的数字带有乐观偏差，属于不可复现的结论。
      标准做法是在验证集上选超参、在测试集上只评估一次。
      随之而来的要求是：**所有算法都必须用 train 训练**，
      否则「基线少看了一条交互、模型多看了」这类不对称会让对比失去意义。

    只有 1 条交互的用户：进 test，没有 val；只有 2 条的用户：val + test，train 为空。
    ml-1m 每人至少 20 条，实际不会触发这两个边界。
    """
    ordered = df.sort_values(["uidx", "timestamp"]).reset_index(drop=True)
    rev = ordered.groupby("uidx").cumcount(ascending=False)   # 最后一条 = 0
    test = ordered[rev == 0].reset_index(drop=True)
    val = ordered[rev == 1].reset_index(drop=True)
    train = ordered[rev > 1].reset_index(drop=True)
    return train, val, test


def build_user_items(df: pd.DataFrame, n_users: int) -> dict[int, set[int]]:
    """每个用户已交互物品的集合（用 iidx）。"""
    d: dict[int, set[int]] = {u: set() for u in range(n_users)}
    for u, i in zip(df["uidx"].values, df["iidx"].values):
        d[int(u)].add(int(i))
    return d


# --------------------------------------------------------------------------- #
# 负采样
# --------------------------------------------------------------------------- #

def sample_negatives(rng: np.random.Generator, exclude: set[int],
                     n_items: int, n: int) -> list[int]:
    """
    从"未交互物品"里随机采 n 个。exclude 必须包含该用户**全部**已交互物品。

    ★ 这里是最容易出错的地方：评测时的 exclude 必须是 训练集 ∪ 测试集。
      若只用训练集，测试正样本本身可能被采成负样本（实测约 2.7% 的用户会命中），
      候选集里就出现重复的正样本，指标口径被污染。
    """
    out: list[int] = []
    seen_extra: set[int] = set()
    guard = 0
    while len(out) < n:
        j = int(rng.integers(0, n_items))
        if j in exclude or j in seen_extra:
            guard += 1
            # 理论上不会触发；留着防止 n_items 过小时死循环
            if guard > 50 * n + 1000:
                break
            continue
        seen_extra.add(j)
        out.append(j)
    return out


def build_eval_candidates(test_df: pd.DataFrame,
                          all_user_items: dict[int, set[int]],
                          n_items: int,
                          n_neg: int = 99,
                          seed: int = 42,
                          block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    构造留一法评测候选集：每行 = [测试正样本, n_neg 个随机负样本]。

    all_user_items 是「训练集 ∪ 测试集」的已交互集合 —— 调用方负责传对，
    这是防止正样本泄漏成负样本的关键。

    返回 (users_idx, candidates)：
        users_idx  shape (m,)       测试用户（uidx）
        candidates shape (m, 1+n_neg)  第 0 列固定是正样本
    """
    rng = np.random.default_rng(seed)
    tu = test_df["uidx"].values
    ti = test_df["iidx"].values

    users_out = np.empty(len(tu), dtype=np.int64)
    cand = np.empty((len(tu), 1 + n_neg), dtype=np.int64)
    for s in range(0, len(tu), block):
        bu, bi = tu[s:s + block], ti[s:s + block]
        for r, (u, pos) in enumerate(zip(bu, bi)):
            u, pos = int(u), int(pos)
            users_out[s + r] = u
            cand[s + r, 0] = pos
            cand[s + r, 1:] = sample_negatives(
                rng, all_user_items[u], n_items, n_neg)
    return users_out, cand


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def rank_of_positive(scores: np.ndarray) -> int:
    """
    正样本在候选集里的排名（0 起）。约定 scores[0] 是正样本。
    用严格大于计数，并列时不把正样本自己算进去。
    """
    return int((scores > scores[0]).sum())


def evaluate_ranking(scores: np.ndarray, k: int = 10) -> tuple[float, float]:
    """
    计算 HR@k 与 NDCG@k。

    scores: shape (m, 1+n_neg)，第 0 列必须是正样本的分数。
    正样本排第 1 时 rank=0 -> NDCG = 1/log2(2) = 1，符合定义。
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores 必须是二维数组，且第 0 列为正样本")
    ranks = (scores > scores[:, :1]).sum(axis=1)
    hits = (ranks < k).astype(np.float64)
    ndcg = hits / np.log2(ranks + 2.0)
    return float(hits.mean()), float(ndcg.mean())


def dataset_stats(df: pd.DataFrame) -> dict:
    """数据集概况，供脚本开头打印与自查。"""
    return {
        "ratings": len(df),
        "users": df["user_id"].nunique(),
        "items": df["item_id"].nunique(),
        "sparsity": 1 - len(df) / (df["user_id"].nunique() * df["item_id"].nunique()),
    }


def check_dataset(df: pd.DataFrame, strict: bool = True) -> None:
    """
    读取后的自检：规模不对通常意味着分隔符或文件选错了
    （例如误读了 ml-latest 或 ml-100k）。
    """
    s = dataset_stats(df)
    msg = (f"[数据] 评分 {s['ratings']} | 用户 {s['users']} | 影片 {s['items']} | "
           f"稀疏度 {s['sparsity']:.4f}")
    print(msg)
    if not strict:
        return
    if (s["ratings"], s["users"], s["items"]) != (EXPECT_RATINGS, EXPECT_USERS,
                                                  EXPECT_ITEMS):
        raise SystemExit(
            f"[中止] 数据集规模与 ml-1m 不符：实际 {s['ratings']}/{s['users']}/"
            f"{s['items']}，期望 {EXPECT_RATINGS}/{EXPECT_USERS}/{EXPECT_ITEMS}。"
            "确认 --data-dir 指向的是 ml-1m。")


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """三个脚本共用的命令行参数，避免各写一套导致默认值不一致。"""
    ap.add_argument("--data-dir", default="./ml-1m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-neg", type=int, default=99,
                    help="留一法评测里每人的随机负样本数（通用口径为 99）")
    ap.add_argument("--eval-k", type=int, default=10)
