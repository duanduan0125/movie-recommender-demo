#!/usr/bin/env python
"""
统一口径的 Top-N 评测（实验二的数据来源）

★ 为什么需要这个脚本（设计文档_修订版 §3.1）：
   《数据库设计与算法实现说明》3.2 的对比表要求 ItemCF / SVD 也有
   HR@10 / NDCG@10，但 baseline_surprise.py 只算 RMSE/MAE、
   ncf_train.py 只评 NeuMF —— 原有两个脚本产不出那张表。
   更要紧的是：三个算法必须在**同一份留一法划分、同一批负样本**下评测，
   而 baseline_surprise.py 用的是 5 折随机划分，与留一法完全不是一个口径，
   数字放同一张表里对比在方法上不成立。本脚本把口径统一到 common.py。

口径（与 ncf_train.py 完全一致，均由 common.py 保证）：
   划分   三路（--three-way）：test = 每人时间戳最后一条（与留一法同一条），
          val = 倒数第二条（用来选超参），train = 其余
   候选   1 个测试正样本 + 99 个随机负样本，seed 固定
   排除   负采样排除「训练集 ∪ 验证集 ∪ 测试集」的全部已交互物品
   指标   HR@10 / NDCG@10

用法（实验二主表必须带 --three-way：不加的话基线比模型少看一条交互，
      对比不对称，只能用于快速自查）：
    python eval_topn.py --data-dir ./ml-1m --algo itemcf --three-way
    python eval_topn.py --data-dir ./ml-1m --algo svd   --three-way
    python eval_topn.py --data-dir ./ml-1m --algo neumf --ckpt ncf_model.pt --three-way
    python eval_topn.py --data-dir ./ml-1m --algo itemknn --tune-k --three-way
    python eval_topn.py --algo itemcf --limit-users 500     # 快速自查（二路口径）
   （--data_dir 的下划线写法同样接受，照旧文档抄的命令不会报错）

产出：
    控制台打印指标，并追加写入 topn_result.csv（algo / HR@10 / NDCG@10 / 用户数 / 耗时）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    build_eval_candidates,
    build_user_items,
    check_dataset,
    evaluate_ranking,
    leave_one_out,
    leave_one_out_with_val,
    load_ratings,
    reindex,
)

OUT_CSV = "topn_result.csv"


# --------------------------------------------------------------------------- #
# 打分器：统一签名 score(u_idx_array, i_idx_array) -> np.ndarray
#
# u_idx / i_idx 都是 reindex 后的 0..n-1 连续索引，打分器内部自己决定
# 是否需要还原成 MovieLens 原始 id（surprise 需要，NeuMF 不需要）。
# --------------------------------------------------------------------------- #

def make_surprise_scorer(algo_name: str, train_df, users_orig, items_orig,
                         cache_dir: str):
    """
    用 surprise 训练并在给定 (uidx, iidx) 上预测评分作为排序分数。

    注意：surprise 内部用原始 id，所以这里要映射回去；
    训练只用训练集，绝不碰测试集。
    """
    from surprise import AlgoBase, BaselineOnly, Dataset, KNNWithMeans, Reader, SVD

    reader = Reader(rating_scale=(1, 5))
    # 只用训练集构造 trainset —— 若传全量数据就会把测试集带进模型，属于标签泄漏
    data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    algos: dict[str, AlgoBase] = {
        "itemcf": KNNWithMeans(sim_options={"user_based": False, "name": "cosine"},
                               verbose=False),
        "usercf": KNNWithMeans(sim_options={"user_based": True, "name": "cosine"},
                               verbose=False),
        "svd": SVD(random_state=42),
        "baseline": BaselineOnly(),
    }
    if algo_name not in algos:
        raise SystemExit(f"[中止] 未知算法 {algo_name}，可选：{list(algos)}")

    algo = algos[algo_name]
    print(f"[训练] {algo_name} 拟合 {len(train_df)} 条训练评分 ...")
    t0 = time.time()
    algo.fit(trainset)
    print(f"[训练] 完成，耗时 {time.time() - t0:.1f}s")

    def score(uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        out = np.empty(len(uidx), dtype=np.float64)
        for k, (u, i) in enumerate(zip(uidx, iidx)):
            out[k] = algo.predict(int(users_orig[u]), int(items_orig[i])).est
        return out

    return score


# --------------------------------------------------------------------------- #
# 排序式基线（不需要训练）
#
# ★ 为什么必须补这两个（设计文档_修订版 §3.2 的对比公平性）：
#   SVD / KNNWithMeans 是**评分预测**模型。拿它们的预测分去排序时，大量
#   (用户,候选) 会回落到物品或用户的均值，在 99 个随机负样本里几乎没有区分度
#   —— 实测 HR@10 只有 0.22~0.25，与文献锚点（ItemCF ≈ 0.60~0.63）差了一个
#   量级。那个锚点对应的是标准的 ItemKNN 排序实现，两者不是一回事。
#   把"评分预测模型按预测分排序"和"面向排序训练的 NeuMF"放同一张表对比，
#   对被对比方不公平，结论也站不住。因此补上同为排序目标的两个基线：
#       mostpop  热门度，作为下限参照
#       itemknn  二值交互矩阵算物品相似度，对用户历史做相似度求和打分
# --------------------------------------------------------------------------- #

def build_binary_matrix(uidx, iidx, n_users: int, n_items: int):
    """由训练集交互构造二值「用户×物品」稀疏矩阵（有交互记 1，忽略评分值）"""
    from scipy.sparse import csr_matrix
    data = np.ones(len(uidx), dtype=np.float32)
    return csr_matrix((data, (uidx, iidx)), shape=(n_users, n_items))


def item_item_cosine(X) -> np.ndarray:
    """
    物品-物品余弦相似度：sim(i,j) = |U_i ∩ U_j| / sqrt(|U_i| · |U_j|)

    X 为 用户×物品 二值稀疏矩阵；返回 n_items×n_items 的 float32 稠密矩阵。
    对角线为 1（物品与自身完全相似）。冷门物品（无人交互）不参与，避免除零。
    """
    Xt = X.T.tocsr()                                        # 物品×用户
    norms = np.sqrt(np.asarray(Xt.multiply(Xt).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    Xt = Xt.multiply(1.0 / norms[:, None]).tocsr()           # 按物品归一化
    return (Xt @ Xt.T).toarray().astype(np.float32)


def itemknn_scores(S: np.ndarray, X) -> np.ndarray:
    """
    打分 = 该物品与"用户历史物品"的相似度之和。
    返回 (n_items, n_users) 的 float32 矩阵，取值时用 scores[iidx, uidx]。
    """
    return (S @ X.T.toarray()).astype(np.float32)


def mostpop_scores(X) -> np.ndarray:
    """热度分 = 物品在训练集中的交互次数（与用户无关）。"""
    return np.asarray(X.sum(axis=0)).ravel().astype(np.float32)


def make_matrix_scorer(score_matrix: np.ndarray):
    """把 (n_items, n_users) 的分数矩阵包装成统一的 score(uidx, iidx) 签名"""
    def score(uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        return score_matrix[iidx, uidx].astype(np.float64)
    return score


def truncate_topk_similarity(S: np.ndarray, k: int) -> np.ndarray:
    """
    每个物品只保留相似度最高的 k 个邻居，其余置 0 —— 即文献里 ItemKNN 常用的
    "top-K 邻居"设定。截断能去掉大量噪声相似度（两个冷门物品偶然共享一个用户
    就会产生一个不小的余弦值），通常对排序效果有提升。

    · 行 i 的 top-K **不含 i 自己**（先把对角线置 -inf 再取，避免自己占名额）
    · k <= 0 表示不截断，原样返回
    · k 超过 n-1 时按 n-1 处理（最多只能有这么多别的物品）
    """
    if k <= 0:
        return S
    n = S.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return S
    work = S.copy()
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(-work, k - 1, axis=1)[:, :k]
    out = np.zeros_like(S)
    rows = np.arange(n)[:, None]
    out[rows, idx] = work[rows, idx]
    return out


def make_itemknn_scorer(train_df, n_users: int, n_items: int, topk_sim: int = 0):
    X = build_binary_matrix(train_df["uidx"].values, train_df["iidx"].values,
                            n_users, n_items)
    print(f"[基线] 训练集二值矩阵 {X.shape}，非零 {X.nnz} 个")
    t0 = time.time()
    S = item_item_cosine(X)
    print(f"[基线] 物品相似度矩阵 {S.shape} 计算完成，耗时 {time.time() - t0:.1f}s")
    if topk_sim > 0:
        t0 = time.time()
        S = truncate_topk_similarity(S, topk_sim)
        kept = int((S > 0).sum())
        print(f"[基线] 保留每个物品的 top-{topk_sim} 邻居，"
              f"相似度非零项 {kept} 个，耗时 {time.time() - t0:.1f}s")
    t0 = time.time()
    full = itemknn_scores(S, X)
    print(f"[基线] 全量打分矩阵 {full.shape} 计算完成，耗时 {time.time() - t0:.1f}s")
    return make_matrix_scorer(full)


def make_mostpop_scorer(train_df, n_users: int, n_items: int):
    X = build_binary_matrix(train_df["uidx"].values, train_df["iidx"].values,
                            n_users, n_items)
    pop = mostpop_scores(X)

    def score(uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        return pop[iidx].astype(np.float64)

    return score


def make_neumf_scorer(ckpt: str, n_users: int, n_items: int, dim: int,
                      device: str, block: int = 8192):
    """
    从 ncf_train.py 训练出的权重加载 NeuMF 并打分。

    直接复用 ncf_train.NeuMF，避免为了评测再复制一份模型定义
    （两份定义一旦不同步，评测的就不是同一个模型了）。
    """
    import torch
    from ncf_train import NeuMF

    if not os.path.exists(ckpt):
        raise SystemExit(f"[中止] 找不到权重文件 {ckpt}，先跑 ncf_train.py")

    model = NeuMF(n_users, n_items, dim=dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[模型] 已加载 {ckpt}（dim={dim}, device={device}）")

    @torch.no_grad()
    def score(uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        out = np.empty(len(uidx), dtype=np.float64)
        for s in range(0, len(uidx), block):
            bu = torch.tensor(uidx[s:s + block], dtype=torch.long, device=device)
            bi = torch.tensor(iidx[s:s + block], dtype=torch.long, device=device)
            out[s:s + block] = model(bu, bi).cpu().numpy()
        return out

    return score


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """命令行定义单独抽出来，便于单测直接验证参数名与默认值。"""
    ap = argparse.ArgumentParser(description="统一口径的 Top-N 评测")
    ap.add_argument("--algo", required=True,
                    choices=["itemcf", "usercf", "svd", "baseline", "neumf",
                             "itemknn", "mostpop"],
                    help="itemknn / mostpop 是排序式基线（无需训练）；"
                         "svd / itemcf / usercf 是评分预测模型按其预测分排序")
    # 设计文档 §3.2 的用法写的是 --data_dir（下划线）。文档已统一为 --data-dir，
    # 这里两种都收，保证照着文档抄下来的老命令照样能跑通
    # （baseline_surprise.py / ncf_train.py 早已如此）。
    ap.add_argument("--data-dir", "--data_dir", dest="data_dir",
                    default="./ml-1m",
                    help="ml-1m 目录（兼容旧的 --data_dir 写法）")
    ap.add_argument("--ckpt", default="ncf_model.pt", help="neumf 的权重文件")
    ap.add_argument("--dim", type=int, default=32, help="neumf 的嵌入维度，与训练一致")
    ap.add_argument("--n-neg", type=int, default=99)
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--limit-users", type=int, default=0,
                    help="只评测前 N 个用户（快速试跑用，0 表示全部）")
    ap.add_argument("--topk-sim", type=int, default=0,
                    help="仅对 itemknn 生效：每个物品只保留相似度最高的 K 个邻居，"
                         "0 表示不截断")
    ap.add_argument("--label", default=None,
                    help="结果表里的算法名（默认用 --algo）。做参数扫描时用它区分，"
                         "例如 --algo itemknn --topk-sim 200 --label itemknn_top200")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--three-way", action="store_true",
                    help="改用三路划分（train=除最后两条 / val=倒数第二条 / test=最后一条），"
                         "所有算法都用 train 训练。选超参时必须打开，"
                         "否则基线少看一条交互、模型多看了，对比不对称")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="在哪个划分上报指标，默认 test；val 用于选超参")
    ap.add_argument("--tune-k", action="store_true",
                    help="仅 itemknn：在验证集上扫 K 并选出最优，再到测试集上只评估一次")
    ap.add_argument("--k-grid", default="1,2,3,5,10,20,30,50,75,100,200,500,0",
                    help="选参时扫描的 K 列表，0 表示不截断")
    ap.add_argument("--out", default=OUT_CSV)
    return ap


def main():
    args = build_parser().parse_args()

    if args.topk_sim and args.algo != "itemknn":
        print(f"[提示] --topk-sim 仅对 itemknn 生效，当前算法 {args.algo} 将忽略它")
    if args.tune_k and args.algo != "itemknn":
        raise SystemExit("[中止] --tune-k 只对 itemknn 有意义")

    df = load_ratings(args.data_dir)
    check_dataset(df)
    df, users_orig, items_orig = reindex(df)
    n_users, n_items = len(users_orig), len(items_orig)

    # 负采样排除集合恒为「该用户全部交互」：只排除训练集的话，
    # 参与评测的那个正样本可能被采成负样本，候选集里出现重复正样本。
    all_items = build_user_items(df, n_users)

    if args.three_way:
        train, val, test = leave_one_out_with_val(df)
        print(f"[划分] 三路：train {len(train)} / val {len(val)} / test {len(test)}")
        if args.tune_k:
            return tune_k(train, val, test, all_items, n_users, n_items, args)
        eval_df = test if args.split == "test" else val
        split_name = args.split
    else:
        train, test = leave_one_out(df)
        eval_df, split_name = test, "test"
        print("[划分] 二路留一法：train = 除最后一条，test = 最后一条。")
        print("       ★ 该口径下模型比基线多看一条交互，只能用于快速自查；"
              "实验二主表必须加 --three-way")

    users_idx, cand = build_eval_candidates(eval_df, all_items, n_items,
                                           n_neg=args.n_neg, seed=args.seed)
    if args.limit_users:
        users_idx, cand = users_idx[:args.limit_users], cand[:args.limit_users]
    print(f"[评测] {args.algo} | 划分 {split_name} | 用户 {len(users_idx)} | "
          f"候选 {cand.shape[1]} 个/人 | HR@{args.eval_k} / NDCG@{args.eval_k}")

    score = build_scorer(args, train, n_users, n_items, users_orig, items_orig)
    hr, ndcg, elapsed, total = score_and_evaluate(score, users_idx, cand, args.eval_k)

    label = args.label or args.algo
    print(f"\n[结果] {label}  HR@{args.eval_k}={hr:.4f}  "
          f"NDCG@{args.eval_k}={ndcg:.4f}  用户={total}  耗时={elapsed:.0f}s")

    write_row(args.out, label, hr, ndcg, total, elapsed, args)
    return 0


def build_scorer(args, train, n_users: int, n_items: int, users_orig, items_orig):
    """按算法名构造统一的打分函数；所有算法都用传入的 train 训练"""
    if args.algo == "neumf":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            raise SystemExit("[中止] 缺少 torch")
        return make_neumf_scorer(args.ckpt, n_users, n_items, args.dim, device)
    if args.algo == "itemknn":
        return make_itemknn_scorer(train, n_users, n_items, topk_sim=args.topk_sim)
    if args.algo == "mostpop":
        return make_mostpop_scorer(train, n_users, n_items)
    return make_surprise_scorer(args.algo, train, users_orig, items_orig, args.out)


def score_and_evaluate(score, users_idx, cand, k: int):
    """
    逐行打分后算 HR@k / NDCG@k。每行第 0 列是正样本，其余是负样本。
    返回 (hr, ndcg, 耗时秒, 用户数)。
    """
    t0 = time.time()
    scores = np.empty(cand.shape, dtype=np.float64)
    total = len(cand)
    for r in range(total):
        scores[r] = score(np.full(cand.shape[1], users_idx[r], dtype=np.int64),
                          cand[r])
        if (r + 1) % 1000 == 0 or r + 1 == total:
            el = time.time() - t0
            print(f"  进度 {r + 1}/{total}  耗时 {el:.0f}s  "
                  f"预计剩余 {el / (r + 1) * (total - r - 1):.0f}s")
    hr, ndcg = evaluate_ranking(scores, k=k)
    return hr, ndcg, time.time() - t0, total


def tune_k(train, val, test, all_items, n_users: int, n_items: int, args) -> int:
    """
    在验证集上选 ItemKNN 的邻居数 K，再到测试集上只评估一次。

    ★ 这是选超参唯一站得住的做法。若拿测试集扫一遍 K 再挑最好的那个报出来，
      等于在测试集上调参，数字带乐观偏差且不可复现 —— 上一轮就是这么做的，
      扫出 K=1 时 HR@10=0.9310，远超已发表工作，纯属口径失真。

    选参依据用 NDCG@k：它对排序质量比 HR@k 更敏感（HR 只要进前 k 就算命中），
    在候选池是「1 正 + 99 随机负」时更稳定。两个指标都会打印出来备查。
    """
    grid = [int(x) for x in args.k_grid.split(",") if x.strip() != ""]
    X = build_binary_matrix(train["uidx"].values, train["iidx"].values,
                            n_users, n_items)
    print(f"[选参] 训练集二值矩阵 {X.shape}，非零 {X.nnz} 个")
    S = item_item_cosine(X)

    v_users, v_cand = build_eval_candidates(val, all_items, n_items,
                                           n_neg=args.n_neg, seed=args.seed)
    print(f"[验证集] 用户 {len(v_users)}，候选 {v_cand.shape[1]} 个/人，"
          f"K 网格 {[('不截断' if g <= 0 else g) for g in grid]}")

    sweep_path = (args.out[:-4] + "_val_sweep.csv" if args.out.endswith(".csv")
                  else args.out + "_val_sweep.csv")
    best_k, best = 0, (-1.0, -1.0)
    for kk in grid:
        t0 = time.time()
        score = make_matrix_scorer(itemknn_scores(truncate_topk_similarity(S, kk), X))
        hr, ndcg, _, _ = score_and_evaluate(score, v_users, v_cand, args.eval_k)
        name = "不截断" if kk <= 0 else f"K={kk}"
        print(f"  {name:<8} HR@{args.eval_k}={hr:.4f}  NDCG@{args.eval_k}={ndcg:.4f}"
              f"  ({time.time() - t0:.0f}s)")
        write_row(sweep_path, f"itemknn_{'none' if kk <= 0 else kk}", hr, ndcg,
                  len(v_users), time.time() - t0, args)
        if ndcg > best[1]:
            best_k, best = kk, (hr, ndcg)

    print(f"\n[选参] 验证集最优 {'不截断' if best_k <= 0 else f'K={best_k}'}"
          f"（HR@{args.eval_k}={best[0]:.4f}, NDCG@{args.eval_k}={best[1]:.4f}）")
    print(f"[选参] K 扫描结果已写入 {sweep_path}")

    t_users, t_cand = build_eval_candidates(test, all_items, n_items,
                                           n_neg=args.n_neg, seed=args.seed)
    score = make_matrix_scorer(itemknn_scores(truncate_topk_similarity(S, best_k), X))
    hr, ndcg, elapsed, total = score_and_evaluate(score, t_users, t_cand, args.eval_k)

    label = args.label or ("itemknn" if best_k <= 0 else f"itemknn(K={best_k})")
    print(f"\n[结果] {label}（K 由验证集选出，测试集只评一次）  "
          f"HR@{args.eval_k}={hr:.4f}  NDCG@{args.eval_k}={ndcg:.4f}  "
          f"用户={total}  耗时={elapsed:.0f}s")
    write_row(args.out, label, hr, ndcg, total, elapsed, args)
    return 0


def write_row(path: str, algo: str, hr: float, ndcg: float, n_users: int,
              elapsed: float, args) -> None:
    """
    追加写入结果表；同一 algo 重复跑会覆盖旧行（避免表里出现同一算法的多行）。
    """
    rows: list[list] = []
    header = ["algo", "HR@k", "NDCG@k", "users", "k", "n_neg", "耗时(s)"]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            existing = list(reader)
        if existing:
            header = existing[0]
            rows = [r for r in existing[1:] if r and r[0] != algo]

    rows.append([algo, f"{hr:.4f}", f"{ndcg:.4f}", n_users, args.eval_k,
                 args.n_neg, f"{elapsed:.0f}"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[保存] {path}")


if __name__ == "__main__":
    sys.exit(main())
