"""
神经协同过滤 (NeuMF) 训练脚本  ——  MovieLens-1M

用法：
    python ncf_train.py --data-dir ./ml-1m --epochs 20 --topk 50

产出：
    1. 控制台打印每轮 loss / HR@10 / NDCG@10
    2. rec_result.csv    每位用户的 Top-50 推荐（原始 movie_id，可直接导入数据库）
    3. ncf_model.pt      最佳轮次的模型权重（不是最后一轮）
    4. metrics.txt       最终指标与超参，供论文使用

说明：
    · 全程 CPU 可跑（有 CUDA 会自动使用）
    · ml-1m 上单轮训练约 30~60 秒（CPU），20 轮含评估约 20~30 分钟
      （原注释写"15 分钟"没算上每轮 60 万次前向的评估开销）
    · 数据划分、负采样与 HR/NDCG 全部复用 common.py，
      与 eval_topn.py 同口径，否则两个脚本的数字不能放同一张表里对比
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================ 基础工具

# 数据读取、留一法划分、负采样、HR/NDCG 全部复用 common.py，
# 与 eval_topn.py 共用同一份实现（设计文档_修订版 §0 原则 6）。
# 原先这些函数在本文件里单独实现了一份，与评测脚本各写一套 ——
# 一旦某处的排除集合不一致（例如负采样漏了测试集），指标就不可比且不会报错。
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================ 数据集

class ImplicitDataset(Dataset):
    """正负样本 1:neg_ratio；负样本从未交互物品中随机采样（一次性固定）"""

    def __init__(self, train_df, n_items, user_items, neg_ratio=4, seed=42):
        rng = np.random.default_rng(seed)
        users, items, labels = [], [], []
        for u, i in zip(train_df["uidx"].values, train_df["iidx"].values):
            u, i = int(u), int(i)
            users.append(u)
            items.append(i)
            labels.append(1.0)
            cnt = 0
            seen = user_items[u]
            while cnt < neg_ratio:
                j = int(rng.integers(0, n_items))
                if j not in seen:
                    users.append(u)
                    items.append(j)
                    labels.append(0.0)
                    cnt += 1
        self.users = np.asarray(users, dtype=np.int64)
        self.items = np.asarray(items, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ============================================================ 模型

class NeuMF(nn.Module):
    """GMF 通路（线性交互） + MLP 通路（非线性交互）"""

    def __init__(self, n_users, n_items, dim=32, mlp_layers=(64, 32, 16), dropout=0.2):
        super().__init__()
        self.emb_user_gmf = nn.Embedding(n_users, dim)
        self.emb_item_gmf = nn.Embedding(n_items, dim)
        self.emb_user_mlp = nn.Embedding(n_users, dim)
        self.emb_item_mlp = nn.Embedding(n_items, dim)

        layers, in_dim = [], dim * 2
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim + in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for emb in (self.emb_user_gmf, self.emb_item_gmf,
                    self.emb_user_mlp, self.emb_item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, u, i):
        gmf = self.emb_user_gmf(u) * self.emb_item_gmf(i)
        mlp = self.mlp(torch.cat([self.emb_user_mlp(u), self.emb_item_mlp(i)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)   # 输出 logit


# ============================================================ 评估（留一法）

@torch.no_grad()
def evaluate(model, test_df, n_items, exclude_items, device,
             k=10, n_neg=99, seed=42, block=2048):
    """
    留一法评测：每人的测试正样本 + n_neg 个随机负样本，算 HR@k / NDCG@k。

    ★ exclude_items 必须是「训练集 ∪ 测试集」的全部已交互物品。
      原实现只传训练集，于是测试正样本本身可能被采成负样本
      （实测约 2.7% 的用户会命中），候选集里出现重复正样本。
      现在候选构造与指标计算都走 common.py，与 eval_topn.py 完全同口径，
      两个脚本的数字才能放同一张表里对比。
    """
    model.eval()
    users_idx, cand = build_eval_candidates(test_df, exclude_items, n_items,
                                            n_neg=n_neg, seed=seed)

    scores = np.empty(cand.shape, dtype=np.float64)
    for s in range(0, len(cand), block):
        chunk = cand[s:s + block]
        u_t = torch.tensor(np.repeat(users_idx[s:s + block], chunk.shape[1]),
                           dtype=torch.long, device=device)
        i_t = torch.tensor(chunk.reshape(-1), dtype=torch.long, device=device)
        scores[s:s + block] = model(u_t, i_t).view(len(chunk), -1).cpu().numpy()

    return evaluate_ranking(scores, k=k)


# ============================================================ 导出 Top-K

@torch.no_grad()
def export_topk(model, n_users, n_items, user_items, users_orig, items_orig,
                k=50, out="rec_result.csv", device="cpu", block=256):
    model.eval()
    rows = []
    for s in range(0, n_users, block):
        us = list(range(s, min(s + block, n_users)))
        u_t = torch.tensor(np.repeat(np.asarray(us), n_items), dtype=torch.long, device=device)
        i_t = torch.arange(n_items, dtype=torch.long, device=device).repeat(len(us))
        sc = model(u_t, i_t).view(len(us), n_items).cpu().numpy()

        for r, u in enumerate(us):
            if user_items[u]:
                sc[r, list(user_items[u])] = -1e9          # 排除已看过的
            top = np.argpartition(-sc[r], k - 1)[:k]
            top = top[np.argsort(-sc[r][top])]
            for rank, j in enumerate(top, start=1):
                rows.append((int(users_orig[u]), int(items_orig[j]), float(sc[r][j]), rank))

    pd.DataFrame(rows, columns=["user_id", "movie_id", "score", "rank_num"]).to_csv(out, index=False)
    print(f"[导出] {len(rows)} 条推荐结果 -> {out}")


# ============================================================ 主流程

def build_parser() -> argparse.ArgumentParser:
    """
    单独抽出来是为了能对参数名与默认值做单测 —— 参数口径写错（例如
    --data_dir 与 --data-dir 只认一个）会让"照着文档敲命令"直接失败。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", "--data_dir", dest="data_dir",
                    default="./ml-1m",
                    help="ml-1m 目录（兼容旧的 --data_dir 写法）")
    ap.add_argument("--three-way", action="store_true",
                    help="改用三路划分训练（倒数第二条作验证集，不参与训练）。"
                         "与基线做对比实验时必须打开，否则模型比基线多看了一条交互")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=".",
                    help="输出目录：ncf_model.pt / metrics.txt / rec_result.csv 都写到这里，"
                         "默认当前目录（原实现写死当前目录）。"
                         "★ 刻意不叫 --out：本仓库另两个脚本的 --out 指的是"
                         "「输出 CSV 的文件路径」，同名不同义容易误操作")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--neg_ratio", type=int, default=4)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[环境] device={device}  torch={torch.__version__}")

    df = load_ratings(args.data_dir)
    check_dataset(df)          # 规模不符直接中止，避免拿错数据集跑出的指标写进论文
    df, users_orig, items_orig = reindex(df)
    n_users, n_items = len(users_orig), len(items_orig)

    train_df, test_df = leave_one_out(df)
    if args.three_way:
        # 与基线保持一致：训练只用 train（除最后两条），
        # 倒数第二条留作验证集（供 ItemKNN 之类的基线选超参）。
        # 若模型用全部历史、基线却少看一条，对比就不对称了。
        train_df, val_df, test_df = leave_one_out_with_val(df)
        print(f"[划分] 三路：train {len(train_df)} / val {len(val_df)} / "
              f"test {len(test_df)}（训练只用 train）")
    # user_items     训练用：负采样时只需排除"看过的"
    # all_user_items 评测与导出用：必须排除 训练集 ∪ 验证集 ∪ 测试集 的全部已评分影片
    #                （库侧校验 D2 要求推荐结果里不得出现用户已评分的影片）
    user_items = build_user_items(train_df, n_users)
    all_user_items = build_user_items(df, n_users)
    print(f"[划分] 训练集 {len(train_df)} | 测试集 {len(test_df)} | "
          f"用户 {n_users} | 影片 {n_items}")

    print("[采样] 正在构造负样本，首次约需 30 秒...")
    t0 = time.time()
    ds = ImplicitDataset(train_df, n_items, user_items, neg_ratio=args.neg_ratio, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[采样] 完成，共 {len(ds)} 条样本，耗时 {time.time() - t0:.1f}s")

    model = NeuMF(n_users, n_items, dim=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = (-1.0, -1.0)
    best_epoch = 0
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for u, i, y in loader:
            u = u.to(device)
            i = i.to(device)
            y = y.to(device)
            loss = loss_fn(model(u, i), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        if ep % args.eval_every == 0 or ep == args.epochs:
            hr, ndcg = evaluate(model, test_df, n_items, all_user_items, device)
            if ndcg > best[1]:
                best = (hr, ndcg)
                best_epoch = ep
                # ★ 同时把当时的参数留下来。
                #   原实现只记数字，保存与导出的却是最后一轮的权重 ——
                #   于是"论文写最佳指标、实际跑的是另一个模型"，这是个静默的不一致。
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            print(f"[Epoch {ep:>2}/{args.epochs}] loss={total / nb:.4f}  "
                  f"HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

    print(f"\n[最佳] 第 {best_epoch} 轮  HR@10={best[0]:.4f}  NDCG@10={best[1]:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print("[权重] 已回滚到最佳轮次的参数，再保存与导出")
    else:
        print("[警告] 全程没有完成过评估，保存的仍是最后一轮权重")

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "ncf_model.pt")
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    rec_path = os.path.join(args.out_dir, "rec_result.csv")

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"NeuMF  HR@10={best[0]:.4f}  NDCG@10={best[1]:.4f}  "
                f"best_epoch={best_epoch}  epochs={args.epochs}  "
                f"dim={args.dim}  neg_ratio={args.neg_ratio}  seed={args.seed}\n")
    print(f"[保存] {model_path} / {metrics_path}")

    export_topk(model, n_users, n_items, all_user_items, users_orig, items_orig,
                k=args.topk, out=rec_path, device=device)


if __name__ == "__main__":
    main()
