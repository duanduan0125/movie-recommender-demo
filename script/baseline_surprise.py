#!/usr/bin/env python
"""
评分预测基线（scikit-surprise）—— 实验一的数据来源

用途：UserCF / ItemCF / SVD / NMF / BaselineOnly 的 RMSE 与 MAE。

★ 修订说明（设计文档_修订版 §3.4）。原实现有四处会影响结论可信度的问题：

  1. 每个算法用了**不同的随机划分**（最隐蔽的一处）
     原代码 `cross_validate(algo, data, cv=5)` 传的是整数，surprise 会自己
     构造 shuffle=True、random_state=None 的 KFold —— 于是五个算法各自
     随机切了一遍数据。RMSE 之间的差异里混进了"划分不同"带来的噪声，
     严格来说这些数字不可直接比较。现在统一传同一个 KFold 对象。

  2. KNNBasic → KNNWithMeans
     KNNBasic 是原始加权平均，ml-1m 上 RMSE 通常 0.97~0.98，
     与文档给出的 0.92 锚点对不上，会让人误判"自己跑错了"；
     0.92 量级对应的是 KNNWithMeans。

  3. 固定随机种子
     SVD / NMF 的随机初始化会让每次结果都不同，论文数据不可复现。
     现在统一 random_state=42（交叉验证与算法本身都固定）。

  4. 折数默认从 5 降到 3
     KNN 与 NMF 在 100 万评分上跑 5 折非常慢（实测几十分钟量级，
     不是原注释写的"几分钟"）。需要 5 折时显式传 --cv 5。

  5. 逐折进度默认打开（--quiet 可关）
     UserCF 在 100 万评分上单折就要几分钟（实测 3 折共 709.6 秒），
     只打印"开始/完成"会让中间十几分钟毫无输出、看起来像卡死 ——
     设计文档 §3.4 第 5 条明确要求打印进度。

基线范围：
    BaselineOnly / UserCF / ItemCF / SVD / NMF 五项，
    与开题报告的基线清单及实验一的表格逐项对应。

用法：
    python baseline_surprise.py
    python baseline_surprise.py --cv 5
    python baseline_surprise.py --quiet          # 不要逐折进度

产出：控制台对比表 + baseline_result.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from surprise import BaselineOnly, Dataset, KNNWithMeans, NMF, Reader, SVD
from surprise.model_selection import KFold, cross_validate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import check_dataset, load_ratings  # noqa: E402  与 Top-N 评测共用读取与自检

DEFAULT_DATA_DIR = "./ml-1m"


def build_algos() -> dict:
    """
    基线五项：BaselineOnly / UserCF / ItemCF / SVD / NMF。

    UserCF / ItemCF 用 KNNWithMeans（不是 KNNBasic，理由见模块说明）；
    SVD / NMF 固定随机种子，保证可复现。
    """
    return {
        "BaselineOnly": BaselineOnly(),
        "UserCF": KNNWithMeans(
            sim_options={"name": "cosine", "user_based": True}, verbose=False),
        "ItemCF": KNNWithMeans(
            sim_options={"name": "cosine", "user_based": False}, verbose=False),
        "SVD": SVD(random_state=42),
        "NMF": NMF(random_state=42),
    }


def build_parser() -> argparse.ArgumentParser:
    """命令行定义单独抽出来，便于单测直接验证参数名与默认值。"""
    ap = argparse.ArgumentParser(description="评分预测基线对比（实验一）")
    ap.add_argument("--data-dir", "--data_dir", dest="data_dir",
                    default=DEFAULT_DATA_DIR,
                    help="ml-1m 目录（兼容旧的 --data_dir 写法）")
    ap.add_argument("--cv", type=int, default=3,
                    help="交叉验证折数。默认 3；KNN/NMF 在 100 万评分上跑 5 折很慢，"
                         "需要与文档一致时显式传 5")
    ap.add_argument("--seed", type=int, default=42)
    # 设计文档 §3.4 要求「打印进度（verbose），否则长时间无输出会误以为卡死」。
    # UserCF 在 100 万评分上单折就要几分钟，所以逐折进度默认打开。
    ap.add_argument("--quiet", action="store_true",
                    help="关闭交叉验证的逐折进度输出（默认打开）")
    ap.add_argument("--out", default="baseline_result.csv")
    return ap


def main():
    args = build_parser().parse_args()

    df = load_ratings(args.data_dir)
    check_dataset(df)
    print(f"[数据] 评分 {len(df)} | 用户 {df['user_id'].nunique()} | "
          f"影片 {df['item_id'].nunique()}")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    # ★ 所有算法共用同一个 KFold 对象：固定 seed 后每次迭代给出完全相同的划分，
    #   这样算法之间的 RMSE 差异才只来自算法本身。
    cv = KFold(n_splits=args.cv, random_state=args.seed, shuffle=True)
    print(f"[划分] {args.cv} 折交叉验证，shuffle=True，random_state={args.seed}"
          f"（所有算法共用同一划分）")

    algos = build_algos()
    print(f"[基线] 共 {len(algos)} 项：{' / '.join(algos)}")

    rows = []
    for name, algo in algos.items():
        t0 = time.time()
        print(f"[{name}] 开始 {args.cv} 折交叉验证 ...")
        res = cross_validate(algo, data, measures=["RMSE", "MAE"],
                             cv=cv, verbose=not args.quiet)
        row = {
            "算法": name,
            "RMSE": round(float(res["test_rmse"].mean()), 4),
            "MAE": round(float(res["test_mae"].mean()), 4),
            "耗时(s)": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  完成 {name}：RMSE={row['RMSE']}  MAE={row['MAE']}  "
              f"耗时 {row['耗时(s)']}s")

    out = pd.DataFrame(rows)
    print("\n================ 对比实验结果（实验一：评分预测） ================")
    print(out.to_string(index=False))
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {args.out}")

    # 自查锚点：SVD 的 RMSE 应在 0.87 附近，明显偏差说明流程有问题
    svd = next((r for r in rows if r["算法"] == "SVD"), None)
    if svd:
        if 0.84 <= svd["RMSE"] <= 0.92:
            print(f"[自查] SVD RMSE={svd['RMSE']} 落在 ml-1m 的常见区间（≈0.87），"
                  f"流程正确")
        else:
            print(f"[自查] SVD RMSE={svd['RMSE']} 偏离常见区间（≈0.87），"
                  f"请检查数据与划分")


if __name__ == "__main__":
    main()
