"""实时口味重排的离线对照实验（EXP-04）。

背景与要回答的问题
==================
`app.recommend.taste-rerank` 默认关闭，因为它要改的是"线上顺序直接取召回结果"这条口径（R16），
而那条口径是**用实验立的**（EXP-03：LLM 重排显著变差）。所以这一项也必须先拿证据，
再决定默认值。实验要回答三个问题：

1. **能顶多远**：口味权重给到多少，才会把召回里靠后的片子顶进前 10？
2. **代价多大**：模型分（协同过滤辛苦学出来的排序信号）损失了多少？
3. **代价换来了什么**：口味匹配率与类目多样性各提升了多少？

方法（为什么这么设计）
======================
- **对照必须固定候选池**：同一批用户、同一个召回 Top-50，只允许顺序不同 ——
  否则比较的就不是"重排"而是"另一套召回"。这与 EXP-03 的方法论一致。
- **不做 HR@10 类指标**：rec_result **本身就排除了用户评分过的影片**
  （`db/verify.sql` D2 断言为 0），所以"用户喜欢的某部片"不可能出现在候选池里，
  救回率类的指标在这套数据上没有定义 —— 硬算只会得到一个恒为 0 的假结论。
  因此改用**配对的可比指标**（见下），它们在同一候选池上都有定义、且能直接回答上面的问题。
- **没有信息泄漏**：口味信号取自用户的评分历史，而候选池里**不含任何他评分过的影片**，
  两边天然不相交，不存在"用答案算答案"。
- **权重扫描**而不是单点：只有这样才知道"先撞到天花板的是收益还是代价"。

指标（都是"前 10 条"的均值）
============================
| 指标 | 含义 | 期望方向 |
|---|---|---|
| `model_keep` | 前 10 的归一化模型分均值（min-max 后）| 越高越好（= 代价越小）|
| `taste_match` | 前 10 的口味匹配率均值（命中权重 ÷ 全部权重）| 越高越好（= 收益）|
| `ild` | 前 10 的类型多样性（两两不同类型的对数占比）| 越高越好（= 别越推越窄）|
| `churn` | 前 10 与基线不同的条数（0~10）| 反映"改了多少"，本身无好坏 |

用法
====
    python script/taste_rerank_ablation.py --sample 300 --seed 42
    python script/taste_rerank_ablation.py --weights 0,0.1,0.2,0.3,0.5   # 自定义扫描点

只依赖 pymysql（与 db/verify.sql 的执行环境一致）。**全流程只读**，不写任何表。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from itertools import combinations

import pymysql

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'backend', 'src', 'main', 'resources', 'application-local.yml')


def connect():
    """按后端同一份 application-local.yml 连库（与 db/ 下其它脚本一致的做法）。

    ★ 已装依赖只有 pymysql，所以本脚本**刻意不依赖 pandas / numpy / scipy** ——
      配对检验用标准库自己实现（符号检验是二项分布，math.comb 就够）。
    """
    cfg = open(CONFIG_PATH, encoding='utf-8').read()
    url = re.search(r'url:\s*"([^"]+)"', cfg).group(1)
    user = re.search(r'username:\s*(\S+)', cfg).group(1)
    pwd = re.search(r'password:\s*"([^"]+)"', cfg).group(1)
    db = re.search(r'/(\w+)\?', url).group(1)
    return pymysql.connect(host='127.0.0.1', port=3306, user=user,
                           password=pwd, database=db, charset='utf8mb4')


def load_genre_map(cur):
    """movie_id → 类型 id 集合（一次读全量，内存里建索引；数据量小、避免 N+1）。"""
    cur.execute('SELECT movie_id, genre_id FROM movie_genre')
    genres = {}
    for movie_id, genre_id in cur.fetchall():
        genres.setdefault(movie_id, set()).add(genre_id)
    return genres


def load_users(cur, sample, seed):
    """确定性抽样：user_id 升序取模，不用 RAND()（要可复现，也要能在论文里写清口径）。"""
    cur.execute('SELECT DISTINCT user_id FROM rec_result ORDER BY user_id')
    users = [row[0] for row in cur.fetchall()]
    step = max(1, len(users) // sample)
    picked = users[::step][:sample]
    print(f'  抽样：{len(users)} 位有离线结果的用户中取 {len(picked)} 位'
          f'（每 {step} 位取一位，user_id 升序；seed={seed} 仅记录、不参与抽样）')
    return picked


def load_pool(cur, user_id):
    """该用户最新批次的召回 Top-50：[(movie_id, score)]，按名次升序。"""
    cur.execute(
        'SELECT movie_id, score FROM rec_result WHERE user_id = %s '
        'AND batch_date = (SELECT MAX(batch_date) FROM rec_result WHERE user_id = %s) '
        'ORDER BY rank_num ASC LIMIT 50', (user_id, user_id))
    return [(row[0], float(row[1])) for row in cur.fetchall()]


def load_taste(cur, user_id, recent, min_score):
    """口味权重：最近 N 条评分里 ≥min_score 的影片，按类型计数。返回 {genre_id: 权重}。"""
    cur.execute('SELECT movie_id, score FROM rating WHERE user_id = %s '
                'ORDER BY rate_time DESC, id DESC LIMIT %s', (user_id, recent))
    liked = [row[0] for row in cur.fetchall() if row[1] is not None and row[1] >= min_score]
    return liked


def norm_scores(pool):
    """min-max 归一化（与线上实现同一口径）。★ 不能用"除以最大值"：
    rec_result.score 是 NeuMF 的 logit（实测榜首 3.2356），出现负值时那种做法会反向排序。"""
    values = [s for _, s in pool]
    lo, hi = min(values), max(values)
    span = max(1e-9, hi - lo)
    return {movie_id: (score - lo) / span for movie_id, score in pool}


def taste_ratio(movie_id, taste, genres, total):
    if total == 0:
        return 0.0
    hit = sum(taste.get(g, 0) for g in genres.get(movie_id, ()))
    return min(1.0, hit / total)


def ild(top_ids, genres):
    """类目多样性：两两不同类型（无交集）的对数占比。1 = 全不同，0 = 全同类。"""
    pairs = list(combinations(top_ids, 2))
    if not pairs:
        return 0.0
    different = sum(1 for a, b in pairs if not (genres.get(a, set()) & genres.get(b, set())))
    return different / len(pairs)


def sign_test_p(wins, losses):
    """配对符号检验（双侧精确二项检验）。只报"方向是否稳定"，不假装成显著性结论。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def overlap(movie_id, chosen, genres):
    """这部片与"已选项"的最大类型重合度（Jaccard，0~1）。没有类型信息时记 0。"""
    g1 = genres.get(movie_id, set())
    if not g1:
        return 0.0
    best = 0.0
    for other in chosen:
        g2 = genres.get(other, set())
        if not g2:
            continue
        inter = len(g1 & g2)
        if inter:
            best = max(best, inter / len(g1 | g2))
    return best


def mmr_order(pool, norm, taste, genres, taste_total, w_taste, lam, window=20):
    """候选池的展示顺序。

    **前 window 条走 MMR 贪心**（每轮取 argmax(相关性 − λ·与已选项的类型重合度)），
    其余按相关性序接上 —— 池尾的顺序对首页没有影响，不必为它付 O(n³) 的代价。

    ★ 为什么要 MMR：EXP-04 的第一条结论是"按口味重排会把类目多样性压掉 13%~36%"。
    打分只问"这条像不像你喜欢的"，MMR 再问一句"它跟前面几条是不是**重复**" ——
    推荐的价值本来有一部分在"发现"上，这一句就是为它加的。
    """
    relevance = {m: (1 - w_taste) * norm[m] + w_taste * taste_ratio(m, taste, genres, taste_total)
                 for m, _ in pool}
    order_index = {m: i for i, (m, _) in enumerate(pool)}
    remaining = [m for m, _ in pool]
    chosen = []
    while remaining and len(chosen) < window:
        best = max(remaining, key=lambda m: (relevance[m] - lam * overlap(m, chosen, genres),
                                             -order_index[m]))
        chosen.append(best)
        remaining.remove(best)
    chosen.extend(sorted(remaining, key=lambda m: (-relevance[m], order_index[m])))
    return chosen


def evaluate(pool, taste_total, taste, genres, configs):
    """在一个候选池上跑 (w_taste, λ) 扫描，返回 {(w, lam): 指标字典}。"""
    norm = norm_scores(pool)
    baseline = [movie_id for movie_id, _ in pool][:10]
    out = {}
    for w, lam in configs:
        order = mmr_order(pool, norm, taste, genres, taste_total, w, lam)
        top = order[:10]
        out[(w, lam)] = {
            'model_keep': statistics.fmean(norm[m] for m in top),
            'taste_match': statistics.fmean(taste_ratio(m, taste, genres, taste_total) for m in top),
            'ild': ild(top, genres),
            'churn': len(set(baseline) ^ set(top)) // 2,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description='实时口味重排的离线对照（EXP-04）')
    ap.add_argument('--sample', type=int, default=300, help='抽样用户数')
    ap.add_argument('--seed', type=int, default=42, help='记录用（本脚本用确定性抽样）')
    ap.add_argument('--recent', type=int, default=30, help='口味信号取最近多少条评分')
    ap.add_argument('--min-score', type=int, default=4, help='视为喜欢的分数下限')
    ap.add_argument('--min-ratings', type=int, default=3, help='少于这么多条喜欢的评分则跳过该用户')
    ap.add_argument('--weights', default='0,0.1,0.2,0.25,0.35,0.5',
                    help='口味权重扫描点（逗号分隔，0 = 不做重排的基线）')
    ap.add_argument('--lambdas', default='0,0.1,0.2,0.3',
                    help='MMR 多样性惩罚 λ 的扫描点（0 = 不做多样性约束）')
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)),
                    help='结果 CSV 输出目录（默认 script/，与 EXP-01/03 同一套惯例）')
    ap.add_argument('--no-out', action='store_true', help='只打印、不写 CSV')
    args = ap.parse_args()
    weights = [float(w) for w in args.weights.split(',')]
    lambdas = [float(x) for x in args.lambdas.split(',')]
    configs = [(w, lam) for w in weights for lam in lambdas]

    conn = connect()
    cur = conn.cursor()
    genres = load_genre_map(cur)
    users = load_users(cur, args.sample, args.seed)

    rows = []
    user_ids = []
    skipped = 0
    for user_id in users:
        pool = load_pool(cur, user_id)
        if len(pool) < 10:
            skipped += 1
            continue
        liked = load_taste(cur, user_id, args.recent, args.min_score)
        if len(liked) < args.min_ratings:
            skipped += 1
            continue
        taste = {}
        for movie_id in liked:
            for g in genres.get(movie_id, ()):
                taste[g] = taste.get(g, 0) + 1
        rows.append((pool, sum(taste.values()), taste))
        user_ids.append(user_id)
    print(f'  参与评测的用户 {len(rows)} 位（跳过 {skipped} 位：召回不足 10 条或口味信号太薄）')

    print()
    print()
    print(f'  {"w_taste":>7} {"λ":>5} │ {"model_keep":>10} │ {"taste_match":>11} │ {"ILD":>6} │ {"churn":>5}')
    print('  ' + '-' * 58)
    per_cfg = {}
    detail_rows = []
    for cfg in configs:
        metrics = [evaluate(pool, total, taste, genres, [cfg])[cfg] for pool, total, taste in rows]
        per_cfg[cfg] = metrics
        w, lam = cfg
        means = {k: statistics.fmean(m[k] for m in metrics)
                 for k in ('model_keep', 'taste_match', 'ild', 'churn')}
        print(f'  {w:>7.2f} {lam:>5.2f} │ {means["model_keep"]:>10.4f} │'
              f' {means["taste_match"]:>11.4f} │ {means["ild"]:>6.3f} │ {means["churn"]:>5.2f}')
        for user_id, m in zip(user_ids, metrics):
            detail_rows.append({'user_id': user_id, 'w_taste': w, 'lambda': lam, **m})

    base = per_cfg[configs[0]]
    base_mean = {k: statistics.fmean(m[k] for m in base)
                 for k in ('model_keep', 'taste_match', 'ild', 'churn')}
    print()
    print('  相对基线（w=0, λ=0）—— 达标定义：口味匹配率↑、多样性不低于基线、模型分落幅 <1%、改动 >=1 条')
    print(f'  {"w_taste":>7} {"λ":>5} │ {"Δmodel":>8} │ {"Δtaste":>8} │ {"ΔILD":>7} │ {"churn":>5} │ 判定')
    print('  ' + '-' * 58)
    summary = []
    for cfg in configs:
        means = {k: statistics.fmean(m[k] for m in per_cfg[cfg])
                 for k in ('model_keep', 'taste_match', 'ild', 'churn')}
        row = {'w_taste': cfg[0], 'lambda': cfg[1], **means}
        row['d_model'] = means['model_keep'] - base_mean['model_keep']
        row['d_taste'] = means['taste_match'] - base_mean['taste_match']
        row['d_ild'] = means['ild'] - base_mean['ild']
        row['ok'] = (row['d_taste'] > 0 and row['d_ild'] >= 0
                     and row['d_model'] >= -0.01 and row['churn'] >= 1.0)
        row['verdict'] = '达标' if row['ok'] else '-'
        summary.append(row)
        print(f'  {cfg[0]:>7.2f} {cfg[1]:>5.2f} │ {row["d_model"]:>+8.4f} │'
              f' {row["d_taste"]:>+8.4f} │ {row["d_ild"]:>+7.3f} │'
              f' {row["churn"]:>5.2f} │ {row["verdict"]}')

    ok_rows = [r for r in summary if r['ok']]
    print()
    print('  逐用户配对符号检验（相对基线，双侧精确二项检验）——只列达标组合，最多 4 个：')
    if not ok_rows:
        print('    （没有达标组合：多样性代价压不下去，或改动量不足）')
    for row in ok_rows[:4]:
        cfg = (row['w_taste'], row['lambda'])
        for metric, label in (('model_keep', '模型分保持'), ('taste_match', '口味匹配率'),
                              ('ild', '类目多样性')):
            wins = sum(1 for a, b in zip(per_cfg[cfg], base) if a[metric] > b[metric] + 1e-12)
            losses = sum(1 for a, b in zip(per_cfg[cfg], base) if a[metric] < b[metric] - 1e-12)
            print(f'    w={row["w_taste"]:<4} λ={row["lambda"]:<4} {label:<6}:'
                  f' 升 {wins:>3} / 降 {losses:>3} / 平 {len(per_cfg[cfg]) - wins - losses:>3}'
                  f'   p={sign_test_p(wins, losses):.2e}')

    if not args.no_out:
        result_path = os.path.join(args.out_dir, 'taste_rerank_result.csv')
        detail_path = os.path.join(args.out_dir, 'taste_rerank_detail.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as f:
            cols = ['w_taste', 'lambda', 'model_keep', 'taste_match', 'ild', 'churn',
                    'd_model', 'd_taste', 'd_ild', 'verdict']
            f.write(','.join(cols) + '\n')
            for row in summary:
                f.write(','.join(
                    f'{row[c]:.4f}' if isinstance(row[c], float) else str(row[c]) for c in cols) + '\n')
        with open(detail_path, 'w', encoding='utf-8', newline='') as f:
            cols = ['user_id', 'w_taste', 'lambda', 'model_keep', 'taste_match', 'ild', 'churn']
            f.write(','.join(cols) + '\n')
            for row in detail_rows:
                f.write(','.join(
                    f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c]) for c in cols) + '\n')
        print()
        print(f'  结果文件：{result_path}')
        print(f'  明细文件：{detail_path}（每人一行 × 每个组合，可复核、可画图）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
