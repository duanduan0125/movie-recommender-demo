#!/usr/bin/env python
"""
ncf_train.py 的命令行参数口径测试。

★ 为什么用子进程而不是直接 import ncf_train
  ncf_train 在模块级 `import torch`。在同一个 pytest 进程里，torch 会在已经加载了
  （surprise/sklearn/scipy 带进来的）MKL、OpenMP 运行时之后再加载自己的那一份，
  实测会直接 **Fatal Python error: Aborted**（退出码 3，整个测试进程崩掉）。
  单独跑本文件是正常的、单独跑 torch 也是正常的，只有混在一起才崩 ——
  属于典型的原生库冲突，Python 层没法捕获。
  因此本文件全部通过 `subprocess` 调真实命令行来验证参数口径：
  既避开冲突，又顺便测了"用户实际敲的命令能不能被接受"。

为什么值得为参数写测试：参数名或默认值写错，"照着设计文档敲命令"会直接失败，
或者在更糟的情况下静默用了别的默认值 —— 而训练要跑 20 多分钟，发现就晚了。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "ncf_train.py")
TIMEOUT = 120


def run_cli(args: list[str], timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=HERE,
    )


def parser_snapshot() -> dict:
    """在子进程里把解析器的参数名与默认值导出成 JSON（子进程里加载 torch 无妨）"""
    code = (
        "import json,sys;sys.path.insert(0,'.');"
        "import ncf_train as m;"
        "p=m.build_parser();"
        "print(json.dumps({"
        "'options':[o for a in p._actions for o in a.option_strings],"
        "'defaults':{a.dest:a.default for a in p._actions if a.dest!='help'},"
        "}, ensure_ascii=False, default=str))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=TIMEOUT, cwd=HERE)
    assert r.returncode == 0, f"子进程导出参数失败：{r.stderr[-800:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_help_lists_documented_flags():
    """设计文档里出现的三种写法都要能被接受，否则照文档敲命令会直接报错"""
    r = run_cli(["--help"])
    assert r.returncode == 0
    for flag in ("--data-dir", "--data_dir", "--three-way",
                 "--out-dir", "--out_dir", "--epochs", "--topk", "--seed"):
        assert flag in r.stdout, f"帮助信息里缺少 {flag}"


def test_out_dir_defaults_to_current_directory():
    """原实现把输出路径写死在当前目录，默认值必须保持"."，不能悄悄改行为"""
    snap = parser_snapshot()
    assert snap["defaults"]["out_dir"] == "."


def test_out_flag_is_deliberately_absent():
    """
    刻意不提供 --out：另两个脚本的 --out 指「输出 CSV 的文件路径」，
    而本脚本要写三个文件（权重/指标/推荐结果），语义是目录。
    同名不同义会造成误操作，因此只提供 --out-dir。
    """
    options = parser_snapshot()["options"]
    assert "--out" not in options
    assert "--out-dir" in options and "--out_dir" in options


def test_defaults_match_documented_hyperparameters():
    """论文里写的是 20 轮 / Top-50 / seed 42 / dim 32，不要被默认值悄悄改掉"""
    d = parser_snapshot()["defaults"]
    assert (d["epochs"], d["topk"], d["seed"], d["dim"]) == (20, 50, 42, 32)


def test_three_way_flag_defaults_off():
    """三路划分要显式打开：默认关闭时模型会比基线多看一条交互（不对称）"""
    assert parser_snapshot()["defaults"]["three_way"] is False


def test_unknown_flag_exits_with_error():
    r = run_cli(["--definitely-not-a-flag"])
    assert r.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
