#!/usr/bin/env python3
"""
批量对「一次大规模采集」根目录下的每个实验子文件夹调用 calculate_fall_risk.py，
合并各次 summary，并写出整体 / 按推力方向的统计。

典型目录结构（与 automated_collection*.sh 一致）::

    /path/to/only_active_push_1600/8dir-200.0N-0.4s-20260317_235134/
        forward-200.0N-1/
            20260317_235138/   # 实际 csv/bin 常在这一层（连续采集切换路径时）
        forward_left-200.0N-2/
        ...

支持两种布局：日志直接在 ``forward-200.0N-1/`` 下，或在 ``forward-200.0N-1/<时间戳>/`` 下。

用法示例::

    cd /path/to/engineai_ros2_workspace
    python3 scripts/batch_calculate_fall_risk.py \\
        --root /home/wang22/data/mujoco_logs/only_active_push_1600/8dir-200.0N-0.4s-20260317_235134

单次 risk 结果仍在各实验子目录（与原始 csv/bin 同文件夹）。
批量汇总默认写在 --root（与各子实验文件夹同级）::

    all_runs_summary.csv   # 1 行：n_runs + 各分项跨 run 的 mean/median/…（列名如 contact_risk_median；单次 summary 仅 episode 标量无此类列）
    per_run_risk.csv       # 每行一次实验的明细（原合并表）
    stats_by_direction.csv
    stats_overall.txt
    policy_switch_walking_combined.csv  # 各子目录 policy_switch 中仅 from_mode=walking 的行（可加 --no-combine-policy-switch 跳过）

长时 calculate_fall_risk 子进程运行期间，stderr 会周期性刷新一行，显示**当前子任务已用时**与**本批预计剩余**（不含当前子任务，有历史平均耗时后才有）。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 与 calculate_fall_risk 同目录，保证可 import mujoco_data_io
SCRIPTS_DIR = Path(__file__).resolve().parent
CALCULATE_FALL_RISK = SCRIPTS_DIR / "calculate_fall_risk.py"

try:
    from mujoco_data_io import read_binary_policy_switch, resolve_paths_from_log_dir
except ImportError:
    resolve_paths_from_log_dir = None  # type: ignore
    read_binary_policy_switch = None  # type: ignore


def summarize_scalar_across_runs(values: np.ndarray) -> Dict[str, float]:
    """
    对多次实验的同一标量 risk（如 contact_risk）做跨 run 描述统计。
    键名用于 all_runs_summary：`contact_risk_median` 等（跨多次实验）；单次 `*_summary.csv` 无此类列，仅有 `contact_risk` 等 episode 标量。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    p01, p05, p25, p50, p75, p95, p99 = (
        float(np.quantile(v, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    )
    return {
        "mean": float(np.mean(v)),
        "median": p50,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "iqr": float(p75 - p25),
        "p25": p25,
        "p75": p75,
        "p01": p01,
        "p05": p05,
        "p95": p95,
        "p99": p99,
        "p95_minus_p05": float(p95 - p05),
        "p99_minus_p01": float(p99 - p01),
    }


def is_valid_run_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    if resolve_paths_from_log_dir is None:
        return any(d.glob("contact_data*.csv")) or any(d.glob("contact_data*.bin"))
    r = resolve_paths_from_log_dir(str(d))
    need = ("contact", "sensor_vibration", "joint_state", "joint_forces")
    return all(r.get(k) for k in need)


def parse_run_folder_name(name: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    例如 forward_left-200.0N-42 -> direction=forward_left, force=200.0, run_index=42
    解析失败则 direction 为原名，force/run_index 为 None。
    """
    m = re.match(r"^(.+)-([\d.]+)N-(\d+)$", name)
    if not m:
        return name, None, None
    return m.group(1), m.group(2), int(m.group(3))


def discover_run_dirs(root: Path) -> List[Path]:
    """收集含完整日志的目录：一级子目录，或 一级/时间戳 二级目录。"""
    found: List[Path] = []
    subs = sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    for p in subs:
        if is_valid_run_dir(p):
            found.append(p)
            continue
        nested = sorted(
            [q for q in p.iterdir() if q.is_dir() and not q.name.startswith(".")],
            key=lambda q: q.name,
        )
        for q in nested:
            if is_valid_run_dir(q):
                found.append(q)

    def sort_key(path: Path) -> Tuple[int, str]:
        rel = path.relative_to(root)
        idx = 10**9
        for part in rel.parts:
            _, _, j = parse_run_folder_name(part)
            if j is not None:
                idx = j
                break
        return (idx, str(rel))

    return sorted(found, key=sort_key)


def _norm_policy_mode_cell(x: object) -> str:
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x).strip().strip('"').strip("'")


def combine_policy_switch_walking_rows(
    root: Path,
    out_dir: Path,
    run_dirs: List[Path],
) -> Tuple[int, Optional[Path]]:
    """
    合并各实验目录下的 policy_switch.csv（或 .bin）中 **from_mode == walking** 的行到单一 CSV。

    写入 ``out_dir / "policy_switch_walking_combined.csv"``，并增加列 run_folder、direction、log_path。
    用于分析「从行走/RL 策略切出」的事件（如摔倒切 damping、定时 pdstand）。
    """
    parts: List[pd.DataFrame] = []
    for log_dir in run_dirs:
        csv_p = log_dir / "policy_switch.csv"
        bin_p = log_dir / "policy_switch.bin"
        df: Optional[pd.DataFrame] = None
        if csv_p.is_file():
            try:
                df = pd.read_csv(csv_p)
            except Exception as e:
                print(f"警告: 无法读取 {csv_p}: {e}", file=sys.stderr)
                continue
        elif bin_p.is_file() and read_binary_policy_switch is not None:
            try:
                df = read_binary_policy_switch(str(bin_p))
            except Exception as e:
                print(f"警告: 无法读取 {bin_p}: {e}", file=sys.stderr)
                continue
        else:
            continue
        if df is None or df.empty:
            continue
        if "from_mode" not in df.columns:
            print(f"警告: {log_dir} policy_switch 无 from_mode 列，跳过", file=sys.stderr)
            continue
        sub = df[df["from_mode"].map(_norm_policy_mode_cell) == "walking"].copy()
        if sub.empty:
            continue
        run_folder, direction, _, _ = meta_from_log_dir(log_dir, root)
        sub.insert(0, "run_folder", run_folder)
        sub.insert(1, "direction", direction)
        sub.insert(2, "log_path", str(log_dir))
        parts.append(sub)

    if not parts:
        print(
            "policy_switch 合并: 未发现 csv/bin，或各文件中无 from_mode=walking 的行；跳过写出。",
            file=sys.stderr,
        )
        return 0, None

    merged = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    out_path = out_dir / "policy_switch_walking_combined.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n已写入: {out_path}（仅 from_mode=walking，共 {len(merged)} 行）")
    return len(merged), out_path


def meta_from_log_dir(log_dir: Path, root: Path) -> Tuple[str, str, Optional[str], Optional[int]]:
    """run_folder 相对 root；direction/force/run_index 从路径里形如 xxx-200.0N-12 的一段解析。"""
    try:
        run_folder = str(log_dir.relative_to(root))
    except ValueError:
        run_folder = log_dir.name
    direction: str = run_folder
    force_n: Optional[str] = None
    run_idx: Optional[int] = None
    for part in Path(run_folder).parts:
        d, f, r = parse_run_folder_name(part)
        if r is not None:
            direction, force_n, run_idx = d, f, r
            break
    return run_folder, direction, force_n, run_idx


def _format_eta(sec: Optional[float]) -> str:
    if sec is None or sec < 0 or (sec != sec):  # nan
        return "—"
    if sec >= 3600:
        h, r = int(sec // 3600), sec % 3600
        return f"{h}h{int(r // 60)}m"
    if sec >= 60:
        m, s = int(sec // 60), sec % 60
        return f"{m}m{int(s)}s"
    return f"{sec:.0f}s"


def run_single_calculate(
    log_dir: Path,
    *,
    ticker_interval_sec: float = 0.0,
    batch_eta_after_current_sec: Optional[float] = None,
) -> int:
    """
    调用 calculate_fall_risk.py。ticker_interval_sec>0 时，在子进程运行期间每隔该秒数向 stderr 用 \\r 刷新一行，
    输出当前子任务已用时与 batch_eta_after_current_sec（本批剩余、不含当前子任务，可为 None）。
    """
    cmd = [sys.executable, str(CALCULATE_FALL_RISK), "--log-dir", str(log_dir)]
    env = {**os.environ}
    if ticker_interval_sec <= 0:
        r = subprocess.run(cmd, cwd=str(SCRIPTS_DIR), env=env)
        return r.returncode

    stop = threading.Event()
    t0 = time.perf_counter()

    def ticker() -> None:
        while not stop.wait(ticker_interval_sec):
            cur = time.perf_counter() - t0
            eta_part = ""
            if batch_eta_after_current_sec is not None and batch_eta_after_current_sec >= 0:
                eta_part = (
                    f" | 本批剩余(不含当前) 约 {_format_eta(batch_eta_after_current_sec)} "
                    f"({batch_eta_after_current_sec:.0f}s)"
                )
            pad = " " * 4
            print(
                f"\r  [batch] calculate_fall_risk 运行中… 当前子任务已用时 {_format_eta(cur)} ({cur:.0f}s)"
                f"{eta_part}{pad}",
                end="",
                flush=True,
                file=sys.stderr,
            )

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        r = subprocess.run(cmd, cwd=str(SCRIPTS_DIR), env=env)
        return r.returncode
    finally:
        stop.set()
        th.join(timeout=2.0)
        print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量对实验子目录调用 calculate_fall_risk.py 并汇总统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="采集批次根目录，其下每个子文件夹为一次实验（含 contact/joint_state 等）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="汇总 CSV/统计输出目录，默认与 --root 相同（与各实验子目录同级）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将处理的子目录，不调用 calculate_fall_risk",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若子目录已有 risk_results_summary.csv 则跳过计算（仅合并/统计时用已有结果）",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="最多处理前 N 个子目录，0 表示不限制（调试用）",
    )
    parser.add_argument(
        "--no-continue-on-error",
        action="store_true",
        help="任一次 calculate 失败则立即退出（默认：失败跳过，继续其余目录）",
    )
    parser.add_argument(
        "--no-combine-policy-switch",
        action="store_true",
        help="不合并各子目录 policy_switch 中 from_mode=walking 的行到 policy_switch_walking_combined.csv",
    )
    parser.add_argument(
        "--progress-ticker-interval",
        type=float,
        default=10.0,
        metavar="SEC",
        help="子进程 calculate_fall_risk 运行期间，每隔 SEC 秒在 stderr 刷新一行（已用时/本批剩余）；0 关闭",
    )
    parser.add_argument(
        "--no-progress-ticker",
        action="store_true",
        help="等价于 --progress-ticker-interval 0，不显示周期性剩余时间行",
    )
    args = parser.parse_args()
    if args.no_progress_ticker:
        args.progress_ticker_interval = 0.0
    continue_on_error = not args.no_continue_on_error

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"错误: 不是目录: {root}", file=sys.stderr)
        return 1

    if not CALCULATE_FALL_RISK.is_file():
        print(f"错误: 未找到 {CALCULATE_FALL_RISK}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root
    if args.output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        n1 = sum(1 for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        print(f"在 {root} 下未发现含完整日志的目录（一级子目录数: {n1}）。")
        print(
            "需要每个实验目录（或 实验/时间戳 子目录）下同时存在 "
            "contact_data*、sensor_vibration_data*、joint_state_data*、joint_forces_data*（csv 或 bin）。",
            file=sys.stderr,
        )
        return 1

    if args.max_runs > 0:
        run_dirs = run_dirs[: args.max_runs]

    print(f"根目录: {root}")
    print(f"将处理 {len(run_dirs)} 个实验子目录")
    print(f"汇总输出目录: {out_dir}")

    if args.dry_run:
        for p in run_dirs:
            try:
                print(f"  {p.relative_to(root)}")
            except ValueError:
                print(f"  {p}")
        return 0

    failed: List[str] = []
    summary_paths: List[Path] = []

    n_total = len(run_dirs)

    def needs_subprocess(ld: Path) -> bool:
        sf = ld / "risk_results_summary.csv"
        if args.skip_existing and sf.is_file():
            return False
        return True

    def count_pending_calc(from_index_0: int) -> int:
        return sum(1 for j in range(from_index_0, n_total) if needs_subprocess(run_dirs[j]))

    calc_durations: List[float] = []
    t_batch0 = time.perf_counter()

    for i, log_dir in enumerate(run_dirs, 1):
        summary_file = log_dir / "risk_results_summary.csv"
        label = str(log_dir.relative_to(root))
        idx0 = i - 1
        remaining_dirs = n_total - i + 1
        pending_calc = count_pending_calc(idx0)
        if calc_durations:
            avg_s = sum(calc_durations) / len(calc_durations)
            # 完成当前项后还剩多少次 subprocess
            after_this = pending_calc - (1 if needs_subprocess(log_dir) else 0)
            eta_sec = max(0.0, avg_s * after_this)
            eta_after_current_sec = max(0.0, avg_s * max(0, pending_calc - 1))
        else:
            eta_sec = None
            eta_after_current_sec = None

        elapsed_batch = time.perf_counter() - t_batch0
        print(
            f"  [进度] 目录剩余 {remaining_dirs}/{n_total} | "
            f"待计算子进程 {pending_calc} 次 | "
            f"已用 {_format_eta(elapsed_batch)} | "
            f"预计剩余 {_format_eta(eta_sec)}",
            flush=True,
        )

        if args.skip_existing and summary_file.is_file():
            print(f"[{i}/{n_total}] 跳过(已有summary): {label}")
            summary_paths.append(summary_file)
            continue

        print(f"[{i}/{n_total}] 计算: {label}")
        t0 = time.perf_counter()
        code = run_single_calculate(
            log_dir,
            ticker_interval_sec=float(args.progress_ticker_interval),
            batch_eta_after_current_sec=eta_after_current_sec,
        )
        dt = time.perf_counter() - t0
        calc_durations.append(dt)
        print(f"  本次耗时 {dt:.1f}s | 最近平均 {sum(calc_durations)/len(calc_durations):.1f}s/次", flush=True)
        if code != 0:
            print(f"  失败 (exit {code}): {label}", file=sys.stderr)
            failed.append(label)
            if not continue_on_error:
                return code
            continue
        if summary_file.is_file():
            summary_paths.append(summary_file)
        else:
            print(f"  警告: 未生成 {summary_file}", file=sys.stderr)
            failed.append(label)

    if not args.no_combine_policy_switch:
        combine_policy_switch_walking_rows(root, out_dir, run_dirs)

    rows = []
    for sp in summary_paths:
        try:
            df = pd.read_csv(sp)
        except Exception as e:
            print(f"警告: 无法读取 {sp}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        log_dir = sp.parent
        run_folder, direction, force_n, run_idx = meta_from_log_dir(log_dir, root)
        row["run_folder"] = run_folder
        row["direction"] = direction
        row["force_N"] = force_n
        row["run_index"] = run_idx
        rows.append(row)

    if not rows:
        print("没有可合并的 summary 行。", file=sys.stderr)
        return 1 if failed else 0

    merged = pd.DataFrame(rows)
    id_cols = ["run_folder", "direction", "force_N", "run_index"]
    rest = [c for c in merged.columns if c not in id_cols]
    merged = merged[[c for c in id_cols if c in merged.columns] + rest]

    detail_csv = out_dir / "per_run_risk.csv"
    merged.to_csv(detail_csv, index=False, float_format="%.6f")
    print(f"\n已写入: {detail_csv} ({len(merged)} 行，逐实验明细)")

    # all_runs_summary：列名与单次 risk_results_summary 一致（contact_risk_mean 等）；contact_risk 仍为跨 run 均值（兼容旧脚本）
    scalar_risk_keys = [
        "contact_risk",
        "vibration_risk",
        "motor_risk",
        "jointforce_risk",
        "all_risk",
    ]
    _distribution_stat_names = (
        "mean",
        "median",
        "min",
        "max",
        "iqr",
        "p25",
        "p75",
        "p01",
        "p05",
        "p95",
        "p99",
        "p95_minus_p05",
        "p99_minus_p01",
    )
    summary_row: dict = {"n_runs": len(merged)}
    for k in scalar_risk_keys:
        if k not in merged.columns:
            continue
        s = pd.to_numeric(merged[k], errors="coerce").dropna()
        stats = summarize_scalar_across_runs(s.to_numpy(dtype=float))
        if not stats:
            continue
        summary_row[k] = stats["mean"]
        for name in _distribution_stat_names:
            summary_row[f"{k}_{name}"] = stats[name]
    summary_df = pd.DataFrame([summary_row])
    all_csv = out_dir / "all_runs_summary.csv"
    summary_df.to_csv(all_csv, index=False, float_format="%.6f")
    print(
        f"已写入: {all_csv}（全批次 1 行；分布列名与单次 summary 一致，取值为跨 run 统计）",
        flush=True,
    )

    if "all_risk" in merged.columns:
        g = merged.groupby("direction", dropna=False)
        agg = g["all_risk"].agg(["count", "mean", "std", "min", "max"])
        agg = agg.rename(columns={"count": "n"})
        agg_path = out_dir / "stats_by_direction.csv"
        agg.to_csv(agg_path, float_format="%.6f")
        print(f"已写入: {agg_path}")

    txt_lines = [
        f"batch_root: {root}",
        f"runs_merged: {len(merged)}",
        f"runs_failed: {len(failed)}",
    ]
    if failed:
        txt_lines.append("failed_folders: " + ", ".join(failed[:20]))
        if len(failed) > 20:
            txt_lines.append(f"... and {len(failed) - 20} more")

    desc = merged.describe(include="all")
    txt_path = out_dir / "stats_overall.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n\n")
        f.write(str(desc))
    print(f"已写入: {txt_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
