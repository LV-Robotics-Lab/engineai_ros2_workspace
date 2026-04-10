#!/usr/bin/env python3
"""
Fig 2-3：四种摔倒工况下的分项 risk 对比（仅跨 run 平均数，无误差棒）。
数据：dataset_risk.csv 中 ``*_risk_mean``；绘图：plot_grouped_risk_bars.plot_grouped_risk_bars_pub。
PNG 写入仓库根目录 ``OutputFigures/``（该目录已加入 .gitignore）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OUTPUT_FIGURES = _REPO_ROOT / "OutputFigures"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_grouped_risk_bars import plot_grouped_risk_bars_pub

_ROW_NAMES = [
    "only_passive_push_1600",
    "only_activemimic_push_1600",
    "protector_all6_passive_push_1600",
    "protector_all6_activemimic_push_1600",
]

_GROUP_LABELS = [
    "only_passive",
    "only_active",
    "protector_passive",
    "protector_active",
]

_RISK_SPECS = [
    ("contact", "Contact"),
    ("jointforce", "Joint"),
    ("vibration", "Vibration"),
    ("motor", "Motor"),
]

_CASE_LABELS = tuple(_GROUP_LABELS)  # 4 个 case（图例）
_RISK_LABELS = tuple(label for _, label in _RISK_SPECS)  # 4 个 risk（横轴）


def _load_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {r["name"]: r for r in reader if r.get("name")}
    return rows


def _build_arrays(
    rows_by_name: dict[str, dict[str, str]],
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """柱高为各分项 ``*_risk_mean``；误差条全 0（不画分位数区间）。"""
    means: list[list[float]] = []
    zero_errors: list[list[list[float]]] = []
    for row_name in _ROW_NAMES:
        if row_name not in rows_by_name:
            raise KeyError(f"dataset_risk.csv 缺少 name={row_name!r}")
        row = rows_by_name[row_name]
        m_row: list[float] = []
        e_row: list[list[float]] = []
        for prefix, _ in _RISK_SPECS:
            m_row.append(float(row[f"{prefix}_risk_mean"]))
            e_row.append([0.0, 0.0])
        means.append(m_row)
        zero_errors.append(e_row)
    return means, zero_errors


def main() -> None:
    csv_path = _SCRIPT_DIR / "dataset_risk.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    rows_by_name = _load_rows(csv_path)
    group_means, group_errors = _build_arrays(rows_by_name)

    # 将维度从：
    #   (n_cases=4, n_risks=4) -> (n_risks=4, n_cases=4)
    # 这样才能满足：横轴是 4 种 risk，标签/图例是 4 种 case。
    n_cases = len(group_means)
    n_risks = len(group_means[0])
    group_means_t = [[group_means[c][r] for c in range(n_cases)] for r in range(n_risks)]
    group_errors_t = [[group_errors[c][r] for c in range(n_cases)] for r in range(n_risks)]

    _OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_FIGURES / "fig2-3_compare_risks_of_4_falling.png"
    fig, _ax = plot_grouped_risk_bars_pub(
        group_names=_RISK_LABELS,  # x-axis：4 risks
        group_medians=group_means_t,
        group_errors=group_errors_t,
        risk_names=_CASE_LABELS,  # legend：4 cases
        # 每组里是 4 个 case，因此颜色需要 4 个
        risk_colors=("#C82423", "#2878B5", "#F8AC8C", "#9AC9DB"),
        ylabel="Mean risk",
        figsize_cm=(9.0, 7.2),
        capsize=0,
        save_path=str(out_path),
        bar_gap=0.03,
        legend_ncol=1,
        legend_loc="upper right",
        # 收回到坐标轴内部，减少整体边距
        legend_bbox_to_anchor=(0.98, 0.98),
    )
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
