from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba


def _get_available_font(font_names):
    """Return the first font name in ``font_names`` that exists on the system."""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in font_names:
        if font_name in available_fonts:
            return font_name
    return "DejaVu Sans"


# 与 plot_contact_grid.py 一致：刻度数字用 Times 系；英文标签/图例等用 Myriad 系
TIMES_FONT = _get_available_font(
    ["Times New Roman", "TimesNewRoman", "Nimbus Roman", "DejaVu Serif"]
)
MYRIAD_FONT = _get_available_font(["Myriad Pro", "MyriadPro", "DejaVu Sans"])

_TICK_FONTSIZE = 8
_LABEL_FONTSIZE = 10
_LEGEND_FONTSIZE = 9

_CM_PER_INCH = 2.54
# 与原先默认 8×4.8 英寸一致
_DEFAULT_FIGSIZE_CM = (8.0 * _CM_PER_INCH, 4.8 * _CM_PER_INCH)


def _set_pub_style():
    """Matplotlib 全局样式；具体刻度字体在绘图后按 Times / Myriad 再设。"""
    plt.rcParams.update({
        "font.family": MYRIAD_FONT,
        "font.size": _LABEL_FONTSIZE,
        "axes.labelsize": _LABEL_FONTSIZE,
        "axes.titlesize": _LABEL_FONTSIZE,
        "xtick.labelsize": _TICK_FONTSIZE,
        "ytick.labelsize": _TICK_FONTSIZE,
        "legend.fontsize": _LEGEND_FONTSIZE,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _apply_tick_fonts(ax):
    """x 轴分组名为英文 → Myriad；y 轴刻度为数字 → Times；均为 8pt（对齐 plot_contact_grid）。"""
    for label in ax.get_xticklabels():
        label.set_fontfamily(MYRIAD_FONT)
        label.set_fontsize(_TICK_FONTSIZE)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(_TICK_FONTSIZE)


def _add_sig_bracket(
    ax,
    x1,
    x2,
    y,
    h,
    text,
    lw=1.0,
    color="#444444",
    fontsize=9,
    fontfamily=None,
):
    """
    Add a significance bracket between x1 and x2.
    """
    if fontfamily is None:
        fontfamily = MYRIAD_FONT
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, c=color, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        color=color,
        fontsize=fontsize,
        fontfamily=fontfamily,
    )

# "#BEB0D0", "#A7C0DE", "#6C91C2", "#A3514F"
def plot_grouped_risk_bars_pub(
    group_names,
    group_medians,
    group_errors,
    risk_names=("Contact force", "Joint wrench", "Vibration", "Motor current"),
    risk_colors=("#C82423", "#2878B5", "#9AC9DB", "#F8AC8C"),
    ylabel="Risk value",
    xlabel=None,
    title=None,
    figsize_cm=_DEFAULT_FIGSIZE_CM,
    bar_width=0.18,
    bar_gap=0.0,
    capsize=3,
    edgecolor="#666666",
    linewidth=0.8,
    errorbar_color="#666666",
    legend=True,
    legend_ncol=2,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 1.10),
    ylim=None,
    yticks=None,
    show_grid=False,
    grid_axis="y",
    sig_brackets=None,
    save_path=None,
    dpi=300,
):
    """
    Publication-style grouped bar plot with asymmetric error bars.

    Parameters
    ----------
    group_names : list[str]
        组名，例如：
        ["No protector", "Normal material", "Non-Newtonian material"]

    group_medians : array-like, shape (n_groups, n_risks)
        每组每个 risk 的中位数

    group_errors : array-like, shape (n_groups, n_risks, 2)
        非对称误差，[lower_error, upper_error]

    risk_names : tuple/list[str]
        每个 risk 的名称

    risk_colors : tuple/list[str]
        每个 risk 的颜色，建议保持 4 个，与 risk 顺序一致

    figsize_cm : tuple[float, float]
        画布大小 ``(width, height)``，单位 **厘米**；内部会除以 2.54 转为英寸传给 matplotlib。
        默认与原先 ``8×4.8`` 英寸一致（``(20.32, 12.192)`` cm）。

    sig_brackets : list[dict] or None
        显著性标记列表。每个元素示例：
        {
            "group1": 0,          # 第一个大组索引
            "risk1": 0,           # 第一个 risk 索引
            "group2": 2,          # 第二个大组索引
            "risk2": 0,           # 第二个 risk 索引
            "text": "***",        # 显著性文字
            "y": 25.0,            # 横线起始高度
            "h": 0.8              # 横线高度
        }

    save_path : str or None
        若给定路径，按 ``plot_contact_grid.py`` 惯例保存 **PNG**（或其它后缀由路径决定）：
        ``dpi`` 默认 300，``bbox_inches=None``, ``pad_inches=0``, ``transparent=True``。

    dpi : int
        保存位图时的分辨率，默认 300（与 ``plot_contact_grid`` 一致）。

    Returns
    -------
    fig, ax
    """
    _set_pub_style()

    group_medians = np.asarray(group_medians, dtype=float)
    group_errors = np.asarray(group_errors, dtype=float)

    if group_medians.ndim != 2:
        raise ValueError("group_medians must have shape (n_groups, n_risks)")
    if group_errors.ndim != 3 or group_errors.shape[2] != 2:
        raise ValueError("group_errors must have shape (n_groups, n_risks, 2)")

    n_groups, n_risks = group_medians.shape

    if len(group_names) != n_groups:
        raise ValueError("len(group_names) must equal n_groups")
    if len(risk_names) != n_risks:
        raise ValueError("len(risk_names) must equal n_risks")
    if len(risk_colors) != n_risks:
        raise ValueError("len(risk_colors) must equal n_risks")
    if group_errors.shape[:2] != group_medians.shape:
        raise ValueError("group_errors first two dims must match group_medians")

    # Matplotlib 的 `ax.bar(color=...)` 支持传入 RGBA 元组。
    # 这里希望：
    # - 填充色使用给定 alpha（例如 0.3）
    # - 柱子边框 edgecolor 不透明（alpha=1.0），并且与填充的 RGB 一致
    fill_alpha = 0.3
    risk_colors_fill = [to_rgba(c, fill_alpha) for c in risk_colors]
    risk_colors_edge = [to_rgba(c, 1.0) for c in risk_colors]

    w_cm, h_cm = float(figsize_cm[0]), float(figsize_cm[1])
    if w_cm <= 0 or h_cm <= 0:
        raise ValueError("figsize_cm 宽高须为正数（厘米）")
    fig, ax = plt.subplots(
        figsize=(w_cm / _CM_PER_INCH, h_cm / _CM_PER_INCH)
    )

    x = np.arange(n_groups)
    if bar_gap < 0:
        raise ValueError("bar_gap must be >= 0")
    # offsets 的步长 = bar_width + bar_gap
    # 这样组内柱子之间会有空隙，而柱子的实际宽度仍保持为 bar_width。
    offsets = (np.arange(n_risks) - (n_risks - 1) / 2.0) * (bar_width + bar_gap)

    ymax_candidates = []

    for i in range(n_risks):
        positions = x + offsets[i]
        heights = group_medians[:, i]
        lower = group_errors[:, i, 0]
        upper = group_errors[:, i, 1]
        yerr = np.vstack([lower, upper])

        ax.bar(
            positions,
            heights,
            width=bar_width,
            color=risk_colors_fill[i],
            edgecolor=risk_colors_edge[i],
            linewidth=linewidth,
            yerr=yerr,
            capsize=capsize,
            label=risk_names[i],
            error_kw=dict(
                ecolor=errorbar_color,
                elinewidth=1.0,
                capthick=1.0,
                zorder=3,
            ),
            zorder=2,
        )

        ymax_candidates.extend((heights + upper).tolist())

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel(ylabel, fontfamily=MYRIAD_FONT, fontsize=_LABEL_FONTSIZE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontfamily=MYRIAD_FONT, fontsize=_LABEL_FONTSIZE)

    if title is not None:
        ax.set_title(title, pad=8, fontfamily=MYRIAD_FONT, fontsize=_LABEL_FONTSIZE)

    if yticks is not None:
        ax.set_yticks(yticks)

    if ylim is None:
        ymax = max(ymax_candidates) if ymax_candidates else np.max(group_medians)
        ax.set_ylim(0, ymax * 1.22)
    else:
        ax.set_ylim(*ylim)

    if show_grid:
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.5,
                color="#DDDDDD", alpha=0.8, zorder=0)

    # Spine styling（与 plot_contact_grid 左/下轴 1.0 线宽一致）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", colors="#333333")
    ax.yaxis.label.set_color("#222222")
    if title is not None:
        ax.title.set_color("#222222")

    _apply_tick_fonts(ax)

    if legend:
        leg = ax.legend(
            frameon=False,
            ncol=legend_ncol,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            handlelength=1.3,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        for text in leg.get_texts():
            text.set_fontfamily(MYRIAD_FONT)
            text.set_fontsize(_LEGEND_FONTSIZE)

    # significance brackets
    if sig_brackets is not None:
        for item in sig_brackets:
            g1 = item["group1"]
            r1 = item["risk1"]
            g2 = item["group2"]
            r2 = item["risk2"]
            text = item.get("text", "*")
            y = item["y"]
            h = item.get("h", 0.5)

            x1 = x[g1] + offsets[r1]
            x2 = x[g2] + offsets[r2]
            _add_sig_bracket(ax, x1, x2, y, h, text)

    # 之前使用 matplotlib 默认 tight_layout pad，会让图的上下左右留白偏大。
    plt.tight_layout(pad=0.3)

    if save_path is not None:
        # 与 plot_contact_grid.py 一致：PNG 常用 300 dpi、透明底、不用 tight 裁边
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches=None,
            pad_inches=0,
            transparent=True,
        )

    return fig, ax



if __name__ == "__main__":
    _out_dir = Path(__file__).resolve().parent.parent / "OutputFigures"
    _out_dir.mkdir(parents=True, exist_ok=True)

    group_names = [
        "No protector",
        "Normal material",
        "Non-Newtonian material"
    ]

    group_medians = [
        [18.2, 14.5, 9.1, 11.3],
        [13.6, 10.8, 7.0, 8.9],
        [9.4,  7.2,  4.8, 6.1]
    ]

    # [lower_error, upper_error]
    group_errors = [
        [[2.1, 3.0], [1.8, 2.4], [0.9, 1.2], [1.1, 1.5]],
        [[1.7, 2.3], [1.4, 1.9], [0.7, 1.0], [0.9, 1.3]],
        [[1.2, 1.6], [1.0, 1.3], [0.5, 0.8], [0.7, 0.9]],
    ]

    fig, ax = plot_grouped_risk_bars_pub(
        group_names=group_names,
        group_medians=group_medians,
        group_errors=group_errors,
        risk_names=("Contact Risk", "Joint Risk", "Vibration Risk", "Motor Risk"),
        ylabel="Median risk",
        save_path=str(_out_dir / "risk_bar_pub.png"),
    )

    plt.show()

    # 添加显著性标记，它的作用是告诉读者：这两个柱子对应的统计比较是显著的。
    sig_brackets = [
        {
            "group1": 0, "risk1": 0,
            "group2": 1, "risk2": 0,
            "text": "*",
            "y": 22.0,
            "h": 0.6,
        },
        {
            "group1": 0, "risk1": 0,
            "group2": 2, "risk2": 0,
            "text": "***",
            "y": 24.0,
            "h": 0.6,
        },
    ]

    fig, ax = plot_grouped_risk_bars_pub(
        group_names=group_names,
        group_medians=group_medians,
        group_errors=group_errors,
        risk_names=("Contact force", "Joint wrench", "Vibration", "Motor current"),
        ylabel="Median risk",
        sig_brackets=sig_brackets,
        save_path=str(_out_dir / "risk_bar_pub_sig.png"),
    )

    plt.show()