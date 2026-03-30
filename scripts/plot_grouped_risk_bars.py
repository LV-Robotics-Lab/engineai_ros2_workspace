import numpy as np
import matplotlib.pyplot as plt


def _set_pub_style():
    """Set a clean publication-style matplotlib theme."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",   # 若你本机有 Arial，可改成 Arial
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _add_sig_bracket(ax, x1, x2, y, h, text, lw=1.0, color="#444444", fontsize=10):
    """
    Add a significance bracket between x1 and x2.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text,
            ha="center", va="bottom", color=color, fontsize=fontsize)


def plot_grouped_risk_bars_pub(
    group_names,
    group_medians,
    group_errors,
    risk_names=("Contact force", "Joint wrench", "Vibration", "Motor current"),
    risk_colors=("#BEB0D0", "#A7C0DE", "#6C91C2", "#A3514F"),
    ylabel="Risk value",
    title=None,
    figsize=(8.0, 4.8),
    bar_width=0.18,
    capsize=3,
    edgecolor="#666666",
    linewidth=0.8,
    errorbar_color="#666666",
    legend=True,
    legend_ncol=2,
    ylim=None,
    yticks=None,
    show_grid=False,
    grid_axis="y",
    sig_brackets=None,
    save_path=None,
    dpi=600,
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

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)
    offsets = (np.arange(n_risks) - (n_risks - 1) / 2.0) * bar_width

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
            color=risk_colors[i],
            edgecolor=edgecolor,
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
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, pad=8)

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

    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(axis="both", colors="#333333")
    ax.yaxis.label.set_color("#222222")
    if title is not None:
        ax.title.set_color("#222222")

    if legend:
        ax.legend(
            frameon=False,
            ncol=legend_ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            handlelength=1.3,
            columnspacing=1.2,
            handletextpad=0.5,
        )

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

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax



if __name__ == "__main__":
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
        risk_names=("Contact force", "Joint wrench", "Vibration", "Motor current"),
        ylabel="Median risk",
        save_path="risk_bar_pub.pdf",
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
        save_path="risk_bar_pub_sig.pdf",
    )

    plt.show()