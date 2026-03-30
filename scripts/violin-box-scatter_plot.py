import numpy as np
import matplotlib.pyplot as plt


def plot_raincloud_like(
    data_dict,
    ylabel="Zero-shot normalized human-like score\n(ImageNet-1K → SALICON-split-1)",
    title=None,
    figsize=(9, 5.8),
    ylim=(-0.5, 1.5),
    xtick_rotation=32,
    save_path=None,
    dpi=400,
):
    """
    画类似 Fig. 6a 的 raincloud-style 图：
    左：jittered points
    中：boxplot
    右：half violin

    Parameters
    ----------
    data_dict : dict[str, array-like]
        例如：
        {
            "Gaussian\\ndistribution": np.array([...]),
            "Centre": np.array([...]),
            ...
            "AdaptiveNN-\\nDeiT-S": np.array([...]),
        }
    """

    plt.rcParams.update({
        "font.family": "DejaVu Sans",   # 若你本机有 Arial，可改成 Arial
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 近似 Fig. 6a 的配色
    palette = [
        "#d9d3c3",  # Gaussian distribution
        "#d7cfc8",  # Centre
        "#cfc8dd",  # corner
        "#b8c9df",  # GradCAM
        "#8aa6c9",  # LayerCAM
        "#9b8fb8",  # GradCAM + GMM
        "#a5514f",  # AdaptiveNN
    ]

    labels = list(data_dict.keys())
    values = [np.asarray(v, dtype=float) for v in data_dict.values()]
    n = len(labels)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(42)

    # 三部分的横向偏移
    point_offset = -0.18
    box_offset = 0.00
    violin_offset = 0.18

    point_jitter = 0.035
    violin_width = 0.42
    box_width = 0.12

    for i, vals in enumerate(values):
        base_x = x[i]
        color = palette[i % len(palette)]

        # 1) 左边 jittered points
        x_points = rng.normal(
            loc=base_x + point_offset,
            scale=point_jitter,
            size=len(vals)
        )
        ax.scatter(
            x_points,
            vals,
            s=12,
            color=color,
            edgecolors="none",
            alpha=0.9,
            zorder=3
        )

        # 2) 中间 boxplot
        ax.boxplot(
            [vals],
            positions=[base_x + box_offset],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#222222", linewidth=1.2),
            boxprops=dict(facecolor="white", edgecolor="#444444", linewidth=0.9),
            whiskerprops=dict(color="#444444", linewidth=0.9),
            capprops=dict(color="#444444", linewidth=0.9),
            zorder=4,
        )

        # 3) 右边 half violin
        violin_pos = base_x + violin_offset
        vp = ax.violinplot(
            [vals],
            positions=[violin_pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        body = vp["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("#444444")
        body.set_linewidth(0.8)
        body.set_alpha(0.55)
        body.set_zorder(2)

        # 裁成右半边 violin
        path = body.get_paths()[0]
        verts = path.vertices
        verts[:, 0] = np.maximum(verts[:, 0], violin_pos)

    # 参考线
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", zorder=0)
    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", zorder=0)

    # 左侧说明文字，模仿 Fig. 6a
    ax.text(
        0.32, 1.02,
        "Expected performance of\nan average individual human observer",
        ha="left", va="bottom", fontsize=9, color="#555555"
    )
    ax.text(
        0.32, 0.02,
        "Randomly\nlocalizing\nvisual fixations",
        ha="left", va="bottom", fontsize=9, color="#555555"
    )

    ax.set_xlim(0.35, n + 0.65)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, pad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#333333")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 示例数据：你直接替换成自己的真实数据就行
    data_dict = {
        "Gaussian\ndistribution": rng.normal(0.15, 0.08, 157),
        "Centre": rng.normal(0.08, 0.07, 157),
        "corner": rng.normal(-0.07, 0.08, 157),
        "GradCAM": rng.normal(0.03, 0.07, 157),
        "LayerCAM": rng.normal(0.46, 0.09, 157),
        "GradCAM\n+ GMM": rng.normal(0.75, 0.10, 157),
        "AdaptiveNN-\nDeiT-S": rng.normal(1.11, 0.08, 157),
    }

    fig, ax = plot_raincloud_like(
        data_dict=data_dict,
        ylabel="Zero-shot normalized human-like score\n(ImageNet-1K → SALICON-split-1)",
        title=None,
        save_path="fig6a_raincloud_like.pdf",
        dpi=400,
    )

    plt.show()