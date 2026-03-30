import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, t, gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================
# Color palette
# =========================
PALETTE = {
    "Monarch butterfly": "#4C72B0",
    "Balloon": "#55A868",
    "Speedboat": "#C44E52",
    "Ibex": "#8172B2",
    "Snowplough": "#CCB974",
    "Cockatoo": "#64B5CD",
    "default": "#4C72B0",
}


def _half_violin_top(ax, data, color, alpha=0.16, lw=1.0, bw=0.40, scale=0.68):
    data = np.asarray(data, dtype=float)
    xs = np.linspace(np.min(data), np.max(data), 200)
    kde = gaussian_kde(data, bw_method=bw)
    ys = kde(xs)
    ys = scale * ys / ys.max() if ys.max() > 0 else ys

    ax.fill_between(xs, 0, ys, color=color, alpha=alpha, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=lw)
    ax.set_ylim(0, 1.02)
    ax.axis("off")


def _half_violin_right(ax, data, color, alpha=0.16, lw=1.0, bw=0.40, scale=0.68):
    data = np.asarray(data, dtype=float)
    ys = np.linspace(np.min(data), np.max(data), 200)
    kde = gaussian_kde(data, bw_method=bw)
    xs = kde(ys)
    xs = scale * xs / xs.max() if xs.max() > 0 else xs

    ax.fill_betweenx(ys, 0, xs, color=color, alpha=alpha, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=lw)
    ax.set_xlim(0, 1.02)
    ax.axis("off")


def plot_correlation_with_marginals(
    x,
    y,
    title="Monarch butterfly",
    xlabel="Difficulty level predicted by model (normalized)",
    ylabel="Human-assessed difficulty level (normalized)",
    point_size=28,
    point_alpha=0.9,
    ci_alpha=0.18,
    marginal_size="11%",
    marginal_pad=0.04,
    save_path=None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) < 3:
        raise ValueError("Need at least 3 paired points.")

    color = PALETTE.get(title, PALETTE["default"])

    fig, ax = plt.subplots(figsize=(4.8, 4.6))

    # scatter
    ax.scatter(
        x, y,
        s=point_size,
        alpha=point_alpha,
        color=color,
        edgecolors="none",
        zorder=3
    )

    # regression
    lr = linregress(x, y)
    slope, intercept = lr.slope, lr.intercept
    x_fit = np.linspace(np.min(x), np.max(x), 200)
    y_fit = intercept + slope * x_fit

    # 95% CI of mean prediction
    n = len(x)
    x_mean = np.mean(x)
    y_pred = intercept + slope * x
    s_err = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
    sxx = np.sum((x - x_mean) ** 2)
    t_val = t.ppf(0.975, df=n - 2)

    conf = t_val * s_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
    y_lower = y_fit - conf
    y_upper = y_fit + conf

    ax.plot(x_fit, y_fit, color=color, linewidth=2.0, zorder=4)
    ax.fill_between(x_fit, y_lower, y_upper, color=color, alpha=ci_alpha, zorder=2)

    # stats
    rho, pval = pearsonr(x, y)
    p_text = "P < 0.0001" if pval < 1e-4 else f"P = {pval:.4f}"
    ax.text(
        0.05, 0.95,
        f"\u03c1 = {rho:.2f}\n{p_text}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        color="#333333"
    )

    # main axis style
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")

    # marginal half-violins
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=marginal_size, pad=marginal_pad, sharex=ax)
    ax_right = divider.append_axes("right", size=marginal_size, pad=marginal_pad, sharey=ax)

    _half_violin_top(ax_top, x, color=color)
    _half_violin_right(ax_right, y, color=color)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 只跑一个图
    title = "Monarch butterfly"

    x = np.linspace(-1, 3, 14)
    y = 0.80 * x + rng.normal(0, 0.35, 14)

    fig, ax = plot_correlation_with_marginals(
        x=x,
        y=y,
        title=title,
        save_path=None,   # 例如改成 "single_panel.pdf"
    )

    plt.show()