import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator


def smooth_curve(t, y, smooth_window=11, polyorder=3, num_dense=1000):
    t = np.asarray(t)
    y = np.asarray(y)

    if len(y) < 5:
        return t, y

    win = min(smooth_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    win = max(win, 5)
    if win % 2 == 0:
        win -= 1
    if polyorder >= win:
        polyorder = win - 2

    y_smooth = savgol_filter(y, window_length=win, polyorder=polyorder)
    t_dense = np.linspace(t.min(), t.max(), num_dense)

    interp = PchipInterpolator(t, y_smooth)
    y_dense = interp(t_dense)
    return t_dense, y_dense


def add_scalebar(ax, x0, length, label="5s", y_frac=0.08, text_offset_frac=0.05, lw=1.2):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    y0 = ymin + y_frac * yr

    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=lw, clip_on=False)
    ax.text(
        x0 + length / 2,
        y0 - text_offset_frac * yr,
        label,
        ha="center",
        va="top",
        fontsize=10
    )


def style_axis_like_paper(ax, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="out", length=3, width=1, labelsize=10)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)


def plot_rounded_trace(
    ax,
    t,
    y,
    label=None,
    lw=2.6,
    alpha=1.0,
    color=None,
    show_raw=False,
    smooth_window=11,
    polyorder=3,
    num_dense=1000,
):
    t_dense, y_dense = smooth_curve(
        t, y, smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )

    if show_raw:
        ax.scatter(t, y, s=10, alpha=0.15)

    ax.plot(
        t_dense,
        y_dense,
        linewidth=lw,
        alpha=alpha,
        color=color,
        label=label,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True
    )


def plot_multiple_traces(ax, t, ys, labels=None, lw=2.6, show_raw=False):
    if labels is None:
        labels = [None] * len(ys)

    for y, label in zip(ys, labels):
        plot_rounded_trace(ax, t, y, label=label, lw=lw, show_raw=show_raw)


def main():
    np.random.seed(7)
    t = np.linspace(0, 20, 180)

    angle1 = -12 + 10 * np.sin(2 * np.pi * 0.7 * t) + 4 * np.sin(2 * np.pi * 1.7 * t + 0.4)
    angle1 += np.random.normal(scale=2.0, size=len(t))

    angle2 = -10 + 9 * np.sin(2 * np.pi * 0.7 * t + 0.2) + 3.5 * np.sin(2 * np.pi * 1.7 * t + 0.7)
    angle2 += np.random.normal(scale=1.6, size=len(t))

    angle3 = -8 + 7 * np.sin(2 * np.pi * 0.68 * t + 0.5) + 2.5 * np.sin(2 * np.pi * 1.5 * t + 1.0)
    angle3 += np.random.normal(scale=1.3, size=len(t))

    torque1 = 0.18 * np.sin(2 * np.pi * 0.95 * t + 0.8) + 0.10 * np.sin(2 * np.pi * 2.2 * t)
    torque1 += np.random.normal(scale=0.07, size=len(t))

    torque2 = 0.16 * np.sin(2 * np.pi * 0.95 * t + 1.1) + 0.08 * np.sin(2 * np.pi * 2.0 * t + 0.3)
    torque2 += np.random.normal(scale=0.06, size=len(t))

    torque3 = 0.12 * np.sin(2 * np.pi * 1.05 * t + 0.4) + 0.07 * np.sin(2 * np.pi * 2.4 * t + 0.7)
    torque3 += np.random.normal(scale=0.05, size=len(t))

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.6), dpi=200)

    plot_multiple_traces(
        axes[0],
        t,
        [angle1, angle2, angle3],
        labels=["E1 Angle", "E2 Angle", "E3 Angle"],
        lw=2.8
    )
    style_axis_like_paper(axes[0], ylabel="Angle (deg)")
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    add_scalebar(axes[0], x0=1.0, length=5.0, label="5s")

    plot_multiple_traces(
        axes[1],
        t,
        [torque1, torque2, torque3],
        labels=["E1 Torque", "E2 Torque", "E3 Torque"],
        lw=2.4
    )
    style_axis_like_paper(axes[1], ylabel="Torque (N·m)")
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")
    add_scalebar(axes[1], x0=1.0, length=5.0, label="5s")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()