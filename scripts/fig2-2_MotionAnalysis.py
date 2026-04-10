from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib import transforms
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_curves import plot_rounded_trace, style_axis_like_paper


TIME_MIN = 3.0
TIME_MAX = 5.0
COLOR_ACTIVE = "#2878B5"
COLOR_PASSIVE = "#C82423"
FONT_TICK = "Times New Roman"
FONT_OTHER = "Myriad Pro"
FONT_SIZE_TICK = 8
FONT_SIZE_OTHER = 10
AXIS_LW = 1


def _load_csv(log_dir: Path, name: str) -> pd.DataFrame:
    path = log_dir / name
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    return pd.read_csv(path)


def _resolve_font(preferred: str, fallback: str) -> str:
    try:
        fm.findfont(preferred, fallback_to_default=False)
        return preferred
    except ValueError:
        return fallback


def _add_scalebar_below_axis(ax, x0: float, length: float, label: str = "0.2s") -> None:
    # x 使用数据坐标，y 使用轴坐标；确保比例尺在上下方向远离 y 轴原点
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_line = -0.18
    y_text = -0.24
    ax.plot([x0, x0 + length], [y_line, y_line], color="black", lw=AXIS_LW, transform=trans, clip_on=False)
    ax.text(x0 + length / 2, y_text, label, ha="center", va="top", transform=trans, clip_on=False)


def _time_window(df: pd.DataFrame, t_col: str = "timestamp") -> pd.DataFrame:
    return df[(df[t_col] >= TIME_MIN) & (df[t_col] <= TIME_MAX)].copy()


def _extract_torso_speed(link_ke_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "LINK_TORSO_YAW_vel_x", "LINK_TORSO_YAW_vel_y", "LINK_TORSO_YAW_vel_z"]
    miss = [c for c in cols if c not in link_ke_df.columns]
    if miss:
        raise KeyError(f"link_kinetic_energy_data.csv 缺少列: {miss}")
    out = link_ke_df[cols].copy()
    # 按需求：取 z 方向速度并取负号
    out["torso_speed"] = -out["LINK_TORSO_YAW_vel_z"]
    return out[["timestamp", "torso_speed"]]


def _extract_total_pe(link_ke_df: pd.DataFrame) -> pd.DataFrame:
    if "total_PE" not in link_ke_df.columns:
        raise KeyError("link_kinetic_energy_data.csv 缺少列: total_PE")
    return link_ke_df[["timestamp", "total_PE"]].copy()


def _group_contact_force(contact_df: pd.DataFrame, keywords: tuple[str, ...]) -> pd.DataFrame:
    if "timestamp" not in contact_df.columns or "force_normal" not in contact_df.columns:
        raise KeyError("contact_data.csv 缺少 timestamp 或 force_normal")
    if "body1_name" not in contact_df.columns or "body2_name" not in contact_df.columns:
        raise KeyError("contact_data.csv 缺少 body1_name 或 body2_name")

    b1 = contact_df["body1_name"].fillna("").astype(str)
    b2 = contact_df["body2_name"].fillna("").astype(str)

    mask = False
    for kw in keywords:
        mask = mask | b1.str.contains(kw, case=False) | b2.str.contains(kw, case=False)

    hit = contact_df.loc[mask, ["timestamp", "force_normal"]].copy()
    if hit.empty:
        # 空数据时返回全 0，占位以保证两条曲线可画
        base = contact_df[["timestamp"]].drop_duplicates().copy()
        base["force_normal"] = 0.0
        return base.sort_values("timestamp")

    # 用同一时刻目标部位接触法向力之和，表示该部位总冲击力
    agg = hit.groupby("timestamp", as_index=False)["force_normal"].sum()
    return agg.sort_values("timestamp")


def _align_to_timebase(series_df: pd.DataFrame, timebase: np.ndarray, value_col: str) -> np.ndarray:
    s = series_df.sort_values("timestamp")
    t = s["timestamp"].to_numpy()
    y = s[value_col].to_numpy()
    if len(t) == 0:
        return np.zeros_like(timebase)
    return np.interp(timebase, t, y, left=0.0, right=0.0)


def _prepare_dataset(log_dir: Path) -> dict:
    link_ke = _time_window(_load_csv(log_dir, "link_kinetic_energy_data.csv"))
    contact = _time_window(_load_csv(log_dir, "contact_data.csv"))

    torso_speed_df = _extract_torso_speed(link_ke)
    pe_df = _extract_total_pe(link_ke)
    knee_df = _group_contact_force(contact, ("KNEE",))
    elbow_df = _group_contact_force(contact, ("ELBOW",))
    torso_df = _group_contact_force(contact, ("TORSO",))
    head_df = _group_contact_force(contact, ("HEAD",))

    # 公共时间轴用 link_kinetic_energy 的时间戳
    t = torso_speed_df["timestamp"].to_numpy()
    torso_force = _align_to_timebase(torso_df, t, "force_normal")
    head_force = _align_to_timebase(head_df, t, "force_normal")
    vulnerable_force = np.maximum(torso_force, head_force)

    return {
        "t": t,
        "torso_speed": torso_speed_df["torso_speed"].to_numpy(),
        "pe": _align_to_timebase(pe_df, t, "total_PE"),
        "knee_force": _align_to_timebase(knee_df, t, "force_normal"),
        "elbow_force": _align_to_timebase(elbow_df, t, "force_normal"),
        "vulnerable_force": vulnerable_force,
    }


def main() -> None:
    tick_font = _resolve_font(FONT_TICK, "DejaVu Serif")
    other_font = _resolve_font(FONT_OTHER, "DejaVu Sans")

    plt.rcParams.update({
        "font.family": other_font,
        "font.size": FONT_SIZE_OTHER,
    })

    # 曲线平滑参数（可按观感微调）
    smooth_window = 25
    polyorder = 3
    num_dense = 1200

    # 主动摔 / 被动摔 日志目录
    active_dir = Path("/home/wang22/data/mujoco_logs/fig2-2_TrueFalling_mimic")
    passive_dir = Path("/home/wang22/data/mujoco_logs/fig2-2_PassiveFalling")

    active = _prepare_dataset(active_dir)
    passive = _prepare_dataset(passive_dir)

    t = active["t"]
    if len(t) == 0:
        raise RuntimeError("3.0s~5.0s 时间窗口内没有数据")

    fig_width_cm = 9.33
    fig_height_cm = 18
    cm_to_inch = 1.0 / 2.54
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(fig_width_cm * cm_to_inch, fig_height_cm * cm_to_inch),
        dpi=300,
    )
    ax_a, ax_b, ax_c, ax_d = axes

    # a) Torso 速度对比
    plot_rounded_trace(
        ax_a, t, active["torso_speed"], label="Active", lw=2.5, color=COLOR_ACTIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_a, t, passive["torso_speed"], label="Passive", lw=2.5, color=COLOR_PASSIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    style_axis_like_paper(ax_a, ylabel="Torso Speed (m/s)")
    # ax_a.set_title("a. ")
    ax_a.set_xlim(TIME_MIN, TIME_MAX)
    ax_a.margins(x=0)
    ax_a.legend(frameon=False, fontsize=FONT_SIZE_OTHER, loc="upper right")
    _add_scalebar_below_axis(ax_a, x0=TIME_MIN, length=0.2, label="0.2s")

    # b) PE 对比
    plot_rounded_trace(
        ax_b, t, active["pe"], label="Active", lw=2.5, color=COLOR_ACTIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_b, t, passive["pe"], label="Passive", lw=2.5, color=COLOR_PASSIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    style_axis_like_paper(ax_b, ylabel="Potential Energy (J)")
    # ax_b.set_title("b. ")
    ax_b.set_xlim(TIME_MIN, TIME_MAX)
    ax_b.margins(x=0)
    # b 图不显示图例
    _add_scalebar_below_axis(ax_b, x0=TIME_MIN, length=0.2, label="0.2s")

    # c) Knee / Elbow 分开冲击力对比（不再相加）
    plot_rounded_trace(
        ax_c, t, active["knee_force"] / 1000.0, lw=2.5, alpha=0.5, color=COLOR_ACTIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_c, t, passive["knee_force"] / 1000.0, lw=2.5, alpha=0.5, color=COLOR_PASSIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_c, t, active["elbow_force"] / 1000.0, lw=2.0, alpha=0.9, color=COLOR_ACTIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_c, t, passive["elbow_force"] / 1000.0, lw=2.0, alpha=0.9, color=COLOR_PASSIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    style_axis_like_paper(ax_c, ylabel="Knee / Elbow\nCollision (kN)")
    # ax_c.set_title("c. ")
    ax_c.set_xlim(TIME_MIN, TIME_MAX)
    ax_c.margins(x=0)
    active_pair = (
        Line2D([0], [0], color=COLOR_ACTIVE, lw=2.5, alpha=0.5),
        Line2D([0], [0], color=COLOR_ACTIVE, lw=2.0, alpha=0.9),
    )
    passive_pair = (
        Line2D([0], [0], color=COLOR_PASSIVE, lw=2.5, alpha=0.5),
        Line2D([0], [0], color=COLOR_PASSIVE, lw=2.0, alpha=0.9),
    )
    ax_c.legend(
        [active_pair, passive_pair],
        ["Active", "Passive"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        fontsize=FONT_SIZE_OTHER,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    _add_scalebar_below_axis(ax_c, x0=TIME_MIN, length=0.2, label="0.2s")

    # d) Torso / Head 每帧取最大值（脆弱部位碰撞力）
    plot_rounded_trace(
        ax_d, t, active["vulnerable_force"] / 1000.0, label="Active", lw=2.5, color=COLOR_ACTIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    plot_rounded_trace(
        ax_d, t, passive["vulnerable_force"] / 1000.0, label="Passive", lw=2.5, color=COLOR_PASSIVE,
        smooth_window=smooth_window, polyorder=polyorder, num_dense=num_dense
    )
    style_axis_like_paper(ax_d, ylabel="Vulnerable Part\nCollision (kN)")
    # ax_d.set_title("d. ")
    ax_d.set_xlim(TIME_MIN, TIME_MAX)
    ax_d.margins(x=0)
    # d 图不显示图例
    _add_scalebar_below_axis(ax_d, x0=TIME_MIN, length=0.2, label="0.2s")

    # 统一限制 y 轴主刻度数量为 3
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.spines["left"].set_linewidth(AXIS_LW)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK, direction="in", width=AXIS_LW)
        for tick in ax.get_yticklabels():
            tick.set_fontname(tick_font)
            tick.set_fontsize(FONT_SIZE_TICK)

        # 除刻度以外统一为 10pt + Myriad
        ax.yaxis.label.set_fontname(other_font)
        ax.yaxis.label.set_fontsize(FONT_SIZE_OTHER)
        for txt in ax.texts:
            txt.set_fontname(other_font)
            txt.set_fontsize(FONT_SIZE_OTHER)
        if ax.legend_ is not None:
            for txt in ax.legend_.get_texts():
                txt.set_fontname(other_font)
                txt.set_fontsize(FONT_SIZE_OTHER)

    # c/d 固定为 kN 数值展示
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_d.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    out_dir = Path("/home/wang22/engineai/engineai_ros2_workspace/OutputFigures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2-2_MotionAnalysis.png"

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图像: {out_path}")


if __name__ == "__main__":
    main()
