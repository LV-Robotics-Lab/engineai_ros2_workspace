import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

class ForceCalculator:
    """
    缓冲材料力计算器
    用于根据拟合参数计算缓冲后的冲击力
    """
    
    def __init__(self, params_file="fitted_parameters.json"):
        """
        初始化计算器，加载拟合参数
        
        Args:
            params_file: 参数文件路径
        """
        self.load_parameters(params_file)
    
    def load_parameters(self, params_file):
        """
        从JSON文件加载拟合参数
        
        Args:
            params_file: 参数文件路径
        """
        try:
            with open(self._resolve_params_path(params_file), "r", encoding="utf-8") as f:
                params = json.load(f)
            
            self.C = params["C"]
            self.alpha = params["alpha"]  # 厚度指数
            self.beta = params["beta"]    # 密度指数
            self.gamma = params["gamma"]  # 缓冲前冲击力指数
            
            print(f"参数加载成功:")
            print(f"C = {self.C:.3e}")
            print(f"alpha = {self.alpha:.3f}")
            print(f"beta = {self.beta:.3f}")
            print(f"gamma = {self.gamma:.3f}")
            
        except FileNotFoundError:
            print(f"错误: 找不到参数文件 {params_file}")
            print("请先运行 test.py 生成参数文件")
            raise
        except KeyError as e:
            print(f"错误: 参数文件中缺少必要的参数 {e}")
            raise

    @staticmethod
    def _resolve_params_path(params_file):
        """
        优先按传入路径读取；若不存在，则回退到当前脚本目录。
        """
        if os.path.isabs(params_file) or os.path.exists(params_file):
            return params_file
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), params_file)
    
    def calculate_force_after(self, t, p, F_before):
        """
        计算缓冲后的冲击力
        
        Args:
            t: 材料厚度 (mm)
            p: 材料密度
            F_before: 缓冲前冲击力 (kN)
            
        Returns:
            Fpk: 缓冲后冲击力 (kN)
        """
        Fpk = self.C * (t**self.alpha) * (p**self.beta) * (F_before**self.gamma)
        return Fpk
    
    def calculate_force_reduction(self, t, p, F_before):
        """
        计算力下降百分比
        
        Args:
            t: 材料厚度 (mm)
            p: 材料密度
            F_before: 缓冲前冲击力 (kN)
            
        Returns:
            Fpk: 缓冲后冲击力 (kN)
            reduction_percent: 力下降百分比
        """
        Fpk = self.calculate_force_after(t, p, F_before)
        reduction_percent = (F_before - Fpk) / F_before * 100
        return Fpk, reduction_percent
    
    def design_thickness(self, p, F_before, Fpk_target):
        """
        设计反算：给定密度、缓冲前冲击力和目标缓冲后冲击力，求所需厚度
        
        Args:
            p: 材料密度
            F_before: 缓冲前冲击力 (kN)
            Fpk_target: 目标缓冲后冲击力 (kN)
            
        Returns:
            t_required: 所需厚度 (mm)
        """
        t_required = (Fpk_target / (self.C * (p**self.beta) * (F_before**self.gamma)))**(1.0/self.alpha)
        return t_required
    
    def design_density(self, t, F_before, Fpk_target):
        """
        设计反算：给定厚度、缓冲前冲击力和目标缓冲后冲击力，求所需密度
        
        Args:
            t: 材料厚度 (mm)
            F_before: 缓冲前冲击力 (kN)
            Fpk_target: 目标缓冲后冲击力 (kN)
            
        Returns:
            p_required: 所需密度
        """
        p_required = (Fpk_target / (self.C * (t**self.alpha) * (F_before**self.gamma)))**(1.0/self.beta)
        return p_required
    
    def batch_calculate(self, thicknesses, densities, forces_before):
        """
        批量计算缓冲后的冲击力
        
        Args:
            thicknesses: 厚度数组 (mm)
            densities: 密度数组
            forces_before: 缓冲前冲击力数组 (kN)
            
        Returns:
            forces_after: 缓冲后冲击力数组 (kN)
            reductions: 力下降百分比数组
        """
        forces_after = []
        reductions = []
        
        for t, p, F_before in zip(thicknesses, densities, forces_before):
            Fpk, reduction = self.calculate_force_reduction(t, p, F_before)
            forces_after.append(Fpk)
            reductions.append(reduction)
        
        return np.array(forces_after), np.array(reductions)


def plot_chr_force_decay_curves(
    thicknesses_mm=(1.0, 6.0),
    density=0.4,
    force_min=0.4,
    force_max=170.0,
    num_points=400,
    output_path=None,
    params_file="fitted_parameters.json",
):
    """
    绘制 CHR 方法下不同厚度的力衰减曲线。

    横轴为缓冲前冲击力，纵轴为缓冲后冲击力。
    """
    calculator = ForceCalculator(params_file=params_file)
    force_before = np.linspace(force_min, force_max, num_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        force_before,
        force_before,
        linestyle="--",
        color="0.55",
        linewidth=1.5,
        label="No protection",
    )

    for thickness_mm in thicknesses_mm:
        force_after = calculator.calculate_force_after(thickness_mm, density, force_before)
        ratio = np.divide(
            force_after,
            force_before,
            out=np.zeros_like(force_after),
            where=force_before > 0.0,
        )
        reduction_percent = np.where(force_before > 0.0, (1.0 - ratio) * 100.0, 0.0)
        label = (
            f"CHR, {thickness_mm:g} mm "
            f"(max reduction {np.max(reduction_percent):.1f}%)"
        )
        ax.plot(force_before, force_after, linewidth=2.2, label=label)

    ax.set_title(f"CHR Force Decay Curves (density={density:g})")
    ax.set_xlabel("Unprotected force before impact (kN)")
    ax.set_ylabel("Protected force after impact (kN)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    if output_path is None:
        thickness_tag = "_".join(f"{t:g}mm" for t in thicknesses_mm)
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"force_decay_curves_{thickness_tag}.png",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_chr_force_decay_and_scale_curves(
    thicknesses_mm=(1.0, 6.0),
    density=0.4,
    force_min=0.4,
    force_max=170.0,
    num_points=400,
    output_path=None,
    params_file="fitted_parameters.json",
):
    """
    同时绘制 CHR 力衰减曲线和 scale 曲线。

    上图：缓冲前冲击力 vs 缓冲后冲击力；
    下图：缓冲前冲击力 vs scale(F_after / F_before)。
    """
    calculator = ForceCalculator(params_file=params_file)
    force_before = np.linspace(force_min, force_max, num_points)

    fig, (ax_decay, ax_scale) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    ax_decay.plot(
        force_before,
        force_before,
        linestyle="--",
        color="0.55",
        linewidth=1.5,
        label="No protection",
    )

    for thickness_mm in thicknesses_mm:
        force_after = calculator.calculate_force_after(thickness_mm, density, force_before)
        ratio = np.divide(
            force_after,
            force_before,
            out=np.zeros_like(force_after),
            where=force_before > 0.0,
        )
        reduction_percent = np.where(force_before > 0.0, (1.0 - ratio) * 100.0, 0.0)
        label = (
            f"CHR, {thickness_mm:g} mm "
            f"(max reduction {np.max(reduction_percent):.1f}%)"
        )
        ax_decay.plot(force_before, force_after, linewidth=2.2, label=label)
        ax_scale.plot(force_before, ratio, linewidth=2.2, label=f"{thickness_mm:g} mm")

    ax_decay.set_title(f"CHR Force Decay Curves (density={density:g})")
    ax_decay.set_ylabel("Protected force after impact (kN)")
    ax_decay.grid(True, linestyle="--", alpha=0.35)
    ax_decay.legend()

    ax_scale.set_title(f"CHR Scale Curves (density={density:g})")
    ax_scale.set_xlabel("Unprotected force before impact (kN)")
    ax_scale.set_ylabel("Scale (F_after / F_before)")
    ax_scale.grid(True, linestyle="--", alpha=0.35)
    ax_scale.legend(title="Thickness")

    if output_path is None:
        thickness_tag = "_".join(f"{t:g}mm" for t in thicknesses_mm)
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"force_decay_scale_curves_{thickness_tag}.png",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_argparser():
    parser = argparse.ArgumentParser(description="CHR 缓冲力计算与衰减曲线绘图")
    parser.add_argument(
        "--plot-decay-curves",
        action="store_true",
        help="绘制 CHR 力衰减曲线",
    )
    parser.add_argument(
        "--plot-scale-curves",
        action="store_true",
        help="绘制 CHR scale 曲线 (F_after / F_before)",
    )
    parser.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        default=[1.0, 6.0],
        help="要绘制的厚度列表，单位 mm，默认: 1 6",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.4,
        help="材料密度，默认: 0.4",
    )
    parser.add_argument(
        "--force-min",
        type=float,
        default=0.4,
        help="缓冲前冲击力最小值，单位 kN，默认: 0.4",
    )
    parser.add_argument(
        "--force-max",
        type=float,
        default=170.0,
        help="缓冲前冲击力最大值，单位 kN，默认: 170.0",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=400,
        help="曲线采样点数，默认: 400",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出图片路径",
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default="fitted_parameters.json",
        help="拟合参数 JSON 路径",
    )
    return parser

def main():
    """
    主函数：演示计算器的使用
    """
    args = build_argparser().parse_args()

    if args.plot_decay_curves and args.plot_scale_curves:
        output_path = plot_chr_force_decay_and_scale_curves(
            thicknesses_mm=args.thicknesses,
            density=args.density,
            force_min=args.force_min,
            force_max=args.force_max,
            num_points=args.num_points,
            output_path=args.output,
            params_file=args.params_file,
        )
        print(f"已生成 CHR 力衰减+scale 曲线: {output_path}")
        return

    if args.plot_decay_curves:
        output_path = plot_chr_force_decay_curves(
            thicknesses_mm=args.thicknesses,
            density=args.density,
            force_min=args.force_min,
            force_max=args.force_max,
            num_points=args.num_points,
            output_path=args.output,
            params_file=args.params_file,
        )
        print(f"已生成 CHR 力衰减曲线: {output_path}")
        return

    if args.plot_scale_curves:
        output_path = plot_chr_force_decay_and_scale_curves(
            thicknesses_mm=args.thicknesses,
            density=args.density,
            force_min=args.force_min,
            force_max=args.force_max,
            num_points=args.num_points,
            output_path=args.output,
            params_file=args.params_file,
        )
        print(f"已生成 CHR scale 曲线: {output_path}")
        return

    print("=== 缓冲材料力计算器 ===")
    
    # 初始化计算器
    try:
        calculator = ForceCalculator()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    print("\n=== 单次计算示例 ===")
    
    # 示例1：计算缓冲后的力
    t = 0.012  # 厚度 12mm
    p = 0.3    # 密度
    F_before = 20.0  # 缓冲前冲击力 20kN
    
    Fpk, reduction = calculator.calculate_force_reduction(t, p, F_before)
    
    print(f"输入条件:")
    print(f"  厚度: {t} m")
    print(f"  密度: {p}")
    print(f"  缓冲前冲击力: {F_before} kN")
    print(f"结果:")
    print(f"  缓冲后冲击力: {Fpk:.3f} kN")
    print(f"  力下降: {reduction:.2f}%")
    
    print("\n=== 设计反算示例 ===")
    
    # 示例2：设计反算厚度
    p_design = 0.3
    F_before_design = 25.0
    Fpk_target = 5.0
    
    t_required = calculator.design_thickness(p_design, F_before_design, Fpk_target)
    
    print(f"设计条件:")
    print(f"  密度: {p_design}")
    print(f"  缓冲前冲击力: {F_before_design} kN")
    print(f"  目标缓冲后冲击力: {Fpk_target} kN")
    print(f"所需厚度: {t_required:.4f} m ({t_required*1000:.1f} mm)")
    
    # 验证计算结果
    Fpk_verify, reduction_verify = calculator.calculate_force_reduction(t_required, p_design, F_before_design)
    print(f"验证: 计算得到的缓冲后冲击力 = {Fpk_verify:.3f} kN")
    
    print("\n=== 批量计算示例 ===")
    
    # 示例3：批量计算
    thicknesses = [0.006, 0.012, 0.018, 0.024]  # 不同厚度
    densities = [0.3, 0.3, 0.3, 0.3]  # 相同密度
    forces_before = [15.0, 20.0, 25.0, 30.0]  # 不同冲击力
    
    forces_after, reductions = calculator.batch_calculate(thicknesses, densities, forces_before)
    
    print("批量计算结果:")
    print("厚度(mm)  密度   缓冲前(kN)  缓冲后(kN)  下降(%)")
    print("-" * 50)
    for i in range(len(thicknesses)):
        print(f"{thicknesses[i]*1000:6.1f}    {densities[i]:4.1f}    {forces_before[i]:8.1f}    {forces_after[i]:8.2f}    {reductions[i]:6.1f}")

if __name__ == "__main__":
    main()
