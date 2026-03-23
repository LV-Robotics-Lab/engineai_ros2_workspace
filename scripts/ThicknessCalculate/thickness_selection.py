import json
import numpy as np
from force_calculator import ForceCalculator

class ThicknessSelector:
    """
    厚度选型器
    根据缓冲前的力，从可选厚度中选择合适的厚度，使衰减后的力降到2kN以下
    """
    
    def __init__(self, params_file="fitted_parameters.json", default_density=0.4):
        """
        初始化选型器
        
        Args:
            params_file: 参数文件路径
            default_density: 默认材料密度
        """
        self.calculator = ForceCalculator(params_file)
        self.default_density = default_density
        # 可选厚度：6, 12, 18, 24mm，转换为米
        self.available_thicknesses = [6, 12, 18, 24]  # 单位：mm
        self.target_force = 3.0  # 目标：衰减后的力 < 2kN
    
    def select_thickness(self, F_before, density=None, target_force=None):
        """
        选择合适的厚度
        
        Args:
            F_before: 缓冲前冲击力 (kN)
            density: 材料密度，如果为None则使用默认密度
            target_force: 目标衰减后的力 (kN)，如果为None则使用默认值2kN
            
        Returns:
            selected_thickness: 选中的厚度 (mm)，如果所有厚度都不满足要求则返回None
            Fpk: 衰减后的力 (kN)
            reduction_percent: 力下降百分比
            all_results: 所有可选厚度的计算结果列表
        """
        if density is None:
            density = self.default_density
        
        if target_force is None:
            target_force = self.target_force
        
        all_results = []

        # 0mm 表示不加护具，缓冲后力等于缓冲前力
        # 若已满足目标，直接返回 0mm
        all_results.append({
            'thickness_mm': 0,
            'Fpk': F_before,
            'reduction_percent': 0.0,
            'meets_target': F_before < target_force
        })

        if F_before < target_force:
            return 0, F_before, 0.0, all_results
        
        # 遍历所有可选厚度，计算衰减后的力
        for t in self.available_thicknesses:
            Fpk, reduction = self.calculator.calculate_force_reduction(t, density, F_before)
            all_results.append({
                'thickness_mm': t,
                'Fpk': Fpk,
                'reduction_percent': reduction,
                'meets_target': Fpk < target_force
            })
        
        # 找到满足目标的最小厚度（优先选择较薄的）
        selected_result = None
        for result in all_results:
            if result['meets_target']:
                if selected_result is None or result['thickness_mm'] < selected_result['thickness_mm']:
                    selected_result = result
        
        if selected_result is None:
            return None, None, None, all_results
        
        return (selected_result['thickness_mm'], 
                selected_result['Fpk'], 
                selected_result['reduction_percent'],
                all_results)
    
    def print_selection_result(self, F_before, density=None, target_force=None):
        """
        打印选型结果
        
        Args:
            F_before: 缓冲前冲击力 (kN)
            density: 材料密度，如果为None则使用默认密度
            target_force: 目标衰减后的力 (kN)，如果为None则使用默认值2kN
        """
        if density is None:
            density = self.default_density
        
        print("=" * 60)
        print("厚度选型结果")
        print("=" * 60)
        print(f"输入条件:")
        print(f"  缓冲前冲击力: {F_before:.2f} kN")
        print(f"  材料密度: {density:.3f}")
        print(f"  目标衰减后力: < {target_force if target_force else self.target_force:.2f} kN")
        print()
        
        selected_thickness, Fpk, reduction, all_results = self.select_thickness(
            F_before, density, target_force
        )
        
        print("所有可选厚度的计算结果:")
        print("-" * 60)
        print(f"{'厚度(mm)':<12} {'衰减后力(kN)':<15} {'力下降(%)':<12} {'是否满足目标':<12}")
        print("-" * 60)
        
        for result in all_results:
            status = "✓ 满足" if result['meets_target'] else "✗ 不满足"
            highlight = ">>>" if result['thickness_mm'] == selected_thickness else "   "
            print(f"{highlight} {result['thickness_mm']:>6.0f}      "
                  f"{result['Fpk']:>10.3f}        "
                  f"{result['reduction_percent']:>8.2f}      "
                  f"{status}")
        
        print()
        
        if selected_thickness is not None:
            print("=" * 60)
            print("推荐选型:")
            print("=" * 60)
            print(f"  推荐厚度: {selected_thickness:.0f} mm")
            print(f"  衰减后力: {Fpk:.3f} kN")
            print(f"  力下降: {reduction:.2f}%")
            print(f"  满足目标: ✓ (衰减后力 < {target_force if target_force else self.target_force:.2f} kN)")
        else:
            print("=" * 60)
            print("选型结果:")
            print("=" * 60)
            print("  ⚠️  警告: 所有可选厚度都无法满足目标要求！")
            print(f"  即使使用最厚的 {self.available_thicknesses[-1]*1000:.0f}mm，")
            print(f"  衰减后的力仍有 {all_results[-1]['Fpk']:.3f} kN，")
            print(f"  超过目标值 {target_force if target_force else self.target_force:.2f} kN")
            print()
            print("  建议:")
            print("  1. 增加材料厚度（超出可选范围）")
            print("  2. 使用更高密度的材料")
            print("  3. 降低缓冲前的冲击力")
        
        print("=" * 60)


def main():
    """
    主函数：演示厚度选型器的使用
    """
    print("=== 厚度选型程序 ===\n")
    
    # 初始化选型器
    try:
        selector = ThicknessSelector()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 示例1：输入缓冲前的力，自动选型
    print("\n示例1：输入缓冲前冲击力 20 kN")
    selector.print_selection_result(20.0)
    
    # 示例2：输入缓冲前的力 30 kN
    print("\n\n示例2：输入缓冲前冲击力 30 kN")
    selector.print_selection_result(30.0)
    
    # 示例3：输入缓冲前的力 50 kN（可能无法满足要求）
    print("\n\n示例3：输入缓冲前冲击力 50 kN")
    selector.print_selection_result(50.0)
    
    # 示例4：自定义密度
    print("\n\n示例4：输入缓冲前冲击力 25 kN，密度 0.4")
    selector.print_selection_result(25.0, density=0.4)
    
    # 交互式输入
    print("\n\n" + "=" * 60)
    print("交互式选型")
    print("=" * 60)
    
    try:
        F_before_input = float(input("请输入缓冲前冲击力 (kN): "))
        density_input = input(f"请输入材料密度 (直接回车使用默认值 {selector.default_density}): ")
        density_input = float(density_input) if density_input.strip() else None
        
        selector.print_selection_result(F_before_input, density=density_input)
    except ValueError:
        print("输入格式错误，请输入有效的数字")
    except KeyboardInterrupt:
        print("\n程序已取消")


# 全局计算器实例（延迟初始化）
_calculator = None

def select_thickness_simple(F_before, density=0.4, target_force=3.0):
    """
    精简的厚度选型函数
    
    Args:
        F_before: 缓冲前冲击力 (kN)
        density: 材料密度，默认0.4
        target_force: 目标衰减后的力 (kN)，默认3.0
        
    Returns:
        thickness_mm: 选中的厚度 (mm)，如果所有厚度都不满足要求则返回None
    """
    global _calculator
    
    # 延迟初始化计算器（静默模式，不打印信息）
    if _calculator is None:
        import sys
        import os
        from io import StringIO
        
        # 参数文件使用本模块所在目录，避免从其他目录运行脚本时找不到
        _params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitted_parameters.json")
        # 临时重定向stdout以抑制打印
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            _calculator = ForceCalculator(params_file=_params_file)
        finally:
            sys.stdout = old_stdout
    
    # 0mm 表示不加护具：若已满足目标，直接返回 0mm
    if F_before < target_force:
        return 0

    # 可选厚度：6, 12, 18, 24mm
    available_thicknesses_mm = [6, 12, 18, 24]
    
    # 遍历所有可选厚度，找到满足目标的最小厚度
    for t_mm in available_thicknesses_mm:
        Fpk = _calculator.calculate_force_after(t_mm, density, F_before)
        if Fpk < target_force:
            return t_mm
    
    # 如果所有厚度都不满足要求，返回None
    return None


if __name__ == "__main__":
    main()

