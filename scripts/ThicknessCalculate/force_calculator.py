import json
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
            with open(params_file, "r", encoding="utf-8") as f:
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

def main():
    """
    主函数：演示计算器的使用
    """
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
