import pandas as pd
import numpy as np
import os


class ThicknessFromPressure:
    """
    基于RT-FEM数据的厚度计算器
    根据冲击力和期望压强，从RT-FEM.tsv中选择合适的厚度
    """
    
    def __init__(self, tsv_file="RT-FEM.tsv"):
        """
        初始化计算器，加载RT-FEM数据
        
        Args:
            tsv_file: RT-FEM数据文件路径
        """
        self.tsv_file = tsv_file
        self.data = None
        self.available_thicknesses = [6, 12, 18, 24]  # 单位：mm
        self.load_data()
    
    def load_data(self):
        """
        从TSV文件加载数据
        """
        try:
            # 获取脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, self.tsv_file)
            
            # 读取TSV文件
            self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            
            # 验证数据格式
            expected_columns = ['冲击头质量（kg)', '冲击力（kN）', '6mm', '12mm', '18mm', '24mm']
            if list(self.data.columns) != expected_columns:
                raise ValueError(f"数据文件格式不正确，期望列名: {expected_columns}")
            
        except FileNotFoundError:
            print(f"错误: 找不到数据文件 {file_path}")
            raise
        except Exception as e:
            print(f"错误: 加载数据文件失败: {e}")
            raise
    
    def find_closest_force_row(self, input_force):
        """
        根据输入的冲击力，找到最接近的数据行
        
        Args:
            input_force: 输入的冲击力 (kN)
            
        Returns:
            row_index: 最接近的数据行索引
            row_data: 该行的数据（Series）
            actual_force: 该行对应的实际冲击力 (kN)
        """
        # 获取冲击力列
        force_column = self.data['冲击力（kN）']
        
        # 计算与输入冲击力的差值
        differences = np.abs(force_column - input_force)
        
        # 找到最小差值的索引
        closest_idx = differences.idxmin()
        closest_row = self.data.iloc[closest_idx]
        actual_force = closest_row['冲击力（kN）']
        
        return closest_idx, closest_row, actual_force
    
    def calculate_thickness(self, input_force, target_pressure):
        """
        根据冲击力和期望压强计算厚度
        
        Args:
            input_force: 输入的冲击力 (kN)
            target_pressure: 期望减小后的压强 (MPa)
            
        Returns:
            selected_thickness: 选中的厚度 (mm)，如果所有厚度都不满足要求则返回None
            actual_pressure: 该厚度对应的实际压强 (MPa)
            closest_force: 最接近的冲击力值 (kN)
            all_results: 所有可选厚度的计算结果列表
        """
        # 找到最接近的冲击力行
        row_idx, row_data, closest_force = self.find_closest_force_row(input_force)
        
        all_results = []
        
        # 遍历所有可选厚度，获取对应的压强
        for thickness in self.available_thicknesses:
            # 从数据行中获取该厚度对应的压强
            pressure_key = f"{thickness}mm"
            actual_pressure = row_data[pressure_key]
            
            # 判断是否满足目标压强要求（压强应该小于等于目标值）
            meets_target = actual_pressure <= target_pressure
            
            all_results.append({
                'thickness_mm': thickness,
                'pressure_mpa': actual_pressure,
                'meets_target': meets_target
            })
        
        # 找到满足目标的最小厚度（优先选择较薄的）
        selected_result = None
        for result in all_results:
            if result['meets_target']:
                if selected_result is None or result['thickness_mm'] < selected_result['thickness_mm']:
                    selected_result = result
        
        if selected_result is None:
            return None, None, closest_force, all_results
        
        return (selected_result['thickness_mm'], 
                selected_result['pressure_mpa'],
                closest_force,
                all_results)
    
    def print_calculation_result(self, input_force, target_pressure):
        """
        打印计算结果
        
        Args:
            input_force: 输入的冲击力 (kN)
            target_pressure: 期望减小后的压强 (MPa)
        """
        print("=" * 60)
        print("基于RT-FEM数据的厚度计算")
        print("=" * 60)
        print(f"输入条件:")
        print(f"  冲击力: {input_force:.2f} kN")
        print(f"  期望压强: ≤ {target_pressure:.2f} MPa")
        print()
        
        selected_thickness, actual_pressure, closest_force, all_results = self.calculate_thickness(
            input_force, target_pressure
        )
        
        print(f"数据匹配:")
        print(f"  最接近的冲击力: {closest_force:.2f} kN")
        print()
        
        print("所有可选厚度的压强结果:")
        print("-" * 60)
        print(f"{'厚度(mm)':<12} {'压强(MPa)':<15} {'是否满足目标':<12}")
        print("-" * 60)
        
        for result in all_results:
            status = "✓ 满足" if result['meets_target'] else "✗ 不满足"
            highlight = ">>>" if result['thickness_mm'] == selected_thickness else "   "
            print(f"{highlight} {result['thickness_mm']:>6.0f}      "
                  f"{result['pressure_mpa']:>10.3f}        "
                  f"{status}")
        
        print()
        
        if selected_thickness is not None:
            print("=" * 60)
            print("推荐选型:")
            print("=" * 60)
            print(f"  推荐厚度: {selected_thickness:.0f} mm")
            print(f"  实际压强: {actual_pressure:.3f} MPa")
            print(f"  满足目标: ✓ (压强 ≤ {target_pressure:.2f} MPa)")
        else:
            print("=" * 60)
            print("计算结果:")
            print("=" * 60)
            print("  ⚠️  警告: 所有可选厚度都无法满足目标要求！")
            print(f"  即使使用最厚的 {self.available_thicknesses[-1]}mm，")
            print(f"  压强仍有 {all_results[-1]['pressure_mpa']:.3f} MPa，")
            print(f"  超过目标值 {target_pressure:.2f} MPa")
            print()
            print("  建议:")
            print("  1. 增加材料厚度（超出可选范围）")
            print("  2. 降低冲击力")
            print("  3. 提高可接受的压强阈值")
        
        print("=" * 60)


def calculate_thickness_from_pressure(input_force, target_pressure, tsv_file="RT-FEM.tsv"):
    """
    简化的厚度计算函数
    
    Args:
        input_force: 输入的冲击力 (kN)
        target_pressure: 期望减小后的压强 (MPa)
        tsv_file: RT-FEM数据文件路径，默认为"RT-FEM.tsv"
        
    Returns:
        selected_thickness: 选中的厚度 (mm)，如果所有厚度都不满足要求则返回None
    """
    calculator = ThicknessFromPressure(tsv_file)
    selected_thickness, _, _, _ = calculator.calculate_thickness(input_force, target_pressure)
    return selected_thickness


def main():
    """
    主函数：演示厚度计算器的使用
    """
    print("=== 基于RT-FEM数据的厚度计算程序 ===\n")
    
    # 初始化计算器
    try:
        calculator = ThicknessFromPressure()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 示例1：输入冲击力 100 kN，期望压强 50 MPa
    print("\n示例1：输入冲击力 100 kN，期望压强 50 MPa")
    calculator.print_calculation_result(100.0, 50.0)
    
    # 示例2：输入冲击力 85 kN，期望压强 30 MPa
    print("\n\n示例2：输入冲击力 85 kN，期望压强 30 MPa")
    calculator.print_calculation_result(85.0, 30.0)
    
    # 示例3：输入冲击力 170 kN，期望压强 40 MPa
    print("\n\n示例3：输入冲击力 170 kN，期望压强 40 MPa")
    calculator.print_calculation_result(170.0, 40.0)
    
    # 交互式输入
    print("\n\n" + "=" * 60)
    print("交互式计算")
    print("=" * 60)
    
    try:
        input_force = float(input("请输入冲击力 (kN): "))
        target_pressure = float(input("请输入期望压强 (MPa): "))
        
        calculator.print_calculation_result(input_force, target_pressure)
    except ValueError:
        print("输入格式错误，请输入有效的数字")
    except KeyboardInterrupt:
        print("\n程序已取消")


if __name__ == "__main__":
    main()

