# from force_interpolation import get_protected_force

# # 计算防护后的力
# force_protected = get_protected_force(100.0, 12.0)
# print(f"防护后的力: {force_protected:.3f} kN")

import csv
import os
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import numpy as np

try:
    from scipy.interpolate import interp2d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_rt_fem_data(tsv_file="RT-FEM.tsv"):
    """
    加载RT-FEM数据文件
    
    Args:
        tsv_file: RT-FEM数据文件路径，默认为"RT-FEM.tsv"
        
    Returns:
        data: 数据字典或DataFrame，包含RT-FEM数据
        force_values: 无防护的冲击力数组 (kN)
        thickness_values: 厚度数组 (mm)
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, tsv_file)
    
    if HAS_PANDAS:
        # 使用pandas读取
        data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        force_values = data['冲击力（kN）'].values
        thickness_columns = ['6mm', '12mm', '18mm', '24mm']
        thickness_values = np.array([int(col.replace('mm', '')) for col in thickness_columns])
    else:
        # 使用标准库csv读取
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
        
        # 解析表头
        header = rows[0]
        force_col_idx = header.index('冲击力（kN）')
        thickness_columns = ['6mm', '12mm', '18mm', '24mm']
        thickness_col_indices = [header.index(col) for col in thickness_columns]
        thickness_values = np.array([int(col.replace('mm', '')) for col in thickness_columns])
        
        # 解析数据
        data = []
        force_values = []
        for row in rows[1:]:
            force_values.append(float(row[force_col_idx]))
            data_row = {}
            for i, col in enumerate(thickness_columns):
                data_row[col] = float(row[thickness_col_indices[i]])
            data.append(data_row)
        
        force_values = np.array(force_values)
        # 将data转换为类似DataFrame的结构（字典列表）
        data = {'rows': data, 'columns': thickness_columns, 'force_col_idx': force_col_idx}
    
    return data, force_values, thickness_values


def bilinear_interpolation(force_unprotected, thickness, tsv_file="RT-FEM.tsv"):
    """
    使用双线性插值方法，根据无防护的力和厚度，计算防护后的力
    
    Args:
        force_unprotected: 无防护的冲击力 (kN)
        thickness: 防护材料厚度 (mm)
        tsv_file: RT-FEM数据文件路径，默认为"RT-FEM.tsv"
        
    Returns:
        force_protected: 防护后的冲击力 (kN)
    """
    # 加载数据
    data, force_values, thickness_values = load_rt_fem_data(tsv_file)
    
    # 检查输入是否在数据范围内
    force_min, force_max = force_values.min(), force_values.max()
    thickness_min, thickness_max = thickness_values.min(), thickness_values.max()
    
    if force_unprotected < force_min or force_unprotected > force_max:
        raise ValueError(
            f"输入的冲击力 {force_unprotected} kN 超出数据范围 [{force_min}, {force_max}] kN"
        )
    
    if thickness < thickness_min or thickness > thickness_max:
        raise ValueError(
            f"输入的厚度 {thickness} mm 超出数据范围 [{thickness_min}, {thickness_max}] mm"
        )
    
    # 准备插值数据矩阵
    # 行：不同的冲击力值
    # 列：不同的厚度值
    thickness_columns = ['6mm', '12mm', '18mm', '24mm']
    
    if HAS_PANDAS:
        force_protected_matrix = data[thickness_columns].values
    else:
        # 从字典格式构建矩阵
        force_protected_matrix = np.array([[row[col] for col in thickness_columns] 
                                          for row in data['rows']])
    
    if not HAS_SCIPY:
        raise ImportError("scipy未安装，请使用 bilinear_interpolation_manual 函数")
    
    # 使用scipy的interp2d进行双线性插值
    # 注意：interp2d的参数顺序是 (x, y, z)，其中：
    # x: 第一个维度（厚度）
    # y: 第二个维度（冲击力）
    # z: 函数值（防护后的力）
    interp_func = interp2d(
        thickness_values,  # x: 厚度
        force_values,      # y: 冲击力
        force_protected_matrix.T,  # z: 防护后的力矩阵（需要转置）
        kind='linear'
    )
    
    # 执行插值
    force_protected = interp_func(thickness, force_unprotected)[0]
    
    return force_protected


def bilinear_interpolation_manual(force_unprotected, thickness, tsv_file="RT-FEM.tsv"):
    """
    手动实现双线性插值方法（不依赖scipy）
    根据无防护的力和厚度，计算防护后的力
    
    Args:
        force_unprotected: 无防护的冲击力 (kN)
        thickness: 防护材料厚度 (mm)
        tsv_file: RT-FEM数据文件路径，默认为"RT-FEM.tsv"
        
    Returns:
        force_protected: 防护后的冲击力 (kN)
    """
    # 加载数据
    data, force_values, thickness_values = load_rt_fem_data(tsv_file)
    
    # 检查输入是否在数据范围内
    force_min, force_max = force_values.min(), force_values.max()
    thickness_min, thickness_max = thickness_values.min(), thickness_values.max()
    
    if force_unprotected < force_min or force_unprotected > force_max:
        raise ValueError(
            f"输入的冲击力 {force_unprotected} kN 超出数据范围 [{force_min}, {force_max}] kN"
        )
    
    if thickness < thickness_min or thickness > thickness_max:
        raise ValueError(
            f"输入的厚度 {thickness} mm 超出数据范围 [{thickness_min}, {thickness_max}] mm"
        )
    
    # 检查是否正好匹配数据表中的值
    force_exact_match = np.isclose(force_values, force_unprotected, atol=1e-6)
    thickness_exact_match = np.isclose(thickness_values, thickness, atol=1e-6)
    
    # 如果两个维度都精确匹配，直接返回对应的值
    if np.any(force_exact_match) and np.any(thickness_exact_match):
        force_idx = np.where(force_exact_match)[0][0]
        thickness_idx = np.where(thickness_exact_match)[0][0]
        thickness_columns = ['6mm', '12mm', '18mm', '24mm']
        col = thickness_columns[thickness_idx]
        
        if HAS_PANDAS:
            return data.iloc[force_idx][col]
        else:
            return data['rows'][force_idx][col]
    
    # 找到冲击力维度上的两个相邻点
    force_idx = np.searchsorted(force_values, force_unprotected)
    
    # 处理边界情况
    if force_idx == 0:
        force_idx = 1
    elif force_idx >= len(force_values):
        force_idx = len(force_values) - 1
    
    force_low = force_values[force_idx - 1]
    force_high = force_values[force_idx]
    
    # 找到厚度维度上的两个相邻点
    thickness_idx = np.searchsorted(thickness_values, thickness)
    
    # 处理边界情况
    if thickness_idx == 0:
        thickness_idx = 1
    elif thickness_idx >= len(thickness_values):
        thickness_idx = len(thickness_values) - 1
    
    thickness_low = thickness_values[thickness_idx - 1]
    thickness_high = thickness_values[thickness_idx]
    
    # 获取四个角点的值
    thickness_columns = ['6mm', '12mm', '18mm', '24mm']
    
    # 找到对应的列索引
    col_low = thickness_columns[thickness_idx - 1]
    col_high = thickness_columns[thickness_idx]
    
    # 获取四个角点的防护后力值
    if HAS_PANDAS:
        f11 = data.iloc[force_idx - 1][col_low]   # (force_low, thickness_low)
        f12 = data.iloc[force_idx - 1][col_high]  # (force_low, thickness_high)
        f21 = data.iloc[force_idx][col_low]       # (force_high, thickness_low)
        f22 = data.iloc[force_idx][col_high]      # (force_high, thickness_high)
    else:
        # 从字典格式获取值
        f11 = data['rows'][force_idx - 1][col_low]   # (force_low, thickness_low)
        f12 = data['rows'][force_idx - 1][col_high]  # (force_low, thickness_high)
        f21 = data['rows'][force_idx][col_low]       # (force_high, thickness_low)
        f22 = data['rows'][force_idx][col_high]      # (force_high, thickness_high)
    
    # 计算插值权重
    # 在厚度维度上的插值权重
    if thickness_high == thickness_low:
        t_weight = 0.0
    else:
        t_weight = (thickness - thickness_low) / (thickness_high - thickness_low)
    
    # 在冲击力维度上的插值权重
    if force_high == force_low:
        f_weight = 0.0
    else:
        f_weight = (force_unprotected - force_low) / (force_high - force_low)
    
    # 双线性插值
    # 先在厚度维度上插值
    f1 = f11 * (1 - t_weight) + f12 * t_weight  # 在force_low处的插值
    f2 = f21 * (1 - t_weight) + f22 * t_weight  # 在force_high处的插值
    
    # 再在冲击力维度上插值
    force_protected = f1 * (1 - f_weight) + f2 * f_weight
    
    return force_protected


def get_protected_force(force_unprotected, thickness, tsv_file="RT-FEM.tsv", use_scipy=False):
    """
    根据无防护的力和厚度，计算防护后的力（主函数）
    
    使用双线性插值方法，从RT-FEM.tsv数据表中查找并插值计算防护后的力。
    
    Args:
        force_unprotected: 无防护的冲击力 (kN)
        thickness: 防护材料厚度 (mm)
        tsv_file: RT-FEM数据文件路径，默认为"RT-FEM.tsv"
        use_scipy: 是否使用scipy的插值函数，True使用scipy，False使用手动实现（默认False）
        
    Returns:
        force_protected: 防护后的冲击力 (kN)
        
    Raises:
        ValueError: 当输入的力或厚度超出数据范围时
        
    Example:
        >>> from force_interpolation import get_protected_force
        >>> force_protected = get_protected_force(100.0, 12.0)
        >>> print(f"防护后的力: {force_protected:.3f} kN")
    """
    if use_scipy and HAS_SCIPY:
        try:
            return bilinear_interpolation(force_unprotected, thickness, tsv_file)
        except (ImportError, ValueError) as e:
            # 如果scipy插值失败，回退到手动实现
            return bilinear_interpolation_manual(force_unprotected, thickness, tsv_file)
    else:
        return bilinear_interpolation_manual(force_unprotected, thickness, tsv_file)


def main():
    """
    主函数：演示插值函数的使用
    """
    print("=== RT-FEM数据双线性插值计算器 ===\n")
    
    # 测试用例
    test_cases = [
        (100.0, 12.0),   # 100 kN, 12mm
        (85.0, 18.0),    # 85 kN, 18mm
        (150.0, 6.0),    # 150 kN, 6mm
        (50.0, 24.0),    # 50 kN, 24mm
        (120.0, 15.0),   # 120 kN, 15mm (需要插值)
        (90.0, 10.0),    # 90 kN, 10mm (需要插值)
    ]
    
    print("测试用例:")
    print("-" * 60)
    print(f"{'无防护力(kN)':<15} {'厚度(mm)':<12} {'防护后力(kN)':<15}")
    print("-" * 60)
    
    for force_unprotected, thickness in test_cases:
        try:
            force_protected = get_protected_force(force_unprotected, thickness, use_scipy=False)
            print(f"{force_unprotected:>12.1f}    {thickness:>8.1f}    {force_protected:>12.3f}")
        except ValueError as e:
            print(f"{force_unprotected:>12.1f}    {thickness:>8.1f}    错误: {e}")
    
    print("\n" + "=" * 60)
    print("交互式计算")
    print("=" * 60)
    
    try:
        force_input = float(input("请输入无防护的冲击力 (kN): "))
        thickness_input = float(input("请输入防护材料厚度 (mm): "))
        
        force_result = get_protected_force(force_input, thickness_input, use_scipy=False)
        
        print("\n计算结果:")
        print(f"  无防护的冲击力: {force_input:.2f} kN")
        print(f"  防护材料厚度: {thickness_input:.2f} mm")
        print(f"  防护后的冲击力: {force_result:.3f} kN")
        print(f"  力下降: {force_input - force_result:.3f} kN ({((force_input - force_result) / force_input * 100):.2f}%)")
        
    except ValueError as e:
        print(f"输入错误: {e}")
    except KeyboardInterrupt:
        print("\n程序已取消")


if __name__ == "__main__":
    main()

