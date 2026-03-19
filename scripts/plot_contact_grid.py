#!/usr/bin/env python3
"""
绘制接触力数据的网格颜色图
x轴: robot_frame_y
y轴: robot_frame_z
颜色: force_magnitude 和 protector thickness
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath
import matplotlib.font_manager as fm
import numpy as np
import argparse
import os
import re
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import gc

# 尝试导入psutil用于内存监控
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("提示: 安装psutil可以显示内存使用情况 (pip install psutil)")


def get_memory_usage_mb():
    """获取当前进程的内存使用量（MB）"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    else:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux上单位是KB
        except:
            return 0


def print_memory_usage(stage=""):
    """打印内存使用情况"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        mem_percent = process.memory_percent()
        print(f"      内存使用: {mem_mb:.1f} MB ({mem_percent:.1f}%) {stage}")
    else:
        mem_mb = get_memory_usage_mb()
        if mem_mb > 0:
            print(f"      内存使用: {mem_mb:.1f} MB {stage}")

# 检查并设置可用字体
def get_available_font(font_names):
    """检查字体是否可用，返回第一个可用的字体名称"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in font_names:
        if font_name in available_fonts:
            return font_name
    # 如果都不可用，返回默认字体
    return 'DejaVu Sans'

# 设置字体
TIMES_FONT = get_available_font(['Times New Roman', 'TimesNewRoman', 'Nimbus Roman', 'DejaVu Serif'])
MYRIAD_FONT = get_available_font(['Myriad Pro', 'MyriadPro', 'DejaVu Sans'])

print(f"字体设置: Times字体={TIMES_FONT}, Myriad字体={MYRIAD_FONT}")

# 尝试导入进度条库
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装tqdm可以显示进度条 (pip install tqdm)")

# 尝试导入STL处理库
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    try:
        from stl import mesh
        HAS_STL = True
        HAS_TRIMESH = False
    except ImportError:
        HAS_STL = False
        HAS_TRIMESH = False
        print("警告: 未找到trimesh或numpy-stl库，表面积计算功能将不可用")
        print("      请安装: pip install trimesh 或 pip install numpy-stl")

# 导入分类模块
try:
    from classify_body_part import get_body_part
except ImportError:
    print("警告: 无法导入classify_body_part模块，将使用默认分类方法")
    # 如果导入失败，提供一个默认函数
    def get_body_part(row):
        return "Unknown"

# 添加ThicknessCalculate目录到路径，以便导入thickness_selection
script_dir = Path(__file__).parent
thickness_dir = script_dir / 'ThicknessCalculate'
sys.path.insert(0, str(thickness_dir))

try:
    from thickness_selection import select_thickness_simple
    # 获取参数文件路径
    params_file_path = thickness_dir / 'fitted_parameters.json'
except ImportError:
    print("警告: 无法导入thickness_selection模块，厚度计算功能将不可用")
    select_thickness_simple = None
    params_file_path = None

try:
    from thickness_from_pressure import calculate_thickness_from_pressure
except ImportError:
    print("警告: 无法导入thickness_from_pressure模块，基于压强的厚度计算功能将不可用")
    calculate_thickness_from_pressure = None


def create_white_to_red_cmap():
    """创建从淡红色到红色的颜色映射（最小值是淡红色，白色用于无数据区域）"""
    # 淡红色 RGB: (1.0, 0.8, 0.8) 或更淡一些 (1.0, 0.85, 0.85)
    # 红色 RGB: (1.0, 0.0, 0.0)
    colors = [(1.0, 0.85, 0.85), (1.0, 0.0, 0.0)]  # 从淡红色到红色
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('light_red_to_red', colors, N=n_bins)
    return cmap


def create_orange_cmap():
    """创建从白色到深橙的颜色映射（0是白色，24是最橙色）"""
    # 定义颜色渐变：从白色(1.0, 1.0, 1.0)到深橙(255, 140, 0)
    colors = [(1.0, 1.0, 1.0), (1.0, 0.55, 0.0)]  # 白色到深橙
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('orange', colors, N=n_bins)
    return cmap


def create_discrete_orange_cmap():
    """创建离散的橙色颜色映射，用于5个标准厚度值（0, 6, 12, 18, 24）"""
    # 定义5种颜色：从白色到深橙色
    # 0: 白色, 6: 浅橙, 12: 中浅橙, 18: 中橙, 24: 深橙
    colors = [
        (1.0, 1.0, 1.0),      # 0 - 白色 #FFFFFF
        (1.0, 0.9, 0.7),      # 6 - 浅橙 #FFE6B3
        (1.0, 0.8, 0.5),      # 12 - 中浅橙 #FFCC80
        (1.0, 0.65, 0.25),    # 18 - 中橙 #FFA640
        (1.0, 0.55, 0.0)      # 24 - 深橙 #FF8C00
    ]
    # 使用 ListedColormap 创建真正的离散颜色映射
    cmap = ListedColormap(colors, name='discrete_orange')
    return cmap


def create_blue_cmap():
    """创建从浅蓝到深蓝的颜色映射"""
    colors = [(0.7, 0.85, 1.0), (0.0, 0.3, 0.8)]  # 浅蓝到深蓝
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue', colors, N=n_bins)
    return cmap


def load_stl_mesh(stl_path, convert_to_meters=True):
    """
    加载STL文件并返回网格对象
    
    参数:
        stl_path: STL文件路径
        convert_to_meters: 是否将STL坐标转换为米（默认True，假设STL是毫米单位）
    
    返回:
        mesh: 网格对象（trimesh或numpy-stl），坐标已转换为米
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL文件不存在: {stl_path}")
    
    if HAS_TRIMESH:
        mesh_obj = trimesh.load(str(stl_path))
        # 检查STL文件的单位：如果最大坐标值大于10，可能是毫米单位
        if convert_to_meters and mesh_obj.vertices.size > 0:
            max_coord = np.abs(mesh_obj.vertices).max()
            if max_coord > 10:
                print(f"检测到STL文件坐标范围较大（最大: {max_coord:.2f}），假设为毫米单位，转换为米")
                # 将毫米转换为米
                mesh_obj.vertices = mesh_obj.vertices / 1000.0
            else:
                print(f"STL文件坐标范围: {max_coord:.2f}，假设为米单位")
        return mesh_obj
    elif HAS_STL:
        mesh_obj = mesh.Mesh.from_file(str(stl_path))
        # 检查STL文件的单位
        if convert_to_meters and mesh_obj.vectors.size > 0:
            max_coord = np.abs(mesh_obj.vectors).max()
            if max_coord > 10:
                print(f"检测到STL文件坐标范围较大（最大: {max_coord:.2f}），假设为毫米单位，转换为米")
                # 将毫米转换为米
                mesh_obj.vectors = mesh_obj.vectors / 1000.0
            else:
                print(f"STL文件坐标范围: {max_coord:.2f}，假设为米单位")
        return mesh_obj
    else:
        raise ImportError("无法加载STL文件：未找到trimesh或numpy-stl库")


def calculate_surface_area_for_grid_cells(grid_centers, stl_path, search_radius=0.01):
    """
    为每个网格单元计算表面积
    
    参数:
        grid_centers: 网格单元中心坐标数组，形状为 (N, 3)，单位：米
        stl_path: STL文件路径
        search_radius: 搜索半径（米），用于计算每个网格单元周围的表面积
    
    返回:
        surface_areas: 每个网格单元的表面积数组（单位：m²）
    """
    if not HAS_TRIMESH and not HAS_STL:
        raise ImportError("无法计算表面积：未找到trimesh或numpy-stl库")
    
    print(f"正在加载STL文件: {stl_path}")
    mesh_obj = load_stl_mesh(stl_path, convert_to_meters=True)
    
    # 检查网格和接触点的坐标范围是否匹配
    if HAS_TRIMESH:
        mesh_coords = mesh_obj.vertices
    elif HAS_STL:
        mesh_coords = mesh_obj.vectors.reshape(-1, 3)
    else:
        mesh_coords = np.array([])
    
    if mesh_coords.size > 0:
        mesh_max = np.abs(mesh_coords).max()
        grid_max = np.abs(grid_centers).max()
        print(f"STL网格坐标范围: ±{mesh_max:.3f} m")
        print(f"接触点坐标范围: ±{grid_max:.3f} m")
        if mesh_max > 10 * grid_max or grid_max > 10 * mesh_max:
            print(f"警告: STL网格和接触点的坐标范围差异较大，可能存在单位不匹配问题")
    
    print(f"正在计算 {len(grid_centers)} 个网格单元的表面积...")
    surface_areas = np.zeros(len(grid_centers))
    
    # 尝试导入tqdm显示进度
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("提示: 安装tqdm可以显示进度条 (pip install tqdm)")
    
    if HAS_TRIMESH:
        # 使用trimesh计算
        # 获取所有三角形
        triangles = mesh_obj.triangles
        triangle_centers = triangles.mean(axis=1)
        triangle_areas = mesh_obj.area_faces
        
        # 使用KD树加速最近邻搜索
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(triangle_centers)
            use_kdtree = True
        except ImportError:
            use_kdtree = False
            print("提示: 安装scipy可以加速计算 (pip install scipy)")
        
        # 对于每个网格单元，找到附近的三角形并累加面积
        iterator = tqdm(enumerate(grid_centers), total=len(grid_centers), desc="计算表面积") if has_tqdm else enumerate(grid_centers)
        
        for i, center in iterator:
            if use_kdtree:
                # 使用KD树查找搜索半径内的点
                indices = tree.query_ball_point(center, search_radius)
                if len(indices) > 0:
                    surface_areas[i] = triangle_areas[indices].sum()
                else:
                    # 如果找不到附近的三角形，使用最近三角形的面积
                    dist, nearest_idx = tree.query(center)
                    surface_areas[i] = triangle_areas[nearest_idx]
            else:
                # 计算点到所有三角形中心的距离
                distances = np.linalg.norm(triangle_centers - center, axis=1)
                # 找到在搜索半径内的三角形
                nearby_mask = distances < search_radius
                if np.any(nearby_mask):
                    # 累加附近三角形的面积
                    surface_areas[i] = triangle_areas[nearby_mask].sum()
                else:
                    # 如果找不到附近的三角形，使用最近三角形的面积
                    nearest_idx = np.argmin(distances)
                    surface_areas[i] = triangle_areas[nearest_idx]
    
    elif HAS_STL:
        # 使用numpy-stl计算
        # 获取所有三角形
        triangles = mesh_obj.vectors  # 形状: (N, 3, 3)
        triangle_centers = triangles.mean(axis=1)  # 形状: (N, 3)
        
        # 计算每个三角形的面积
        def triangle_area(triangle):
            """计算三角形面积"""
            v1 = triangle[1] - triangle[0]
            v2 = triangle[2] - triangle[0]
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        triangle_areas = np.array([triangle_area(tri) for tri in triangles])
        
        # 使用KD树加速最近邻搜索
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(triangle_centers)
            use_kdtree = True
        except ImportError:
            use_kdtree = False
            print("提示: 安装scipy可以加速计算 (pip install scipy)")
        
        # 对于每个网格单元，找到附近的三角形并累加面积
        iterator = tqdm(enumerate(grid_centers), total=len(grid_centers), desc="计算表面积") if has_tqdm else enumerate(grid_centers)
        
        for i, center in iterator:
            if use_kdtree:
                # 使用KD树查找搜索半径内的点
                indices = tree.query_ball_point(center, search_radius)
                if len(indices) > 0:
                    surface_areas[i] = triangle_areas[indices].sum()
                else:
                    # 如果找不到附近的三角形，使用最近三角形的面积
                    dist, nearest_idx = tree.query(center)
                    surface_areas[i] = triangle_areas[nearest_idx]
            else:
                # 计算点到所有三角形中心的距离
                distances = np.linalg.norm(triangle_centers - center, axis=1)
                # 找到在搜索半径内的三角形
                nearby_mask = distances < search_radius
                if np.any(nearby_mask):
                    # 累加附近三角形的面积
                    surface_areas[i] = triangle_areas[nearby_mask].sum()
                else:
                    # 如果找不到附近的三角形，使用最近三角形的面积
                    nearest_idx = np.argmin(distances)
                    surface_areas[i] = triangle_areas[nearest_idx]
    
    print(f"表面积计算完成，范围: {surface_areas.min():.6f} - {surface_areas.max():.6f} m²")
    return surface_areas


def round_thickness_to_standard(thicknesses):
    """
    将厚度值四舍五入到最接近的标准值：0, 6, 12, 18, 24
    
    参数:
        thicknesses: 厚度数组（单位：mm），可能包含nan
    
    返回:
        rounded_thicknesses: 四舍五入后的厚度数组
    """
    # 标准厚度值
    standard_values = np.array([0, 6, 12, 18, 24])
    
    # 创建结果数组，保持原始数据类型
    rounded = np.full_like(thicknesses, np.nan, dtype=float)
    
    # 处理nan值
    valid_mask = ~np.isnan(thicknesses)
    
    if np.any(valid_mask):
        # 使用向量化操作找到最接近的标准值
        valid_thicknesses = thicknesses[valid_mask]
        # 对于每个有效值，计算到所有标准值的距离
        # 使用广播：valid_thicknesses[:, None] 与 standard_values[None, :] 比较
        distances = np.abs(valid_thicknesses[:, None] - standard_values[None, :])
        # 找到每个值最接近的标准值索引
        nearest_indices = np.argmin(distances, axis=1)
        # 赋值
        rounded[valid_mask] = standard_values[nearest_indices]
    
    return rounded


def upgrade_thickness_one_level(thicknesses):
    """
    将厚度值提升一个级别
    
    参数:
        thicknesses: 厚度数组（单位：mm），可能包含nan，值应该是标准值：0, 6, 12, 18, 24
    
    返回:
        upgraded_thicknesses: 提升后的厚度数组
        提升规则：0->6, 6->12, 12->18, 18->24, 24->24, nan->nan
    """
    # 标准厚度值
    standard_values = np.array([0, 6, 12, 18, 24])
    # 提升后的厚度值
    upgraded_values = np.array([6, 12, 18, 24, 24])
    
    # 创建结果数组，保持原始数据类型
    upgraded = thicknesses.copy()
    
    # 处理nan值
    valid_mask = ~np.isnan(thicknesses)
    
    if np.any(valid_mask):
        valid_thicknesses = thicknesses[valid_mask]
        # 使用向量化操作找到每个值对应的标准值索引
        # 计算每个有效值到所有标准值的距离
        distances = np.abs(valid_thicknesses[:, None] - standard_values[None, :])
        # 找到每个值最接近的标准值索引
        nearest_indices = np.argmin(distances, axis=1)
        # 使用提升后的值
        upgraded[valid_mask] = upgraded_values[nearest_indices]
    
    return upgraded


def limit_force_in_z_range(force_kn, z_values, z_min=0.8, z_max=1.0, max_force=20.0):
    """
    将z坐标在指定范围内的力值限制为小于等于max_force
    
    参数:
        force_kn: 力值数组（单位：kN）
        z_values: z坐标数组（单位：m）
        z_min: z坐标最小值（默认0.8）
        z_max: z坐标最大值（默认1.0）
        max_force: 最大力值（单位：kN，默认20.0）
    
    返回:
        limited_force_kn: 限制后的力值数组
    """
    # 创建结果数组
    limited = force_kn.copy()
    
    # 识别z坐标在指定范围内的点
    z_mask = (z_values >= z_min) & (z_values <= z_max)
    
    if np.any(z_mask):
        # 找到需要限制的点（力值大于max_force的点）
        force_mask = force_kn > max_force
        combined_mask = z_mask & force_mask
        
        if np.any(combined_mask):
            limited_count = np.sum(combined_mask)
            print(f"      检测到 {limited_count} 个z坐标在({z_min}, {z_max})范围内且力值大于{max_force}kN的点，将力值限制为{max_force}kN...")
            # 将满足条件的点的力值限制为max_force
            limited[combined_mask] = max_force
            print(f"      已限制 {limited_count} 个点的力值")
    
    return limited


def set_thickness_to_12mm_in_z_range(thicknesses, df, z_values):
    """
    将任何部位z坐标在0.3~0.5范围内的厚度直接设置为12mm
    
    参数:
        thicknesses: 厚度数组（单位：mm），可能包含nan
        df: DataFrame（保留参数以保持接口一致性，但不再使用）
        z_values: z坐标数组（单位：m）
    
    返回:
        modified_thicknesses: 修改后的厚度数组
    """
    # 创建结果数组，保持原始数据类型
    modified = thicknesses.copy()
    
    # 识别z坐标在(0.32, 0.48)范围内的点（任何部位）
    z_mask = (z_values >= 0.32) & (z_values <= 0.48)
    
    if np.any(z_mask):
        z_count = np.sum(z_mask)
        print(f"      检测到 {z_count} 个z坐标在(0.32, 0.48)范围内的点（任何部位），将厚度设置为12mm...")
        # 将满足条件的点的厚度设置为12mm
        modified[z_mask] = 12.0
        print(f"      已修改 {z_count} 个点的厚度为12mm")
    
    return modified


def set_elbow_thickness_to_0_in_z_range(thicknesses, df, z_values, z_min=0.5, z_max=0.7):
    """
    将elbow部位z坐标在指定范围内的厚度强制设置为0mm
    
    参数:
        thicknesses: 厚度数组（单位：mm），可能包含nan
        df: DataFrame，包含body_part列
        z_values: z坐标数组（单位：m）
        z_min: z坐标最小值（默认0.5）
        z_max: z坐标最大值（默认0.7）
    
    返回:
        modified_thicknesses: 修改后的厚度数组
    """
    # 创建结果数组，保持原始数据类型
    modified = thicknesses.copy()
    
    # 识别elbow组
    elbow_mask = df['body_part'].isin(['Left_Elbow', 'Right_Elbow'])
    # 识别z坐标在指定范围内的点
    z_mask = (z_values >= z_min) & (z_values <= z_max)
    # 组合条件：elbow组 AND z坐标在指定范围内
    elbow_z_mask = elbow_mask & z_mask
    
    if np.any(elbow_z_mask):
        elbow_z_count = np.sum(elbow_z_mask)
        print(f"      检测到 {elbow_z_count} 个elbow组中z坐标在({z_min}, {z_max})范围内的点，将厚度强制设置为0mm...")
        # 将满足条件的点的厚度设置为0mm
        modified[elbow_z_mask] = 0.0
        print(f"      已修改 {elbow_z_count} 个点的厚度为0mm")
    
    return modified


def set_elbow_thickness_to_6mm(thicknesses, df, z_values):
    """
    将elbow和shoulder部位z坐标在0.8~1.0范围内的厚度设置为6mm
    
    参数:
        thicknesses: 厚度数组（单位：mm），可能包含nan
        df: DataFrame，包含body_part列
        z_values: z坐标数组（单位：m）
    
    返回:
        modified_thicknesses: 修改后的厚度数组
    """
    # 创建结果数组，保持原始数据类型
    modified = thicknesses.copy()
    
    # 识别elbow和shoulder组
    elbow_mask = df['body_part'].isin(['Left_Elbow', 'Right_Elbow'])
    shoulder_mask = df['body_part'].isin(['Left_Shoulder', 'Right_Shoulder'])
    # 识别z坐标在(0.8, 1.0)范围内的点
    z_mask = (z_values >= 0.8) & (z_values <= 1.0)
    # 组合条件：(elbow组 OR shoulder组) AND z坐标在(0.8, 1.0)范围内
    elbow_z_mask = elbow_mask & z_mask
    shoulder_z_mask = shoulder_mask & z_mask
    combined_mask = elbow_z_mask | shoulder_z_mask
    
    if np.any(combined_mask):
        elbow_z_count = np.sum(elbow_z_mask)
        shoulder_z_count = np.sum(shoulder_z_mask)
        total_count = np.sum(combined_mask)
        print(f"      检测到 {total_count} 个elbow/shoulder组中z坐标在(0.8, 1.0)区域的点（elbow: {elbow_z_count}, shoulder: {shoulder_z_count}），将厚度设置为6mm...")
        # 将满足条件的点的厚度设置为6mm
        modified[combined_mask] = 6.0
        print(f"      已修改 {total_count} 个点的厚度为6mm")
    
    return modified


def calculate_thicknesses(force_magnitudes, density=0.4, target_force=3.0, 
                          method='chr', target_pressure=None):
    """
    计算每个接触点的保护层厚度，并四舍五入到标准值（0, 6, 12, 18, 24）
    
    参数:
        force_magnitudes: 力的大小数组（单位：N）
        density: 材料密度（默认0.4，仅用于chr方法）
        target_force: 目标衰减后的力（kN，默认3.0，仅用于chr方法）
        method: 计算方法，'chr'使用thickness_selection，'zzq'使用thickness_from_pressure（默认'chr'）
        target_pressure: 期望减小后的压强（MPa，仅用于zzq方法）
    
    返回:
        thicknesses: 厚度数组（单位：mm），已四舍五入到标准值，如果无法满足要求则为nan
    """
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到ThicknessCalculate目录
        os.chdir(str(thickness_dir))
        
        thicknesses = []
        
        if method == 'chr':
            # 使用thickness_selection方法
            if select_thickness_simple is None:
                raise ImportError("无法导入select_thickness_simple函数，请检查ThicknessCalculate模块")
            
            # 使用进度条显示计算进度
            iterator = tqdm(force_magnitudes, desc="计算厚度", disable=not HAS_TQDM, ncols=80) if HAS_TQDM else force_magnitudes
            
            for force_n in iterator:
                # 将力从N转换为kN
                force_kn = force_n / 1000.0
                # 计算厚度
                thickness_mm = select_thickness_simple(force_kn, density=density, target_force=target_force)
                # 如果返回None，转换为nan以便numpy处理
                if thickness_mm is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(thickness_mm))
        
        elif method == 'zzq':
            # 使用thickness_from_pressure方法
            if calculate_thickness_from_pressure is None:
                raise ImportError("无法导入calculate_thickness_from_pressure函数，请检查ThicknessCalculate模块")
            
            if target_pressure is None:
                raise ValueError("使用zzq方法时必须提供target_pressure参数")
            
            # 导入ThicknessFromPressure类，创建单个实例以复用（避免重复加载数据）
            from thickness_from_pressure import ThicknessFromPressure
            calculator = ThicknessFromPressure()
            
            # 使用进度条显示计算进度
            iterator = tqdm(force_magnitudes, desc="计算厚度", disable=not HAS_TQDM, ncols=80) if HAS_TQDM else force_magnitudes
            
            for force_n in iterator:
                # 将力从N转换为kN
                force_kn = force_n / 1000.0
                # 使用计算器实例计算厚度（避免重复加载数据）
                selected_thickness, _, _, _ = calculator.calculate_thickness(force_kn, target_pressure)
                # 如果返回None，转换为nan以便numpy处理
                if selected_thickness is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(selected_thickness))
        
        else:
            raise ValueError(f"未知的计算方法: {method}，必须是'chr'或'zzq'")
        
        thicknesses = np.array(thicknesses)
        
        # 将厚度值四舍五入到标准值
        thicknesses = round_thickness_to_standard(thicknesses)
        
        return thicknesses
    
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


def calculate_part_statistics(df, density=0.4, target_force=1.0, force_column='force_normal',
                              method='chr', target_pressure=None):
    """
    计算每个身体部分的最大力和厚度
    
    参数:
        df: DataFrame，包含body2_name, robot_frame_y, robot_frame_z, force_magnitude或force_normal
        density: 材料密度（默认0.4，仅用于chr方法）
        target_force: 目标衰减后的力（kN，默认1.0，仅用于chr方法）
        force_column: 使用的力列名，'force_normal'或'force_magnitude'（默认'force_normal'）
        method: 计算方法，'chr'使用thickness_selection，'zzq'使用thickness_from_pressure（默认'chr'）
        target_pressure: 期望减小后的压强（MPa，仅用于zzq方法）
    
    返回:
        stats: DataFrame，包含每个身体部分的统计信息
        unsatisfied_points: DataFrame，包含无法满足目标要求的接触点信息
    """
    # 检查必要的列
    if 'body2_name' not in df.columns:
        print("警告: 未找到body2_name列，无法按身体部分分组")
        return None, pd.DataFrame()
    
    # 检查力列是否存在
    if force_column not in df.columns:
        # 尝试使用另一个列
        if force_column == 'force_normal' and 'force_magnitude' in df.columns:
            print(f"警告: 未找到{force_column}列，使用force_magnitude代替")
            force_column = 'force_magnitude'
        elif force_column == 'force_magnitude' and 'force_normal' in df.columns:
            print(f"警告: 未找到{force_column}列，使用force_normal代替")
            force_column = 'force_normal'
        else:
            print(f"错误: 未找到力列 {force_column}，也无法找到替代列")
            return None, pd.DataFrame()
    
    # 添加身体部分列（使用多进程加速，如果还没有body_part列）
    df = df.copy()
    if 'body_part' not in df.columns:
        if len(df) > 10000:
            n_jobs = max(1, cpu_count() - 2)
            print(f"      正在分类身体部分（{len(df)} 行数据，使用 {n_jobs} 个进程）...")
        else:
            n_jobs = 1
            print(f"      正在分类身体部分（{len(df)} 行数据）...")
        df = apply_body_part_multiprocess(df, n_jobs=n_jobs)
    else:
        print(f"      跳过分类（已有body_part列）")
    
    # 按身体部分分组，计算最大力，并找到最大力对应的索引
    def get_max_force_idx(group):
        """返回最大力对应的索引"""
        return group[force_column].idxmax()
    
    # 按身体部分分组，计算统计信息
    part_stats = df.groupby('body_part').agg({
        force_column: ['max', 'mean', 'count']
    }).reset_index()
    
    # 扁平化列名
    part_stats.columns = ['body_part', 'max_force_n', 'mean_force_n', 'count']
    
    # 找到每个部分最大力对应的索引
    max_indices = df.groupby('body_part', group_keys=False).apply(get_max_force_idx, include_groups=False).reset_index()
    max_indices.columns = ['body_part', 'max_idx']
    
    # 合并统计信息和索引
    part_stats = part_stats.merge(max_indices, on='body_part')
    
    # 获取最大力对应的xyz坐标
    part_stats['max_x'] = part_stats['max_idx'].apply(lambda idx: df.loc[idx, 'robot_frame_x'] if 'robot_frame_x' in df.columns else 0.0)
    part_stats['max_y'] = part_stats['max_idx'].apply(lambda idx: df.loc[idx, 'robot_frame_y'] if 'robot_frame_y' in df.columns else 0.0)
    part_stats['max_z'] = part_stats['max_idx'].apply(lambda idx: df.loc[idx, 'robot_frame_z'] if 'robot_frame_z' in df.columns else 0.0)
    
    # 获取最大力对应的body1和body2名称
    part_stats['max_body1_name'] = part_stats['max_idx'].apply(lambda idx: str(df.loc[idx, 'body1_name']) if 'body1_name' in df.columns and idx in df.index else 'N/A')
    part_stats['max_body2_name'] = part_stats['max_idx'].apply(lambda idx: str(df.loc[idx, 'body2_name']) if 'body2_name' in df.columns and idx in df.index else 'N/A')
    
    # 获取最大力对应的fail-type（使用fall_type_info列）
    if 'fall_type_info' in df.columns:
        part_stats['max_fail_type'] = part_stats['max_idx'].apply(
            lambda idx: str(df.loc[idx, 'fall_type_info']) if idx in df.index and pd.notna(df.loc[idx, 'fall_type_info']) else 'N/A'
        )
    else:
        part_stats['max_fail_type'] = 'N/A'
    
    # 删除临时列
    part_stats = part_stats.drop('max_idx', axis=1)
    
    # 将最大力从N转换为kN
    part_stats['max_force_kn'] = part_stats['max_force_n'] / 1000.0
    part_stats['mean_force_kn'] = part_stats['mean_force_n'] / 1000.0
    
    # 计算每个部分的最大力对应的厚度
    unsatisfied_points = pd.DataFrame()  # 存储无法满足要求的点
    original_cwd = os.getcwd()
    try:
        os.chdir(str(thickness_dir))
        thicknesses = []
        iterator = tqdm(part_stats['max_force_kn'], desc="计算厚度", disable=not HAS_TQDM) if HAS_TQDM else part_stats['max_force_kn']
        
        if method == 'chr':
            if select_thickness_simple is None:
                raise ImportError("无法导入select_thickness_simple函数，请检查ThicknessCalculate模块")
            for force_kn in iterator:
                thickness_mm = select_thickness_simple(force_kn, density=density, target_force=target_force)
                if thickness_mm is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(thickness_mm))
        elif method == 'zzq':
            if calculate_thickness_from_pressure is None:
                raise ImportError("无法导入calculate_thickness_from_pressure函数，请检查ThicknessCalculate模块")
            if target_pressure is None:
                raise ValueError("使用zzq方法时必须提供target_pressure参数")
            # 导入ThicknessFromPressure类，创建单个实例以复用（避免重复加载数据）
            from thickness_from_pressure import ThicknessFromPressure
            calculator = ThicknessFromPressure()
            for force_kn in iterator:
                # 使用计算器实例计算厚度（避免重复加载数据）
                selected_thickness, _, _, _ = calculator.calculate_thickness(force_kn, target_pressure)
                if selected_thickness is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(selected_thickness))
        else:
            raise ValueError(f"未知的计算方法: {method}，必须是'chr'或'zzq'")
        
        # 将厚度值四舍五入到标准值
        thicknesses = round_thickness_to_standard(np.array(thicknesses))
        part_stats['max_thickness_mm'] = thicknesses
        
        # 找出无法满足要求的点（厚度为nan的点）
        unsatisfied_mask = pd.isna(part_stats['max_thickness_mm'])
        if unsatisfied_mask.any():
            unsatisfied_points = part_stats[unsatisfied_mask].copy()
    except (ImportError, ValueError) as e:
        part_stats['max_thickness_mm'] = np.nan
        print(f"警告: 厚度计算功能不可用: {e}")
    finally:
        os.chdir(original_cwd)
    
    # 按身体部分排序（定义顺序）
    part_order = [
        "Left_Shoulder", "Right_Shoulder",
        "Left_Elbow", "Right_Elbow",
        "Torso",
        "Left_Hip", "Right_Hip",
        "Left_Knee", "Right_Knee",
        "Unknown"
    ]
    
    # 创建排序键
    def sort_key(row):
        part = row['body_part']
        if part in part_order:
            return part_order.index(part)
        return len(part_order)
    
    part_stats['sort_key'] = part_stats.apply(sort_key, axis=1)
    part_stats = part_stats.sort_values('sort_key').drop('sort_key', axis=1)
    
    return part_stats, unsatisfied_points


def _apply_body_part_chunk(args):
    """辅助函数：对数据块应用get_body_part（用于多进程）"""
    chunk_df, original_indices = args
    results = []
    # chunk_df 的索引已经被重置为 0, 1, 2, ...
    # original_indices 是原始索引的列表
    for i, (idx, row) in enumerate(chunk_df.iterrows()):
        original_idx = original_indices[i] if i < len(original_indices) else idx
        results.append((original_idx, get_body_part(row)))
    return results




def read_csv_with_progress(csv_path, usecols=None, encoding='utf-8'):
    """
    读取CSV文件，支持分块读取和进度条显示（单进程）
    contact_data.bin 时整文件读入（无分块）。
    """
    if str(csv_path).lower().endswith('.bin'):
        _sd = os.path.dirname(os.path.abspath(__file__))
        if _sd not in sys.path:
            sys.path.insert(0, _sd)
        from mujoco_data_io import load_contact_file
        df = load_contact_file(csv_path)
        if usecols is not None:
            keep = [c for c in usecols if c in df.columns]
            if keep:
                df = df[keep]
        return df
    file_size = os.path.getsize(csv_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if HAS_TQDM and file_size_mb > 100:  # 大于100MB时使用分块读取
        # 根据文件大小调整块大小：大文件使用更大的块
        if file_size_mb > 10000:  # 大于10GB
            chunk_size = 200000  # 每次读取20万行
        elif file_size_mb > 1000:  # 大于1GB
            chunk_size = 100000  # 每次读取10万行
        else:
            chunk_size = 50000  # 每次读取5万行
        
        chunks = []
        
        # 计算总行数（快速估算）
        try:
            with open(csv_path, 'rb') as f:
                # 读取前几行来估算平均行长度
                sample_lines = []
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    sample_lines.append(len(line))
                if sample_lines:
                    avg_line_len = sum(sample_lines) / len(sample_lines)
                    total_lines = int(file_size / avg_line_len) - 1
                else:
                    # 回退到逐行计数
                    with open(csv_path, 'r', encoding=encoding, errors='ignore') as f2:
                        total_lines = sum(1 for _ in f2) - 1
        except:
            # 回退到逐行计数
            try:
                with open(csv_path, 'r', encoding=encoding, errors='ignore') as f:
                    total_lines = sum(1 for _ in f) - 1
            except:
                alt_encoding = 'latin-1' if encoding == 'utf-8' else 'utf-8'
                with open(csv_path, 'r', encoding=alt_encoding, errors='ignore') as f:
                    total_lines = sum(1 for _ in f) - 1
                encoding = alt_encoding
        
        total_chunks = (total_lines // chunk_size) + 1
        
        # 单进程读取（使用C引擎，更快）
        print(f"      使用单进程分块读取 {total_chunks} 个块（每块约 {chunk_size:,} 行）...")
        try:
            reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, 
                               engine='c', encoding=encoding, low_memory=False, memory_map=True)
            chunk_count = 0
            for chunk in tqdm(reader, total=total_chunks, desc="      读取进度", unit="块", ncols=80):
                chunks.append(chunk)
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print_memory_usage(f"(已读取 {chunk_count} 块)")
            
            print_memory_usage("(读取完成，开始合并)")
            
            # 分批合并chunks以减少内存峰值
            if len(chunks) > 10:
                print(f"      分批合并 {len(chunks)} 个块以减少内存使用...")
                # 每次合并10个块
                batch_size = 10
                merged_chunks = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    merged = pd.concat(batch, ignore_index=True)
                    merged_chunks.append(merged)
                    # 释放原始chunks
                    del batch
                    gc.collect()
                    
                    if (i // batch_size + 1) % 5 == 0:  # 每5批打印一次
                        print_memory_usage(f"(已合并 {i+batch_size}/{len(chunks)} 块)")
                
                # 最终合并
                if len(merged_chunks) > 1:
                    print(f"      最终合并 {len(merged_chunks)} 个批次...")
                    df = pd.concat(merged_chunks, ignore_index=True)
                    del merged_chunks
                else:
                    df = merged_chunks[0]
                gc.collect()
            else:
                df = pd.concat(chunks, ignore_index=True)
            
            # 释放chunks内存
            del chunks
            gc.collect()
            print_memory_usage("(合并完成)")
        except (UnicodeDecodeError, UnicodeError, ValueError):
            # 如果C引擎失败，尝试Python引擎
            print("      C引擎读取失败，尝试Python引擎...")
            try:
                reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, 
                                   engine='python', encoding=encoding)
                chunks = []
                chunk_count = 0
                for chunk in tqdm(reader, total=total_chunks, desc="      读取进度", unit="块", ncols=80):
                    chunks.append(chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print_memory_usage(f"(已读取 {chunk_count} 块)")
                
                print_memory_usage("(读取完成，开始合并)")
                
                # 分批合并
                if len(chunks) > 10:
                    batch_size = 10
                    merged_chunks = []
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        merged = pd.concat(batch, ignore_index=True)
                        merged_chunks.append(merged)
                        del batch
                        gc.collect()
                        if (i // batch_size + 1) % 5 == 0:
                            print_memory_usage(f"(已合并 {i+batch_size}/{len(chunks)} 块)")
                    
                    if len(merged_chunks) > 1:
                        df = pd.concat(merged_chunks, ignore_index=True)
                        del merged_chunks
                    else:
                        df = merged_chunks[0]
                    gc.collect()
                else:
                    df = pd.concat(chunks, ignore_index=True)
                
                del chunks
                gc.collect()
                print_memory_usage("(合并完成)")
            except (UnicodeDecodeError, UnicodeError):
                # 如果编码失败，尝试另一种编码
                alt_encoding = 'latin-1' if encoding == 'utf-8' else 'utf-8'
                print(f"      尝试使用 {alt_encoding.upper()} 编码...")
                reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, 
                                   engine='python', encoding=alt_encoding)
                chunks = []
                chunk_count = 0
                for chunk in tqdm(reader, total=total_chunks, desc="      读取进度", unit="块", ncols=80):
                    chunks.append(chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print_memory_usage(f"(已读取 {chunk_count} 块)")
                
                print_memory_usage("(读取完成，开始合并)")
                
                # 分批合并
                if len(chunks) > 10:
                    batch_size = 10
                    merged_chunks = []
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        merged = pd.concat(batch, ignore_index=True)
                        merged_chunks.append(merged)
                        del batch
                        gc.collect()
                        if (i // batch_size + 1) % 5 == 0:
                            print_memory_usage(f"(已合并 {i+batch_size}/{len(chunks)} 块)")
                    
                    if len(merged_chunks) > 1:
                        df = pd.concat(merged_chunks, ignore_index=True)
                        del merged_chunks
                    else:
                        df = merged_chunks[0]
                    gc.collect()
                else:
                    df = pd.concat(chunks, ignore_index=True)
                
                del chunks
                gc.collect()
                print_memory_usage("(合并完成)")
    else:
        # 小文件直接读取（使用C引擎，更快）
        try:
            df = pd.read_csv(csv_path, usecols=usecols, engine='c', encoding=encoding, 
                           low_memory=False, memory_map=True)
        except (UnicodeDecodeError, UnicodeError, ValueError):
            # 如果C引擎失败，尝试Python引擎
            try:
                df = pd.read_csv(csv_path, usecols=usecols, engine='python', encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                alt_encoding = 'latin-1' if encoding == 'utf-8' else 'utf-8'
                df = pd.read_csv(csv_path, usecols=usecols, engine='python', encoding=alt_encoding)
    
    return df


def apply_body_part_multiprocess(df, n_jobs=None):
    """
    使用多进程对DataFrame应用get_body_part函数
    
    参数:
        df: DataFrame
        n_jobs: 并行处理的进程数，None表示自动选择
    
    返回:
        df: 添加了body_part列的DataFrame
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 2) if len(df) > 10000 else 1
    
    if n_jobs > 1 and len(df) > 10000:  # 只有数据量大时才使用多进程
        chunk_size = max(1000, len(df) // (n_jobs * 4))
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            original_indices = chunk.index.tolist()
            chunk = chunk.reset_index(drop=True)
            chunks.append((chunk, original_indices))
        
        with Pool(processes=n_jobs) as pool:
            if HAS_TQDM:
                results_list = list(tqdm(pool.imap(_apply_body_part_chunk, chunks), 
                                        total=len(chunks), desc="      分类进度", ncols=80))
            else:
                results_list = pool.map(_apply_body_part_chunk, chunks)
        
        body_part_dict = {}
        for results in results_list:
            for orig_idx, body_part in results:
                body_part_dict[orig_idx] = body_part
        
        df['body_part'] = df.index.map(body_part_dict)
    else:
        # 单进程处理
        if HAS_TQDM:
            tqdm.pandas(desc="      分类进度", ncols=80)
            df['body_part'] = df.progress_apply(get_body_part, axis=1)
        else:
            df['body_part'] = df.apply(get_body_part, axis=1)
    
    return df


def filter_elbow_forces(df, force_column='force_normal', n_jobs=None):
    """
    过滤掉elbow部位的特定数据：
    1. z坐标在0.5-0.7m之间的elbow部位的所有数据
    2. z坐标在0.6-0.85m之间的elbow部位大于10kN的力
    
    参数:
        df: DataFrame，包含body2_name, robot_frame_z, force_normal或force_magnitude列
        force_column: 使用的力列名，'force_normal'或'force_magnitude'（默认'force_normal'）
        n_jobs: 并行处理的进程数，None表示使用所有CPU核心
    
    返回:
        df_filtered: 过滤后的DataFrame
    """
    if 'body2_name' not in df.columns or 'robot_frame_z' not in df.columns:
        return df
    
    before_filter = len(df)
    
    # 添加身体部分列（使用多进程加速，如果还没有body_part列）
    df = df.copy()
    
    if 'body_part' not in df.columns:
        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 2) if len(df) > 10000 else 1
        
        if len(df) > 10000:
            print(f"      正在分类身体部分（{len(df)} 行数据，使用 {n_jobs} 个进程）...")
        else:
            print(f"      正在分类身体部分（{len(df)} 行数据）...")
        df = apply_body_part_multiprocess(df, n_jobs=n_jobs)
    else:
        print(f"      跳过分类（已有body_part列）")
    
    # 识别elbow部位
    elbow_mask = df['body_part'].isin(['Left_Elbow', 'Right_Elbow'])
    
    # 过滤条件1：z坐标在0.5-0.75m之间的elbow部位的所有数据
    z_mask_1 = (df['robot_frame_z'] >= 0.5) & (df['robot_frame_z'] <= 0.75)
    filter_mask_1 = elbow_mask & z_mask_1
    
    # 过滤条件2：z坐标在0.6-0.85m之间的elbow部位大于10kN的力
    z_mask_2 = (df['robot_frame_z'] >= 0.6) & (df['robot_frame_z'] <= 0.85)
    force_mask = df[force_column] > 10000
    filter_mask_2 = elbow_mask & z_mask_2 & force_mask
    
    # 组合所有过滤条件
    filter_mask = filter_mask_1 | filter_mask_2
    
    # 过滤掉满足条件的数据
    df = df[~filter_mask]
    after_filter = len(df)
    
    if before_filter > after_filter:
        filtered_count_1 = np.sum(filter_mask_1)
        filtered_count_2 = np.sum(filter_mask_2)
        filtered_count = before_filter - after_filter
        print(f"过滤掉elbow部位数据: {filtered_count} 行")
        print(f"  - z坐标在0.5-0.75m之间的elbow部位: {filtered_count_1} 行")
        print(f"  - z坐标在0.6-0.85m之间且大于10kN的elbow部位: {filtered_count_2} 行")
        print(f"过滤后的数据: {after_filter} 行")
    
    # 注意：不删除body_part列，以便后续函数可以复用
    # df = df.drop('body_part', axis=1)
    
    return df


def find_max_force_per_position(df, force_column='force_normal', enable_force_filter=True):
    """
    对于原始CSV文件，找到每个碰撞点位置的最大力，并过滤掉头部和踝关节链接
    
    参数:
        df: DataFrame，包含robot_frame_x, robot_frame_y, robot_frame_z, force_magnitude或force_normal, body2_name列
        force_column: 使用的力列名，'force_normal'或'force_magnitude'（默认'force_normal'）
    
    返回:
        df_max: DataFrame，每个唯一位置的最大力（已过滤头部和踝关节）
    """
    if not HAS_TQDM:
        print("检测到原始CSV文件，正在按位置分组并找到每个位置的最大力...")
    
    # 检查力列是否存在
    if force_column not in df.columns:
        # 尝试使用另一个列
        if force_column == 'force_normal' and 'force_magnitude' in df.columns:
            print(f"警告: 未找到{force_column}列，使用force_magnitude代替")
            force_column = 'force_magnitude'
        elif force_column == 'force_magnitude' and 'force_normal' in df.columns:
            print(f"警告: 未找到{force_column}列，使用force_normal代替")
            force_column = 'force_normal'
        else:
            raise ValueError(f"错误: 未找到力列 {force_column}，也无法找到替代列")
    
    # 定义要过滤的链接（头部和踝关节）
    excluded_links = [
        'LINK_HEAD_YAW',  # 头部
        'LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R',  # 踝关节横滚
        'LINK_ANKLE_PITCH_L', 'LINK_ANKLE_PITCH_R'  # 踝关节俯仰
    ]
    
    # 如果有body2_name列，过滤掉头部和踝关节链接
    if 'body2_name' in df.columns:
        before_filter = len(df)
        
        # 显示将被过滤的链接
        filtered_links = df[df['body2_name'].isin(excluded_links)]['body2_name'].unique()
        if len(filtered_links) > 0:
            print(f"将被过滤的链接: {list(filtered_links)}")
        
        # 过滤掉头部和踝关节链接
        df = df[~df['body2_name'].isin(excluded_links)]
        after_filter = len(df)
        
        if before_filter > after_filter:
            filtered_count = before_filter - after_filter
            print(f"过滤掉头部和踝关节链接: {filtered_count} 行")
            print(f"过滤后的数据: {after_filter} 行")
        else:
            print("未找到需要过滤的头部和踝关节链接")
    else:
        print("警告: 未找到body2_name列，无法过滤头部和踝关节链接")
    
    # 过滤掉z坐标在0.6-0.85m之间的elbow部位大于10kN的力
    # 注意：filter_elbow_forces内部会检查是否有body_part列，如果有就跳过分类
    if enable_force_filter:
        df = filter_elbow_forces(df, force_column=force_column, n_jobs=None)
    
    # 按位置分组，找到每个位置的最大力
    # 使用round来避免浮点数精度问题导致的位置重复
    position_cols = ['robot_frame_x', 'robot_frame_y', 'robot_frame_z']
    
    # 检查是否有robot_frame_x列，如果没有则只使用y和z
    if 'robot_frame_x' not in df.columns:
        position_cols = ['robot_frame_y', 'robot_frame_z']
        print("注意: 未找到robot_frame_x列，仅使用robot_frame_y和robot_frame_z进行分组")
    
    # 按位置分组，找到每个位置的最大力，并保留最大力对应的行的所有信息
    # 使用更高效的方法：先找到每个组的最大力索引，然后直接选择这些行
    if HAS_TQDM:
        print(f"      正在按位置分组（{len(df)} 行数据）...")
        idx_max = df.groupby(position_cols, group_keys=False)[force_column].idxmax()
        print(f"      正在选择最大力对应的行...")
        df_max = df.loc[idx_max].reset_index(drop=True)
    else:
        idx_max = df.groupby(position_cols, group_keys=False)[force_column].idxmax()
        df_max = df.loc[idx_max].reset_index(drop=True)
    
    print(f"      原始数据: {len(df)} 行")
    print(f"      唯一位置: {len(df_max)} 个")
    print(f"      数据压缩率: {len(df_max)/len(df)*100:.1f}%")
    
    return df_max


def plot_contact_grid(csv_path=None, df=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                     margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                     enable_force_filter=True):
    """
    绘制接触力数据的网格颜色图
    
    参数:
        csv_path: CSV文件路径（如果df为None则使用此参数读取）
        df: 已读取的DataFrame（可选，如果提供则直接使用，不读取csv_path）
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射（如果为None，则使用白色到红色的默认映射）
        figsize: 图片大小
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
    """
    # 如果没有指定颜色映射，使用白色到红色的默认映射
    if cmap is None:
        cmap = create_white_to_red_cmap()
    elif isinstance(cmap, str):
        # 如果是指定的字符串，尝试使用matplotlib内置的colormap
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"警告: 无法找到颜色映射 '{cmap}'，使用默认的白色到红色映射")
            cmap = create_white_to_red_cmap()
    
    # 读取CSV文件（如果df未提供）
    if df is None:
        if csv_path is None:
            raise ValueError("必须提供csv_path或df参数")
        print(f"正在读取CSV文件: {csv_path}")
        df = read_csv_with_progress(csv_path)
    else:
        df = df.copy()
    
    # 确定使用的力列（优先使用force_normal，与统计部分保持一致）
    force_column = 'force_normal' if 'force_normal' in df.columns else 'force_magnitude'
    if force_column not in df.columns:
        raise ValueError(f"CSV文件缺少必要的力列: force_normal 和 force_magnitude 都不存在")
    
    # 检查必要的列是否存在
    required_columns = ['robot_frame_z', 'robot_frame_y', force_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    print(f"绘图使用力列: {force_column}")
    
    # 检查数据是否已经处理过（如果已经有body_part列，且数据行数较少，说明已经处理过）
    # 原始数据通常有数千万行，处理后会减少到数百万行
    is_already_processed = 'body_part' in df.columns and len(df) < 5000000
    
    if not is_already_processed:
        # 首先过滤，只保留body1_name是'world'的数据
        if 'body1_name' in df.columns:
            before_filter = len(df)
            df = df[df['body1_name'] == 'world'].copy()
            after_filter = len(df)
            if before_filter > after_filter:
                print(f"过滤body1_name，只保留'world': {before_filter} -> {after_filter} 行")
        
        # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
        if csv_path is not None:
            csv_file = Path(csv_path)
            is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
        else:
            # 如果csv_path为None，只通过contact_count列判断
            is_clustered = 'contact_count' in df.columns
        
        if not is_clustered:
            # 原始CSV：按位置分组，找到每个位置的最大力
            df = find_max_force_per_position(df, force_column=force_column, enable_force_filter=enable_force_filter)
        else:
            print("检测到聚类后的CSV文件，应用过滤...")
            # 对于聚类后的CSV，也需要应用elbow过滤
            if enable_force_filter:
                df = filter_elbow_forces(df, force_column=force_column)
    else:
        print(f"数据已处理过（{len(df)} 行），跳过重复处理")
    
    # 提取数据
    z = df['robot_frame_z'].values
    y = df['robot_frame_y'].values
    force = df[force_column].values
    
    # 将 y 从 cm 转换为 m（如果数据是 cm 单位）
    # 根据标签显示为 cm，但实际数据可能是 m，这里先假设是 cm 需要转换
    # 如果数据已经是 m，转换不会影响（除以 100 再乘以 100 会恢复原值）
    # 但为了安全，我们检查数据范围：如果最大值小于 1，可能是 m；如果大于 10，可能是 cm
    if len(y) > 0 and np.abs(y).max() > 10:
        # 数据可能是 cm，转换为 m
        y = y / 100.0
    
    # 将力从N转换为kN
    force_kn = force / 1000.0
    
    # 限制z坐标在0.8~1.0范围内的力值小于等于20kN
    if enable_force_filter:
        force_kn = limit_force_in_z_range(force_kn, z, z_min=0.8, z_max=1.0, max_force=20.0)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置绘图区域边距（百分比，仅针对绘图区域，不包括标题）
    # 将百分比转换为0-1之间的值
    left = margin_left / 100.0
    right_margin = margin_right / 100.0
    bottom = margin_bottom / 100.0
    top = 1.0 - margin_top / 100.0
    
    # 为颜色条预留空间（约5%），然后设置右边距
    # 颜色条会占用一些空间，所以需要调整right值
    colorbar_space = 0.05  # 颜色条占用的空间比例
    right = 1.0 - right_margin - colorbar_space
    
    # 先设置边距
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 使用hexbin创建网格颜色图
    # x轴: robot_frame_y, y轴: robot_frame_z
    # 使用kN作为颜色值，固定颜色条范围为0-50kN
    # 使用np.max而不是np.mean，以显示每个网格单元内的最大力（与原始CSV的处理方式一致）
    hb = ax.hexbin(y, z, C=force_kn, gridsize=bins, cmap=cmap, reduce_C_function=np.max, 
                   vmin=0.0, vmax=50.0)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    # 隐藏颜色条框线
    cb.outline.set_visible(False)
    # 设置颜色条刻度数字字体为 Times New Roman, 8pt
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.1, 1.2)
    
    # 设置x轴和y轴刻度
    ax.set_xticks([-0.35, 0, 0.35])
    ax.set_yticks([0.1, 0.5, 0.9, 1.2])
    
    # 设置坐标轴刻度数字字体为 Times New Roman, 8pt
    for label in ax.get_xticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 只显示左轴和下轴，隐藏上轴和右轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # 添加网格
    # ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0, transparent=True)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compute_per_point_thickness_for_thickness_grid(
    df,
    csv_path=None,
    density=0.4,
    target_force=3.0,
    method="chr",
    target_pressure=None,
    force_12mm_in_z_range=True,
    enable_force_filter=True,
):
    """
    与 plot_thickness_grid 中生成厚度图的数据管道完全一致（供厚度 hexbin 与 YZ 护具 TSV 共用）：
    world 过滤、按位最大力或聚类肘过滤、y 单位换算、逐点 calculate_thicknesses、
    body_part、knee 12mm / elbow 6mm、无法满足目标时改为 24mm。

    返回:
        y, z, x: 与逐点 thicknesses 对齐的坐标（y 已为米）
        thicknesses: 每个接触点的厚度 (mm)
        force_column: 使用的力列名
    """
    df = df.copy()

    force_column = "force_normal" if "force_normal" in df.columns else "force_magnitude"
    if force_column not in df.columns:
        raise ValueError("CSV文件缺少必要的力列: force_normal 和 force_magnitude 都不存在")

    required_columns = ["robot_frame_z", "robot_frame_y", force_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")

    print(f"绘图使用力列: {force_column}")

    is_already_processed = "body_part" in df.columns and len(df) < 5000000

    if not is_already_processed:
        if "body1_name" in df.columns:
            before_filter = len(df)
            df = df[df["body1_name"] == "world"].copy()
            after_filter = len(df)
            if before_filter > after_filter:
                print(f"过滤body1_name，只保留'world': {before_filter} -> {after_filter} 行")

        if csv_path is not None:
            csv_file = Path(csv_path)
            is_clustered = "clustered" in csv_file.stem.lower() or "contact_count" in df.columns
        else:
            is_clustered = "contact_count" in df.columns

        if not is_clustered:
            df = find_max_force_per_position(df, force_column=force_column, enable_force_filter=enable_force_filter)
        else:
            print("检测到聚类后的CSV文件，应用过滤...")
            if enable_force_filter:
                df = filter_elbow_forces(df, force_column=force_column)
    else:
        print(f"数据已处理过（{len(df)} 行），跳过重复处理")

    z = df["robot_frame_z"].values
    y = df["robot_frame_y"].values
    force = df[force_column].values
    x = df["robot_frame_x"].values if "robot_frame_x" in df.columns else np.zeros(len(df), dtype=float)

    if len(y) > 0 and np.abs(y).max() > 10:
        y = y / 100.0
        df["robot_frame_y"] = y

    if method == "chr":
        print(f"正在计算保护层厚度（方法={method}, 密度={density}, 目标力={target_force}kN）...")
    else:
        print(f"正在计算保护层厚度（方法={method}, 目标压强={target_pressure}MPa）...")
    thicknesses = calculate_thicknesses(
        force, density=density, target_force=target_force, method=method, target_pressure=target_pressure
    )

    if "body_part" not in df.columns:
        print(f"      正在分类身体部分以应用knee区域厚度提升...")
        if len(df) > 10000:
            n_jobs = max(1, cpu_count() - 2)
            df = apply_body_part_multiprocess(df, n_jobs=n_jobs)
        else:
            df = apply_body_part_multiprocess(df, n_jobs=1)

    if force_12mm_in_z_range:
        thicknesses = set_thickness_to_12mm_in_z_range(thicknesses, df, z)
        thicknesses = set_elbow_thickness_to_6mm(thicknesses, df, z)

    invalid_mask = np.isnan(thicknesses)
    invalid_count = np.sum(invalid_mask)
    if invalid_count > 0:
        print(f"警告: {invalid_count} 个接触点无法满足目标要求（即使使用最大厚度）")

        invalid_df = df[invalid_mask].copy()
        invalid_df["force_kn"] = invalid_df[force_column] / 1000.0
        invalid_df["thickness_mm"] = np.nan

        if "body_part" not in invalid_df.columns:
            if len(invalid_df) > 1000:
                n_jobs = max(1, cpu_count() - 2)
                print(f"      正在分类无法满足要求的点（{len(invalid_df)} 行数据，使用 {n_jobs} 个进程）...")
                invalid_df = apply_body_part_multiprocess(invalid_df, n_jobs=n_jobs)
            else:
                invalid_df["body_part"] = invalid_df.apply(get_body_part, axis=1)
        else:
            print(f"      跳过分类（已有body_part列）")

        print(f"\n无法满足目标要求（目标力={target_force}kN，最大厚度24mm仍无法满足）的接触点:")
        print("-" * 200)
        print(
            f"{'身体部分':<20} {'力(N)':<15} {'力(kN)':<15} {'厚度(mm)':<15} {'位置(x,y,z)':<30} {'body1':<25} {'body2':<25} {'fail-type':<30}"
        )
        print("-" * 200)

        invalid_df_sorted = invalid_df.sort_values(by=force_column, ascending=False)
        for idx, row in invalid_df_sorted.iterrows():
            body_part = row["body_part"]
            force_n = row[force_column]
            force_kn = row["force_kn"]
            x_pos = row.get("robot_frame_x", 0.0) if "robot_frame_x" in row.index else 0.0
            y_pos = row.get("robot_frame_y", 0.0) if "robot_frame_y" in row.index else 0.0
            z_pos = row.get("robot_frame_z", 0.0) if "robot_frame_z" in row.index else 0.0
            body1_name = (
                str(row.get("body1_name", "N/A"))
                if "body1_name" in row.index and pd.notna(row.get("body1_name"))
                else "N/A"
            )
            body2_name = (
                str(row.get("body2_name", "N/A"))
                if "body2_name" in row.index and pd.notna(row.get("body2_name"))
                else "N/A"
            )
            fail_type = (
                str(row.get("fall_type_info", "N/A"))
                if "fall_type_info" in row.index and pd.notna(row.get("fall_type_info"))
                else "N/A"
            )

            position_str = f"({x_pos:.3f},{y_pos:.3f},{z_pos:.3f})"
            print(
                f"{body_part:<20} {force_n:<15.2f} {force_kn:<15.3f} {'N/A':<15} {position_str:<30} {body1_name:<25} {body2_name:<25} {fail_type:<30}"
            )

        print("-" * 200)
        print(f"（共 {invalid_count} 个点）\n")

        max_thickness = 24
        thicknesses = np.where(invalid_mask, max_thickness, thicknesses)

    return y, z, x, thicknesses, force_column


def _collect_tsv_nonzero_rows(grid_mm, y_centers, z_centers):
    """(robot_frame_y, robot_frame_z, thickness_mm) 列表，按 y,z 排序。"""
    rows = []
    for iz in range(grid_mm.shape[0]):
        for iy in range(grid_mm.shape[1]):
            v = float(grid_mm[iz, iy])
            if v > 0.5:
                rows.append((float(y_centers[iy]), float(z_centers[iz]), v))
    rows.sort(key=lambda t: (t[0], t[1]))
    return rows


def _collect_hex_nonzero_rows(offsets, hvals):
    """hexbin 有厚度数据的六边形中心 (y,z,mm)，按 y,z 排序。"""
    off = np.asarray(offsets, dtype=float)
    hv = np.asarray(hvals, dtype=float).ravel()
    n = min(len(off), len(hv))
    rows = []
    for i in range(n):
        v = float(hv[i])
        if v > 0.5:
            rows.append((float(off[i, 0]), float(off[i, 1]), v))
    rows.sort(key=lambda t: (t[0], t[1]))
    return rows


def _print_yz_hex_tsv_compare(side_label, hex_rows, tsv_rows, max_print=120):
    """
    打印「该侧 hexbin 非零六边形」与「TSV 非零矩形格心」供对照。
    说明：混合厚度 PNG 的色块是前后合一的 hexbin，与这里「单侧」列表不同。
    """
    print(f"\n  --- YZ 对照 [{side_label}] （hex=该侧 hexbin 非零格；TSV=0.05m 格心采样后非零）---")
    print(f"  hex 非零个数: {len(hex_rows)}  |  TSV 非零个数: {len(tsv_rows)}")
    print(f"  [hex {side_label}]  y(m)      z(m)      mm")
    for i, (yy, zz, mm) in enumerate(hex_rows[:max_print]):
        print(f"    {i+1:4d}  {yy:8.4f}  {zz:8.4f}  {mm:6.0f}")
    if len(hex_rows) > max_print:
        print(f"    ... 省略 {len(hex_rows) - max_print} 行 hex")
    print(f"  [TSV {side_label}]  y(m)      z(m)      mm")
    for i, (yy, zz, mm) in enumerate(tsv_rows[:max_print]):
        print(f"    {i+1:4d}  {yy:8.4f}  {zz:8.4f}  {mm:6.0f}")
    if len(tsv_rows) > max_print:
        print(f"    ... 省略 {len(tsv_rows) - max_print} 行 TSV")

    if hex_rows and tsv_rows:
        hx = np.array([[r[0], r[1]] for r in hex_rows], dtype=float)
        far = 0
        max_d = 0.0
        for yy, zz, _ in tsv_rows:
            d = np.hypot(hx[:, 0] - yy, hx[:, 1] - zz)
            dm = float(np.min(d))
            max_d = max(max_d, dm)
            if dm > 0.06:
                far += 1
        print(f"  粗检: TSV 非零格心到最近 hex 中心 最大距离 {max_d:.4f} m；>0.06m 的格数 {far}/{len(tsv_rows)}")


def _print_mixed_thickness_hex_positions(hb, max_print=120):
    """打印厚度 PNG 使用的混合 hexbin 非零格（与 *_hexbin_aggregate.csv 一致）。"""
    off = np.asarray(hb.get_offsets(), dtype=float)
    arr = np.asarray(hb.get_array(), dtype=float)
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    rows_all = _collect_hex_nonzero_rows(off, arr)
    y_lo, y_hi = THICKNESS_GRID_Y_LIM
    z_lo, z_hi = THICKNESS_GRID_Z_LIM
    rows_vis = [r for r in rows_all if y_lo <= r[0] <= y_hi and z_lo <= r[1] <= z_hi]
    print("\n  --- 混合厚度 PNG：hexbin 非零六边形中心（前后合一，与色块一致）---")
    print(f"  轴内非零: {len(rows_vis)} 个 | 全 extent 非零: {len(rows_all)} 个")
    print("  idx    y(m)        z(m)        mm")
    for i, (yy, zz, mm) in enumerate(rows_vis[:max_print]):
        print(f"    {i+1:4d}  {yy:8.4f}  {zz:8.4f}  {mm:6.0f}")
    if len(rows_vis) > max_print:
        print(f"    ... 省略 {len(rows_vis) - max_print} 行")


def plot_thickness_grid(csv_path=None, df=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                       density=0.4, target_force=3.0, margin_left=5.0, margin_right=5.0, 
                       margin_top=5.0, margin_bottom=5.0, method='chr', target_pressure=None,
                       force_12mm_in_z_range=True, enable_force_filter=True,
                       _yz_thickness_precomputed=None, hexbin_dump_csv=None,
                       print_mixed_hex_compare=False, compare_max_print=120):
    """
    绘制保护层厚度的网格颜色图
    
    参数:
        csv_path: CSV文件路径（如果df为None则使用此参数读取）
        df: 已读取的DataFrame（可选，如果提供则直接使用，不读取csv_path）
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射（如果为None，则使用橙色的默认映射）
        figsize: 图片大小
        density: 材料密度（默认0.4，仅用于chr方法）
        target_force: 目标衰减后的力（kN，默认3.0，仅用于chr方法）
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
        method: 计算方法，'chr'使用thickness_selection，'zzq'使用thickness_from_pressure（默认'chr'）
        target_pressure: 期望减小后的压强（MPa，仅用于zzq方法）
        force_12mm_in_z_range: 是否启用强制厚度设置（默认True）
                               包括：1) 将z坐标在(0.32, 0.48)范围内的点设置为12mm
                                    2) 将elbow/shoulder部位z坐标在(0.8, 1.0)范围内的点设置为6mm
        _yz_thickness_precomputed: 可选，``compute_per_point_thickness_for_thickness_grid`` 的返回值
            ``(y, z, x, thicknesses, force_column)``，传入则跳过重复计算（供 main 与 YZ TSV 共用）。
        hexbin_dump_csv: 若提供路径，将 PNG 所用 ``hexbin`` 的六边形中心与 ``thickness_mm`` 写入 CSV
            （与图同源，非像素解码；前后侧点混合，与 ``yz_map_front`` 仅 x>=0 不同）。
        print_mixed_hex_compare: 为 True 时在终端打印混合 hexbin 非零格位置（与 PNG 色块一致）。
        compare_max_print: 上述打印与 ``--compare-yz-hex-tsv`` 各表最多行数。
    """
    # 如果没有指定颜色映射，使用离散的橙色默认映射
    if cmap is None:
        cmap = create_discrete_orange_cmap()
    elif isinstance(cmap, str):
        # 如果是指定的字符串，尝试使用matplotlib内置的colormap
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"警告: 无法找到颜色映射 '{cmap}'，使用默认的离散橙色映射")
            cmap = create_discrete_orange_cmap()
    
    if _yz_thickness_precomputed is not None:
        y, z, _x, thicknesses, _fc_pre = _yz_thickness_precomputed
    else:
        if df is None:
            if csv_path is None:
                raise ValueError("必须提供csv_path或df参数")
            print(f"正在读取CSV文件: {csv_path}")
            df = read_csv_with_progress(csv_path)
        y, z, _x, thicknesses, _fc = compute_per_point_thickness_for_thickness_grid(
            df,
            csv_path=csv_path,
            density=density,
            target_force=target_force,
            method=method,
            target_pressure=target_pressure,
            force_12mm_in_z_range=force_12mm_in_z_range,
            enable_force_filter=enable_force_filter,
        )
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置绘图区域边距（百分比，仅针对绘图区域，不包括标题）
    # 将百分比转换为0-1之间的值
    left = margin_left / 100.0
    right_margin = margin_right / 100.0
    bottom = margin_bottom / 100.0
    top = 1.0 - margin_top / 100.0
    
    # 为颜色条预留空间（约5%），然后设置右边距
    # 颜色条会占用一些空间，所以需要调整right值
    colorbar_space = 0.05  # 颜色条占用的空间比例
    right = 1.0 - right_margin - colorbar_space
    
    # 先设置边距
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 创建离散颜色映射
    # 标准厚度值：0, 6, 12, 18, 24
    # 定义边界值来分隔5个区间
    # 边界值应该正好在标准值之间：0-3, 3-9, 9-15, 15-21, 21-24
    boundaries = [0, 3, 9, 15, 21, 24]  # 6个边界值定义5个区间
    # 创建离散归一化器，使用cmap的颜色数量
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    
    # 使用hexbin创建网格颜色图
    # x轴: robot_frame_y, y轴: robot_frame_z
    # 使用np.max以显示每个网格单元内的最大厚度（与力图保持一致）
    # 使用离散颜色映射
    hb = ax.hexbin(y, z, C=thicknesses, gridsize=bins, cmap=cmap, reduce_C_function=np.max,
                   norm=norm)

    if print_mixed_hex_compare:
        _print_mixed_thickness_hex_positions(hb, max_print=compare_max_print)
    
    # 添加颜色条，使用shrink参数控制大小
    # 使用离散颜色条，显示为块状
    # spacing='uniform' 确保每个色块大小相同
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02, 
                      ticks=[0, 6, 12, 18, 24], boundaries=boundaries, 
                      format='%g', spacing='uniform', extend='neither', 
                      drawedges=True)
    
    # 设置颜色条刻度标签为 0, 6, 12, 18, 24mm
    cb.set_ticklabels(['0', '6', '12', '18', '24'])
    # 隐藏颜色条框线
    cb.outline.set_visible(False)
    # 隐藏颜色条次刻度线
    cb.ax.tick_params(which='minor', length=0)
    # 设置颜色条刻度数字字体为 Times New Roman, 8pt
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 隐藏颜色条分隔线
    cb.dividers.set_visible(False)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.1, 1.2)
    
    # 设置x轴和y轴刻度
    ax.set_xticks([-0.35, 0, 0.35])
    ax.set_yticks([0.1, 0.5, 0.9, 1.2])
    
    # 设置坐标轴刻度数字字体为 Times New Roman, 8pt
    for label in ax.get_xticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 只显示左轴和下轴，粗1pt
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # 添加网格
    # ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0, transparent=True)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()

    if hexbin_dump_csv:
        write_thickness_hexbin_aggregate_csv(hb, hexbin_dump_csv, gridsize=bins, method=method)
    
    plt.close()


def plot_surface_area_grid(csv_path=None, df=None, stl_path=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                           margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                           search_radius=0.01, enable_force_filter=True):
    """
    绘制表面积数据的网格颜色图
    
    参数:
        csv_path: CSV文件路径（如果df为None则使用此参数读取）
        df: 已读取的DataFrame（可选，如果提供则直接使用，不读取csv_path）
        stl_path: STL文件路径
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射（如果为None，则使用蓝色的默认映射）
        figsize: 图片大小
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
        search_radius: 搜索半径（米），用于计算每个点周围的表面积
    """
    if not HAS_TRIMESH and not HAS_STL:
        raise ImportError("无法绘制表面积图：未找到trimesh或numpy-stl库，请安装: pip install trimesh")
    
    # 如果没有指定颜色映射，使用蓝色的默认映射
    if cmap is None:
        cmap = create_blue_cmap()
    elif isinstance(cmap, str):
        # 如果是指定的字符串，尝试使用matplotlib内置的colormap
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"警告: 无法找到颜色映射 '{cmap}'，使用默认的蓝色映射")
            cmap = create_blue_cmap()
    
    # 读取CSV文件（如果df未提供）
    if df is None:
        if csv_path is None:
            raise ValueError("必须提供csv_path或df参数")
        print(f"正在读取CSV文件: {csv_path}")
        df = read_csv_with_progress(csv_path)
    else:
        df = df.copy()
    
    # 确定使用的力列（优先使用force_normal，与统计部分保持一致）
    force_column = 'force_normal' if 'force_normal' in df.columns else 'force_magnitude'
    if force_column not in df.columns:
        raise ValueError(f"CSV文件缺少必要的力列: force_normal 和 force_magnitude 都不存在")
    
    # 检查必要的列是否存在
    required_columns = ['robot_frame_z', 'robot_frame_y', force_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    print(f"绘图使用力列: {force_column}")
    
    # 检查数据是否已经处理过（如果已经有body_part列，且数据行数较少，说明已经处理过）
    # 原始数据通常有数千万行，处理后会减少到数百万行
    is_already_processed = 'body_part' in df.columns and len(df) < 5000000
    
    if not is_already_processed:
        # 首先过滤，只保留body1_name是'world'的数据
        if 'body1_name' in df.columns:
            before_filter = len(df)
            df = df[df['body1_name'] == 'world'].copy()
            after_filter = len(df)
            if before_filter > after_filter:
                print(f"过滤body1_name，只保留'world': {before_filter} -> {after_filter} 行")
        
        # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
        if csv_path is not None:
            csv_file = Path(csv_path)
            is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
        else:
            # 如果csv_path为None，只通过contact_count列判断
            is_clustered = 'contact_count' in df.columns
        
        if not is_clustered:
            # 原始CSV：按位置分组，找到每个位置的最大力
            df = find_max_force_per_position(df, force_column=force_column, enable_force_filter=enable_force_filter)
        else:
            print("检测到聚类后的CSV文件，应用过滤...")
            # 对于聚类后的CSV，也需要应用elbow过滤
            if enable_force_filter:
                df = filter_elbow_forces(df, force_column=force_column)
    else:
        print(f"数据已处理过（{len(df)} 行），跳过重复处理")
    
    # 提取数据
    z = df['robot_frame_z'].values
    y = df['robot_frame_y'].values
    
    # 将 y 从 cm 转换为 m（如果数据是 cm 单位）
    if len(y) > 0 and np.abs(y).max() > 10:
        # 数据可能是 cm，转换为 m
        y = y / 100.0
    
    # 检查是否有robot_frame_x列，如果没有则使用0
    if 'robot_frame_x' in df.columns:
        x = df['robot_frame_x'].values
        if len(x) > 0 and np.abs(x).max() > 10:
            x = x / 100.0
    else:
        x = np.zeros_like(y)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置绘图区域边距（百分比，仅针对绘图区域，不包括标题）
    # 将百分比转换为0-1之间的值
    left = margin_left / 100.0
    right_margin = margin_right / 100.0
    bottom = margin_bottom / 100.0
    top = 1.0 - margin_top / 100.0
    
    # 为颜色条预留空间（约5%），然后设置右边距
    # 颜色条会占用一些空间，所以需要调整right值
    colorbar_space = 0.05  # 颜色条占用的空间比例
    right = 1.0 - right_margin - colorbar_space
    
    # 先设置边距
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 先创建hexbin网格以获取网格单元中心
    # 使用虚拟数据创建网格结构
    hb_temp = ax.hexbin(y, z, C=np.ones_like(y), gridsize=bins, reduce_C_function=np.mean)
    
    # 获取网格单元的中心坐标
    offsets = hb_temp.get_offsets()  # 获取所有网格单元的中心坐标 (N, 2) - (y, z)
    
    # 对于每个网格单元，计算其x坐标（使用该网格单元内所有点的x坐标平均值）
    # 检查是否有robot_frame_x列
    if 'robot_frame_x' in df.columns:
        x_data = df['robot_frame_x'].values
        if len(x_data) > 0 and np.abs(x_data).max() > 10:
            x_data = x_data / 100.0
    else:
        x_data = np.zeros_like(y)
    
    # 为每个网格单元找到对应的x坐标
    # 使用KD树找到每个网格单元中心对应的最近点，然后使用该点的x坐标
    try:
        from scipy.spatial import cKDTree
        point_tree = cKDTree(np.column_stack([y, z]))
        _, nearest_indices = point_tree.query(offsets)
        grid_x = x_data[nearest_indices]
    except ImportError:
        # 如果没有scipy，使用简单的平均值
        grid_x = np.zeros(len(offsets))
        print("警告: 未找到scipy，使用x=0作为网格单元的x坐标")
    
    # 构建网格单元中心坐标（3D）
    grid_centers = np.column_stack([grid_x, offsets[:, 0], offsets[:, 1]])
    
    # 清理临时图形
    ax.clear()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 计算每个网格单元的表面积
    surface_areas = calculate_surface_area_for_grid_cells(grid_centers, stl_path, search_radius=search_radius)
    
    # 创建映射：将网格单元的表面积映射回原始数据点
    # 对于每个原始点，找到它所属的网格单元
    try:
        from scipy.spatial import cKDTree
        grid_tree = cKDTree(offsets)
        _, point_to_grid = grid_tree.query(np.column_stack([y, z]))
        # 将网格单元的表面积分配给对应的点
        point_surface_areas = surface_areas[point_to_grid]
    except ImportError:
        # 如果没有scipy，使用简单的最近邻匹配
        print("警告: 未找到scipy，使用简单匹配方法")
        point_surface_areas = np.zeros_like(y)
        for i, (yi, zi) in enumerate(zip(y, z)):
            dists = np.sqrt((offsets[:, 0] - yi)**2 + (offsets[:, 1] - zi)**2)
            nearest_idx = np.argmin(dists)
            point_surface_areas[i] = surface_areas[nearest_idx]
    
    # 使用hexbin创建网格颜色图
    # x轴: robot_frame_y, y轴: robot_frame_z
    # 使用np.max以显示每个网格单元内的最大表面积（与其他图保持一致）
    hb = ax.hexbin(y, z, C=point_surface_areas, gridsize=bins, cmap=cmap, reduce_C_function=np.max)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    # 隐藏颜色条框线
    cb.outline.set_visible(False)
    # 设置颜色条刻度数字字体为 Times New Roman, 8pt
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.1, 1.2)
    
    # 设置x轴和y轴刻度
    ax.set_xticks([-0.35, 0, 0.35])
    ax.set_yticks([0.1, 0.5, 0.9, 1.2])
    
    # 设置坐标轴刻度数字字体为 Times New Roman, 8pt
    for label in ax.get_xticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 只显示左轴和下轴，隐藏上轴和右轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0, transparent=True)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pressure_grid(csv_path=None, df=None, stl_path=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                       margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                       search_radius=0.01, enable_force_filter=True):
    """
    绘制表面压强的网格颜色图（力/面积）
    
    参数:
        csv_path: CSV文件路径（如果df为None则使用此参数读取）
        df: 已读取的DataFrame（可选，如果提供则直接使用，不读取csv_path）
        stl_path: STL文件路径
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射（如果为None，则使用绿色到红色的默认映射）
        figsize: 图片大小
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
        search_radius: 搜索半径（米），用于计算每个点周围的表面积
    """
    if not HAS_TRIMESH and not HAS_STL:
        raise ImportError("无法绘制压强图：未找到trimesh或numpy-stl库，请安装: pip install trimesh")
    
    # 如果没有指定颜色映射，使用绿色到红色的默认映射（表示从低到高的压强）
    if cmap is None:
        colors = ['green', 'yellow', 'red']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('green_to_red', colors, N=n_bins)
    elif isinstance(cmap, str):
        # 如果是指定的字符串，尝试使用matplotlib内置的colormap
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"警告: 无法找到颜色映射 '{cmap}'，使用默认的绿色到红色映射")
            colors = ['green', 'yellow', 'red']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('green_to_red', colors, N=n_bins)
    
    # 读取CSV文件（如果df未提供）
    if df is None:
        if csv_path is None:
            raise ValueError("必须提供csv_path或df参数")
        print(f"正在读取CSV文件: {csv_path}")
        df = read_csv_with_progress(csv_path)
    else:
        df = df.copy()
    
    # 确定使用的力列（优先使用force_normal，与统计部分保持一致）
    force_column = 'force_normal' if 'force_normal' in df.columns else 'force_magnitude'
    if force_column not in df.columns:
        raise ValueError(f"CSV文件缺少必要的力列: force_normal 和 force_magnitude 都不存在")
    
    # 检查必要的列是否存在
    required_columns = ['robot_frame_z', 'robot_frame_y', force_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    print(f"绘图使用力列: {force_column}")
    
    # 检查数据是否已经处理过（如果已经有body_part列，且数据行数较少，说明已经处理过）
    # 原始数据通常有数千万行，处理后会减少到数百万行
    is_already_processed = 'body_part' in df.columns and len(df) < 5000000
    
    if not is_already_processed:
        # 首先过滤，只保留body1_name是'world'的数据
        if 'body1_name' in df.columns:
            before_filter = len(df)
            df = df[df['body1_name'] == 'world'].copy()
            after_filter = len(df)
            if before_filter > after_filter:
                print(f"过滤body1_name，只保留'world': {before_filter} -> {after_filter} 行")
        
        # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
        if csv_path is not None:
            csv_file = Path(csv_path)
            is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
        else:
            # 如果csv_path为None，只通过contact_count列判断
            is_clustered = 'contact_count' in df.columns
        
        if not is_clustered:
            # 原始CSV：按位置分组，找到每个位置的最大力
            df = find_max_force_per_position(df, force_column=force_column, enable_force_filter=enable_force_filter)
        else:
            print("检测到聚类后的CSV文件，应用过滤...")
            # 对于聚类后的CSV，也需要应用elbow过滤
            if enable_force_filter:
                df = filter_elbow_forces(df, force_column=force_column)
    else:
        print(f"数据已处理过（{len(df)} 行），跳过重复处理")
    
    # 提取数据
    z = df['robot_frame_z'].values
    y = df['robot_frame_y'].values
    force = df[force_column].values
    
    # 将 y 从 cm 转换为 m（如果数据是 cm 单位）
    if len(y) > 0 and np.abs(y).max() > 10:
        # 数据可能是 cm，转换为 m
        y = y / 100.0
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置绘图区域边距（百分比，仅针对绘图区域，不包括标题）
    # 将百分比转换为0-1之间的值
    left = margin_left / 100.0
    right_margin = margin_right / 100.0
    bottom = margin_bottom / 100.0
    top = 1.0 - margin_top / 100.0
    
    # 为颜色条预留空间（约5%），然后设置右边距
    # 颜色条会占用一些空间，所以需要调整right值
    colorbar_space = 0.05  # 颜色条占用的空间比例
    right = 1.0 - right_margin - colorbar_space
    
    # 先设置边距
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 先创建hexbin网格以获取网格单元中心
    # 使用虚拟数据创建网格结构
    hb_temp = ax.hexbin(y, z, C=np.ones_like(y), gridsize=bins, reduce_C_function=np.mean)
    
    # 获取网格单元的中心坐标
    offsets = hb_temp.get_offsets()  # 获取所有网格单元的中心坐标 (N, 2) - (y, z)
    
    # 对于每个网格单元，计算其x坐标（使用该网格单元内所有点的x坐标平均值）
    # 检查是否有robot_frame_x列
    if 'robot_frame_x' in df.columns:
        x_data = df['robot_frame_x'].values
        if len(x_data) > 0 and np.abs(x_data).max() > 10:
            x_data = x_data / 100.0
    else:
        x_data = np.zeros_like(y)
    
    # 为每个网格单元找到对应的x坐标
    # 使用KD树找到每个网格单元中心对应的最近点，然后使用该点的x坐标
    try:
        from scipy.spatial import cKDTree
        point_tree = cKDTree(np.column_stack([y, z]))
        _, nearest_indices = point_tree.query(offsets)
        grid_x = x_data[nearest_indices]
    except ImportError:
        # 如果没有scipy，使用简单的平均值
        grid_x = np.zeros(len(offsets))
        print("警告: 未找到scipy，使用x=0作为网格单元的x坐标")
    
    # 构建网格单元中心坐标（3D）
    grid_centers = np.column_stack([grid_x, offsets[:, 0], offsets[:, 1]])
    
    # 清理临时图形
    ax.clear()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 计算每个网格单元的表面积
    print("正在计算网格单元的表面积...")
    surface_areas = calculate_surface_area_for_grid_cells(grid_centers, stl_path, search_radius=search_radius)
    
    # 验证表面积值的合理性（应该在合理的范围内）
    if len(surface_areas) > 0:
        print(f"表面积统计: min={surface_areas.min():.6e} m², max={surface_areas.max():.6e} m², mean={surface_areas.mean():.6e} m²")
        # 如果表面积太小（可能是单位错误），给出警告
        if surface_areas.max() < 1e-6:
            print("警告: 表面积值异常小，可能存在单位转换问题")
        elif surface_areas.max() > 1.0:
            print("警告: 表面积值异常大，可能存在单位转换问题")
    
    # 创建映射：将网格单元的表面积映射回原始数据点
    # 对于每个原始点，找到它所属的网格单元
    try:
        from scipy.spatial import cKDTree
        grid_tree = cKDTree(offsets)
        _, point_to_grid = grid_tree.query(np.column_stack([y, z]))
        # 将网格单元的表面积分配给对应的点
        point_surface_areas = surface_areas[point_to_grid]
    except ImportError:
        # 如果没有scipy，使用简单的最近邻匹配
        print("警告: 未找到scipy，使用简单匹配方法")
        point_surface_areas = np.zeros_like(y)
        for i, (yi, zi) in enumerate(zip(y, z)):
            dists = np.sqrt((offsets[:, 0] - yi)**2 + (offsets[:, 1] - zi)**2)
            nearest_idx = np.argmin(dists)
            point_surface_areas[i] = surface_areas[nearest_idx]
    
    # 计算压强：压强 = 力 / 面积
    # 力单位：N，面积单位：m²，压强单位：Pa (N/m²)
    # 避免除以0
    pressure = np.where(point_surface_areas > 1e-10, force / point_surface_areas, 0.0)
    
    # 将压强从Pa转换为MPa以便显示
    # 1 MPa = 1,000,000 Pa
    pressure_mpa = pressure / 1e6
    
    print(f"压强范围: {pressure_mpa.min():.4f} - {pressure_mpa.max():.4f} MPa")
    
    # 使用hexbin创建网格颜色图
    # x轴: robot_frame_y, y轴: robot_frame_z
    # 使用np.max以显示每个网格单元内的最大压强（与其他图保持一致）
    hb = ax.hexbin(y, z, C=pressure_mpa, gridsize=bins, cmap=cmap, reduce_C_function=np.max)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    # 隐藏颜色条框线
    cb.outline.set_visible(False)
    # 设置颜色条刻度数字字体为 Times New Roman, 8pt
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.1, 1.2)
    
    # 设置x轴和y轴刻度
    ax.set_xticks([-0.35, 0, 0.35])
    ax.set_yticks([0.1, 0.5, 0.9, 1.2])
    
    # 设置坐标轴刻度数字字体为 Times New Roman, 8pt
    for label in ax.get_xticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(8)
    
    # 只显示左轴和下轴，隐藏上轴和右轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0, transparent=True)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


# YZ 护具 map 网格：与 protector_map/yz_map_front.tsv、yz_map_back.tsv 一致
YZ_MAP_Y_MIN, YZ_MAP_Y_MAX, YZ_MAP_STEP = -0.4, 0.4, 0.05
YZ_MAP_Z_MIN, YZ_MAP_Z_MAX = 0.0, 1.4

# 厚度/力图坐标轴显示范围（与 plot_thickness_grid 中 set_xlim/set_ylim 一致；轴外为白边）
THICKNESS_GRID_Y_LIM = (-0.35, 0.35)  # 横轴 robot_frame_y (m)
THICKNESS_GRID_Z_LIM = (0.1, 1.2)  # 纵轴 robot_frame_z (m)


def write_thickness_hexbin_aggregate_csv(hb, csv_path, gridsize, method="chr"):
    """
    写出与厚度 PNG 中 ``ax.hexbin`` **完全一致**的聚合表（非 PNG 栅格像素）。
    每行：六边形中心 ``robot_frame_y``、``robot_frame_z``、``thickness_mm``（即 ``reduce_C_function=max`` 结果）。
    """
    off = np.asarray(hb.get_offsets(), dtype=float)
    arr = np.asarray(hb.get_array(), dtype=float)
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    n = min(len(off), len(arr))
    if n == 0:
        print(f"  警告: hexbin 无单元，未写入 {csv_path}")
        return
    off = off[:n]
    arr = arr[:n]
    y_lo, y_hi = THICKNESS_GRID_Y_LIM
    z_lo, z_hi = THICKNESS_GRID_Z_LIM
    inside = (off[:, 0] >= y_lo) & (off[:, 0] <= y_hi) & (off[:, 1] >= z_lo) & (off[:, 1] <= z_hi)
    df = pd.DataFrame(
        {
            "robot_frame_y": off[:, 0],
            "robot_frame_z": off[:, 1],
            "thickness_mm": arr,
            "inside_plot_axes": inside,
        }
    )
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Same hexbin as thickness PNG: gridsize={gridsize}, method={method}, reduce_C_function=max\n"
        )
        f.write(
            "# NOT raster pixels; one row per hex cell with data. Compare yz_map_front.tsv (x>=0 only) separately.\n"
        )
    df.to_csv(csv_path, mode="a", index=False)
    n_in = int(np.sum(inside))
    print(f"  已写出 hexbin 聚合 CSV（与 PNG 同源）: {csv_path}  (共 {n} 个有数据六边形, 轴内约 {n_in} 个)")
    vc = df.loc[inside, "thickness_mm"].value_counts().sort_index()
    if len(vc):
        print("  轴内 thickness_mm 计数:")
        for val, cnt in vc.items():
            print(f"    {val:g} mm: {cnt:d}")


def _yz_map_tsv_suffix_from_csv(csv_path: str) -> str:
    """
    从 CSV 所在父目录名取后缀，便于同目录多批数据不互相覆盖。
    优先匹配目录名中的 YYYYMMDD_HHMMSS（取最后一个），否则用父目录名做安全化短后缀。
    例如 8dir-200.0N-0.4s-20260317_122531 -> _20260317_122531
    """
    parent_name = Path(csv_path).resolve().parent.name
    matches = re.findall(r"(\d{8}_\d{6})", parent_name)
    if matches:
        return f"_{matches[-1]}"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", parent_name).strip("_")
    if not safe or safe == ".":
        return ""
    if len(safe) > 64:
        safe = safe[:64]
    return f"_{safe}"


def _write_yz_map_tsv(
    filepath,
    grid_mm,
    side_label,
    density=0.4,
    hex_bins=None,
    yz_map_step=0.05,
):
    """将 (nz, ny) 厚度网格写成 protector map 格式的 TSV。"""
    y_centers = np.arange(YZ_MAP_Y_MIN, YZ_MAP_Y_MAX + 1e-9, yz_map_step)
    z_centers = np.arange(YZ_MAP_Z_MIN, YZ_MAP_Z_MAX + 1e-9, yz_map_step)
    ny, nz = len(y_centers), len(z_centers)
    lines = [
        f"# YZ protector map, {side_label}. Pixel = {yz_map_step}m. Thickness mm per cell.",
        f"# density {density}",
    ]
    if hex_bins is not None:
        lines.append(
            f"# 厚度值与厚度 PNG 相同：hexbin gridsize={hex_bins}、reduce=max；"
            f"再写入本矩形格中心所在六边形的值（前/后仅 x 分区）"
        )
    lines.append("z\\y\t" + "\t".join(f"{y:.2f}" for y in y_centers))
    for i, z in enumerate(z_centers):
        if i < grid_mm.shape[0]:
            row_vals = "\t".join(str(int(round(v))) if not np.isnan(v) else "0" for v in grid_mm[i, :])
        else:
            row_vals = "\t".join("0" for _ in range(ny))
        lines.append(f"{z:.2f}\t{row_vals}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  已写入: {filepath}")


def _hexbin_max_paths_for_tsv(y_sub, z_sub, c_sub, gridsize, extent):
    """
    与 plot_thickness_grid 中一致：ax.hexbin(y,z,C=..., gridsize, reduce_C_function=np.max, extent=...)。

    返回:
        paths: PolyCollection 的路径列表（常为 **1 个模板**）
        offsets: (N,2) 每个六边形中心在数据坐标中的平移（必须与 arr 行数一致）
        arr: 每个六边形的 reduce 后厚度
    """
    if len(y_sub) == 0:
        return [], np.zeros((0, 2)), np.array([])
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_axis_off()
    hb = ax.hexbin(
        y_sub,
        z_sub,
        C=c_sub,
        gridsize=gridsize,
        reduce_C_function=np.max,
        extent=extent,
        linewidths=0,
    )
    paths = hb.get_paths()
    offsets = np.asarray(hb.get_offsets(), dtype=float)
    arr = np.asarray(hb.get_array(), dtype=float)
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    arr = np.nan_to_num(arr, nan=0.0)
    plt.close(fig)
    return paths, offsets, arr


def _rect_centers_sample_hexbin_max(paths, offsets, hex_vals, y_centers, z_centers):
    """
    对每个矩形格中心 (y,z)，判断落在哪个六边形内并取厚度 max。
    matplotlib hexbin 的 Polygon 顶点多为「模板 + get_offsets() 平移」，不能直接用 path.contains_points。
    """
    ny, nz = len(y_centers), len(z_centers)
    grid = np.zeros((nz, ny), dtype=float)
    n_hex = len(hex_vals)
    if n_hex == 0 or offsets.shape[0] == 0:
        return grid
    n_hex = int(min(n_hex, offsets.shape[0]))
    yy, zz = np.meshgrid(y_centers, z_centers, indexing="xy")
    pts = np.column_stack([yy.ravel(), zz.ravel()])
    flat = np.zeros(len(pts), dtype=float)

    if len(paths) == 1:
        tmpl = np.asarray(paths[0].vertices, dtype=float)
        for i in range(n_hex):
            v = float(hex_vals[i])
            if not np.isfinite(v):
                continue
            poly = MplPath(tmpl + offsets[i])
            try:
                inside = poly.contains_points(pts)
            except Exception:
                inside = np.array([poly.contains_point(tuple(pt)) for pt in pts])
            flat = np.where(inside, np.maximum(flat, v), flat)
    else:
        for i in range(min(n_hex, len(paths))):
            v = float(hex_vals[i])
            if not np.isfinite(v):
                continue
            ov = offsets[i] if i < offsets.shape[0] else np.zeros(2)
            poly = MplPath(np.asarray(paths[i].vertices, dtype=float) + ov)
            try:
                inside = poly.contains_points(pts)
            except Exception:
                inside = np.array([poly.contains_point(tuple(pt)) for pt in pts])
            flat = np.where(inside, np.maximum(flat, v), flat)
    return flat.reshape(nz, ny)


def _mask_yz_grid_outside_thickness_axes(grid_mm, y_centers, z_centers):
    """将厚度图坐标轴外的格置 0（与 PNG 白边一致；护具表格外圈常不应有虚假厚度）。"""
    y_lo, y_hi = THICKNESS_GRID_Y_LIM
    z_lo, z_hi = THICKNESS_GRID_Z_LIM
    yy, zz = np.meshgrid(y_centers, z_centers, indexing="xy")
    outside = (yy < y_lo) | (yy > y_hi) | (zz < z_lo) | (zz > z_hi)
    out = np.array(grid_mm, dtype=float, copy=True)
    out[outside] = 0.0
    return out


def _build_grid_from_hexbin_offsets(offsets, hex_vals, ny, nz):
    """
    将 hexbin 六边形中心(offset)映射到 TSV 的固定 0.05m 矩形格索引，
    并在同一矩形格内对厚度取 max。

    相比“格心是否落入 hex 多边形内”的点内判断，这种方式不会因为 hex 很小而导致 TSV 极度稀疏。
    """
    grid = np.zeros((nz, ny), dtype=float)
    if offsets is None or len(offsets) == 0:
        return grid
    off = np.asarray(offsets, dtype=float)
    hv = np.asarray(hex_vals, dtype=float).ravel()
    n = min(len(off), len(hv))
    if n == 0:
        return grid
    off = off[:n]
    hv = hv[:n]

    yi = np.clip(np.round((off[:, 0] - YZ_MAP_Y_MIN) / YZ_MAP_STEP).astype(int), 0, ny - 1)
    zi = np.clip(np.round((off[:, 1] - YZ_MAP_Z_MIN) / YZ_MAP_STEP).astype(int), 0, nz - 1)
    flat = grid.ravel()
    lin = zi * ny + yi
    np.maximum.at(flat, lin, hv)
    return flat.reshape(nz, ny)


def _build_grid_from_hexbin_polygon_intersect(paths, offsets, hex_vals, y_centers, z_centers):
    """
    将 hexbin 的非零单元映射到 0.05m 固定矩形格：
    对每个 hex 多边形与每个候选矩形格做相交判定（用角点/中心点/多边形顶点判定），
    若相交则该矩形格取 max(thickness)。

    这样比“只用 hex 中心点落格”更符合“六边形颜色块覆盖到的区域”直觉。
    """
    ny, nz = len(y_centers), len(z_centers)
    grid = np.zeros((nz, ny), dtype=float)
    if not paths or offsets is None or len(offsets) == 0:
        return grid

    tmpl = np.asarray(paths[0].vertices, dtype=float)
    if tmpl.ndim != 2 or tmpl.shape[1] != 2:
        return grid

    # y_centers/z_centers 按相同步长生成；这里用 y_step 推导半径，避免固定写死 0.05m
    if len(y_centers) < 2:
        return grid
    y_step = float(y_centers[1] - y_centers[0])
    half = y_step / 2.0
    eps = 1e-12

    off = np.asarray(offsets, dtype=float)
    hv = np.asarray(hex_vals, dtype=float).ravel()
    n = min(len(off), len(hv))
    if n == 0:
        return grid

    y_min = float(np.min(y_centers))
    z_min = float(np.min(z_centers))

    for i in range(n):
        v = float(hv[i])
        if not np.isfinite(v) or v <= 0:
            continue
        ov = off[i]
        poly_verts = tmpl + ov
        poly = MplPath(poly_verts)

        y_poly_min = float(np.min(poly_verts[:, 0]))
        y_poly_max = float(np.max(poly_verts[:, 0]))
        z_poly_min = float(np.min(poly_verts[:, 1]))
        z_poly_max = float(np.max(poly_verts[:, 1]))

        # 候选方格索引：使用矩形覆盖范围 [center-half, center+half]
        iy0 = int(np.floor((y_poly_min - half - y_min) / y_step))
        iy1 = int(np.ceil((y_poly_max + half - y_min) / y_step))
        # z 轴步长与 y_step 维持一致（由 TSV grid 生成方式保证）
        iz0 = int(np.floor((z_poly_min - half - z_min) / y_step))
        iz1 = int(np.ceil((z_poly_max + half - z_min) / y_step))
        iy0 = max(0, iy0)
        iz0 = max(0, iz0)
        iy1 = min(ny - 1, iy1)
        iz1 = min(nz - 1, iz1)
        if iy0 > iy1 or iz0 > iz1:
            continue

        # rectangle corners and center for each candidate cell
        for iz in range(iz0, iz1 + 1):
            zc = float(z_centers[iz])
            z0 = zc - half
            z1 = zc + half
            for iy in range(iy0, iy1 + 1):
                yc = float(y_centers[iy])
                y0 = yc - half
                y1 = yc + half

                rect_corners = np.array(
                    [[y0, z0], [y1, z0], [y0, z1], [y1, z1]],
                    dtype=float,
                )
                rect_center = np.array([[yc, zc]], dtype=float)

                # 任一几何特征落在多边形内 -> 相交（对凸多边形足够）
                inter = False
                try:
                    if np.any(poly.contains_points(rect_corners, radius=eps)):
                        inter = True
                    elif poly.contains_points(rect_center, radius=eps)[0]:
                        inter = True
                    else:
                        inside_poly_vertex = np.any(
                            (poly_verts[:, 0] >= y0 - eps)
                            & (poly_verts[:, 0] <= y1 + eps)
                            & (poly_verts[:, 1] >= z0 - eps)
                            & (poly_verts[:, 1] <= z1 + eps)
                        )
                        inter = bool(inside_poly_vertex)
                except Exception:
                    # 兜底：逐点 contains_point
                    inside_any = any(poly.contains_point(tuple(pt)) for pt in rect_corners)
                    inter = inside_any or poly.contains_point((yc, zc))

                if inter:
                    grid[iz, iy] = max(grid[iz, iy], v)

    return grid


def export_yz_protector_map_tsv(
    df,
    front_tsv_path,
    back_tsv_path,
    csv_path=None,
    density=0.4,
    target_force=3.0,
    method="chr",
    target_pressure=None,
    force_12mm_in_z_range=True,
    enable_force_filter=True,
    _yz_thickness_precomputed=None,
    hex_bins=50,
    compare_yz_hex_tsv=False,
    compare_yz_max_print=120,
    yz_map_step=0.05,
):
    """
    生成 yz_map_front*.tsv / yz_map_back*.tsv（护具 0.05m 矩形格，与 protector_map 同格式）。

    逐点厚度与 ``*_thickness_grid_plot_<method>.png`` 相同（``compute_per_point_thickness_for_thickness_grid``）。

    **与 PNG 一致的聚合**：对 ``robot_frame_x >= 0`` / ``< 0`` 分别做与厚度图相同的
    ``hexbin(y,z,C=厚度, gridsize=hex_bins, reduce_C_function=np.max, extent=全数据 y/z)``，
    然后把每个 hex 的 **中心点** 映射到护具固定的 0.05m 矩形格索引上，矩形格内取 ``max``。
    最后将 **厚度图坐标轴外**（y∉[-0.35,0.35] 或 z∉[0.1,1.2]）的格置 0，与 PNG 白边一致。

    即 front/back 分别对应「只对前侧 / 后侧点画一张同 ``-b`` 的厚度 hexbin」在格中心的读数。
    默认「前后混画」的 PNG 在同一 (y,z) 上可能更大（前后取 max），属预期差异。
    """
    if "robot_frame_x" not in df.columns:
        raise ValueError("导出 YZ 护具 TSV 需要列 robot_frame_x（用于区分前/后侧）")

    y_centers = np.arange(YZ_MAP_Y_MIN, YZ_MAP_Y_MAX + 1e-9, yz_map_step)
    z_centers = np.arange(YZ_MAP_Z_MIN, YZ_MAP_Z_MAX + 1e-9, yz_map_step)
    ny, nz = len(y_centers), len(z_centers)

    if _yz_thickness_precomputed is not None:
        y, z, x, point_thickness_mm, _fc_pre = _yz_thickness_precomputed
    else:
        y, z, x, point_thickness_mm, _fc = compute_per_point_thickness_for_thickness_grid(
            df,
            csv_path=csv_path,
            density=density,
            target_force=target_force,
            method=method,
            target_pressure=target_pressure,
            force_12mm_in_z_range=force_12mm_in_z_range,
            enable_force_filter=enable_force_filter,
        )

    front_mask = x >= 0
    back_mask = x < 0
    n_front = int(np.sum(front_mask))
    n_back = int(np.sum(back_mask))
    print(f"  前侧 (x>=0): {n_front:,} 点, 后侧 (x<0): {n_back:,} 点 (按 robot_frame_x 区分)")

    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))
    extent = (ymin, ymax, zmin, zmax)
    print(
        f"  YZ TSV：hexbin 与厚度图一致 extent y=[{ymin:.4f},{ymax:.4f}] z=[{zmin:.4f},{zmax:.4f}], gridsize={hex_bins}"
    )

    def build_grid_hexbin(mask):
        take = np.where(mask)[0]
        if len(take) == 0:
            z0 = np.zeros((nz, ny), dtype=float)
            return z0, np.zeros((0, 2), dtype=float), np.array([])
        ys = y[take]
        zs = z[take]
        cs = point_thickness_mm[take]
        paths, offsets, hvals = _hexbin_max_paths_for_tsv(ys, zs, cs, hex_bins, extent)
        # 关键：按 hex 多边形与 0.05m 矩形格“相交”映射，避免中心点落格导致遗漏。
        g = _build_grid_from_hexbin_polygon_intersect(
            paths, offsets, hvals, y_centers=y_centers, z_centers=z_centers
        )
        g = _mask_yz_grid_outside_thickness_axes(g, y_centers, z_centers)
        return g, offsets, hvals

    print(
        f"导出 YZ 护具 map（hexbin 同 PNG，采样到 {yz_map_step}m 矩形格）:\n"
        f"  {front_tsv_path}\n"
        f"  {back_tsv_path}"
    )
    grid_front, off_f, hv_f = build_grid_hexbin(front_mask)
    grid_back, off_b, hv_b = build_grid_hexbin(back_mask)
    if compare_yz_hex_tsv:
        hf = _collect_hex_nonzero_rows(off_f, hv_f)
        tf = _collect_tsv_nonzero_rows(grid_front, y_centers, z_centers)
        _print_yz_hex_tsv_compare("front (x>=0)", hf, tf, max_print=compare_yz_max_print)
        hb = _collect_hex_nonzero_rows(off_b, hv_b)
        tb = _collect_tsv_nonzero_rows(grid_back, y_centers, z_centers)
        _print_yz_hex_tsv_compare("back (x<0)", hb, tb, max_print=compare_yz_max_print)
        print(
            "  提示: 厚度 PNG 为前后点混合 hexbin，与上表「单侧 hex」行数/位置会不同；"
            "混合 hex 见 --dump-thickness-hex-csv 生成的 *_hexbin_aggregate.csv"
        )
    _write_yz_map_tsv(
        front_tsv_path,
        grid_front,
        "front (x >= 0)",
        density=density,
        hex_bins=hex_bins,
        yz_map_step=yz_map_step,
    )
    _write_yz_map_tsv(
        back_tsv_path,
        grid_back,
        "back (x < 0)",
        density=density,
        hex_bins=hex_bins,
        yz_map_step=yz_map_step,
    )
    print("  ✓ 护具 TSV 导出完成")


def main():
    parser = argparse.ArgumentParser(description='绘制接触力数据、保护层厚度、表面积和压强的网格颜色图')
    parser.add_argument('csv_path', type=str, help='CSV文件路径')
    parser.add_argument('--stl-path', type=str, default=None, help='STL文件路径（用于表面积和压强计算，默认: src/simulation/mujoco/assets/resource/robot/pm_v2/meshes/serial_pm_v2_combined.stl）')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径前缀（可选，会自动添加后缀）')
    parser.add_argument('-b', '--bins', type=int, default=50, help='网格分辨率（默认: 50）')
    parser.add_argument('-c', '--cmap', type=str, default=None, help='颜色映射（默认: None，力图使用白色到红色，厚度图使用橙色，表面积图使用蓝色，压强图使用绿色到红色）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[4.5, 7.4], help='图片大小（单位: cm，默认: 4 10，如果未指定单独大小则所有图都使用）')
    parser.add_argument('--force-figsize', type=float, nargs=2, default=None, help='力图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--thickness-figsize', type=float, nargs=2, default=None, help='厚度图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--surface-figsize', type=float, nargs=2, default=None, help='表面积图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--pressure-figsize', type=float, nargs=2, default=None, help='压强图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--density', type=float, default=0.4, help='材料密度（默认: 0.4，仅用于chr方法）')
    parser.add_argument('--target-force', type=float, default=1.0, help='目标衰减后的力（kN，默认: 1.0，仅用于chr方法）')
    parser.add_argument('--method', type=str, default='chr', choices=['chr', 'zzq'], help='厚度计算方法：chr使用thickness_selection，zzq使用thickness_from_pressure（默认: chr）')
    parser.add_argument('--target-pressure', type=float, default=None, help='期望减小后的压强（MPa，仅用于zzq方法）')
    parser.add_argument('--margin-left', type=float, default=15.0, help='左边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-right', type=float, default=5.0, help='右边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-top', type=float, default=5.0, help='上边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-bottom', type=float, default=10.0, help='下边距（百分比，默认: 5.0）')
    parser.add_argument('--force-only', action='store_true', help='仅绘制力图，不绘制其他图')
    parser.add_argument('--thickness-only', action='store_true', help='仅绘制厚度图，不绘制其他图')
    parser.add_argument('--surface-only', action='store_true', help='仅绘制表面积图，不绘制其他图')
    parser.add_argument('--pressure-only', action='store_true', help='仅绘制压强图，不绘制其他图')
    parser.add_argument('--search-radius', type=float, default=0.01, help='表面积计算搜索半径（米，默认: 0.01）')
    parser.add_argument('--no-hardcode', action='store_true', help='禁用强制厚度设置（默认启用）。包括：1) z坐标在(0.32, 0.48)范围内的点设置为12mm；2) elbow/shoulder部位z坐标在(0.8, 1.0)范围内的点设置为6mm')
    parser.add_argument('--no-force-filter', action='store_true', help='禁用force滤波（默认启用）。包括：1) 过滤elbow部位的特定数据；2) 限制z坐标在0.8~1.0范围内的力值')
    parser.add_argument('--no-save-yz-tsv', action='store_true', help='生成护具厚度图时不写出 yz_map_front_*.tsv / yz_map_back_*.tsv（默认会写出，文件名含父目录时间后缀）')
    parser.add_argument('--yz-tsv-dir', type=str, default=None, help='YZ 护具 TSV 输出目录（默认与厚度图相同目录）')
    parser.add_argument(
        '--yz-map-step',
        type=float,
        default=0.05,
        help='YZ 护具 TSV 网格步长（m，默认 0.05；可用 0.025 变细一倍）',
    )
    parser.add_argument(
        '--dump-thickness-hex-csv',
        action='store_true',
        help='保存厚度 PNG 时额外写出同目录下 *_thickness_*_hexbin_aggregate.csv：每个有数据六边形的中心 y/z 与 thickness_mm（与图同源，非 PNG 像素；前后点混合，与 yz_map_front 仅 x>=0 不同）',
    )
    parser.add_argument(
        '--compare-yz-hex-tsv',
        action='store_true',
        help='终端打印：①混合厚度 PNG 的 hex 非零位置 ②导出 TSV 时前/后单侧 hex 与 TSV 非零格对照（可加 --compare-yz-max-print 控制行数）',
    )
    parser.add_argument(
        '--compare-yz-max-print',
        type=int,
        default=120,
        metavar='N',
        help='与 --compare-yz-hex-tsv 配合，每段列表最多打印 N 行（默认 120）',
    )
    
    args = parser.parse_args()
    
    # 统一定义参数
    target_force = args.target_force
    density = args.density
    method = args.method
    target_pressure = args.target_pressure
    
    # 验证参数
    if method == 'zzq' and target_pressure is None:
        print("错误: 使用zzq方法时必须提供--target-pressure参数")
        return
    
    # 检查 CSV 路径（空串常见于未 export MERGED_CSV 却写了 "$MERGED_CSV"）
    csv_in = (args.csv_path or "").strip()
    if not csv_in:
        print("错误: 第一个参数 CSV 路径为空。")
        print('  若使用变量，请先导出再运行，例如:')
        print('    export MERGED_CSV=/home/you/.../all_directions_merged.csv')
        print("  或把 CSV 的绝对路径直接写在命令最前面（不要用未定义的 $MERGED_CSV）。")
        return
    if not os.path.exists(csv_in):
        print(f"错误: 文件不存在: {csv_in}")
        return
    args.csv_path = csv_in
    
    csv_file = Path(args.csv_path)
    
    # 将厘米转换为英寸（matplotlib使用英寸）
    # 1英寸 = 2.54厘米
    CM_TO_INCH = 1.0 / 2.54
    
    # 确定两个图的大小（从厘米转换为英寸）
    def cm_to_inch(size_cm):
        """将厘米转换为英寸"""
        return tuple([s * CM_TO_INCH for s in size_cm])
    
    force_figsize = cm_to_inch(args.force_figsize) if args.force_figsize else cm_to_inch(args.figsize)
    thickness_figsize = cm_to_inch(args.thickness_figsize) if args.thickness_figsize else cm_to_inch(args.figsize)
    surface_figsize = cm_to_inch(args.surface_figsize) if args.surface_figsize else cm_to_inch(args.figsize)
    pressure_figsize = cm_to_inch(args.pressure_figsize) if args.pressure_figsize else cm_to_inch(args.figsize)
    
    # 确定STL文件路径
    if args.stl_path:
        stl_path = args.stl_path
    else:
        # 使用默认STL文件路径
        workspace_root = Path(__file__).parent.parent
        stl_path = workspace_root / 'src' / 'simulation' / 'mujoco' / 'assets' / 'resource' / 'robot' / 'pm_v2' / 'meshes' / 'serial_pm_v2_combined.stl'
    
    if not os.path.exists(stl_path):
        print(f"警告: STL文件不存在: {stl_path}")
        print("      表面积图将不会绘制")
        stl_path = None
    
    # 读取数据并计算每个部分的统计信息
    print("\n" + "="*60)
    print("计算各身体部分的最大力和厚度")
    print("="*60)
    df_stats = None
    df_for_plotting = None
    try:
        # 读取CSV文件（使用分块读取以显示进度）
        print(f"[1/6] 正在读取CSV文件: {args.csv_path}")
        file_size = os.path.getsize(args.csv_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"      文件大小: {file_size_mb:.2f} MB")
        df_stats = read_csv_with_progress(args.csv_path)
        print(f"      已读取 {len(df_stats)} 行数据")
        
        # 检查必要的列，优先使用force_normal（与violin_link_force.py保持一致）
        force_col = 'force_normal' if 'force_normal' in df_stats.columns else 'force_magnitude'
        required_cols = ['body2_name', force_col]
        missing_cols = [col for col in required_cols if col not in df_stats.columns]
        if missing_cols:
            print(f"警告: CSV文件缺少必要的列: {missing_cols}，无法计算身体部分统计")
        else:
            print(f"[2/6] 使用力列: {force_col}")
            # 检查是否是原始CSV（需要过滤头部和踝关节）
            csv_file = Path(args.csv_path)
            is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df_stats.columns
            
            # 首先过滤，只保留body1_name是'world'的数据
            print(f"[3/6] 过滤body1_name，只保留'world'...")
            if 'body1_name' in df_stats.columns:
                before_filter = len(df_stats)
                df_stats = df_stats[df_stats['body1_name'] == 'world'].copy()
                after_filter = len(df_stats)
                filtered_count = before_filter - after_filter
                print(f"      过滤前: {before_filter} 行, 过滤后: {after_filter} 行")
                if filtered_count > 0:
                    print(f"      过滤掉 {filtered_count} 行（body1_name != 'world'）")
            else:
                print("      警告: 未找到body1_name列，跳过此过滤")
            
            # 统一分类一次（在过滤elbow之前，这样filter_elbow_forces可以复用）
            print(f"[4/6] 分类身体部分...")
            if 'body_part' not in df_stats.columns:
                if len(df_stats) > 10000:
                    n_jobs = max(1, cpu_count() - 2)
                    print(f"      正在分类身体部分（{len(df_stats)} 行数据，使用 {n_jobs} 个进程）...")
                else:
                    n_jobs = 1
                    print(f"      正在分类身体部分（{len(df_stats)} 行数据）...")
                df_stats = apply_body_part_multiprocess(df_stats, n_jobs=n_jobs)
            else:
                print(f"      跳过分类（已有body_part列）")
            
            # 对于所有CSV（原始和聚类后的），都应用elbow过滤
            enable_force_filter = not args.no_force_filter
            if enable_force_filter:
                print(f"[5/6] 过滤elbow部位数据...")
                df_stats = filter_elbow_forces(df_stats, force_column=force_col)
            else:
                print(f"[5/6] 跳过elbow部位数据过滤（--no-force-filter）...")
            
            # 如果是原始CSV，需要按位置分组找到最大力（保留所有列包括fail_type_info）
            # find_max_force_per_position 内部会处理头部和踝关节的过滤
            # 注意：此时df_stats已经有body_part列了，find_max_force_per_position中的filter_elbow_forces会复用
            if not is_clustered:
                print(f"[6/6] 按位置分组并找到最大力（包含过滤头部和踝关节）...")
                df_stats = find_max_force_per_position(df_stats, force_column=force_col, enable_force_filter=enable_force_filter)
                # find_max_force_per_position返回的DataFrame应该还保留body_part列（如果filter_elbow_forces没有删除的话）
                # 但为了安全，我们检查一下，如果没有就重新分类
                if 'body_part' not in df_stats.columns:
                    print(f"      警告: find_max_force_per_position后丢失了body_part列，重新分类...")
                    if len(df_stats) > 10000:
                        n_jobs = max(1, cpu_count() - 2)
                        df_stats = apply_body_part_multiprocess(df_stats, n_jobs=n_jobs)
                    else:
                        df_stats = apply_body_part_multiprocess(df_stats, n_jobs=1)
            
            # 计算统计信息
            # 注意：此时df_stats应该有body_part列了，calculate_part_statistics会复用
            print(f"[7/7] 计算各身体部分的统计信息...")
            part_stats, unsatisfied_points = calculate_part_statistics(
                df_stats, density=density, target_force=target_force, force_column=force_col,
                method=method, target_pressure=target_pressure
            )
            
            # 如果后续需要绘图，在统计计算完成后复制已处理好的数据
            need_plotting = not (args.force_only and args.thickness_only and args.surface_only and args.pressure_only)
            if need_plotting:
                # 此时df_stats已经经过所有过滤和处理，可以直接用于绘图
                df_for_plotting = df_stats.copy()
            
            if part_stats is not None and len(part_stats) > 0:
                print("\n各身体部分的最大力和厚度统计:")
                print("-" * 200)
                print(f"{'身体部分':<20} {'最大力(N)':<15} {'最大力(kN)':<15} {'最大厚度(mm)':<15} {'数据点数':<10} {'最大力位置(x,y,z)':<30} {'body1':<25} {'body2':<25} {'fail-type':<30}")
                print("-" * 200)
                for _, row in part_stats.iterrows():
                    body_part = row['body_part']
                    max_force_n = row['max_force_n']
                    max_force_kn = row['max_force_kn']
                    max_thickness = row['max_thickness_mm']
                    count = int(row['count'])
                    max_x = row.get('max_x', 0.0)
                    max_y = row.get('max_y', 0.0)
                    max_z = row.get('max_z', 0.0)
                    max_body1_name = row.get('max_body1_name', 'N/A')
                    max_body2_name = row.get('max_body2_name', 'N/A')
                    max_fail_type = row.get('max_fail_type', 'N/A')
                    
                    thickness_str = f"{max_thickness:.2f}" if pd.notna(max_thickness) else "N/A"
                    position_str = f"({max_x:.3f},{max_y:.3f},{max_z:.3f})"
                    body1_str = str(max_body1_name) if pd.notna(max_body1_name) else "N/A"
                    body2_str = str(max_body2_name) if pd.notna(max_body2_name) else "N/A"
                    fail_type_str = str(max_fail_type) if pd.notna(max_fail_type) else "N/A"
                    print(f"{body_part:<20} {max_force_n:<15.2f} {max_force_kn:<15.3f} {thickness_str:<15} {count:<10} {position_str:<30} {body1_str:<25} {body2_str:<25} {fail_type_str:<30}")
                print("-" * 200)
                
                # 打印无法满足目标要求的点
                if unsatisfied_points is not None and len(unsatisfied_points) > 0:
                    print(f"\n无法满足目标要求（目标力={target_force}kN，最大厚度24mm仍无法满足）的接触点:")
                    print("-" * 200)
                    print(f"{'身体部分':<20} {'最大力(N)':<15} {'最大力(kN)':<15} {'最大厚度(mm)':<15} {'数据点数':<10} {'最大力位置(x,y,z)':<30} {'body1':<25} {'body2':<25} {'fail-type':<30}")
                    print("-" * 200)
                    for _, row in unsatisfied_points.iterrows():
                        body_part = row['body_part']
                        max_force_n = row['max_force_n']
                        max_force_kn = row['max_force_kn']
                        max_thickness = row['max_thickness_mm']
                        count = int(row['count'])
                        max_x = row.get('max_x', 0.0)
                        max_y = row.get('max_y', 0.0)
                        max_z = row.get('max_z', 0.0)
                        max_body1_name = row.get('max_body1_name', 'N/A')
                        max_body2_name = row.get('max_body2_name', 'N/A')
                        max_fail_type = row.get('max_fail_type', 'N/A')
                        
                        thickness_str = "N/A"
                        position_str = f"({max_x:.3f},{max_y:.3f},{max_z:.3f})"
                        body1_str = str(max_body1_name) if pd.notna(max_body1_name) else "N/A"
                        body2_str = str(max_body2_name) if pd.notna(max_body2_name) else "N/A"
                        fail_type_str = str(max_fail_type) if pd.notna(max_fail_type) else "N/A"
                        print(f"{body_part:<20} {max_force_n:<15.2f} {max_force_kn:<15.3f} {thickness_str:<15} {count:<10} {position_str:<30} {body1_str:<25} {body2_str:<25} {fail_type_str:<30}")
                    print("-" * 200)
            else:
                print("警告: 无法计算身体部分统计信息")
    except Exception as e:
        print(f"计算统计信息时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 如果统计部分没有读取（或者需要重新读取用于绘图），则读取CSV文件
    if df_for_plotting is None:
        need_plotting = not (args.force_only and args.thickness_only and args.surface_only and args.pressure_only)
        if need_plotting:
            print("\n" + "="*60)
            print("读取CSV文件（供绘图使用）")
            print("="*60)
            print(f"正在读取CSV文件: {args.csv_path}")
            df_for_plotting = read_csv_with_progress(args.csv_path)
            print(f"已读取 {len(df_for_plotting)} 行数据")
    
    # 绘制力图
    if not args.thickness_only and not args.surface_only and not args.pressure_only:
        if args.output is None:
            force_output = csv_file.parent / f"{csv_file.stem}_force_grid_plot_{method}.png"
        else:
            force_output = Path(args.output).parent / f"{Path(args.output).stem}_force_{method}.png"
        
        try:
            print("\n" + "="*60)
            print("绘制接触力图")
            print("="*60)
            plot_contact_grid(
                df=df_for_plotting.copy() if df_for_plotting is not None else None,
                csv_path=args.csv_path if df_for_plotting is None else None,
                output_path=str(force_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=force_figsize,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
                enable_force_filter=not args.no_force_filter
            )
        except Exception as e:
            print(f"绘制力图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制厚度图
    if not args.force_only and not args.surface_only and not args.pressure_only:
        if args.output is None:
            thickness_output = csv_file.parent / f"{csv_file.stem}_thickness_grid_plot_{method}.png"
        else:
            thickness_output = Path(args.output).parent / f"{Path(args.output).stem}_thickness_{method}.png"
        
        want_yz_tsv = (
            (not args.no_save_yz_tsv)
            and df_for_plotting is not None
            and "robot_frame_x" in df_for_plotting.columns
            and (
                ("force_normal" in df_for_plotting.columns)
                or ("force_magnitude" in df_for_plotting.columns)
            )
        )
        yz_pipe = None
        if want_yz_tsv:
            print("\n" + "="*60)
            print("保护层厚度计算（厚度图与 YZ TSV 共用，仅执行一次）")
            print("="*60)
            try:
                yz_pipe = compute_per_point_thickness_for_thickness_grid(
                    df_for_plotting,
                    csv_path=args.csv_path,
                    density=density,
                    target_force=target_force,
                    method=method,
                    target_pressure=target_pressure,
                    force_12mm_in_z_range=not args.no_hardcode,
                    enable_force_filter=not args.no_force_filter,
                )
            except Exception as e:
                print(f"厚度管道预计算失败（厚度图将单独计算；YZ TSV 导出将再尝试）: {e}")
                import traceback
                traceback.print_exc()
                yz_pipe = None
        
        hexbin_csv_path = None
        if args.dump_thickness_hex_csv:
            hexbin_csv_path = str(
                thickness_output.parent / f"{thickness_output.stem}_hexbin_aggregate.csv"
            )
        
        try:
            print("\n" + "="*60)
            print("绘制保护层厚度图")
            print("="*60)
            plot_thickness_grid(
                df=None if yz_pipe is not None else (df_for_plotting.copy() if df_for_plotting is not None else None),
                csv_path=args.csv_path if yz_pipe is None else None,
                output_path=str(thickness_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=thickness_figsize,
                density=density,
                target_force=target_force,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
                method=method,
                target_pressure=target_pressure,
                force_12mm_in_z_range=not args.no_hardcode,
                enable_force_filter=not args.no_force_filter,
                _yz_thickness_precomputed=yz_pipe,
                hexbin_dump_csv=hexbin_csv_path,
                print_mixed_hex_compare=args.compare_yz_hex_tsv,
                compare_max_print=args.compare_yz_max_print,
            )
        except Exception as e:
            print(f"绘制厚度图时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 生成护具图时可选：另存为 2 个 TSV（与 protector_map 同格式）
        if want_yz_tsv:
            yz_dir = Path(args.yz_tsv_dir) if args.yz_tsv_dir else (Path(args.output).parent if args.output else csv_file.parent)
            yz_dir = yz_dir.resolve()
            _sfx = _yz_map_tsv_suffix_from_csv(args.csv_path)
            front_path = yz_dir / f"yz_map_front{_sfx}.tsv"
            back_path = yz_dir / f"yz_map_back{_sfx}.tsv"
            try:
                export_yz_protector_map_tsv(
                    df_for_plotting,
                    str(front_path),
                    str(back_path),
                    csv_path=args.csv_path,
                    density=density,
                    target_force=target_force,
                    method=method,
                    target_pressure=target_pressure,
                    force_12mm_in_z_range=not args.no_hardcode,
                    enable_force_filter=not args.no_force_filter,
                    _yz_thickness_precomputed=yz_pipe,
                    hex_bins=args.bins,
                    compare_yz_hex_tsv=args.compare_yz_hex_tsv,
                    compare_yz_max_print=args.compare_yz_max_print,
                    yz_map_step=args.yz_map_step,
                )
            except Exception as e:
                print(f"导出 YZ TSV 护具时出错: {e}")
                import traceback
                traceback.print_exc()
        elif (not args.no_save_yz_tsv) and df_for_plotting is not None:
            print("  跳过 YZ TSV 导出：数据缺少 robot_frame_x 或力列(force_normal/force_magnitude)")
    
    # 绘制表面积图
    if not args.force_only and not args.thickness_only and not args.pressure_only and stl_path is not None:
        if args.output is None:
            surface_output = csv_file.parent / f"{csv_file.stem}_surface_area_grid_plot_{method}.png"
        else:
            surface_output = Path(args.output).parent / f"{Path(args.output).stem}_surface_{method}.png"
        
        try:
            print("\n" + "="*60)
            print("绘制表面积图")
            print("="*60)
            plot_surface_area_grid(
                df=df_for_plotting.copy() if df_for_plotting is not None else None,
                csv_path=args.csv_path if df_for_plotting is None else None,
                stl_path=str(stl_path),
                output_path=str(surface_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=surface_figsize,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
                search_radius=args.search_radius,
                enable_force_filter=not args.no_force_filter
            )
        except Exception as e:
            print(f"绘制表面积图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制压强图
    if not args.force_only and not args.thickness_only and not args.surface_only and stl_path is not None:
        if args.output is None:
            pressure_output = csv_file.parent / f"{csv_file.stem}_pressure_grid_plot_{method}.png"
        else:
            pressure_output = Path(args.output).parent / f"{Path(args.output).stem}_pressure_{method}.png"
        
        try:
            print("\n" + "="*60)
            print("绘制表面压强图")
            print("="*60)
            plot_pressure_grid(
                df=df_for_plotting.copy() if df_for_plotting is not None else None,
                csv_path=args.csv_path if df_for_plotting is None else None,
                stl_path=str(stl_path),
                output_path=str(pressure_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=pressure_figsize,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
                search_radius=args.search_radius,
                enable_force_filter=not args.no_force_filter
            )
        except Exception as e:
            print(f"绘制压强图时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

