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
import matplotlib.font_manager as fm
import numpy as np
import argparse
import os
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


def create_white_to_red_cmap():
    """创建从白色到红色的颜色映射"""
    colors = ['white', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('white_to_red', colors, N=n_bins)
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
        (1.0, 1.0, 1.0),      # 0 - 白色
        (1.0, 0.9, 0.7),      # 6 - 浅橙
        (1.0, 0.8, 0.5),      # 12 - 中浅橙
        (1.0, 0.65, 0.25),    # 18 - 中橙
        (1.0, 0.55, 0.0)      # 24 - 深橙
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


def calculate_thicknesses(force_magnitudes, density=0.4, target_force=3.0):
    """
    计算每个接触点的保护层厚度，并四舍五入到标准值（0, 6, 12, 18, 24）
    
    参数:
        force_magnitudes: 力的大小数组（单位：N）
        density: 材料密度（默认0.4）
        target_force: 目标衰减后的力（kN，默认3.0）
    
    返回:
        thicknesses: 厚度数组（单位：mm），已四舍五入到标准值，如果无法满足要求则为nan
    """
    if select_thickness_simple is None:
        raise ImportError("无法导入select_thickness_simple函数，请检查ThicknessCalculate模块")
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到ThicknessCalculate目录，以便ForceCalculator能找到参数文件
        os.chdir(str(thickness_dir))
        
        thicknesses = []
        for force_n in force_magnitudes:
            # 将力从N转换为kN
            force_kn = force_n / 1000.0
            # 计算厚度
            thickness_mm = select_thickness_simple(force_kn, density=density, target_force=target_force)
            # 如果返回None，转换为nan以便numpy处理
            if thickness_mm is None:
                thicknesses.append(np.nan)
            else:
                thicknesses.append(float(thickness_mm))
        
        thicknesses = np.array(thicknesses)
        
        # 将厚度值四舍五入到标准值
        thicknesses = round_thickness_to_standard(thicknesses)
        
        return thicknesses
    
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


def calculate_part_statistics(df, density=0.4, target_force=1.0, force_column='force_normal'):
    """
    计算每个身体部分的最大力和厚度
    
    参数:
        df: DataFrame，包含body2_name, robot_frame_y, robot_frame_z, force_magnitude或force_normal
        density: 材料密度（默认0.4）
        target_force: 目标衰减后的力（kN，默认1.0）
        force_column: 使用的力列名，'force_normal'或'force_magnitude'（默认'force_normal'）
    
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
    if select_thickness_simple is not None:
        original_cwd = os.getcwd()
        try:
            os.chdir(str(thickness_dir))
            thicknesses = []
            iterator = tqdm(part_stats['max_force_kn'], desc="计算厚度", disable=not HAS_TQDM) if HAS_TQDM else part_stats['max_force_kn']
            for force_kn in iterator:
                thickness_mm = select_thickness_simple(force_kn, density=density, target_force=target_force)
                if thickness_mm is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(thickness_mm))
            # 将厚度值四舍五入到标准值
            thicknesses = round_thickness_to_standard(np.array(thicknesses))
            part_stats['max_thickness_mm'] = thicknesses
            
            # 找出无法满足要求的点（厚度为nan的点）
            unsatisfied_mask = pd.isna(part_stats['max_thickness_mm'])
            if unsatisfied_mask.any():
                unsatisfied_points = part_stats[unsatisfied_mask].copy()
        finally:
            os.chdir(original_cwd)
    else:
        part_stats['max_thickness_mm'] = np.nan
        print("警告: 无法导入thickness_selection模块，厚度计算功能不可用")
    
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
    
    参数:
        csv_path: CSV文件路径
        usecols: 要读取的列（可选）
        encoding: 编码方式，默认'utf-8'
    
    返回:
        df: DataFrame
    """
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
    过滤掉z坐标在0.6-0.85m之间的elbow部位大于10kN的力
    
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
    
    # 检查z坐标是否在0.6-0.85m之间
    z_mask = (df['robot_frame_z'] >= 0.6) & (df['robot_frame_z'] <= 0.85)
    
    # 检查力是否大于10kN（10000N）
    force_mask = df[force_column] > 10000
    
    # 组合条件：elbow部位 AND z坐标在0.6-0.85m之间 AND 力大于10kN
    filter_mask = elbow_mask & z_mask & force_mask
    
    # 过滤掉满足条件的数据
    df = df[~filter_mask]
    after_filter = len(df)
    
    if before_filter > after_filter:
        filtered_count = before_filter - after_filter
        print(f"过滤掉z坐标在0.6-0.85m之间的elbow部位大于10kN的力: {filtered_count} 行")
        print(f"过滤后的数据: {after_filter} 行")
    
    # 注意：不删除body_part列，以便后续函数可以复用
    # df = df.drop('body_part', axis=1)
    
    return df


def find_max_force_per_position(df, force_column='force_normal'):
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
                     margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0):
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
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，应用过滤...")
        # 对于聚类后的CSV，也需要应用elbow过滤
        df = filter_elbow_forces(df, force_column=force_column)
    
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
    cb.set_label('Force Magnitude (kN)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('y (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title('Collision Force Distribution', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.4)
    
    # 添加网格
    # ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_thickness_grid(csv_path=None, df=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                       density=0.4, target_force=3.0, margin_left=5.0, margin_right=5.0, 
                       margin_top=5.0, margin_bottom=5.0):
    """
    绘制保护层厚度的网格颜色图
    
    参数:
        csv_path: CSV文件路径（如果df为None则使用此参数读取）
        df: 已读取的DataFrame（可选，如果提供则直接使用，不读取csv_path）
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射（如果为None，则使用橙色的默认映射）
        figsize: 图片大小
        density: 材料密度（默认0.4）
        target_force: 目标衰减后的力（kN，默认3.0）
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
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
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，应用过滤...")
        # 对于聚类后的CSV，也需要应用elbow过滤
        df = filter_elbow_forces(df, force_column=force_column)
    
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
    
    # 计算每个接触点的厚度
    print(f"正在计算保护层厚度（密度={density}, 目标力={target_force}kN）...")
    thicknesses = calculate_thicknesses(force, density=density, target_force=target_force)
    
    # 统计无法满足要求的点（None值已被转换为nan）
    invalid_mask = np.isnan(thicknesses)
    invalid_count = np.sum(invalid_mask)
    if invalid_count > 0:
        print(f"警告: {invalid_count} 个接触点无法满足目标要求（即使使用最大厚度）")
        
        # 收集无法满足要求的点的详细信息
        invalid_df = df[invalid_mask].copy()
        invalid_df['force_kn'] = invalid_df[force_column] / 1000.0
        invalid_df['thickness_mm'] = np.nan
        
        # 添加身体部分列（使用多进程加速，如果还没有body_part列）
        if 'body_part' not in invalid_df.columns:
            if len(invalid_df) > 1000:
                n_jobs = max(1, cpu_count() - 2)
                print(f"      正在分类无法满足要求的点（{len(invalid_df)} 行数据，使用 {n_jobs} 个进程）...")
                invalid_df = apply_body_part_multiprocess(invalid_df, n_jobs=n_jobs)
            else:
                invalid_df['body_part'] = invalid_df.apply(get_body_part, axis=1)
        else:
            print(f"      跳过分类（已有body_part列）")
        
        # 打印无法满足要求的点
        print(f"\n无法满足目标要求（目标力={target_force}kN，最大厚度24mm仍无法满足）的接触点:")
        print("-" * 200)
        print(f"{'身体部分':<20} {'力(N)':<15} {'力(kN)':<15} {'厚度(mm)':<15} {'位置(x,y,z)':<30} {'body1':<25} {'body2':<25} {'fail-type':<30}")
        print("-" * 200)
        
        # 按力从大到小排序，打印所有无法满足要求的点
        invalid_df_sorted = invalid_df.sort_values(by=force_column, ascending=False)
        for idx, row in invalid_df_sorted.iterrows():
            body_part = row['body_part']
            force_n = row[force_column]
            force_kn = row['force_kn']
            x_pos = row.get('robot_frame_x', 0.0) if 'robot_frame_x' in row.index else 0.0
            y_pos = row.get('robot_frame_y', 0.0) if 'robot_frame_y' in row.index else 0.0
            z_pos = row.get('robot_frame_z', 0.0) if 'robot_frame_z' in row.index else 0.0
            body1_name = str(row.get('body1_name', 'N/A')) if 'body1_name' in row.index and pd.notna(row.get('body1_name')) else 'N/A'
            body2_name = str(row.get('body2_name', 'N/A')) if 'body2_name' in row.index and pd.notna(row.get('body2_name')) else 'N/A'
            fail_type = str(row.get('fall_type_info', 'N/A')) if 'fall_type_info' in row.index and pd.notna(row.get('fall_type_info')) else 'N/A'
            
            position_str = f"({x_pos:.3f},{y_pos:.3f},{z_pos:.3f})"
            print(f"{body_part:<20} {force_n:<15.2f} {force_kn:<15.3f} {'N/A':<15} {position_str:<30} {body1_name:<25} {body2_name:<25} {fail_type:<30}")
        
        print("-" * 200)
        print(f"（共 {invalid_count} 个点）\n")
        
        # 将None/nan替换为最大厚度值，以便绘图
        max_thickness = 24  # 最大可选厚度
        thicknesses = np.where(invalid_mask, max_thickness, thicknesses)
    
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
    
    # 添加颜色条，使用shrink参数控制大小
    # 使用离散颜色条，显示为块状
    # spacing='uniform' 确保每个色块大小相同
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02, 
                      ticks=[0, 6, 12, 18, 24], boundaries=boundaries, 
                      format='%g', spacing='uniform', extend='neither', 
                      drawedges=True)
    cb.set_label('Protector Thickness (mm)', fontsize=12, fontfamily=MYRIAD_FONT)
    
    # 设置颜色条刻度标签为 0, 6, 12, 18, 24mm
    cb.set_ticklabels(['0', '6', '12', '18', '24'])
    # 设置颜色条刻度数字字体为 Times New Roman, 10pt
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(10)
    
    # 设置颜色条边缘线颜色，使离散块更明显
    cb.outline.set_edgecolor('black')
    cb.dividers.set_color('black')
    cb.dividers.set_linewidth(1.5)
    
    # 设置标签和标题（Myriad Pro, 12pt）
    ax.set_xlabel('y (m)', fontsize=12, fontfamily=MYRIAD_FONT)
    ax.set_ylabel('z (m)', fontsize=12, fontfamily=MYRIAD_FONT)
    ax.set_title('Protector Thickness Distribution', fontsize=12, fontfamily=MYRIAD_FONT)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.4)
    
    # 设置x轴刻度标签为 -0.4, -0.2, 0, 0.2, 0.4
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    
    # 设置坐标轴刻度数字字体为 Times New Roman, 10pt
    # 需要在设置坐标轴范围之后设置，以确保刻度标签已生成
    for label in ax.get_xticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(10)
    for label in ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)
        label.set_fontsize(10)
    
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
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_surface_area_grid(csv_path=None, df=None, stl_path=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                           margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                           search_radius=0.01):
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
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，应用过滤...")
        # 对于聚类后的CSV，也需要应用elbow过滤
        df = filter_elbow_forces(df, force_column=force_column)
    
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
    cb.set_label('Surface Area (m²)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('y (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title('Surface Area Distribution', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.4)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pressure_grid(csv_path=None, df=None, stl_path=None, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                       margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                       search_radius=0.01):
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
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，应用过滤...")
        # 对于聚类后的CSV，也需要应用elbow过滤
        df = filter_elbow_forces(df, force_column=force_column)
    
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
    cb.set_label('Pressure (MPa)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('y (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title('Surface Pressure Distribution', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 1.4)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 确保边距设置正确
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    
    # 保存或显示（不使用bbox_inches='tight'以保持设置的边距）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0)
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制接触力数据、保护层厚度、表面积和压强的网格颜色图')
    parser.add_argument('csv_path', type=str, help='CSV文件路径')
    parser.add_argument('--stl-path', type=str, default=None, help='STL文件路径（用于表面积和压强计算，默认: src/simulation/mujoco/assets/resource/robot/pm_v2/meshes/serial_pm_v2_combined.stl）')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径前缀（可选，会自动添加后缀）')
    parser.add_argument('-b', '--bins', type=int, default=50, help='网格分辨率（默认: 50）')
    parser.add_argument('-c', '--cmap', type=str, default=None, help='颜色映射（默认: None，力图使用白色到红色，厚度图使用橙色，表面积图使用蓝色，压强图使用绿色到红色）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[15, 15], help='图片大小（单位: cm，默认: 4 10，如果未指定单独大小则所有图都使用）')
    parser.add_argument('--force-figsize', type=float, nargs=2, default=None, help='力图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--thickness-figsize', type=float, nargs=2, default=None, help='厚度图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--surface-figsize', type=float, nargs=2, default=None, help='表面积图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--pressure-figsize', type=float, nargs=2, default=None, help='压强图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--density', type=float, default=0.4, help='材料密度（默认: 0.4）')
    parser.add_argument('--target-force', type=float, default=1.0, help='目标衰减后的力（kN，默认: 1.0）')
    parser.add_argument('--margin-left', type=float, default=15.0, help='左边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-right', type=float, default=5.0, help='右边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-top', type=float, default=5.0, help='上边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-bottom', type=float, default=10.0, help='下边距（百分比，默认: 5.0）')
    parser.add_argument('--force-only', action='store_true', help='仅绘制力图，不绘制其他图')
    parser.add_argument('--thickness-only', action='store_true', help='仅绘制厚度图，不绘制其他图')
    parser.add_argument('--surface-only', action='store_true', help='仅绘制表面积图，不绘制其他图')
    parser.add_argument('--pressure-only', action='store_true', help='仅绘制压强图，不绘制其他图')
    parser.add_argument('--search-radius', type=float, default=0.01, help='表面积计算搜索半径（米，默认: 0.01）')
    
    args = parser.parse_args()
    
    # 统一定义参数
    target_force = args.target_force
    density = args.density
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: 文件不存在: {args.csv_path}")
        return
    
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
        
        # 如果后续需要绘图，复用这个DataFrame
        need_plotting = not (args.force_only and args.thickness_only and args.surface_only and args.pressure_only)
        if need_plotting:
            df_for_plotting = df_stats.copy()
        
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
            print(f"[5/6] 过滤elbow部位数据...")
            df_stats = filter_elbow_forces(df_stats, force_column=force_col)
            
            # 如果是原始CSV，需要按位置分组找到最大力（保留所有列包括fail_type_info）
            # find_max_force_per_position 内部会处理头部和踝关节的过滤
            # 注意：此时df_stats已经有body_part列了，find_max_force_per_position中的filter_elbow_forces会复用
            if not is_clustered:
                print(f"[6/6] 按位置分组并找到最大力（包含过滤头部和踝关节）...")
                df_stats = find_max_force_per_position(df_stats, force_column=force_col)
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
            part_stats, unsatisfied_points = calculate_part_statistics(df_stats, density=density, target_force=target_force, force_column=force_col)
            
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
            force_output = csv_file.parent / f"{csv_file.stem}_force_grid_plot.png"
        else:
            force_output = Path(args.output).parent / f"{Path(args.output).stem}_force.png"
        
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
                margin_bottom=args.margin_bottom
            )
        except Exception as e:
            print(f"绘制力图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制厚度图
    if not args.force_only and not args.surface_only and not args.pressure_only:
        if args.output is None:
            thickness_output = csv_file.parent / f"{csv_file.stem}_thickness_grid_plot.png"
        else:
            thickness_output = Path(args.output).parent / f"{Path(args.output).stem}_thickness.png"
        
        try:
            print("\n" + "="*60)
            print("绘制保护层厚度图")
            print("="*60)
            plot_thickness_grid(
                df=df_for_plotting.copy() if df_for_plotting is not None else None,
                csv_path=args.csv_path if df_for_plotting is None else None,
                output_path=str(thickness_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=thickness_figsize,
                density=density,
                target_force=target_force,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom
            )
        except Exception as e:
            print(f"绘制厚度图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制表面积图
    if not args.force_only and not args.thickness_only and not args.pressure_only and stl_path is not None:
        if args.output is None:
            surface_output = csv_file.parent / f"{csv_file.stem}_surface_area_grid_plot.png"
        else:
            surface_output = Path(args.output).parent / f"{Path(args.output).stem}_surface.png"
        
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
                search_radius=args.search_radius
            )
        except Exception as e:
            print(f"绘制表面积图时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制压强图
    if not args.force_only and not args.thickness_only and not args.surface_only and stl_path is not None:
        if args.output is None:
            pressure_output = csv_file.parent / f"{csv_file.stem}_pressure_grid_plot.png"
        else:
            pressure_output = Path(args.output).parent / f"{Path(args.output).stem}_pressure.png"
        
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
                search_radius=args.search_radius
            )
        except Exception as e:
            print(f"绘制压强图时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

