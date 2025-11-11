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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import argparse
import os
import sys
from pathlib import Path

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
    """创建从浅橙到深橙的颜色映射"""
    # 定义橙色渐变：从浅橙(255, 200, 150)到深橙(255, 140, 0)
    colors = [(1.0, 0.78, 0.59), (1.0, 0.55, 0.0)]  # 浅橙到深橙
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('orange', colors, N=n_bins)
    return cmap


def create_blue_cmap():
    """创建从浅蓝到深蓝的颜色映射"""
    colors = [(0.7, 0.85, 1.0), (0.0, 0.3, 0.8)]  # 浅蓝到深蓝
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue', colors, N=n_bins)
    return cmap


def load_stl_mesh(stl_path):
    """
    加载STL文件并返回网格对象
    
    参数:
        stl_path: STL文件路径
    
    返回:
        mesh: 网格对象（trimesh或numpy-stl）
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL文件不存在: {stl_path}")
    
    if HAS_TRIMESH:
        mesh_obj = trimesh.load(str(stl_path))
        return mesh_obj
    elif HAS_STL:
        mesh_obj = mesh.Mesh.from_file(str(stl_path))
        return mesh_obj
    else:
        raise ImportError("无法加载STL文件：未找到trimesh或numpy-stl库")


def calculate_surface_area_for_points(contact_points, stl_path, search_radius=0.01):
    """
    为每个接触点计算表面积
    
    参数:
        contact_points: 接触点坐标数组，形状为 (N, 3)，单位：米
        stl_path: STL文件路径
        search_radius: 搜索半径（米），用于计算每个点周围的表面积
    
    返回:
        surface_areas: 每个点的表面积数组（单位：m²）
    """
    if not HAS_TRIMESH and not HAS_STL:
        raise ImportError("无法计算表面积：未找到trimesh或numpy-stl库")
    
    print(f"正在加载STL文件: {stl_path}")
    mesh_obj = load_stl_mesh(stl_path)
    
    print(f"正在计算 {len(contact_points)} 个接触点的表面积...")
    surface_areas = np.zeros(len(contact_points))
    
    if HAS_TRIMESH:
        # 使用trimesh计算
        # 获取所有三角形
        triangles = mesh_obj.triangles
        triangle_centers = triangles.mean(axis=1)
        triangle_areas = mesh_obj.area_faces
        
        # 对于每个接触点，找到附近的三角形并累加面积
        for i, point in enumerate(contact_points):
            # 计算点到所有三角形中心的距离
            distances = np.linalg.norm(triangle_centers - point, axis=1)
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
        
        # 对于每个接触点，找到附近的三角形并累加面积
        for i, point in enumerate(contact_points):
            # 计算点到所有三角形中心的距离
            distances = np.linalg.norm(triangle_centers - point, axis=1)
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


def calculate_thicknesses(force_magnitudes, density=0.4, target_force=3.0):
    """
    计算每个接触点的保护层厚度
    
    参数:
        force_magnitudes: 力的大小数组（单位：N）
        density: 材料密度（默认0.4）
        target_force: 目标衰减后的力（kN，默认3.0）
    
    返回:
        thicknesses: 厚度数组（单位：mm），如果无法满足要求则为nan
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
        
        return np.array(thicknesses)
    
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


def get_body_part(row):
    """
    根据body2_name和坐标识别身体部分
    
    参数:
        row: DataFrame的一行，包含body2_name, robot_frame_y, robot_frame_z
    
    返回:
        body_part: 身体部分名称，如 "Left_Shoulder", "Right_Elbow", "Torso" 等
    """
    link_name = str(row.get('body2_name', '')).lower()
    robot_y = row.get('robot_frame_y', 0) if pd.notna(row.get('robot_frame_y')) else 0
    robot_z = row.get('robot_frame_z', 0) if pd.notna(row.get('robot_frame_z')) else 0
    
    # 确定左右侧
    is_left = False
    if any(x in link_name for x in ["left", "l_", "_l", "l_shoulder", "l_elbow", "l_hip", "l_knee"]):
        is_left = True
    elif any(x in link_name for x in ["right", "r_", "_r", "r_shoulder", "r_elbow", "r_hip", "r_knee"]):
        is_left = False
    elif "base" in link_name:
        # For base: y+ is left, y- is right
        is_left = robot_y > 0 if pd.notna(robot_y) else False
    else:
        # Default: y+ is left, y- is right
        if pd.notna(robot_y):
            is_left = robot_y > 0
    
    side = "Left" if is_left else "Right"
    
    # 1. Shoulder: shoulder_pitch, shoulder_roll
    if "shoulder_pitch" in link_name or "shoulder_roll" in link_name:
        return f"{side}_Shoulder"
    
    # 2. Elbow: shoulder_yaw, elbow_yaw, elbow_pitch
    if "shoulder_yaw" in link_name or "elbow_yaw" in link_name or "elbow_pitch" in link_name:
        return f"{side}_Elbow"
    
    # 3. Torso: torso, 不分左右
    if "torso" in link_name:
        return "Torso"
    
    # 4. Hip: base（根据robot_frame_y区分，y+是左，y-是右）, hip_pitch, hip_roll, hip_yaw（robot_frame_z > 0.55m)
    if "base" in link_name:
        return f"{side}_Hip"
    if "hip_pitch" in link_name or "hip_roll" in link_name:
        return f"{side}_Hip"
    if "hip_yaw" in link_name and robot_z > 0.55:
        return f"{side}_Hip"
    
    # 5. Knee: hip_yaw(robot_frame_z < 0.55m), knee_pitch
    if "hip_yaw" in link_name and robot_z < 0.55:
        return f"{side}_Knee"
    if "knee_pitch" in link_name:
        return f"{side}_Knee"
    
    # 默认：返回未知
    return "Unknown"


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
    """
    # 检查必要的列
    if 'body2_name' not in df.columns:
        print("警告: 未找到body2_name列，无法按身体部分分组")
        return None
    
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
            return None
    
    # 添加身体部分列
    df = df.copy()
    df['body_part'] = df.apply(get_body_part, axis=1)
    
    # 按身体部分分组，计算最大力
    part_stats = df.groupby('body_part').agg({
        force_column: ['max', 'mean', 'count']
    }).reset_index()
    
    # 扁平化列名
    part_stats.columns = ['body_part', 'max_force_n', 'mean_force_n', 'count']
    
    # 将最大力从N转换为kN
    part_stats['max_force_kn'] = part_stats['max_force_n'] / 1000.0
    part_stats['mean_force_kn'] = part_stats['mean_force_n'] / 1000.0
    
    # 计算每个部分的最大力对应的厚度
    if select_thickness_simple is not None:
        original_cwd = os.getcwd()
        try:
            os.chdir(str(thickness_dir))
            thicknesses = []
            for force_kn in part_stats['max_force_kn']:
                thickness_mm = select_thickness_simple(force_kn, density=density, target_force=target_force)
                if thickness_mm is None:
                    thicknesses.append(np.nan)
                else:
                    thicknesses.append(float(thickness_mm))
            part_stats['max_thickness_mm'] = thicknesses
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
    
    return part_stats


def find_max_force_per_position(df, force_column='force_normal'):
    """
    对于原始CSV文件，找到每个碰撞点位置的最大力，并过滤掉头部和踝关节链接
    
    参数:
        df: DataFrame，包含robot_frame_x, robot_frame_y, robot_frame_z, force_magnitude或force_normal, body2_name列
        force_column: 使用的力列名，'force_normal'或'force_magnitude'（默认'force_normal'）
    
    返回:
        df_max: DataFrame，每个唯一位置的最大力（已过滤头部和踝关节）
    """
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
    
    # 按位置分组，找到每个位置的最大力
    # 使用round来避免浮点数精度问题导致的位置重复
    position_cols = ['robot_frame_x', 'robot_frame_y', 'robot_frame_z']
    
    # 检查是否有robot_frame_x列，如果没有则只使用y和z
    if 'robot_frame_x' not in df.columns:
        position_cols = ['robot_frame_y', 'robot_frame_z']
        print("注意: 未找到robot_frame_x列，仅使用robot_frame_y和robot_frame_z进行分组")
    
    # 按位置分组，找到每个位置的最大力
    df_max = df.groupby(position_cols, as_index=False)[force_column].max()
    
    print(f"原始数据: {len(df)} 行")
    print(f"唯一位置: {len(df_max)} 个")
    print(f"数据压缩率: {len(df_max)/len(df)*100:.1f}%")
    
    return df_max


def plot_contact_grid(csv_path, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                     margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0):
    """
    绘制接触力数据的网格颜色图
    
    参数:
        csv_path: CSV文件路径
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
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
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
    
    # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
    csv_file = Path(csv_path)
    is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
    
    if not is_clustered:
        # 原始CSV：按位置分组，找到每个位置的最大力
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，直接使用数据")
    
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
    ax.set_xlabel('robot_frame_y (m)', fontsize=12)
    ax.set_ylabel('robot_frame_z (m)', fontsize=12)
    ax.set_title('Contact Force Distribution (Grid Color Plot)', fontsize=12)
    
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


def plot_thickness_grid(csv_path, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                       density=0.4, target_force=3.0, margin_left=5.0, margin_right=5.0, 
                       margin_top=5.0, margin_bottom=5.0):
    """
    绘制保护层厚度的网格颜色图
    
    参数:
        csv_path: CSV文件路径
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
    # 如果没有指定颜色映射，使用橙色的默认映射
    if cmap is None:
        cmap = create_orange_cmap()
    elif isinstance(cmap, str):
        # 如果是指定的字符串，尝试使用matplotlib内置的colormap
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"警告: 无法找到颜色映射 '{cmap}'，使用默认的橙色映射")
            cmap = create_orange_cmap()
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
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
    
    # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
    csv_file = Path(csv_path)
    is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
    
    if not is_clustered:
        # 原始CSV：按位置分组，找到每个位置的最大力
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，直接使用数据")
    
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
    
    # 使用hexbin创建网格颜色图
    # x轴: robot_frame_y, y轴: robot_frame_z
    # 使用np.max以显示每个网格单元内的最大厚度（与力图保持一致）
    # 设置颜色范围以匹配标准厚度值 6, 12, 18, 24mm
    hb = ax.hexbin(y, z, C=thicknesses, gridsize=bins, cmap=cmap, reduce_C_function=np.max,
                   vmin=0.0, vmax=24.0)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('Protector Thickness (mm)', fontsize=12)
    
    # 设置颜色条刻度标签为 0, 6, 12, 18, 24mm
    cb.set_ticks([0, 6, 12, 18, 24])
    cb.set_ticklabels(['0', '6', '12', '18', '24'])
    
    # 设置标签和标题
    ax.set_xlabel('robot_frame_y (m)', fontsize=12)
    ax.set_ylabel('robot_frame_z (m)', fontsize=12)
    ax.set_title('Protector Thickness Distribution (Grid Color Plot)', fontsize=12)
    
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


def plot_surface_area_grid(csv_path, stl_path, output_path=None, bins=50, cmap=None, figsize=(10, 8), 
                           margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0,
                           search_radius=0.01):
    """
    绘制表面积数据的网格颜色图
    
    参数:
        csv_path: CSV文件路径
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
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
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
    
    # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
    csv_file = Path(csv_path)
    is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
    
    if not is_clustered:
        # 原始CSV：按位置分组，找到每个位置的最大力
        df = find_max_force_per_position(df, force_column=force_column)
    else:
        print("检测到聚类后的CSV文件，直接使用数据")
    
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
    
    # 构建接触点坐标（3D）
    contact_points = np.column_stack([x, y, z])
    
    # 计算每个接触点的表面积
    surface_areas = calculate_surface_area_for_points(contact_points, stl_path, search_radius=search_radius)
    
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
    # 使用np.max以显示每个网格单元内的最大表面积（与其他图保持一致）
    hb = ax.hexbin(y, z, C=surface_areas, gridsize=bins, cmap=cmap, reduce_C_function=np.max)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('Surface Area (m²)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('robot_frame_y (m)', fontsize=12)
    ax.set_ylabel('robot_frame_z (m)', fontsize=12)
    ax.set_title('Surface Area Distribution (Grid Color Plot)', fontsize=12)
    
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
    parser = argparse.ArgumentParser(description='绘制接触力数据、保护层厚度和表面积的网格颜色图')
    parser.add_argument('csv_path', type=str, help='CSV文件路径')
    parser.add_argument('--stl-path', type=str, default=None, help='STL文件路径（用于表面积计算，默认: src/simulation/mujoco/assets/resource/robot/pm_v2/meshes/serial_pm_v2_combined.stl）')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径前缀（可选，会自动添加后缀）')
    parser.add_argument('-b', '--bins', type=int, default=50, help='网格分辨率（默认: 50）')
    parser.add_argument('-c', '--cmap', type=str, default=None, help='颜色映射（默认: None，力图使用白色到红色，厚度图使用橙色，表面积图使用蓝色）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[15, 15], help='图片大小（单位: cm，默认: 4 10，如果未指定单独大小则两个图都使用）')
    parser.add_argument('--force-figsize', type=float, nargs=2, default=None, help='力图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--thickness-figsize', type=float, nargs=2, default=None, help='厚度图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--surface-figsize', type=float, nargs=2, default=None, help='表面积图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--density', type=float, default=0.4, help='材料密度（默认: 0.4）')
    parser.add_argument('--target-force', type=float, default=1.0, help='目标衰减后的力（kN，默认: 3.0）')
    parser.add_argument('--margin-left', type=float, default=15.0, help='左边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-right', type=float, default=5.0, help='右边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-top', type=float, default=5.0, help='上边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-bottom', type=float, default=10.0, help='下边距（百分比，默认: 5.0）')
    parser.add_argument('--force-only', action='store_true', help='仅绘制力图，不绘制厚度图和表面积图')
    parser.add_argument('--thickness-only', action='store_true', help='仅绘制厚度图，不绘制力图和表面积图')
    parser.add_argument('--surface-only', action='store_true', help='仅绘制表面积图，不绘制力图和厚度图')
    parser.add_argument('--search-radius', type=float, default=0.01, help='表面积计算搜索半径（米，默认: 0.01）')
    
    args = parser.parse_args()
    
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
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {args.csv_path}")
        df_stats = pd.read_csv(args.csv_path)
        
        # 检查必要的列，优先使用force_normal（与violin_link_force.py保持一致）
        force_col = 'force_normal' if 'force_normal' in df_stats.columns else 'force_magnitude'
        required_cols = ['body2_name', force_col]
        missing_cols = [col for col in required_cols if col not in df_stats.columns]
        if missing_cols:
            print(f"警告: CSV文件缺少必要的列: {missing_cols}，无法计算身体部分统计")
        else:
            print(f"使用力列: {force_col}")
            # 检查是否是原始CSV（需要过滤头部和踝关节）
            csv_file = Path(args.csv_path)
            is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df_stats.columns
            
            if not is_clustered:
                # 原始CSV：先过滤头部和踝关节
                excluded_links = [
                    'LINK_HEAD_YAW',
                    'LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R',
                    'LINK_ANKLE_PITCH_L', 'LINK_ANKLE_PITCH_R'
                ]
                if 'body2_name' in df_stats.columns:
                    before_filter = len(df_stats)
                    df_stats = df_stats[~df_stats['body2_name'].isin(excluded_links)]
                    print(f"过滤头部和踝关节: {before_filter} -> {len(df_stats)} 行")
            
            # 计算统计信息
            part_stats = calculate_part_statistics(df_stats, density=args.density, target_force=args.target_force, force_column=force_col)
            
            if part_stats is not None and len(part_stats) > 0:
                print("\n各身体部分的最大力和厚度统计:")
                print("-" * 80)
                print(f"{'身体部分':<20} {'最大力(N)':<15} {'最大力(kN)':<15} {'最大厚度(mm)':<15} {'数据点数':<10}")
                print("-" * 80)
                for _, row in part_stats.iterrows():
                    body_part = row['body_part']
                    max_force_n = row['max_force_n']
                    max_force_kn = row['max_force_kn']
                    max_thickness = row['max_thickness_mm']
                    count = int(row['count'])
                    
                    thickness_str = f"{max_thickness:.2f}" if pd.notna(max_thickness) else "N/A"
                    print(f"{body_part:<20} {max_force_n:<15.2f} {max_force_kn:<15.3f} {thickness_str:<15} {count:<10}")
                print("-" * 80)
            else:
                print("警告: 无法计算身体部分统计信息")
    except Exception as e:
        print(f"计算统计信息时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 绘制力图
    if not args.thickness_only and not args.surface_only:
        if args.output is None:
            force_output = csv_file.parent / f"{csv_file.stem}_force_grid_plot.png"
        else:
            force_output = Path(args.output).parent / f"{Path(args.output).stem}_force.png"
        
        try:
            print("\n" + "="*60)
            print("绘制接触力图")
            print("="*60)
            plot_contact_grid(
                args.csv_path,
                str(force_output),
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
    if not args.force_only and not args.surface_only:
        if args.output is None:
            thickness_output = csv_file.parent / f"{csv_file.stem}_thickness_grid_plot.png"
        else:
            thickness_output = Path(args.output).parent / f"{Path(args.output).stem}_thickness.png"
        
        try:
            print("\n" + "="*60)
            print("绘制保护层厚度图")
            print("="*60)
            plot_thickness_grid(
                args.csv_path,
                str(thickness_output),
                bins=args.bins,
                cmap=args.cmap,
                figsize=thickness_figsize,
                density=args.density,
                target_force=args.target_force,
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
    if not args.force_only and not args.thickness_only and stl_path is not None:
        if args.output is None:
            surface_output = csv_file.parent / f"{csv_file.stem}_surface_area_grid_plot.png"
        else:
            surface_output = Path(args.output).parent / f"{Path(args.output).stem}_surface.png"
        
        try:
            print("\n" + "="*60)
            print("绘制表面积图")
            print("="*60)
            plot_surface_area_grid(
                args.csv_path,
                str(stl_path),
                str(surface_output),
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


if __name__ == '__main__':
    main()

