#!/usr/bin/env python3
"""
绘制接触力数据的网格颜色图
x轴: robot_frame_y
y轴: robot_frame_z
颜色: force_magnitude 和 protector thickness
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from pathlib import Path

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


def find_max_force_per_position(df):
    """
    对于原始CSV文件，找到每个碰撞点位置的最大力，并过滤掉头部和踝关节链接
    
    参数:
        df: DataFrame，包含robot_frame_x, robot_frame_y, robot_frame_z, force_magnitude, body2_name列
    
    返回:
        df_max: DataFrame，每个唯一位置的最大力（已过滤头部和踝关节）
    """
    print("检测到原始CSV文件，正在按位置分组并找到每个位置的最大力...")
    
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
    df_max = df.groupby(position_cols, as_index=False)['force_magnitude'].max()
    
    print(f"原始数据: {len(df)} 行")
    print(f"唯一位置: {len(df_max)} 个")
    print(f"数据压缩率: {len(df_max)/len(df)*100:.1f}%")
    
    return df_max


def plot_contact_grid(csv_path, output_path=None, bins=50, cmap='viridis', figsize=(10, 8), 
                     margin_left=5.0, margin_right=5.0, margin_top=5.0, margin_bottom=5.0):
    """
    绘制接触力数据的网格颜色图
    
    参数:
        csv_path: CSV文件路径
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射
        figsize: 图片大小
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
    """
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    required_columns = ['robot_frame_z', 'robot_frame_y', 'force_magnitude']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
    csv_file = Path(csv_path)
    is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
    
    if not is_clustered:
        # 原始CSV：按位置分组，找到每个位置的最大力
        df = find_max_force_per_position(df)
    else:
        print("检测到聚类后的CSV文件，直接使用数据")
    
    # 提取数据
    z = df['robot_frame_z'].values
    y = df['robot_frame_y'].values
    force = df['force_magnitude'].values
    
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
    hb = ax.hexbin(y, z, C=force, gridsize=bins, cmap=cmap, reduce_C_function=np.mean)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('Force Magnitude (N)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('robot_frame_y (cm)', fontsize=12)
    ax.set_ylabel('robot_frame_z (cm)', fontsize=12)
    ax.set_title('Contact Force Distribution (Grid Color Plot)', fontsize=12)
    
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


def plot_thickness_grid(csv_path, output_path=None, bins=50, cmap='plasma', figsize=(10, 8), 
                       density=0.4, target_force=3.0, margin_left=5.0, margin_right=5.0, 
                       margin_top=5.0, margin_bottom=5.0):
    """
    绘制保护层厚度的网格颜色图
    
    参数:
        csv_path: CSV文件路径
        output_path: 输出图片路径（可选，如果为None则显示图片）
        bins: 网格分辨率
        cmap: 颜色映射
        figsize: 图片大小
        density: 材料密度（默认0.4）
        target_force: 目标衰减后的力（kN，默认3.0）
        margin_left: 左边距（百分比，默认5.0）
        margin_right: 右边距（百分比，默认5.0）
        margin_top: 上边距（百分比，默认5.0）
        margin_bottom: 下边距（百分比，默认5.0）
    """
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    required_columns = ['robot_frame_z', 'robot_frame_y', 'force_magnitude']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    # 检查是否是原始CSV（通过检查是否有contact_count列，或者文件名是否包含"clustered"）
    csv_file = Path(csv_path)
    is_clustered = 'clustered' in csv_file.stem.lower() or 'contact_count' in df.columns
    
    if not is_clustered:
        # 原始CSV：按位置分组，找到每个位置的最大力
        df = find_max_force_per_position(df)
    else:
        print("检测到聚类后的CSV文件，直接使用数据")
    
    # 提取数据
    z = df['robot_frame_z'].values
    y = df['robot_frame_y'].values
    force = df['force_magnitude'].values
    
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
    hb = ax.hexbin(y, z, C=thicknesses, gridsize=bins, cmap=cmap, reduce_C_function=np.mean)
    
    # 添加颜色条，使用shrink参数控制大小
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('Protector Thickness (mm)', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('robot_frame_y (cm)', fontsize=12)
    ax.set_ylabel('robot_frame_z (cm)', fontsize=12)
    ax.set_title('Protector Thickness Distribution (Grid Color Plot)', fontsize=12)
    
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
    parser = argparse.ArgumentParser(description='绘制接触力数据和保护层厚度的网格颜色图')
    parser.add_argument('csv_path', type=str, help='CSV文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径前缀（可选，会自动添加后缀）')
    parser.add_argument('-b', '--bins', type=int, default=50, help='网格分辨率（默认: 50）')
    parser.add_argument('-c', '--cmap', type=str, default='viridis', help='颜色映射（默认: viridis，两个图都使用）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[15, 15], help='图片大小（单位: cm，默认: 4 10，如果未指定单独大小则两个图都使用）')
    parser.add_argument('--force-figsize', type=float, nargs=2, default=None, help='力图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--thickness-figsize', type=float, nargs=2, default=None, help='厚度图大小（单位: cm，宽 高，默认使用--figsize）')
    parser.add_argument('--density', type=float, default=0.4, help='材料密度（默认: 0.4）')
    parser.add_argument('--target-force', type=float, default=3.0, help='目标衰减后的力（kN，默认: 3.0）')
    parser.add_argument('--margin-left', type=float, default=10.0, help='左边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-right', type=float, default=5.0, help='右边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-top', type=float, default=5.0, help='上边距（百分比，默认: 5.0）')
    parser.add_argument('--margin-bottom', type=float, default=10.0, help='下边距（百分比，默认: 5.0）')
    parser.add_argument('--force-only', action='store_true', help='仅绘制力图，不绘制厚度图')
    parser.add_argument('--thickness-only', action='store_true', help='仅绘制厚度图，不绘制力图')
    
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
    
    # 绘制力图
    if not args.thickness_only:
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
    if not args.force_only:
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


if __name__ == '__main__':
    main()

