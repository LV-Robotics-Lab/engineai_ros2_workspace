#!/usr/bin/env python3
"""
Joint Forces Plotting Script
绘制所有关节的M_eq（综合破坏载荷）、M_bend（弯矩）和F_shear（剪切力）与时间的关系曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from mujoco_data_io import load_data_file

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_joint_forces_csv(csv_file):
    """加载关节反力CSV文件"""
    print(f"Loading joint forces data: {csv_file}")
    
    try:
        df = load_data_file(csv_file)
        print(f"Successfully loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def plot_joint_meq(df, output_file=None, figsize=(16, 10)):
    """绘制所有关节的M_eq、M_bend和F_shear与时间曲线"""
    
    # 获取所有唯一的关节
    unique_joints = sorted(df['joint_name'].unique())
    num_joints = len(unique_joints)
    
    print(f"Found {num_joints} unique joints")
    
    # 创建渐变色colormap（使用彩虹色）
    try:
        # 新版本matplotlib API
        cmap = plt.colormaps.get_cmap('rainbow')
    except AttributeError:
        # 旧版本matplotlib API
        cmap = plt.cm.get_cmap('rainbow')
    colors = [cmap(i / max(1, num_joints - 1)) for i in range(num_joints)]
    
    # 计算子图布局（尽量接近正方形）
    cols = int(np.ceil(np.sqrt(num_joints)))
    rows = int(np.ceil(num_joints / cols))
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Joint Forces: M_eq, M_bend, and F_shear vs Time', fontsize=16, fontweight='bold')
    
    # 如果只有一个关节，axes不是数组
    if num_joints == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 为每个关节绘制曲线
    for idx, joint_name in enumerate(unique_joints):
        # 获取该关节的数据
        joint_data = df[df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        
        # 选择子图
        ax = axes[idx]
        
        # 创建双y轴：左轴用于力矩（N·m），右轴用于力（N）
        ax2 = ax.twinx()
        
        # 绘制M_eq曲线（左y轴）
        line1 = ax.plot(joint_data['timestamp'], joint_data['M_eq'], 
                       linewidth=1.5, label='M_eq', color='#1f77b4', alpha=0.8)
        
        # 绘制M_bend曲线（左y轴）
        line2 = ax.plot(joint_data['timestamp'], joint_data['M_bend_mag'], 
                       linewidth=1.5, label='M_bend', color='#ff7f0e', alpha=0.8, linestyle='--')
        
        # 绘制F_shear曲线（右y轴）
        line3 = ax2.plot(joint_data['timestamp'], joint_data['F_shear_mag'], 
                        linewidth=1.5, label='F_shear', color='#2ca02c', alpha=0.8, linestyle=':')
        
        # 设置标签和标题
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Moment (N·m)', fontsize=10, color='#1f77b4')
        ax2.set_ylabel('Force (N)', fontsize=10, color='#2ca02c')
        ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 设置y轴颜色
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        
        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        # 设置y轴从0开始
        y_max_moment = max(joint_data['M_eq'].max(), joint_data['M_bend_mag'].max())
        y_max_force = joint_data['F_shear_mag'].max()
        if y_max_moment > 0:
            ax.set_ylim(bottom=0, top=y_max_moment * 1.1)
        if y_max_force > 0:
            ax2.set_ylim(bottom=0, top=y_max_force * 1.1)
    
    # 隐藏多余的子图
    for idx in range(num_joints, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def plot_joint_meq_overlay(df, output_file=None, figsize=(14, 8)):
    """绘制所有关节的M_eq、M_bend和F_shear曲线叠加在同一张图上"""
    
    # 获取所有唯一的关节
    unique_joints = sorted(df['joint_name'].unique())
    num_joints = len(unique_joints)
    
    print(f"Found {num_joints} unique joints")
    
    # 创建渐变色colormap（使用彩虹色）
    try:
        # 新版本matplotlib API
        cmap = plt.colormaps.get_cmap('rainbow')
    except AttributeError:
        # 旧版本matplotlib API
        cmap = plt.cm.get_cmap('rainbow')
    colors = [cmap(i / max(1, num_joints - 1)) for i in range(num_joints)]
    
    # 创建图形，使用三个子图分别显示M_eq、M_bend和F_shear
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle('All Joints: M_eq, M_bend, and F_shear vs Time', 
                 fontsize=14, fontweight='bold')
    
    # 绘制M_eq
    ax1 = axes[0]
    for idx, joint_name in enumerate(unique_joints):
        joint_data = df[df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        ax1.plot(joint_data['timestamp'], joint_data['M_eq'], 
                linewidth=1.5, label=joint_name, alpha=0.8, color=colors[idx])
    ax1.set_ylabel('M_eq (N·m)', fontsize=12)
    ax1.set_title('M_eq (Equivalent Bending Moment)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    y_max = df['M_eq'].max()
    if y_max > 0:
        ax1.set_ylim(bottom=0, top=y_max * 1.1)
    
    # 绘制M_bend
    ax2 = axes[1]
    for idx, joint_name in enumerate(unique_joints):
        joint_data = df[df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        ax2.plot(joint_data['timestamp'], joint_data['M_bend_mag'], 
                linewidth=1.5, label=joint_name, alpha=0.8, color=colors[idx], linestyle='--')
    ax2.set_ylabel('M_bend (N·m)', fontsize=12)
    ax2.set_title('M_bend (Bending Moment)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    y_max = df['M_bend_mag'].max()
    if y_max > 0:
        ax2.set_ylim(bottom=0, top=y_max * 1.1)
    
    # 绘制F_shear
    ax3 = axes[2]
    for idx, joint_name in enumerate(unique_joints):
        joint_data = df[df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        ax3.plot(joint_data['timestamp'], joint_data['F_shear_mag'], 
                linewidth=1.5, label=joint_name, alpha=0.8, color=colors[idx], linestyle=':')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('F_shear (N)', fontsize=12)
    ax3.set_title('F_shear (Shear Force)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    y_max = df['F_shear_mag'].max()
    if y_max > 0:
        ax3.set_ylim(bottom=0, top=y_max * 1.1)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot M_eq, M_bend, and F_shear vs Time for all joints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all joints in separate subplots
  python plot_joint_meq.py joint_forces_data_20231201_120000.csv
  
  # Plot all joints overlaid on one plot
  python plot_joint_meq.py joint_forces_data_20231201_120000.csv --overlay
  
  # Save to specific output file
  python plot_joint_meq.py joint_forces_data_20231201_120000.csv -o output.png
        """
    )
    
    parser.add_argument('csv_file', type=str, help='Path to joint forces CSV file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output image file path (default: auto-generate from CSV filename)')
    parser.add_argument('--overlay', action='store_true',
                       help='Plot all joints on one graph (overlay mode)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 10],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 16 10)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # 加载数据
    df = load_joint_forces_csv(args.csv_file)
    
    # 检查必要的列是否存在
    required_columns = ['timestamp', 'joint_name', 'M_eq', 'M_bend_mag', 'F_shear_mag']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # 生成输出文件名
    if args.output is None:
        csv_path = Path(args.csv_file)
        if args.overlay:
            output_file = csv_path.parent / f"{csv_path.stem}_meq_overlay.png"
        else:
            output_file = csv_path.parent / f"{csv_path.stem}_meq.png"
    else:
        output_file = args.output
    
    # 绘制图形
    if args.overlay:
        plot_joint_meq_overlay(df, output_file, figsize=tuple(args.figsize))
    else:
        plot_joint_meq(df, output_file, figsize=tuple(args.figsize))
    
    print("Plotting completed!")

if __name__ == '__main__':
    main()

