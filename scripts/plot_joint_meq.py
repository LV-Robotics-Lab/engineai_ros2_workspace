#!/usr/bin/env python3
"""
Joint M_eq Plotting Script
绘制所有关节的M_eq（综合破坏载荷）与时间的关系曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_joint_forces_csv(csv_file):
    """加载关节反力CSV文件"""
    print(f"Loading joint forces data: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def plot_joint_meq(df, output_file=None, figsize=(16, 10)):
    """绘制所有关节的M_eq与时间曲线"""
    
    # 获取所有唯一的关节
    unique_joints = sorted(df['joint_name'].unique())
    num_joints = len(unique_joints)
    
    print(f"Found {num_joints} unique joints")
    
    # 创建渐变色colormap
    cmap = plt.cm.get_cmap('viridis')  # 使用viridis渐变色
    colors = [cmap(i / max(1, num_joints - 1)) for i in range(num_joints)]
    
    # 计算子图布局（尽量接近正方形）
    cols = int(np.ceil(np.sqrt(num_joints)))
    rows = int(np.ceil(num_joints / cols))
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Joint M_eq (Equivalent Bending Moment) vs Time', fontsize=16, fontweight='bold')
    
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
        
        # 绘制曲线，使用渐变色
        ax.plot(joint_data['timestamp'], joint_data['M_eq'], 
                linewidth=1.5, label=joint_name, color=colors[idx])
        
        # 设置标签和标题
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('M_eq (N·m)', fontsize=10)
        ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # 设置y轴从0开始（可选）
        y_min = joint_data['M_eq'].min()
        y_max = joint_data['M_eq'].max()
        if y_max > 0:
            ax.set_ylim(bottom=0, top=y_max * 1.1)
    
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
    """绘制所有关节的M_eq曲线叠加在同一张图上"""
    
    # 获取所有唯一的关节
    unique_joints = sorted(df['joint_name'].unique())
    num_joints = len(unique_joints)
    
    print(f"Found {num_joints} unique joints")
    
    # 创建渐变色colormap
    cmap = plt.cm.get_cmap('viridis')  # 使用viridis渐变色
    colors = [cmap(i / max(1, num_joints - 1)) for i in range(num_joints)]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 为每个关节绘制曲线
    for idx, joint_name in enumerate(unique_joints):
        # 获取该关节的数据
        joint_data = df[df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        
        # 绘制曲线，使用渐变色
        ax.plot(joint_data['timestamp'], joint_data['M_eq'], 
                linewidth=1.5, label=joint_name, alpha=0.8, color=colors[idx])
    
    # 设置标签和标题
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('M_eq (N·m)', fontsize=12)
    ax.set_title('All Joints M_eq (Equivalent Bending Moment) vs Time', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # 设置y轴从0开始
    y_max = df['M_eq'].max()
    if y_max > 0:
        ax.set_ylim(bottom=0, top=y_max * 1.1)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot M_eq (Equivalent Bending Moment) vs Time for all joints',
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
    required_columns = ['timestamp', 'joint_name', 'M_eq']
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

