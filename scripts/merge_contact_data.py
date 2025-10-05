#!/usr/bin/env python3
"""
合并多个contact CSV文件的脚本
"""

import os
import sys
import pandas as pd
import glob
from datetime import datetime

def merge_contact_csv_files(log_dir, output_file=None):
    """
    合并指定目录下的所有contact_data_*.csv文件
    
    Args:
        log_dir: 包含CSV文件的目录
        output_file: 输出文件名，如果为None则自动生成
    """
    print(f"正在合并目录: {log_dir}")
    
    # 查找所有contact_data_*.csv文件
    pattern = os.path.join(log_dir, "contact_data_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"在目录 {log_dir} 中没有找到contact_data_*.csv文件")
        return None
    
    # 按文件名排序
    csv_files.sort()
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # 读取并合并所有CSV文件
    all_dataframes = []
    total_rows = 0
    
    for i, csv_file in enumerate(csv_files):
        print(f"\n正在处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"  读取了 {len(df)} 行数据")
            
            # 添加源文件信息 - 使用文件名（不含扩展名）作为标识
            source_name = os.path.splitext(os.path.basename(csv_file))[0]  # 去掉.csv扩展名
            df['source_file'] = source_name
            
            all_dataframes.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"  错误: 无法读取文件 {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("没有成功读取任何CSV文件")
        return None
    
    # 合并所有数据框
    print(f"\n正在合并 {len(all_dataframes)} 个数据框...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"合并完成!")
    print(f"总行数: {len(merged_df)}")
    print(f"总列数: {len(merged_df.columns)}")
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取原文件夹名称
        folder_name = os.path.basename(os.path.abspath(log_dir))
        output_file = os.path.join(log_dir, f"merged_contact_data_{folder_name}_{timestamp}.csv")
    elif not os.path.isabs(output_file):
        # 如果输出文件不是绝对路径，则保存到log_dir目录中
        output_file = os.path.join(log_dir, output_file)
    
    # 保存合并后的文件
    print(f"\n正在保存到: {output_file}")
    merged_df.to_csv(output_file, index=False)
    
    print(f"合并完成! 输出文件: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # 显示数据统计
    print(f"\n=== 合并后的数据统计 ===")
    print(f"总行数: {len(merged_df)}")
    print(f"时间范围: {merged_df['timestamp'].min():.3f} - {merged_df['timestamp'].max():.3f} 秒")
    print(f"力范围: {merged_df['force_magnitude'].min():.3f} - {merged_df['force_magnitude'].max():.3f} N")
    print(f"平均力: {merged_df['force_magnitude'].mean():.3f} N")
    print(f"中位数力: {merged_df['force_magnitude'].median():.3f} N")
    
    # 按源文件统计
    print(f"\n=== 按源文件统计 ===")
    source_stats = merged_df.groupby('source_file').size().sort_values(ascending=False)
    for source, count in source_stats.items():
        print(f"  {source}: {count} 行")
    
    # 显示源文件列的信息
    print(f"\n=== 源文件列信息 ===")
    print(f"源文件列已添加到数据中，列名: 'source_file'")
    print(f"包含 {len(source_stats)} 个不同的源文件")
    print(f"源文件列示例值: {list(source_stats.index[:3])}")  # 显示前3个源文件名
    
    return output_file

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 merge_contact_data.py <log_directory> [output_file]")
        print("示例: python3 merge_contact_data.py logs/forward-200.0N-20251005_162754")
        print("示例: python3 merge_contact_data.py logs/forward-200.0N-20251005_162754 merged_contact.csv")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        sys.exit(1)
    
    # 合并CSV文件
    merged_file = merge_contact_csv_files(log_dir, output_file)
    
    if merged_file:
        print(f"\n✅ 合并成功! 输出文件: {merged_file}")
        print(f"\n现在可以使用以下命令来可视化合并后的数据:")
        print(f"python3 scripts/mujoco_xml_contact_display.py {merged_file} src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1500 true 1,2")
    else:
        print("❌ 合并失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
