#!/usr/bin/env python3
"""
测试坐标转换功能的脚本
验证新的URDF坐标转换是否正确考虑了关节角度变化
"""

import pandas as pd
import numpy as np
import sys
import os

def test_coordinate_correction(csv_file):
    """测试坐标转换功能"""
    print(f"测试坐标转换功能: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"CSV文件不存在: {csv_file}")
        return
    
    # 加载数据
    df = pd.read_csv(csv_file)
    print(f"加载了 {len(df)} 行数据")
    
    # 检查是否有新的坐标字段
    has_corrected_coords = ('urdf_corrected_x_body1' in df.columns and 
                           'urdf_corrected_y_body1' in df.columns and 
                           'urdf_corrected_z_body1' in df.columns)
    
    has_legacy_coords = ('urdf_x_body1' in df.columns and 
                        'urdf_y_body1' in df.columns and 
                        'urdf_z_body1' in df.columns)
    
    print(f"\n=== 坐标字段检查 ===")
    print(f"新的修正坐标字段: {'是' if has_corrected_coords else '否'}")
    print(f"旧的坐标字段: {'是' if has_legacy_coords else '否'}")
    
    if has_corrected_coords and has_legacy_coords:
        print("\n=== 坐标比较分析 ===")
        
        # 比较新旧坐标
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # 计算差异
        diff_x = df_sample['urdf_corrected_x_body1'] - df_sample['urdf_x_body1']
        diff_y = df_sample['urdf_corrected_y_body1'] - df_sample['urdf_y_body1']
        diff_z = df_sample['urdf_corrected_z_body1'] - df_sample['urdf_z_body1']
        
        print(f"\n新旧坐标差异统计:")
        print(f"  X轴差异: 均值={diff_x.mean():.6f}, 标准差={diff_x.std():.6f}")
        print(f"  Y轴差异: 均值={diff_y.mean():.6f}, 标准差={diff_y.std():.6f}")
        print(f"  Z轴差异: 均值={diff_z.mean():.6f}, 标准差={diff_z.std():.6f}")
        
        # 检查是否有显著差异
        significant_diff = (abs(diff_x) > 0.001) | (abs(diff_y) > 0.001) | (abs(diff_z) > 0.001)
        print(f"\n显著差异的行数: {significant_diff.sum()}/{sample_size} ({significant_diff.sum()/sample_size*100:.1f}%)")
        
        if significant_diff.any():
            print(f"\n前5个有显著差异的坐标对:")
            diff_rows = df_sample[significant_diff].head(5)
            for i, (_, row) in enumerate(diff_rows.iterrows()):
                print(f"  样本 {i+1}:")
                print(f"    旧坐标: ({row['urdf_x_body1']:.6f}, {row['urdf_y_body1']:.6f}, {row['urdf_z_body1']:.6f})")
                print(f"    新坐标: ({row['urdf_corrected_x_body1']:.6f}, {row['urdf_corrected_y_body1']:.6f}, {row['urdf_corrected_z_body1']:.6f})")
                print(f"    差异:   ({diff_x.iloc[i]:.6f}, {diff_y.iloc[i]:.6f}, {diff_z.iloc[i]:.6f})")
                print()
        else:
            print("所有坐标都相同，可能机器人在仿真过程中没有关节运动")
    
    elif has_corrected_coords:
        print("只有新的修正坐标字段，无法进行比较")
        print("新坐标范围:")
        print(f"  X: {df['urdf_corrected_x_body1'].min():.6f} 到 {df['urdf_corrected_x_body1'].max():.6f}")
        print(f"  Y: {df['urdf_corrected_y_body1'].min():.6f} 到 {df['urdf_corrected_y_body1'].max():.6f}")
        print(f"  Z: {df['urdf_corrected_z_body1'].min():.6f} 到 {df['urdf_corrected_z_body1'].max():.6f}")
    
    elif has_legacy_coords:
        print("只有旧的坐标字段，建议重新运行仿真以生成新的修正坐标")
    
    else:
        print("没有找到任何URDF坐标字段")
    
    # 检查世界坐标
    if 'pos_x' in df.columns:
        print(f"\n=== 世界坐标范围 ===")
        print(f"  X: {df['pos_x'].min():.6f} 到 {df['pos_x'].max():.6f}")
        print(f"  Y: {df['pos_y'].min():.6f} 到 {df['pos_y'].max():.6f}")
        print(f"  Z: {df['pos_z'].min():.6f} 到 {df['pos_z'].max():.6f}")
    
    # 检查接触力数据
    if 'force_magnitude' in df.columns:
        print(f"\n=== 接触力统计 ===")
        print(f"  最大力: {df['force_magnitude'].max():.2f} N")
        print(f"  平均力: {df['force_magnitude'].mean():.2f} N")
        print(f"  力的标准差: {df['force_magnitude'].std():.2f} N")
    
    print(f"\n=== 测试完成 ===")

def main():
    if len(sys.argv) != 2:
        print("用法: python3 test_coordinate_correction.py <csv_file>")
        print("示例: python3 test_coordinate_correction.py logs/contact_data_20250122_123456.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    test_coordinate_correction(csv_file)

if __name__ == "__main__":
    main() 