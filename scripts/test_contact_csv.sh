#!/bin/bash

# 测试接触力CSV每帧保存功能

echo "=== 测试接触力CSV每帧保存功能 ==="

# 创建logs目录（如果不存在）
mkdir -p logs

# 启动仿真器，启用每帧保存
echo "启动仿真器，启用每帧保存..."
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=true \
    csv_save_frequency:=1 \
    headless:=true \
    duration:=5.0 &

# 等待仿真器启动
sleep 3

echo "仿真器已启动，运行5秒..."

# 等待仿真完成
sleep 8

echo "仿真完成，检查CSV文件..."

# 查找最新的CSV文件
CSV_FILE=$(ls -t logs/contact_data_*.csv 2>/dev/null | head -1)

if [ -z "$CSV_FILE" ]; then
    echo "❌ 未找到CSV文件"
    exit 1
fi

echo "✅ 找到CSV文件: $CSV_FILE"

# 检查文件大小
FILE_SIZE=$(du -h "$CSV_FILE" | cut -f1)
echo "文件大小: $FILE_SIZE"

# 检查行数
LINE_COUNT=$(wc -l < "$CSV_FILE")
echo "总行数: $LINE_COUNT"

# 检查时间戳范围
echo "检查时间戳范围..."
head -1 "$CSV_FILE"  # 显示标题行
tail -5 "$CSV_FILE"  # 显示最后5行

# 分析时间戳
echo "分析时间戳..."
python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$CSV_FILE')
    print(f'数据时间范围: {df[\"timestamp\"].min():.6f} - {df[\"timestamp\"].max():.6f}')
    print(f'时间间隔: {df[\"timestamp\"].diff().mean():.6f} 秒')
    print(f'平均每帧接触点数: {len(df) / len(df[\"timestamp\"].unique()):.2f}')
    print(f'总帧数: {len(df[\"timestamp\"].unique())}')
    
    # 检查是否有重复时间戳
    duplicate_times = df['timestamp'].duplicated().sum()
    print(f'重复时间戳数量: {duplicate_times}')
    
    # 检查接触力数据
    force_cols = ['force_x', 'force_y', 'force_z']
    for col in force_cols:
        non_zero = (df[col] != 0).sum()
        print(f'{col} 非零值数量: {non_zero}')
        
except Exception as e:
    print(f'分析失败: {e}')
    sys.exit(1)
"

echo "=== 测试完成 ===" 