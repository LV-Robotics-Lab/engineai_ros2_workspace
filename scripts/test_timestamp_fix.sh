#!/bin/bash

echo "=== 测试时间戳修复 ==="

# 创建logs目录
mkdir -p logs

echo "启动仿真器并保存CSV（5秒后自动停止）..."
echo "请观察CSV文件中的时间戳是否正确递增"

# 启动仿真器
timeout 10s ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true

echo ""
echo "仿真器已停止"
echo "检查生成的CSV文件："

# 查找最新的CSV文件
latest_csv=$(ls -t logs/contact_data_*.csv 2>/dev/null | head -1)

if [[ -n "$latest_csv" ]]; then
    echo "最新CSV文件: $latest_csv"
    echo ""
    echo "前10行数据（检查时间戳）:"
    head -10 "$latest_csv"
    echo ""
    echo "时间戳统计:"
    tail -n +2 "$latest_csv" | cut -d',' -f1 | sort -n | head -5
    echo "..."
    tail -n +2 "$latest_csv" | cut -d',' -f1 | sort -n | tail -5
else
    echo "未找到CSV文件"
fi

echo ""
echo "=== 测试完成 ===" 