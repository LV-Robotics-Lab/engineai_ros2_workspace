#!/bin/bash

echo "=== 快速接触力曲线分析 ==="

# 创建logs目录
mkdir -p logs

echo "步骤1: 启动仿真器生成接触力数据（10秒）..."
echo "请等待仿真器启动并运行..."

# 启动仿真器，运行10秒后停止
timeout 15s ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true

echo ""
echo "仿真器已停止"
echo "步骤2: 分析接触力曲线..."

# 查找最新的CSV文件
latest_csv=$(ls -t logs/contact_data_*.csv 2>/dev/null | head -1)

if [[ -n "$latest_csv" ]]; then
    echo "找到CSV文件: $latest_csv"
    echo "文件大小: $(du -h "$latest_csv" | cut -f1)"
    echo ""
    
    # 显示文件基本信息
    echo "CSV文件基本信息:"
    echo "总行数: $(wc -l < "$latest_csv")"
    echo "前5行数据:"
    head -5 "$latest_csv"
    echo ""
    
    # 运行分析脚本
    echo "正在运行接触力分析..."
    python3 scripts/analyze_contact_forces.py "$latest_csv"
    
else
    echo "未找到CSV文件，请检查仿真是否正常运行"
    echo "可用的CSV文件:"
    ls -la logs/contact_data_*.csv 2>/dev/null || echo "无"
fi

echo ""
echo "=== 分析完成 ===" 