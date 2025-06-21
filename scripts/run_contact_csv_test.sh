#!/bin/bash

# 快速启动接触力CSV每帧保存测试

echo "🚀 启动接触力CSV每帧保存测试"
echo ""

# 设置参数
SAVE_CSV=true
FREQUENCY=1  # 每帧都保存
DURATION=10  # 运行10秒

echo "参数设置:"
echo "  - 保存CSV: $SAVE_CSV"
echo "  - 保存频率: 每${FREQUENCY}帧"
echo "  - 仿真时长: ${DURATION}秒"
echo ""

# 启动仿真器
echo "启动仿真器..."
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=$SAVE_CSV \
    csv_save_frequency:=$FREQUENCY \
    headless:=true \
    duration:=$DURATION

echo ""
echo "✅ 仿真完成！"
echo "检查logs目录中的CSV文件:"
ls -la logs/contact_data_*.csv 2>/dev/null || echo "未找到CSV文件" 