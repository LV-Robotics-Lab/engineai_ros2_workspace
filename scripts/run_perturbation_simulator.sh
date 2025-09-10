#!/bin/bash

# 推倒采样仿真器启动脚本
# Perturbation Sampling Simulator Launch Script

echo "=========================================="
echo "EngineAI 推倒采样仿真器启动脚本"
echo "Perturbation Sampling Simulator Launcher"
echo "=========================================="

# 检查是否在正确的工作目录
if [ ! -f "src/simulation/mujoco/CMakeLists.txt" ]; then
    echo "错误: 请在engineai_ros2_workspace根目录下运行此脚本"
    echo "Error: Please run this script from the engineai_ros2_workspace root directory"
    exit 1
fi

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境，建议激活engineai_ros2环境"
    echo "Warning: No conda environment detected, consider activating engineai_ros2"
fi

# 构建项目
echo "正在构建推倒采样仿真器..."
echo "Building perturbation simulator..."
./scripts/build_nodes.sh sim

if [ $? -ne 0 ]; then
    echo "构建失败，请检查错误信息"
    echo "Build failed, please check error messages"
    exit 1
fi

# 设置环境
echo "设置ROS2环境..."
echo "Setting up ROS2 environment..."
source install/setup.bash

# 显示键盘控制说明
echo ""
echo "=========================================="
echo "键盘控制说明 / Keyboard Controls:"
echo "=========================================="
echo "Shift + F/B: 前后向干扰力 / Forward/Backward force"
echo "Shift + L/R: 左右向干扰力 / Left/Right force"
echo "Shift + U/D: 上下向干扰力 / Up/Down force"
echo "Shift + G/J: X轴干扰力矩 / X-axis torque"
echo "Shift + Y/H: Y轴干扰力矩 / Y-axis torque"
echo "Shift + [/]: Z轴干扰力矩 / Z-axis torque"
echo "Shift + 0: 立即停止所有干扰力 / Stop all forces immediately"
echo "Shift + +/-: 调整干扰力大小 / Adjust force magnitude"
echo "Shift + ,/.: 调整干扰力持续时间 / Adjust force duration"
echo ""
echo "按Ctrl+C退出仿真"
echo "Press Ctrl+C to exit simulation"
echo "=========================================="

# 运行推倒采样仿真器
echo "启动推倒采样仿真器..."
echo "Starting perturbation simulator..."
ros2 run mujoco_simulator perturbation_simulator
