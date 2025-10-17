#!/bin/bash

# 自动化推倒数据采集脚本
# 4000次实验，每次启动RL+MuJoCo、自动推力、等10秒、关闭

TOTAL_EXPERIMENTS=100
EXPERIMENT_DURATION=15
DIRECTIONS=("forward" "backward" "left" "right")

echo "=========================================="
echo "自动化推倒数据采集"
echo "=========================================="
echo "实验次数: $TOTAL_EXPERIMENTS"
echo "实验持续时间: ${EXPERIMENT_DURATION}秒"
echo "=========================================="

# 进入工作目录
cd /home/wang22/engineai/engineai_ros2_workspace

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate engineai_ros2
source install/setup.bash

# 创建实验文件夹
PROGRAM_START_TIME=$(date +"%Y%m%d_%H%M%S")
echo "程序开始时间: $PROGRAM_START_TIME"

# 从YAML文件读取推力参数
FORCE_MAGNITUDE=$(grep "default_force_magnitude:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*default_force_magnitude: *\([0-9.]*\).*/\1/')
DEFAULT_DIRECTION=$(grep "auto_direction:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*auto_direction: *"\([^"]*\)".*/\1/')
# 从YAML文件读取推力持续时间参数
FORCE_DURATION=$(grep "force_duration:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*force_duration: *\([0-9.]*\).*/\1/')


# 创建实验文件夹名称：推力方向-大小-程序开始时间
EXPERIMENT_FOLDER="logs/${DEFAULT_DIRECTION}-${FORCE_MAGNITUDE}N-${FORCE_DURATION}s-${PROGRAM_START_TIME}"

# 创建实验文件夹
mkdir -p "$EXPERIMENT_FOLDER"
echo "实验数据将保存到: $EXPERIMENT_FOLDER"

# 清理函数
cleanup_processes() {
    echo "强制清理所有进程..."
    
    # 关闭MuJoCo窗口
    pkill -f "mujoco" 2>/dev/null
    pkill -f "simulate" 2>/dev/null
    
    # 关闭ROS进程
    pkill -f "rl_basic_example_XZL" 2>/dev/null
    pkill -f "mujoco_simulator" 2>/dev/null
    pkill -f "joint_state_converter" 2>/dev/null
    pkill -f "robot_state_publisher" 2>/dev/null
    pkill -f "static_transform_publisher" 2>/dev/null
    
    # 关闭ROS launch进程
    pkill -f "ros2 launch" 2>/dev/null
    
    # 强制关闭所有相关进程
    pkill -9 -f "mujoco" 2>/dev/null
    pkill -9 -f "simulate" 2>/dev/null
    pkill -9 -f "rl_basic" 2>/dev/null
    
    # 等待进程完全关闭
    sleep 3
    
    echo "进程清理完成"
}

# 循环4000次实验
for ((i=1; i<=TOTAL_EXPERIMENTS; i++)); do
    echo ""
    echo "=========================================="
    echo "实验 $i/$TOTAL_EXPERIMENTS"
    echo "=========================================="
    
    # 从YAML文件读取方向设置
    DIRECTION=$(grep "auto_direction:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*auto_direction: *"\([^"]*\)".*/\1/')
    
    echo "方向: $DIRECTION"
    
    # 修改YAML文件中的自动采样配置
    sed -i "s/auto_sampling: .*/auto_sampling: true/" src/simulation/mujoco/assets/config/pm_v2.yaml
    
    # 启动RL控制器
    echo "启动RL控制器..."
    ros2 launch interface_example rl_basic_example_XZL.launch.py &
    RL_PID=$!
    sleep 3
    
    # 启动MuJoCo仿真器
    echo "启动MuJoCo仿真器..."
    ros2 launch mujoco_simulator mujoco_simulator.launch.py \
        export_contact:=true \
        save_contact_csv:=true \
        save_perturbation_csv:=true \
        csv_file_path:="$EXPERIMENT_FOLDER" &
    MUJOCO_PID=$!

    # 等待10秒后关闭RL控制器模拟断电
    echo "等待10秒后关闭RL控制器模拟断电..."
    sleep 10
    
    # 关闭RL控制器
    echo "关闭RL控制器模拟断电..."
    pkill -f "rl_basic_example_XZL" 2>/dev/null
    echo "RL控制器已关闭"
    
    # 继续运行MuJoCo仿真器直到实验完成（断电后剩余时间）
    REMAINING_TIME=$((EXPERIMENT_DURATION - 10))
    echo "继续运行MuJoCo仿真器${REMAINING_TIME}秒（断电后剩余时间）..."
    sleep $REMAINING_TIME
    
    # 强制清理所有进程
    cleanup_processes
    
    # 短暂休息
    sleep 2
    
    echo "实验 $i 完成"
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "CSV文件保存在: logs/"