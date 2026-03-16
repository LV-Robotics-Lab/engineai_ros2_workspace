#!/bin/bash

# 自动化断电摔倒数据采集脚本
# 每次启动RL+MuJoCo、自动推力(auto_sampling=true)、10秒后关闭RL模拟断电、等待
# pm_v2.yaml 需配置：default_force_magnitude, force_duration, auto_delay
# 8方向循环：0=前, 45=左前, 90=左, 135=左后, 180=后, 225=右后, 270=右, 315=右前

# 运控选择：XZL 或 CHR
CONTROLLER=XZL

# 日志/CSV 根目录（可修改，如 /mnt/ssd/data）
LOG_BASE_DIR="${HOME}/data/mujoco_logs"

TOTAL_EXPERIMENTS=100
EXPERIMENT_DURATION=15
DIRECTION_ANGLES=(0 45 90 135 180 225 270 315)
DIRECTION_NAMES=("forward" "forward_left" "left" "backward_left" "backward" "backward_right" "right" "forward_right")

RL_LAUNCH="rl_basic_example_${CONTROLLER}.launch.py"
RL_PROCESS="rl_basic_example_${CONTROLLER}"

echo "=========================================="
echo "自动化断电摔倒数据采集"
echo "=========================================="
echo "运控: $CONTROLLER ($RL_LAUNCH)"
echo "实验次数: $TOTAL_EXPERIMENTS"
echo "实验持续时间: ${EXPERIMENT_DURATION}秒"
echo "=========================================="

# 进入工作目录
cd /home/wang22/engineai/engineai_ros2_workspace

# 激活conda环境（自动查找 conda 位置）
CONDA_SH=""
if [ -n "$CONDA_EXE" ]; then
  CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(dirname "$(dirname "$(which conda)")")/etc/profile.d/conda.sh"
else
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [ -f "$d/etc/profile.d/conda.sh" ]; then
      CONDA_SH="$d/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
  echo "错误: 未找到 conda，请安装或确保 conda 在 PATH 中"
  exit 1
fi
source "$CONDA_SH"
conda activate engineai_ros2
source install/setup.bash

# 创建实验文件夹
PROGRAM_START_TIME=$(date +"%Y%m%d_%H%M%S")
echo "程序开始时间: $PROGRAM_START_TIME"

# 从YAML文件读取推力参数
FORCE_MAGNITUDE=$(grep "default_force_magnitude:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*default_force_magnitude: *\([0-9.]*\).*/\1/')
# 从YAML文件读取推力持续时间参数
FORCE_DURATION=$(grep "force_duration:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*force_duration: *\([0-9.]*\).*/\1/')


# 创建实验文件夹名称：8方向循环-推力大小-程序开始时间
EXPERIMENT_FOLDER="${LOG_BASE_DIR}/8dir-${FORCE_MAGNITUDE}N-${FORCE_DURATION}s-${PROGRAM_START_TIME}"

# 创建实验文件夹
mkdir -p "$EXPERIMENT_FOLDER"
echo "实验数据将保存到: $EXPERIMENT_FOLDER"

# 清理函数
cleanup_processes() {
    echo "开始清理所有进程..."
    
    # 先尝试优雅关闭（发送SIGTERM信号，允许程序执行清理）
    echo "发送SIGTERM信号，允许程序优雅关闭..."
    pkill -TERM -f "mujoco_simulator" 2>/dev/null
    pkill -TERM -f "$RL_PROCESS" 2>/dev/null
    pkill -TERM -f "ros2 launch" 2>/dev/null
    
    # 等待程序优雅关闭（给异步写入线程时间完成数据写入）
    echo "等待程序优雅关闭（最多10秒）..."
    for i in {1..10}; do
        if ! pgrep -f "mujoco_simulator" > /dev/null && ! pgrep -f "$RL_PROCESS" > /dev/null; then
            echo "程序已优雅关闭"
            break
        fi
        sleep 1
    done
    
    # 如果还有进程在运行，强制关闭
    if pgrep -f "mujoco_simulator" > /dev/null || pgrep -f "$RL_PROCESS" > /dev/null; then
        echo "部分进程未响应，强制关闭..."
        # 关闭MuJoCo窗口
        pkill -f "mujoco" 2>/dev/null
        pkill -f "simulate" 2>/dev/null
        
        # 关闭ROS进程
        pkill -f "$RL_PROCESS" 2>/dev/null
        pkill -f "mujoco_simulator" 2>/dev/null
        pkill -f "joint_state_converter" 2>/dev/null
        pkill -f "robot_state_publisher" 2>/dev/null
        pkill -f "static_transform_publisher" 2>/dev/null
        
        # 关闭ROS launch进程
        pkill -f "ros2 launch" 2>/dev/null
        
        # 最后强制关闭所有相关进程
        pkill -9 -f "mujoco" 2>/dev/null
        pkill -9 -f "simulate" 2>/dev/null
        pkill -9 -f "rl_basic" 2>/dev/null
    fi
    
    # 等待进程完全关闭
    sleep 2
    
    echo "进程清理完成"
}

# 循环4000次实验
for ((i=1; i<=TOTAL_EXPERIMENTS; i++)); do
    echo ""
    echo "=========================================="
    echo "实验 $i/$TOTAL_EXPERIMENTS"
    echo "=========================================="
    
    # 8方向循环：实验 i 使用方向 (i-1) % 8
    IDX=$(( (i - 1) % ${#DIRECTION_ANGLES[@]} ))
    DIRECTION_ANGLE=${DIRECTION_ANGLES[$IDX]}
    DIRECTION=${DIRECTION_NAMES[$IDX]}
    
    echo "方向: $DIRECTION (${DIRECTION_ANGLE}°)"
    
    # 本次实验的 CSV 保存路径：推力方向-大小-编号，文件夹内有多份 CSV（contact、perturbation 等）
    RUN_CSV_DIR="${EXPERIMENT_FOLDER}/${DIRECTION}-${FORCE_MAGNITUDE}N-${i}"
    mkdir -p "$RUN_CSV_DIR"
    
    # 修改YAML：自动采样 + 当前方向角度
    sed -i "s/auto_sampling: .*/auto_sampling: true/" src/simulation/mujoco/assets/config/pm_v2.yaml
    sed -i "s/\(auto_direction_angle: *\)[0-9.]*/\1${DIRECTION_ANGLE}/" src/simulation/mujoco/assets/config/pm_v2.yaml
    
    # 启动RL控制器
    echo "启动RL控制器..."
    ros2 launch interface_example "$RL_LAUNCH" &
    RL_PID=$!
    sleep 3
    
    # 启动MuJoCo仿真器
    echo "启动MuJoCo仿真器..."
    ros2 launch mujoco_simulator mujoco_simulator.launch.py \
        export_contact:=true \
        save_contact_csv:=true \
        save_perturbation_csv:=true \
        save_joint_forces_csv:=true \
        save_sensor_vibration_csv:=true \
        save_joint_state_csv:=true \
        save_link_kinetic_energy_csv:=true \
        save_policy_switch_csv:=true \
        csv_format:=csv \
        csv_file_path:="$RUN_CSV_DIR" &
    MUJOCO_PID=$!

    # 等待10秒后关闭RL控制器模拟断电
    echo "等待10秒后关闭RL控制器模拟断电..."
    sleep 10
    
    # 关闭RL控制器
    echo "关闭RL控制器模拟断电..."
    pkill -f "$RL_PROCESS" 2>/dev/null
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
echo "CSV文件保存在: $LOG_BASE_DIR"