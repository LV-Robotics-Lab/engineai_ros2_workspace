#!/bin/bash

# 自动化推倒数据采集脚本 - 连续模式（方案二）
# MuJoCo 和运控常驻，仅通过 reset + CSV 切换进行多次实验，避免重复启动
# 相比 automated_collection.sh 更快：省去每次实验的进程启动/关闭
# 8方向循环：0=前, 45=左前, 90=左, 135=左后, 180=后, 225=右后, 270=右, 315=右前

# 运控选择：XZL 或 CHR
CONTROLLER=XZL

# 日志/CSV 根目录（可修改，如 /mnt/ssd/data）
LOG_BASE_DIR="${HOME}/data/mujoco_logs"

TOTAL_EXPERIMENTS=1600
EXPERIMENT_DURATION=15
DIRECTION_ANGLES=(0 45 90 135 180 225 270 315)
DIRECTION_NAMES=("forward" "forward_left" "left" "backward_left" "backward" "backward_right" "right" "forward_right")

RL_LAUNCH="rl_basic_example_${CONTROLLER}.launch.py"
RL_PROCESS="rl_basic_example_${CONTROLLER}"

echo "=========================================="
echo "自动化推倒数据采集 - 连续模式"
echo "=========================================="
echo "运控: $CONTROLLER ($RL_LAUNCH)"
echo "实验次数: $TOTAL_EXPERIMENTS"
echo "实验持续时间: ${EXPERIMENT_DURATION}秒"
echo "=========================================="

# 进入工作目录
cd /home/wang22/engineai/engineai_ros2_workspace

# 激活conda环境
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
  echo "错误: 未找到 conda"
  exit 1
fi
source "$CONDA_SH"
conda activate engineai_ros2
source install/setup.bash

# 创建实验文件夹
PROGRAM_START_TIME=$(date +"%Y%m%d_%H%M%S")
FORCE_MAGNITUDE=$(grep "default_force_magnitude:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*default_force_magnitude: *\([0-9.]*\).*/\1/')
FORCE_DURATION=$(grep "force_duration:" src/simulation/mujoco/assets/config/pm_v2.yaml | sed 's/.*force_duration: *\([0-9.]*\).*/\1/')
EXPERIMENT_FOLDER="${LOG_BASE_DIR}/8dir-${FORCE_MAGNITUDE}N-${FORCE_DURATION}s-${PROGRAM_START_TIME}"
mkdir -p "$EXPERIMENT_FOLDER"
echo "实验数据将保存到: $EXPERIMENT_FOLDER"

# 清理函数
cleanup_processes() {
    echo "开始清理所有进程..."
    pkill -TERM -f "mujoco_simulator" 2>/dev/null
    pkill -TERM -f "$RL_PROCESS" 2>/dev/null
    pkill -TERM -f "ros2 launch" 2>/dev/null
    for i in {1..10}; do
        if ! pgrep -f "mujoco_simulator" > /dev/null && ! pgrep -f "$RL_PROCESS" > /dev/null; then
            break
        fi
        sleep 1
    done
    if pgrep -f "mujoco_simulator" > /dev/null || pgrep -f "$RL_PROCESS" > /dev/null; then
        pkill -9 -f "mujoco" 2>/dev/null
        pkill -9 -f "rl_basic" 2>/dev/null
    fi
    sleep 2
    echo "进程清理完成"
}

# 启动 RL 控制器（只启动一次）
sed -i "s/auto_sampling: .*/auto_sampling: true/" src/simulation/mujoco/assets/config/pm_v2.yaml
echo "启动RL控制器..."
ros2 launch interface_example "$RL_LAUNCH" &
RL_PID=$!
sleep 3

# 实验循环
for ((i=1; i<=TOTAL_EXPERIMENTS; i++)); do
    echo ""
    echo "=========================================="
    echo "实验 $i/$TOTAL_EXPERIMENTS"
    echo "=========================================="

    IDX=$(( (i - 1) % ${#DIRECTION_ANGLES[@]} ))
    DIRECTION_ANGLE=${DIRECTION_ANGLES[$IDX]}
    DIRECTION=${DIRECTION_NAMES[$IDX]}
    echo "方向: $DIRECTION (${DIRECTION_ANGLE}°)"

    RUN_CSV_DIR="${EXPERIMENT_FOLDER}/${DIRECTION}-${FORCE_MAGNITUDE}N-${i}"
    mkdir -p "$RUN_CSV_DIR"

    sed -i "s/\(auto_direction_angle: *\)[0-9.]*/\1${DIRECTION_ANGLE}/" src/simulation/mujoco/assets/config/pm_v2.yaml

    if [ $i -eq 1 ]; then
        # 第一次：启动 MuJoCo（方向已在上面 sed 写入 YAML）
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
    else
        # 后续：先发布方向，再发布路径，触发 CSV 切换 + reset
        echo "切换新实验目录并触发 reset..."
        ros2 topic pub --once /mujoco/next_direction_angle std_msgs/msg/Float64 "{data: ${DIRECTION_ANGLE}}"
        sleep 0.2
        ros2 topic pub --once /mujoco/new_experiment std_msgs/msg/String "{data: '$RUN_CSV_DIR'}"
        sleep 1
    fi

    echo "等待${EXPERIMENT_DURATION}秒..."
    sleep $EXPERIMENT_DURATION

    echo "实验 $i 完成"
done

cleanup_processes

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "CSV文件保存在: $EXPERIMENT_FOLDER"
