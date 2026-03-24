#!/bin/bash

# 自动化推倒数据采集脚本
# 每次启动RL+MuJoCo、自动推力(auto_sampling=true)、等待、关闭
# pm_v2.yaml 需配置：default_force_magnitude, force_duration, auto_delay
# 8方向循环：0=前, 45=左前, 90=左, 135=左后, 180=后, 225=右后, 270=右, 315=右前

# 运控选择：XZL 或 CHR
CONTROLLER=XZL

# 保存格式：csv（文本）或 bin（二进制，体积小、写入快）
SAVE_FORMAT=csv

# 日志/数据根目录（可修改，如 /mnt/ssd/data）
LOG_BASE_DIR="${HOME}/data/mujoco_logs"

SAVE_FORMAT=$(echo "$SAVE_FORMAT" | tr '[:upper:]' '[:lower:]')
case "$SAVE_FORMAT" in
  csv|bin) ;;
  *) echo "错误: SAVE_FORMAT 仅支持 csv 或 bin，当前: $SAVE_FORMAT"; exit 1 ;;
esac

TOTAL_EXPERIMENTS=80
EXPERIMENT_DURATION=15
DIRECTION_ANGLES=(0 45 90 135 180 225 270 315)
DIRECTION_NAMES=("forward" "forward_left" "left" "backward_left" "backward" "backward_right" "right" "forward_right")

RL_LAUNCH="rl_basic_example_${CONTROLLER}.launch.py"
RL_PROCESS="rl_basic_example_${CONTROLLER}"

echo "=========================================="
echo "自动化推倒数据采集"
echo "=========================================="
echo "运控: $CONTROLLER ($RL_LAUNCH)"
echo "实验次数: $TOTAL_EXPERIMENTS"
echo "实验持续时间: ${EXPERIMENT_DURATION}秒"
echo "保存格式: $SAVE_FORMAT"
echo "=========================================="

# 工作目录：脚本所在目录的上一级
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(realpath -s "$SCRIPT_DIR/..")"
cd "$WORKSPACE_DIR" || { echo "错误: 无法进入工作目录 $WORKSPACE_DIR"; exit 1; }

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

# 保证 rl_basic_example_* 运行时能找到 libglog.so.0（CHR 需要；系统或 conda）
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

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
        csv_format:=$SAVE_FORMAT \
        csv_file_path:="$RUN_CSV_DIR" &
    MUJOCO_PID=$!

    # 等待实验完成
    echo "等待${EXPERIMENT_DURATION}秒..."
    sleep $EXPERIMENT_DURATION
    
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
echo "数据文件（$SAVE_FORMAT）保存在: $LOG_BASE_DIR"

echo ""
echo "批量摔倒风险汇总（batch_calculate_fall_risk）..."
python3 scripts/batch_calculate_fall_risk.py \
  --root "$EXPERIMENT_FOLDER"

echo ""
echo "合并接触数据（按方向 + 摔倒类型 + 最小力阈值）..."
python3 scripts/merge_contact_data.py \
  "$EXPERIMENT_FOLDER" \
  --add-fall-type \
  --group-by-direction \
  --min-force-n 400

echo ""
echo "合并为 all_directions_merged.csv..."
python3 scripts/merge_contact_data.py \
  "$EXPERIMENT_FOLDER" \
  all_directions_merged.csv \
  --pattern "merged_contact_data_*.csv" \
  --fast-append

MERGED_CSV="$EXPERIMENT_FOLDER/all_directions_merged.csv"
echo ""
echo "绘制接触网格图（plot_contact_grid）..."
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --no-hardcode --no-force-filter