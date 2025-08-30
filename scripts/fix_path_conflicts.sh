#!/bin/bash

# 解决Python路径冲突的脚本

echo "=== 解决Python路径冲突 ==="

# 1. 确保使用正确的conda环境
if [[ "$CONDA_DEFAULT_ENV" != "engineai_ros2" ]]; then
    echo "激活engineai_ros2环境..."
    conda activate engineai_ros2
fi

# 2. 设置正确的Python路径
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"

# 3. 设置正确的库路径
export LD_LIBRARY_PATH="/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:$LD_LIBRARY_PATH"

# 4. 确保ROS2环境正确加载
source /opt/ros/humble/setup.bash

# 5. 加载工作空间
source /home/wang22/engineai/engineai_ros2_workspace/install/setup.bash

echo "=== 环境设置完成 ==="
echo "Python路径: $(which python3)"
echo "Python版本: $(python3 --version)"
echo "ROS2版本: $(ros2 --version)"

# 6. 清理可能的冲突
echo "清理可能的Python包冲突..."
pip uninstall -y catkin_pkg empy || true

# 7. 重新安装必要的包
echo "重新安装必要的包..."
pip install catkin_pkg empy

echo "=== 路径冲突解决完成 ===" 