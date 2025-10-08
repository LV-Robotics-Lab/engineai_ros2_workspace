#!/bin/bash

# 完全清理的构建脚本，避免 conda 环境干扰
echo "Starting clean build without conda interference"

# 获取源码目录
root_dir="$(realpath -s $(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/..)"

# 完全清理环境变量
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PYTHON_EXE
unset CONDA_EXE
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset CONDA_ROOT
unset CONDA_BACKUP_PATH
unset CONDA_MANAGED_STATE

# 设置纯净的 PATH
export PATH="/usr/bin:/usr/local/bin:/usr/sbin:/usr/local/sbin:/sbin:/bin"

# 设置编译器
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LD=/usr/bin/ld

# 设置 Python 环境
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"
export ROSIDL_ADAPTER_PYTHON_EXECUTABLE="/usr/bin/python3"
export AMENT_PYTHON_EXECUTABLE=/usr/bin/python3

# 设置 ROS 环境
export AMENT_TRACE_SETUP_FILES=0
export CMAKE_PREFIX_PATH="/opt/ros/humble:/usr/lib/x86_64-linux-gnu/cmake:$CMAKE_PREFIX_PATH"
export iceoryx_binding_c_DIR="/opt/ros/humble/lib/x86_64-linux-gnu/cmake/iceoryx_binding_c"
export iceoryx_hoofs_DIR="/opt/ros/humble/lib/x86_64-linux-gnu/cmake/iceoryx_hoofs"
export iceoryx_posh_DIR="/opt/ros/humble/lib/x86_64-linux-gnu/cmake/iceoryx_posh"

# 清理构建目录
echo "Cleaning build directory..."
rm -rf "$root_dir/build"
rm -rf "$root_dir/install"
mkdir -p "$root_dir/build"

# 设置目标节点
NODES=("mujoco_simulator" "interface_protocol" "interface_example")
PACKAGES_ARG="--packages-select ${NODES[*]}"

echo "Building with clean environment..."
echo "PATH: $PATH"
echo "CC: $CC"
echo "CXX: $CXX"

# 运行构建
cd "$root_dir" && \
colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --cmake-args -DCMAKE_PREFIX_PATH="/opt/ros/humble:/usr/lib/x86_64-linux-gnu/cmake:$CMAKE_PREFIX_PATH" \
    --cmake-args -Diceoryx_binding_c_DIR="/opt/ros/humble/lib/x86_64-linux-gnu/cmake/iceoryx_binding_c" \
    --build-base build \
    --install-base install \
    $PACKAGES_ARG

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "To use the built packages, run:"
    echo "source /opt/ros/humble/setup.bash"
    echo "source $root_dir/install/setup.bash"
else
    echo "Build failed!"
    exit 1
fi



