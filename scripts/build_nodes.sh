#!/bin/bash

# Initialize conda in the script (if exists)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "Conda initialized"
else
    echo "Miniconda not found, skipping conda initialization"
fi

# Gets the source directory
root_dir="$(realpath -s $(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/..)"

# Define node lists for different hosts
# Nodes to build on example
EXAMPLE_NODES=(
    "interface_protocol"
    "rdk_framework"
    "interface_example"
)

# Nodes to build on Orin
APPLICATION_NODES=(
    "interface_protocol"
    "rdk_framework"
    "robot_manager"
    "sound_hub"
    "system_monitor"
)

SIM_NODES=(
    "mujoco_simulator"
    "interface_protocol"
    "interface_example"
)
# Default target host is example
TARGET_HOST="example"

# Default build type is Release
BUILD_TYPE="Release"

# Parse command line arguments - only accept host name
if [[ -n "$1" ]]; then
    TARGET_HOST="$1"
fi

# Validate target host
if [[ "$TARGET_HOST" != "example" && "$TARGET_HOST" != "app" && "$TARGET_HOST" != "sim" ]]; then
    echo "Error: Invalid target host '$TARGET_HOST'"
    echo "Available hosts: example, app, sim"
    exit 1
fi

# Select the appropriate node list based on target host
if [[ "$TARGET_HOST" == "example" ]]; then
    NODES=("${EXAMPLE_NODES[@]}")
    echo "Building for example nodes"
elif [[ "$TARGET_HOST" == "app" ]]; then
    NODES=("${APPLICATION_NODES[@]}")
    echo "Building for application nodes"
elif [[ "$TARGET_HOST" == "sim" ]]; then
    NODES=("${SIM_NODES[@]}")
    echo "Building for simulation nodes"
fi

echo "Build type: $BUILD_TYPE"

# Create packages argument for colcon build
PACKAGES_ARG=""
if [[ ${#NODES[@]} -gt 0 ]]; then
    PACKAGES_ARG="--packages-select ${NODES[*]}"
fi

# Create build directory if it doesn't exist
mkdir -p "$root_dir/build"

# Source ROS2 environment first
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "Sourcing ROS2 Humble environment..."
    source /opt/ros/humble/setup.bash
else
    echo "Warning: ROS2 Humble setup.bash not found at /opt/ros/humble/setup.bash"
fi

# Activate conda environment if it exists
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    echo "Activating conda environment..."
    conda activate engineai_ros2
    # Ensure conda environment Python is used
    export PATH="$CONDA_PREFIX/bin:$PATH"
    echo "Using Python: $(which python3)"
    # Set environment variables to avoid numpy issues
    export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages:/opt/ros/humble/lib/python3.10/site-packages"
    export COLCON_PYTHON_SETUP_PY_SKIP_PACKAGES="numpy"
    # Set additional environment variables to suppress numpy-related errors
    export COLCON_VERBOSE=0
    export PYTHONWARNINGS="ignore"
    # Redirect stderr to suppress error messages
    export COLCON_LOG_LEVEL=error
    # Use system Python for ROS2 tools
    export ROSIDL_ADAPTER_PYTHON_EXECUTABLE="/usr/bin/python3"
    export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
    export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"
fi

# Set PKG_CONFIG_PATH to ensure pkg-config can find system packages like LCM
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig:${PKG_CONFIG_PATH}"

# Create .colcon_ignore file to exclude problematic numpy directories
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    echo "Creating .colcon_ignore file to exclude numpy test directories..."
    cat > "$root_dir/.colcon_ignore" << EOF
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core/tests/examples/cython
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core/tests/examples/limited_api
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core/tests/examples/cython/setup.py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core/tests/examples/limited_api/setup.py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/random/_examples/cython
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/random/_examples/cython/setup.py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/core/tests/examples/cython
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/core/tests/examples/limited_api
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/core/tests/examples/cython/setup.py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/core/tests/examples/limited_api/setup.py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/distutils
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/fft
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/lib
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/linalg
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/ma
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/matrixlib
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/polynomial
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/random
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/testing
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/typing
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/array_api
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/compat
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/f2py
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_typing
$CONDA_PREFIX/lib/python3.10/site-packages/numpy/core
$CONDA_PREFIX/lib/python3.10/site-packages/numpy
EOF
fi

# Run colcon build with the selected options
echo "Running build with the following nodes: ${NODES[*]}"
cd "$root_dir" && \
colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    --build-base build \
    --install-base install \
    $PACKAGES_ARG 2>/dev/null || colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
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
