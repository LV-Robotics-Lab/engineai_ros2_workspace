#!/bin/bash
# Gets the source directory
ROOT_DIR="$(realpath -s $(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/..)"

# Define node lists for different hosts
# Nodes to build on protocol
PROTOCOL_NODES=(
    "interface_protocol"
)

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
    "data_recorder"
    "dashboard"
)

SIM_NODES=(
    "mujoco_simulator"
    "interface_protocol"
    "interface_example"
)

# ALL_APPLICATION_NODES=("${APPLICATION_NODES[@]}" "${WEB_NODES[@]}")

WEB_NODES=(
    "dashboard_frontend"
    "foxglove_studio"
)

declare -A WEB_NODES_FUNC=(
    ["dashboard_frontend"]="${ROOT_DIR}/src/applications/dashboard/scripts/build_frontend.sh"
    ["foxglove_studio"]="${ROOT_DIR}/src/applications/dashboard/scripts/build_foxglove.sh"
)

usage() {
    echo "Usage: $0 <target>"
    echo
    echo "Targets:"
    echo "  app        - Build main application ROS 2 nodes."
    echo "  web        - Build web components (Dashboard Frontend, Foxglove)."
    echo "  all        - Build both 'app' and 'web' components."
    echo "  sim        - Build simulation ROS 2 nodes."
    echo "  example    - Build example ROS 2 nodes."
    echo "  msg        - Build protocol ROS 2 nodes."
    echo
}

build_ros2_nodes() {
    declare -n nodes_ref=$1
    
    if [ ${#nodes_ref[@]} -eq 0 ]; then
        echo "No ROS 2 nodes to build for this target. Skipping."
        return 0
    fi

    echo "Starting ROS 2 Nodes Build"

    local packages_arg="--packages-select ${nodes_ref[*]}"
    echo "Running build with the following nodes: ${nodes_ref[*]}"

    cd "$ROOT_DIR" && \
    colcon build \
        --cmake-args -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        --build-base build \
        --install-base install \
        --allow-overriding interface_protocol \
        $packages_arg
    
    return $?
}

build_web_nodes() {
    echo "Starting Web Components Build"
    local all_successful=true

    for node in "${WEB_NODES[@]}"; do
        script="${WEB_NODES_FUNC[$node]}"
        if [[ -x "$script" ]]; then
            if ! "$script"; then
                echo "Failed to build $node."
                all_successful=false
            fi
        else
            echo "Warning: Script for $node not found."
        fi
    done

    $all_successful
}

# Default target host is example
TARGET_HOST="example"

# Default build type is Release
BUILD_TYPE="Release"

# Parse command line arguments - only accept host name
if [[ -n "$1" ]]; then
    TARGET_HOST="$1"
fi

if [[ -n "$2" ]]; then
    BUILD_TYPE="$2"
fi

echo "Building for $TARGET_HOST nodes"
echo "Build type: $BUILD_TYPE"

mkdir -p "$ROOT_DIR/build"

build_status=0
case "$TARGET_HOST" in
    "msg")
        build_ros2_nodes "PROTOCOL_NODES"
        build_status=$?
        ;;
    "example")
        build_ros2_nodes "EXAMPLE_NODES"
        build_status=$?
        ;;
    "app")
        build_ros2_nodes "APPLICATION_NODES"
        build_status=$?
        ;;
    "sim")
        build_ros2_nodes "SIM_NODES"
        build_status=$?
        ;;
    "web")
        build_web_nodes
        build_status=$?
        ;;
    "all")
        build_web_nodes
        build_ros2_nodes "APPLICATION_NODES"
        build_status=$?
        ;;
    *)
        echo "Error: Invalid target host '$TARGET_HOST'"
        echo "Available hosts: example, app, sim, web, all"
        exit 1
        ;;
esac

if [ $build_status -eq 0 ]; then
    echo "Build successful!"
    echo "To use the built packages, run:"
    echo "source /opt/ros/humble/setup.bash"
    echo "source $ROOT_DIR/install/setup.bash"
else
    echo "Build failed!"
    exit 1
fi
