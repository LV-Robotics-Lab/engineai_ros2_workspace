from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package path using get_package_share_directory
    package_dir = get_package_share_directory('interface_example')

    # Get product from environment variable or use default 'pm01'
    product = os.environ.get('PRODUCT', 'pm01')
    
    # Build config file path using os.path.join - use XZL config
    config_dir = os.path.join(package_dir, 'config',
                              product, 'rl_basic', 'basic')
    print(config_dir)
    # Ensure config directory exists
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Check if XZL config file exists
    xzl_config_file = os.path.join(config_dir, 'rl_basic_param_XZL.yaml')
    if not os.path.exists(xzl_config_file):
        raise FileNotFoundError(f"XZL config file not found: {xzl_config_file}")
    print(f"Using XZL config file: {xzl_config_file}")

    # Append dynamic library path
    ENGINEAI_ROBOTICS_THIRD_PARTY = "/opt/engineai_robotics_third_party"
    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        ENGINEAI_ROBOTICS_THIRD_PARTY, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    # Create node - using rl_basic_example but with runner logic
    hardware_node = Node(
        package='interface_example',
        executable='rl_basic_example_XZL',
        name='rl_basic_example_XZL',
        arguments=[config_dir, 'rl_basic_param_XZL.yaml'],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        hardware_node
    ])
