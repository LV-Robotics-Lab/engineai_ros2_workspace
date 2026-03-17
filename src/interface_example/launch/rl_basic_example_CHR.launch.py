from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package path using get_package_share_directory
    package_dir = get_package_share_directory('interface_example')

    # Get product from environment variable or use default 'pm01'
    product = os.environ.get('PRODUCT', 'pm01')
    
    # Build config file path using os.path.join - use CHR config
    config_dir = os.path.join(package_dir, 'config',
                              product, 'rl_basic', 'basic')
    print(config_dir)
    # Ensure config directory exists
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Check if CHR config file exists
    chr_config_file = os.path.join(config_dir, 'rl_basic_param_CHR.yaml')
    if not os.path.exists(chr_config_file):
        raise FileNotFoundError(f"CHR config file not found: {chr_config_file}")
    print(f"Using CHR config file: {chr_config_file}")

    # rl_basic_example_CHR 依赖 libglog.so.0，需在 LD_LIBRARY_PATH 中能找到（系统或 conda）
    ENGINEAI_ROBOTICS_THIRD_PARTY = "/opt/engineai_robotics_third_party"
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    system_lib = '/usr/lib/x86_64-linux-gnu'
    ld_parts = [os.path.join(ENGINEAI_ROBOTICS_THIRD_PARTY, 'lib')]
    if conda_prefix:
        ld_parts.insert(0, os.path.join(conda_prefix, 'lib'))
    ld_parts.insert(0, system_lib)
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_parts) + (':' + existing if existing else '')

    # Create node - using rl_basic_example_CHR
    hardware_node = Node(
        package='interface_example',
        executable='rl_basic_example_CHR',
        name='rl_basic_example_CHR',
        arguments=[config_dir, 'rl_basic_param_CHR.yaml'],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        hardware_node
    ])
