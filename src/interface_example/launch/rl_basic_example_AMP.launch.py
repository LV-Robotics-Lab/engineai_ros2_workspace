from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('interface_example')

    product = os.environ.get('PRODUCT', 'pm01')

    config_dir = os.path.join(package_dir, 'config', product, 'rl_basic', 'basic')
    print(config_dir)
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    amp_config_file = os.path.join(config_dir, 'rl_basic_param_AMP.yaml')
    if not os.path.exists(amp_config_file):
        raise FileNotFoundError(f"AMP config file not found: {amp_config_file}")
    print(f"Using AMP config file: {amp_config_file}")

    ENGINEAI_ROBOTICS_THIRD_PARTY = "/opt/engineai_robotics_third_party"
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    system_lib = '/usr/lib/x86_64-linux-gnu'
    ld_parts = [os.path.join(ENGINEAI_ROBOTICS_THIRD_PARTY, 'lib')]
    if conda_prefix:
        ld_parts.insert(0, os.path.join(conda_prefix, 'lib'))
    ld_parts.insert(0, system_lib)
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_parts) + (':' + existing if existing else '')

    hardware_node = Node(
        package='interface_example',
        executable='rl_basic_example_AMP',
        name='rl_basic_example_AMP',
        arguments=[config_dir, 'rl_basic_param_AMP.yaml'],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        hardware_node
    ])
