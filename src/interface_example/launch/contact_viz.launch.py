from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    package_dir = get_package_share_directory('interface_example')
    mujoco_dir = get_package_share_directory('mujoco_simulator')
    # URDF文件路径
    urdf_path = os.path.join(
        mujoco_dir,
        'assets/resource/robot/pm_v2/urdf/serial_pm_v2.urdf'
    )
    # RViz配置文件路径
    rviz_config_file = os.path.join(package_dir, 'launch', 'contact_points.rviz')

    # 读取URDF内容
    with open(urdf_path, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    return LaunchDescription([
        # 启动robot_state_publisher，订阅mujoco的joint_states话题
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_content}],
            remappings=[
                ('joint_states', '/hardware/joint_state')
            ]
        ),
        # 启动接触点可视化节点
        Node(
            package='interface_example',
            executable='contact_viz_and_log.py',
            name='contact_visualizer',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        # 启动RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen',
            parameters=[{'use_sim_time': True}]
        )
    ]) 