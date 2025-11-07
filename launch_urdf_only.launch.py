#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    package_dir = get_package_share_directory('mujoco_simulator')
    
    # URDF文件路径
    urdf_file = os.path.join(package_dir, 'assets', 'resource', 'robot', 'pm_v2', 'urdf', 'serial_pm_v2.urdf')
    
    # 读取URDF文件内容
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # robot_state_publisher节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
            'publish_frequency': 50.0  # 降低发布频率到50Hz
        }]
    )
    
    # 静态变换发布器
    static_transform_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'LINK_BASE'],
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        static_transform_publisher_node
    ])
