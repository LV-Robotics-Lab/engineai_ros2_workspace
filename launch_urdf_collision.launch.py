#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 获取 launch 文件所在目录
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 声明 launch 参数
    declare_urdf_file_arg = DeclareLaunchArgument(
        'urdf_file',
        default_value='/home/wang22/engineai/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/pm01/urdf/serial_pm_v2_primitive.urdf',
        description='URDF文件的绝对路径'
    )
    
    # 使用 OpaqueFunction 来在运行时获取参数值
    return LaunchDescription([
        declare_urdf_file_arg,
        OpaqueFunction(function=lambda context: launch_setup(context, launch_file_dir))
    ])

def launch_setup(context, launch_file_dir, *args, **kwargs):
    # 获取 launch 参数值
    urdf_file = LaunchConfiguration('urdf_file').perform(context)
    
    # 检查文件是否存在
    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"URDF文件不存在: {urdf_file}")
    
    print(f"使用URDF文件: {urdf_file}")
    
    # 读取URDF文件内容
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # 将URDF中的相对路径转换为绝对路径
    # 支持各种相对路径格式（../meshes/, ./meshes/, meshes/ 等）
    urdf_dir = os.path.dirname(os.path.abspath(urdf_file))
    
    import re
    
    def convert_to_absolute_path(match):
        """将相对路径转换为绝对路径"""
        relative_path = match.group(1)
        
        # 如果已经是绝对路径或包含 file://，直接返回
        if os.path.isabs(relative_path) or relative_path.startswith('file://'):
            return match.group(0)
        
        # 处理 package:// 协议
        if relative_path.startswith('package://'):
            # 移除 package:// 前缀
            package_path = relative_path.replace('package://', '')
            # 从 URDF 文件路径推断 package 根目录
            # 假设 package 根目录在 assets/ 目录下
            # URDF 路径: .../assets/resource/robot/pm_v2/urdf/xxx.urdf
            # package://resource/... 应该指向 .../assets/resource/...
            assets_dir = os.path.dirname(urdf_dir)  # 获取 pm_v2 目录
            assets_dir = os.path.dirname(assets_dir)  # 获取 robot 目录
            assets_dir = os.path.dirname(assets_dir)  # 获取 resource 目录
            assets_dir = os.path.dirname(assets_dir)  # 获取 assets 目录
            abs_path = os.path.abspath(os.path.join(assets_dir, package_path))
            return f'filename="file://{abs_path}"'
        
        # 将相对路径转换为绝对路径
        # 相对于 URDF 文件所在目录
        abs_path = os.path.abspath(os.path.join(urdf_dir, relative_path))
        return f'filename="file://{abs_path}"'
    
    # 匹配所有 filename="..." 中的相对路径
    # 匹配模式：filename="相对路径" 或 filename="绝对路径"
    pattern = r'filename="([^"]+)"'
    
    # 统计转换前的相对路径和 package:// 路径数量
    all_paths = re.findall(pattern, robot_description)
    relative_count = sum(1 for p in all_paths if not os.path.isabs(p) and not p.startswith('file://'))
    
    # 替换所有相对路径为绝对路径
    robot_description = re.sub(pattern, convert_to_absolute_path, robot_description)
    
    # 验证转换结果
    remaining_relative = len([p for p in re.findall(pattern, robot_description) 
                             if not os.path.isabs(p.replace('file://', '')) and not p.startswith('file://') and not p.startswith('package://')])
    
    if remaining_relative > 0:
        print(f"警告: 仍有 {remaining_relative} 个相对路径未转换")
    else:
        print(f"成功: 已转换 {relative_count} 个相对路径为绝对路径")
    
    # 从 URDF 中提取根 link 名称（第一个 link）
    root_link_match = re.search(r'<link name="([^"]+)"', robot_description)
    if root_link_match:
        root_link_name = root_link_match.group(1)
        print(f"检测到根 link: {root_link_name}")
    else:
        # 如果无法找到，使用默认值
        root_link_name = 'link_base'
        print(f"警告: 无法从 URDF 中提取根 link 名称，使用默认值: {root_link_name}")
    
    # robot_state_publisher节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
            'publish_frequency': 50.0
        }]
    )
    
    # 静态变换发布器 - 从world到根link（动态获取）
    static_transform_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', root_link_name],
        output='screen'
    )
    
    # 零位关节状态发布节点 - 发布所有关节的零位状态，使robot_state_publisher能够发布TF
    script_path = os.path.join(launch_file_dir, 'scripts', 'publish_zero_joint_states.py')
    zero_joint_state_node = ExecuteProcess(
        cmd=['python3', script_path],
        output='screen'
    )
    
    # RViz节点 - 显示collision model
    rviz_config_file = os.path.join(launch_file_dir, 'pm01_collision.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )
    
    return [
        # 确保 robot_state_publisher 先启动，这样 rviz2 才能正确读取 robot_description
        robot_state_publisher_node,
        static_transform_publisher_node,
        zero_joint_state_node,
        # 延迟启动 rviz2，确保 robot_state_publisher 已经设置好参数并发布到参数服务器
        TimerAction(
            period=2.0,
            actions=[rviz_node]
        )
    ]

