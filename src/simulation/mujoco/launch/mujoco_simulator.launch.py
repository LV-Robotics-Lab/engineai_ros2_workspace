# 导入LaunchDescription，用于描述整个launch文件的内容
from launch import LaunchDescription
# 导入声明参数和设置环境变量的动作类
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
# 导入参数替换、路径拼接和环境变量获取的工具
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
# 导入Node类，用于定义ROS 2节点的启动方式
from launch_ros.actions import Node
# 导入OpaqueFunction，用于在launch时动态获取参数
from launch.actions import OpaqueFunction
# 导入获取ROS 2包路径的工具函数
from ament_index_python.packages import get_package_share_directory
# 导入os模块，进行操作系统相关操作
import os

# 生成launch描述的主函数
def generate_launch_description():

    # 获取mujoco_simulator包的安装路径
    package_dir = get_package_share_directory('mujoco_simulator')

    # 声明launch参数
    declare_export_contact_arg = DeclareLaunchArgument(
        'export_contact',
        default_value='true',
        description='是否导出接触力信息 (true/false)'
    )
    
    declare_contact_topic_arg = DeclareLaunchArgument(
        'contact_topic',
        default_value='/mujoco/contact_forces',
        description='接触力话题名称'
    )

    declare_save_csv_arg = DeclareLaunchArgument(
        'save_contact_csv',
        default_value='false',
        description='是否保存接触力数据到CSV文件 (true/false)'
    )
    
    declare_save_perturbation_csv_arg = DeclareLaunchArgument(
        'save_perturbation_csv',
        default_value='false',
        description='是否保存推力数据到CSV文件 (true/false)'
    )
    
    declare_save_joint_forces_csv_arg = DeclareLaunchArgument(
        'save_joint_forces_csv',
        default_value='false',
        description='是否保存关节反力数据到CSV文件 (true/false)'
    )
    
    declare_save_sensor_vibration_csv_arg = DeclareLaunchArgument(
        'save_sensor_vibration_csv',
        default_value='false',
        description='是否保存传感器震动数据到CSV文件（持续记录base_link和head的加速度） (true/false)'
    )
    
    declare_save_joint_state_csv_arg = DeclareLaunchArgument(
        'save_joint_state_csv',
        default_value='false',
        description='是否保存关节状态数据到CSV文件（持续记录关节位置、速度、力矩） (true/false)'
    )
    
    declare_save_link_kinetic_energy_csv_arg = DeclareLaunchArgument(
        'save_link_kinetic_energy_csv',
        default_value='false',
        description='是否保存Link动能数据到CSV文件（持续记录每个link的速度和动能） (true/false)'
    )
    
    declare_save_policy_switch_csv_arg = DeclareLaunchArgument(
        'save_policy_switch_csv',
        default_value='false',
        description='是否保存RL policy切换事件到CSV文件（walking↔mimic↔damping） (true/false)'
    )
    
    declare_csv_format_arg = DeclareLaunchArgument(
        'csv_format',
        default_value='csv',
        description='日志格式：csv 或 binary（.bin，含 contact/joint/perturbation/policy_switch 等）'
    )

    declare_csv_min_force_n_arg = DeclareLaunchArgument(
        'csv_min_force_n',
        default_value='0.0',
        description='contact：接触力模长≥此值才写入（N）。joint：与 csv_joint_forces_min_torque_nm 为 OR；0=该侧不作用'
    )
    declare_csv_joint_forces_min_torque_nm_arg = DeclareLaunchArgument(
        'csv_joint_forces_min_torque_nm',
        default_value='0.0',
        description='joint_forces：子link反力 ||M||≥此值(N·m)或与 csv_min_force_n 满足力条件则写入；0=不按力矩过滤'
    )
    
    declare_csv_path_arg = DeclareLaunchArgument(
        'csv_file_path',
        default_value='',
        description='CSV文件保存路径（留空则使用默认路径）'
    )

    declare_save_console_log_arg = DeclareLaunchArgument(
        'save_console_log',
        default_value='false',
        description='是否将终端输出同时保存到 ROS 2 launch 日志文件 (true/false)'
    )

    declare_console_log_dir_arg = DeclareLaunchArgument(
        'console_log_dir',
        default_value='',
        description='终端输出日志目录；仅在 save_console_log=true 时生效，留空则使用默认 ~/.ros/log'
    )

    declare_headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='是否以隐藏窗口模式运行 MuJoCo（true/false）'
    )

    # 添加URDF文件路径参数
    declare_urdf_file_arg = DeclareLaunchArgument(
        'urdf_file',
        default_value=PathJoinSubstitution([package_dir, 'assets', 'resource', 'robot', 'pm_v2', 'urdf', 'serial_pm_v2.urdf']),
        description='URDF文件路径'
    )

    # 推倒采样相关参数
    declare_perturb_force_arg = DeclareLaunchArgument(
        'perturb_force_magnitude',
        default_value='20.0',
        description='推倒采样干扰力大小 (N)'
    )

    declare_perturb_torque_arg = DeclareLaunchArgument(
        'perturb_torque_magnitude',
        default_value='5.0',
        description='推倒采样干扰力矩大小 (N.m)'
    )

    declare_perturb_duration_arg = DeclareLaunchArgument(
        'perturb_duration',
        default_value='0.2',
        description='推倒采样干扰力持续时间（秒）'
    )

    declare_perturb_body_arg = DeclareLaunchArgument(
        'perturb_body_name',
        default_value='LINK_TORSO_YAW',
        description='施加干扰力的物体名称'
    )

    # 创建环境变量列表，供仿真器使用
    env_vars = [
        # 设置产品型号为pm_v2
        SetEnvironmentVariable('PRODUCT', 'pm_v2'),
        # 设置MuJoCo资源文件路径
        SetEnvironmentVariable('MUJOCO_ASSETS_PATH',
                               PathJoinSubstitution([package_dir, 'assets'])),
        # 设置动态库搜索路径，包含第三方库和ROS库
        SetEnvironmentVariable('LD_LIBRARY_PATH', [
                               '/opt/engineai_robotics_third_party/lib:/opt/ros/humble/lib:', EnvironmentVariable(name='LD_LIBRARY_PATH', default_value='')])
    ]

    # 根据headless参数创建节点配置（此处暂未用到headless参数，保留扩展性）
    def launch_setup(context, *args, **kwargs):

        # 获取launch参数值
        export_contact_str = LaunchConfiguration('export_contact').perform(context)
        contact_topic = LaunchConfiguration('contact_topic').perform(context)
        save_csv_str = LaunchConfiguration('save_contact_csv').perform(context)
        save_perturbation_csv_str = LaunchConfiguration('save_perturbation_csv').perform(context)
        save_joint_forces_csv_str = LaunchConfiguration('save_joint_forces_csv').perform(context)
        save_sensor_vibration_csv_str = LaunchConfiguration('save_sensor_vibration_csv').perform(context)
        save_joint_state_csv_str = LaunchConfiguration('save_joint_state_csv').perform(context)
        save_link_kinetic_energy_csv_str = LaunchConfiguration('save_link_kinetic_energy_csv').perform(context)
        save_policy_switch_csv_str = LaunchConfiguration('save_policy_switch_csv').perform(context)
        csv_format = LaunchConfiguration('csv_format').perform(context)
        csv_file_path = LaunchConfiguration('csv_file_path').perform(context)
        save_console_log_str = LaunchConfiguration('save_console_log').perform(context)
        console_log_dir = LaunchConfiguration('console_log_dir').perform(context)
        headless_str = LaunchConfiguration('headless').perform(context)
        csv_min_force_n_str = LaunchConfiguration('csv_min_force_n').perform(context)
        csv_joint_torque_str = LaunchConfiguration('csv_joint_forces_min_torque_nm').perform(context)
        urdf_file = LaunchConfiguration('urdf_file').perform(context)
        perturb_force = LaunchConfiguration('perturb_force_magnitude').perform(context)
        perturb_torque = LaunchConfiguration('perturb_torque_magnitude').perform(context)
        perturb_duration = LaunchConfiguration('perturb_duration').perform(context)
        perturb_body = LaunchConfiguration('perturb_body_name').perform(context)
        
        # 将字符串转换为布尔值
        export_contact = export_contact_str.lower() == 'true'
        save_csv = save_csv_str.lower() == 'true'
        save_perturbation_csv = save_perturbation_csv_str.lower() == 'true'
        save_joint_forces_csv = save_joint_forces_csv_str.lower() == 'true'
        save_sensor_vibration_csv = save_sensor_vibration_csv_str.lower() == 'true'
        save_joint_state_csv = save_joint_state_csv_str.lower() == 'true'
        save_link_kinetic_energy_csv = save_link_kinetic_energy_csv_str.lower() == 'true'
        save_policy_switch_csv = save_policy_switch_csv_str.lower() == 'true'
        save_console_log = save_console_log_str.lower() == 'true'
        headless = headless_str.lower() == 'true'

        node_output_mode = 'both' if save_console_log else 'screen'

        extra_actions = []
        if save_console_log and console_log_dir:
            extra_actions.append(SetEnvironmentVariable('ROS_LOG_DIR', console_log_dir))

        # 读取URDF文件内容
        try:
            with open(urdf_file, 'r') as f:
                robot_description = f.read()
        except FileNotFoundError:
            print(f"URDF文件未找到: {urdf_file}")
            robot_description = ""

        try:
            csv_min_force_n = float(csv_min_force_n_str)
        except ValueError:
            print(f"无效的 csv_min_force_n: {csv_min_force_n_str!r}，使用 0.0")
            csv_min_force_n = 0.0
        try:
            csv_joint_forces_min_torque_nm = float(csv_joint_torque_str)
        except ValueError:
            print(f"无效的 csv_joint_forces_min_torque_nm: {csv_joint_torque_str!r}，使用 0.0")
            csv_joint_forces_min_torque_nm = 0.0

        # 节点启动参数列表
        args = []
        
        # 如果启用接触力导出，添加相关参数
        if export_contact:
            args.extend([
                '--export_contact',
                '--contact_topic', contact_topic
            ])
        
        # 如果启用CSV保存，添加相关参数
        if save_csv:
            args.extend([
                '--save_contact_csv'
            ])
            if csv_file_path:
                args.extend(['--csv_file_path', csv_file_path])

        if headless:
            args.append('--headless')

        # 定义MuJoCo仿真器节点
        mujoco_node = Node(
            package='mujoco_simulator',           # 节点所属包
            executable='mujoco_simulator',        # 可执行文件名
            name='mujoco_simulator',              # 节点名称
            output=node_output_mode,              # 输出到终端；可选同时写 ROS 2 launch 日志
            emulate_tty=True,                     # 终端仿真，便于日志显示
            arguments=args,                       # 启动参数
            parameters=[
                {'use_sim_time': True},           # 使用仿真时间
                {'export_contact': export_contact},  # 是否导出接触力
                {'contact_topic': contact_topic},    # 接触力话题名称
                {'save_contact_csv': save_csv},      # 是否保存CSV
                {'save_perturbation_csv': save_perturbation_csv},  # 是否保存推力CSV
                {'save_joint_forces_csv': save_joint_forces_csv},  # 是否保存关节反力CSV
                {'save_sensor_vibration_csv': save_sensor_vibration_csv},  # 是否保存传感器震动CSV
                {'save_joint_state_csv': save_joint_state_csv},  # 是否保存关节状态CSV
                {'save_link_kinetic_energy_csv': save_link_kinetic_energy_csv},  # 是否保存Link动能CSV
                {'save_policy_switch_csv': save_policy_switch_csv},  # 是否保存RL policy切换CSV
                {'csv_format': csv_format},  # CSV格式：csv 或 binary
                {'csv_file_path': csv_file_path},    # CSV文件路径
                {'csv_min_force_n': csv_min_force_n},
                {'csv_joint_forces_min_torque_nm': csv_joint_forces_min_torque_nm},
                {'perturb_force_magnitude': float(perturb_force)},      # 干扰力大小
                {'perturb_torque_magnitude': float(perturb_torque)},    # 干扰力矩大小
                {'perturb_duration': float(perturb_duration)},          # 干扰力持续时间
                {'perturb_body_name': perturb_body},                    # 干扰力目标物体
            ]
        )

        # 定义joint_state转换节点
        joint_state_converter_node = Node(
            package='interface_example',
            executable='joint_state_converter.py',
            name='joint_state_converter',
            output=node_output_mode,
            parameters=[{'use_sim_time': True}]
        )

        # 定义robot_state_publisher节点
        robot_state_publisher_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output=node_output_mode,
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True
            }]
        )

        # 定义静态变换发布器，发布从world到LINK_BASE的变换
        static_transform_publisher_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'LINK_BASE'],
            output=node_output_mode
        )

        # 返回节点对象列表
        return [*extra_actions, mujoco_node, joint_state_converter_node, robot_state_publisher_node, static_transform_publisher_node]

    # 使用OpaqueFunction以便在launch时动态获取参数
    mujoco_launch = OpaqueFunction(function=launch_setup)

    # 返回LaunchDescription对象，包含所有环境变量和节点启动操作
    return LaunchDescription([
        declare_export_contact_arg,
        declare_contact_topic_arg,
        declare_save_csv_arg,
        declare_save_perturbation_csv_arg,
        declare_save_joint_forces_csv_arg,
        declare_save_sensor_vibration_csv_arg,
        declare_save_joint_state_csv_arg,
        declare_save_link_kinetic_energy_csv_arg,
        declare_save_policy_switch_csv_arg,
        declare_csv_format_arg,
        declare_csv_min_force_n_arg,
        declare_csv_joint_forces_min_torque_nm_arg,
        declare_csv_path_arg,
        declare_save_console_log_arg,
        declare_console_log_dir_arg,
        declare_headless_arg,
        declare_urdf_file_arg,
        declare_perturb_force_arg,
        declare_perturb_torque_arg,
        declare_perturb_duration_arg,
        declare_perturb_body_arg,
        *env_vars,
        mujoco_launch
    ])
