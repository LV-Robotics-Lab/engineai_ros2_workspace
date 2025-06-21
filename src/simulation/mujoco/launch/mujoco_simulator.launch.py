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
# 导入os模块，进行操作系统相关操作（本文件未实际用到）
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
        default_value='true',
        description='是否保存接触力数据到CSV文件 (true/false)'
    )
    
    declare_csv_path_arg = DeclareLaunchArgument(
        'csv_file_path',
        default_value='',
        description='CSV文件保存路径（留空则使用默认路径）'
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
        csv_file_path = LaunchConfiguration('csv_file_path').perform(context)
        
        # 将字符串转换为布尔值
        export_contact = export_contact_str.lower() == 'true'
        save_csv = save_csv_str.lower() == 'true'

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

        # 定义MuJoCo仿真器节点
        mujoco_node = Node(
            package='mujoco_simulator',           # 节点所属包
            executable='mujoco_simulator',        # 可执行文件名
            name='mujoco_simulator',              # 节点名称
            output='screen',                      # 输出到终端
            emulate_tty=True,                     # 终端仿真，便于日志显示
            arguments=args,                       # 启动参数
            parameters=[
                {'use_sim_time': True},           # 使用仿真时间
                {'export_contact': export_contact},  # 是否导出接触力
                {'contact_topic': contact_topic},    # 接触力话题名称
                {'save_contact_csv': save_csv},      # 是否保存CSV
                {'csv_file_path': csv_file_path},    # CSV文件路径
            ]
        )

        # 返回节点对象列表
        return [mujoco_node]

    # 使用OpaqueFunction以便在launch时动态获取参数
    mujoco_launch = OpaqueFunction(function=launch_setup)

    # 返回LaunchDescription对象，包含所有环境变量和节点启动操作
    return LaunchDescription([
        declare_export_contact_arg,
        declare_contact_topic_arg,
        declare_save_csv_arg,
        declare_csv_path_arg,
        *env_vars,
        mujoco_launch
    ])
