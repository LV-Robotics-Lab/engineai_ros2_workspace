import urdfpy
import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

def parse_xml_joint_positions(xml_path):
    """解析XML文件中的关节初始位置"""
    print(f"正在解析XML文件: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 查找keyframe中的qpos
    keyframe = root.find('.//keyframe/key')
    if keyframe is not None:
        qpos_str = keyframe.get('qpos', '')
        if qpos_str:
            # 解析关节位置数组
            all_positions = [float(x) for x in qpos_str.split()]
            print(f"找到总位置数据: {len(all_positions)} 个值")
            
            # 前7个值是自由关节（floating base），跳过它们
            if len(all_positions) >= 31:
                joint_positions = all_positions[7:]  # 取第8个值开始
                print(f"提取关节角度: {len(joint_positions)} 个值")
                print(f"原始关节角度数据: {joint_positions}")
                
                # 按照XML中的定义，关节顺序为（总共24个关节）：
                # 左腿关节: 0-5 (J00-J05)
                # 右腿关节: 6-11 (J06-J11) 
                # 腰部关节: 12 (J12)
                # 左臂关节: 13-17 (J13-J17)
                # 右臂关节: 19-23 (J18-J22)
                # 头部关节: 24 (J23)
                
                print("关节角度分布:")
                print(f"  左腿 (J00-J05): {joint_positions[0:6]}")
                print(f"  右腿 (J06-J11): {joint_positions[6:12]}")
                print(f"  腰部 (J12): {joint_positions[12]}")
                print(f"  左臂 (J13-J17): {joint_positions[13:18]}")
                print(f"  右臂 (J18-J22): {joint_positions[18:23]}")
                print(f"  头部 (J23): {joint_positions[23] if len(joint_positions) > 23 else 'N/A'}")
                
                # 调试：打印完整的关节角度数组
                print(f"完整关节角度数组: {joint_positions}")
                print(f"数组长度: {len(joint_positions)}")
                
                return joint_positions
            else:
                print(f"数据长度不足，期望31个值，实际{len(all_positions)}个")
                return None
    
    print("未找到关节初始位置，使用默认值")
    return None

def parse_xml_joint_axes(xml_path):
    """从XML文件中解析关节轴信息"""
    print(f"正在解析XML文件中的关节轴信息: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    joint_axes = {}
    
    # 查找所有joint元素
    joints = root.findall('.//joint')
    print(f"找到 {len(joints)} 个关节定义")
    
    for joint in joints:
        joint_name = joint.get('name')
        axis_str = joint.get('axis')
        
        if joint_name and axis_str:
            # 解析轴向量
            try:
                axis_values = [float(x) for x in axis_str.split()]
                if len(axis_values) == 3:
                    joint_axes[joint_name] = np.array(axis_values)
                    print(f"  {joint_name}: {axis_values}")
                else:
                    print(f"  警告: {joint_name} 的轴向量格式不正确: {axis_str}")
            except ValueError as e:
                print(f"  错误: 无法解析 {joint_name} 的轴向量: {axis_str}, 错误: {e}")
    
    print(f"成功解析 {len(joint_axes)} 个关节轴")
    return joint_axes

def urdf_meshes_to_single(urdf_path, xml_path, output_path):
    print(f"开始处理URDF文件: {urdf_path}")
    
    # 1. 解析URDF文件
    robot = urdfpy.URDF.load(urdf_path)
    print(f"成功加载URDF，包含 {len(robot.links)} 个link和 {len(robot.joints)} 个joint")
    
    # 2. 解析XML中的关节初始位置
    joint_positions = parse_xml_joint_positions(xml_path)
    
    # 3. 从XML文件中读取关节轴信息
    # 关节轴信息在serial_links.xml文件中
    links_xml_path = xml_path.replace('serial_pm_v2.xml', 'serial_links.xml')
    joint_axis_mapping = parse_xml_joint_axes(links_xml_path)
    
    if not joint_axis_mapping:
        print("警告: 无法从XML读取关节轴信息，使用默认轴")
        # 如果无法从XML读取，使用默认的轴定义作为备用
        joint_axis_mapping = {
            'J00_HIP_PITCH_L': np.array([0, 0.965926, -0.258819]),
            'J01_HIP_ROLL_L': np.array([1, 0, 0]),
            'J02_HIP_YAW_L': np.array([0, 0, 1]),
            'J03_KNEE_PITCH_L': np.array([0, 1, 0]),
            'J04_ANKLE_PITCH_L': np.array([0, 1, 0]),
            'J05_ANKLE_ROLL_L': np.array([1, 0, 0]),
            
            'J06_HIP_PITCH_R': np.array([0, 0.965926, 0.258819]),
            'J07_HIP_ROLL_R': np.array([1, 0, 0]),
            'J08_HIP_YAW_R': np.array([0, 0, 1]),
            'J09_KNEE_PITCH_R': np.array([0, 1, 0]),
            'J10_ANKLE_PITCH_R': np.array([0, 1, 0]),
            'J11_ANKLE_ROLL_R': np.array([1, 0, 0]),
            
            'J12_WAIST_YAW': np.array([0, 0, 1]),
            
            'J13_SHOULDER_PITCH_L': np.array([0, 0.998027, 0.0627908]),
            'J14_SHOULDER_ROLL_L': np.array([1, 0, 0]),
            'J15_SHOULDER_YAW_L': np.array([0, -0.0628027, 0.998026]),
            'J16_ELBOW_PITCH_L': np.array([0.00272431, 0.998022, 0.0628031]),
            'J17_ELBOW_YAW_L': np.array([-0.214789, -0.0619207, 0.974696]),
            
            'J18_SHOULDER_PITCH_R': np.array([0, 0.998027, -0.0627908]),
            'J19_SHOULDER_ROLL_R': np.array([1, 0, 0]),
            'J20_SHOULDER_YAW_R': np.array([0, 0.0627908, 0.998027]),
            'J21_ELBOW_PITCH_R': np.array([0.00272431, 0.998023, -0.0627902]),
            'J22_ELBOW_YAW_R': np.array([-0.214789, 0.0619088, 0.974696]),
            
            'J23_HEAD_YAW': np.array([0, 0, 1]),
        }
    
    # 4. 计算每个link相对于根坐标系的绝对变换矩阵
    print("正在计算link变换矩阵...")
    link_transforms = {}
    
    # 根link（默认第一个link为根，无父关节）
    root_link = robot.links[0].name
    link_transforms[root_link] = np.eye(4)  # 单位矩阵（原点）
    print(f"根link: {root_link}")
    
    # 遍历关节，计算子link的绝对变换
    joint_idx = 0
    for joint in robot.joints:
        print(f"处理关节 {joint_idx+1}/{len(robot.joints)}: {joint.name}")
        
        parent_link = joint.parent
        child_link = joint.child
        
        # 获取关节的相对变换
        if hasattr(joint.origin, 'xyz'):
            xyz = joint.origin.xyz
        else:
            # 处理不同格式的origin数据
            origin_data = joint.origin
            if isinstance(origin_data, np.ndarray):
                if origin_data.shape == (3, 4):
                    # 如果是3x4矩阵，取最后一列作为xyz
                    xyz = origin_data[:3, 3]
                elif origin_data.shape[0] >= 3:
                    xyz = origin_data[:3]  # 取前3个值
                else:
                    xyz = np.array([0, 0, 0])  # 默认值
            else:
                xyz = np.array([0, 0, 0])  # 默认值
            
        if hasattr(joint.origin, 'rpy'):
            rpy = joint.origin.rpy
        else:
            # 处理不同格式的origin数据
            origin_data = joint.origin
            if isinstance(origin_data, np.ndarray):
                if origin_data.shape == (3, 4):
                    # 如果是3x4矩阵，从矩阵中提取rpy
                    # 通常rpy信息在矩阵的旋转部分，这里我们设为0
                    rpy = np.array([0, 0, 0])
                elif origin_data.shape[0] >= 6:
                    rpy = origin_data[3:6]  # 取第4-6个值作为rpy
                else:
                    rpy = np.array([0, 0, 0])  # 默认值
            else:
                rpy = np.array([0, 0, 0])  # 默认值
        
        # 确保rpy是3个值的数组
        if isinstance(rpy, np.ndarray):
            if rpy.shape[0] > 3:
                rpy = rpy[:3]  # 只取前3个值
            elif len(rpy.shape) > 1:
                # 如果是多维数组，取第一个元素
                rpy = rpy.flatten()[:3]
        
        print(f"  origin xyz: {xyz}, rpy: {rpy}")
        
        # 确保xyz是3个值的数组
        if isinstance(xyz, np.ndarray) and len(xyz.shape) > 1:
            if xyz.shape == (3, 4):
                # 如果是3x4矩阵，取最后一列作为xyz
                xyz = xyz[:3, 3]
            else:
                # 其他情况，取第一个元素
                xyz = xyz.flatten()[:3]
        
        # 旋转矩阵（rpy转xyz顺序的旋转矩阵）
        rot = R.from_euler('xyz', rpy).as_matrix()
        
        # 构建4x4齐次变换矩阵
        joint_tf = np.eye(4)
        joint_tf[:3, :3] = rot
        joint_tf[:3, 3] = xyz
        
        # 如果有关节初始位置，应用关节角度
        if joint_positions and joint_idx < len(joint_positions):
            joint_angle = joint_positions[joint_idx]
            print(f"  关节角度: {joint_angle:.3f} rad")
            
            # 使用预定义的关节轴
            if joint.name in joint_axis_mapping:
                axis = joint_axis_mapping[joint.name]
                print(f"  使用预定义关节轴: {axis}")
            else:
                # 尝试从URDF获取关节轴
                if hasattr(joint, 'axis'):
                    axis = joint.axis
                elif hasattr(joint, 'axis_vector'):
                    axis = joint.axis_vector  # 新版本API
                else:
                    # 尝试其他可能的属性名
                    axis = getattr(joint, 'axis', np.array([0, 0, 1]))  # 默认Z轴
                    
                print(f"  关节轴: {axis}")
            
            # 创建关节旋转矩阵
            joint_rot = R.from_rotvec(axis * joint_angle).as_matrix()
            joint_rot_tf = np.eye(4)
            joint_rot_tf[:3, :3] = joint_rot
            
            # 组合变换：位置变换 × 关节旋转
            joint_tf = joint_tf @ joint_rot_tf
        
        # 子link绝对变换 = 父link绝对变换 × 关节变换
        link_transforms[child_link] = link_transforms[parent_link] @ joint_tf
        joint_idx += 1
    
    # 4. 加载并变换所有mesh
    print("正在加载和变换mesh文件...")
    all_meshes = []
    mesh_count = 0
    
    for i, link in enumerate(robot.links):
        print(f"处理link {i+1}/{len(robot.links)}: {link.name}")
        
        link_tf = link_transforms[link.name]  # 当前link的绝对变换
        
        # 处理visual中的mesh（如需碰撞模型，可同理处理collision）
        for visual in link.visuals:
            if visual.geometry.mesh is not None:
                mesh_path = visual.geometry.mesh.filename
                scale = visual.geometry.mesh.scale or [1, 1, 1]  # 缩放参数
                
                # 处理相对路径：将../meshes/转换为绝对路径
                if mesh_path.startswith("../meshes/"):
                    # 获取URDF文件所在目录
                    urdf_dir = os.path.dirname(urdf_path)
                    # 构建mesh文件的绝对路径
                    mesh_filename = mesh_path.replace("../meshes/", "")
                    mesh_absolute_path = os.path.join(urdf_dir, "..", "meshes", mesh_filename)
                    mesh_absolute_path = os.path.abspath(mesh_absolute_path)
                else:
                    mesh_absolute_path = mesh_path
                
                print(f"  加载mesh: {mesh_filename}")
                
                try:
                    # 加载mesh（支持stl/obj/dae等格式）
                    mesh = trimesh.load(mesh_absolute_path)
                    
                    # 应用缩放（先缩放再旋转平移）
                    mesh.apply_scale(scale)
                    
                    # 应用绝对变换（转换到根坐标系）
                    mesh.apply_transform(link_tf)
                    
                    all_meshes.append(mesh)
                    mesh_count += 1
                    print(f"  ✓ 成功加载mesh: {mesh_filename}")
                    
                except Exception as e:
                    print(f"  ✗ 加载mesh失败: {mesh_filename}, 错误: {e}")
    
    # 5. 合并所有mesh并保存
    print(f"正在合并 {len(all_meshes)} 个mesh...")
    if all_meshes:
        combined_mesh = trimesh.util.concatenate(all_meshes)
        combined_mesh.export(output_path)
        print(f"✓ 合并完成，保存至：{output_path}")
        print(f"  合并了 {len(all_meshes)} 个mesh文件")
    else:
        print("✗ 没有找到可用的mesh文件")

# 示例：合并URDF中的mesh为一个STL文件
if __name__ == "__main__":
    urdf_meshes_to_single(
        urdf_path=r"D:\Test\engineai\engineai_ros2_workspace\src\simulation\mujoco\assets\resource\robot\pm_v2\urdf\serial_pm_v2 copy.urdf",  # 使用修复后的URDF路径
        xml_path=r"D:\Test\engineai\engineai_ros2_workspace\src\simulation\mujoco\assets\resource\robot\pm_v2\xml\serial_pm_v2.xml",  # XML文件路径
        output_path=r"D:\Test\engineai\engineai_ros2_workspace\src\simulation\mujoco\assets\resource\robot\pm_v2\meshes\serial_pm_v2_combined.stl"       # 输出合并后的mesh路径
    )