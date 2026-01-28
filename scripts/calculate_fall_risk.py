#!/usr/bin/env python3
"""
计算每次摔倒的risk。

输入5种 CSV/BIN 文件（接触力、加速度、电机扭矩、关节wrench、扰动），
在 episode 维度上对每帧的风险做离散时间积分。

【电机扭矩风险定义】

给定采样间隔 dt、权重 wp / wr，对每个关节 j、每一帧 t：

1) 过流 / 峰值扭矩风险（peak torque surrogate）：
   r_peak_j(t) = relu( ( |tau_j(t)| - tau_max_j ) / tau_max_j )

2) 反电动势 / 再生功率风险（regenerative power surrogate）：
   P_j(t)       = tau_j(t) * qd_j(t)
   P_regen_j(t) = max( 0, -P_j(t) )
   r_regen_j(t) = relu( ( P_regen_j(t) - P0_j ) / P0_j )

3) 每帧 joint risk：
   r_joint_j(t) = wp * r_peak_j(t) + wr * r_regen_j(t)

【episode 级积分（必须乘采样间隔 dt）】

- 单关节累计风险：
   R_peak_j  = sum_t r_peak_j(t)  * dt
   R_regen_j = sum_t r_regen_j(t) * dt
   R_joint_j = wp * R_peak_j + wr * R_regen_j

- 总 motor 风险：
   R_motor = sum_j R_joint_j

本文件中的 `calculate_motor_torque_risk()` 按上述公式实现，
并返回 episode 级的总风险以及按关节的 breakdown。
"""

import pandas as pd
import numpy as np
import argparse
import struct
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt


def read_binary_contact_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的contact_data文件
    
    二进制格式：
    - 文件头：int32_t num_joints
    - 每条记录：
      - double sim_time
      - int contact_id
      - int32_t body1_name_len + body1_name (string)
      - int32_t body2_name_len + body2_name (string)
      - double[3] red_ball_pos
      - double[3] green_ball_pos
      - double[3] world_forces
      - double force_magnitude
      - double force_normal
      - double[3] world_torques
      - double[3] base_link_pos
      - double[4] base_link_quat
      - double[3] base_link_vel
      - double[3] base_link_angvel
      - double[3] collision_link_pos
      - double[4] collision_link_quat
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        # 读取文件头（关节数量）
        num_joints_bytes = f.read(4)
        if len(num_joints_bytes) < 4:
            return pd.DataFrame()
        num_joints = struct.unpack('i', num_joints_bytes)[0]
        
        # 读取所有记录
        while True:
            # 读取sim_time
            sim_time_bytes = f.read(8)
            if len(sim_time_bytes) < 8:
                break
            sim_time = struct.unpack('d', sim_time_bytes)[0]
            
            # 读取contact_id
            contact_id_bytes = f.read(4)
            if len(contact_id_bytes) < 4:
                break
            contact_id = struct.unpack('i', contact_id_bytes)[0]
            
            # 读取body1_name
            body1_len_bytes = f.read(4)
            if len(body1_len_bytes) < 4:
                break
            body1_len = struct.unpack('i', body1_len_bytes)[0]
            body1_name = f.read(body1_len).decode('utf-8')
            
            # 读取body2_name
            body2_len_bytes = f.read(4)
            if len(body2_len_bytes) < 4:
                break
            body2_len = struct.unpack('i', body2_len_bytes)[0]
            body2_name = f.read(body2_len).decode('utf-8')
            
            # 读取数组数据
            red_ball_pos = struct.unpack('3d', f.read(24))
            green_ball_pos = struct.unpack('3d', f.read(24))
            world_forces = struct.unpack('3d', f.read(24))
            force_magnitude = struct.unpack('d', f.read(8))[0]
            force_normal = struct.unpack('d', f.read(8))[0]
            world_torques = struct.unpack('3d', f.read(24))
            base_link_pos = struct.unpack('3d', f.read(24))
            base_link_quat = struct.unpack('4d', f.read(32))
            base_link_vel = struct.unpack('3d', f.read(24))
            base_link_angvel = struct.unpack('3d', f.read(24))
            collision_link_pos = struct.unpack('3d', f.read(24))
            collision_link_quat = struct.unpack('4d', f.read(32))
            
            data.append({
                'timestamp': sim_time,
                'contact_id': contact_id,
                'body1_name': body1_name,
                'body2_name': body2_name,
                'pos_x': red_ball_pos[0],
                'pos_y': red_ball_pos[1],
                'pos_z': red_ball_pos[2],
                'force_x': world_forces[0],
                'force_y': world_forces[1],
                'force_z': world_forces[2],
                'force_magnitude': force_magnitude,
                'force_normal': force_normal,
                'torque_x': world_torques[0],
                'torque_y': world_torques[1],
                'torque_z': world_torques[2],
                'base_link_x': base_link_pos[0],
                'base_link_y': base_link_pos[1],
                'base_link_z': base_link_pos[2],
                'base_link_qw': base_link_quat[0],
                'base_link_qx': base_link_quat[1],
                'base_link_qy': base_link_quat[2],
                'base_link_qz': base_link_quat[3],
                'base_link_vel_x': base_link_vel[0],
                'base_link_vel_y': base_link_vel[1],
                'base_link_vel_z': base_link_vel[2],
                'base_link_angvel_x': base_link_angvel[0],
                'base_link_angvel_y': base_link_angvel[1],
                'base_link_angvel_z': base_link_angvel[2],
                'collision_link_x': collision_link_pos[0],
                'collision_link_y': collision_link_pos[1],
                'collision_link_z': collision_link_pos[2],
                'collision_link_qw': collision_link_quat[0],
                'collision_link_qx': collision_link_quat[1],
                'collision_link_qy': collision_link_quat[2],
                'collision_link_qz': collision_link_quat[3],
            })
    
    return pd.DataFrame(data)


def read_binary_sensor_vibration_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的sensor_vibration_data文件
    
    二进制格式：
    - 每条记录：
      - double timestamp
      - double[3] base_link_lin_acc
      - double[3] base_link_ang_acc
      - double[3] head_lin_acc
      - double[3] head_ang_acc
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        while True:
            # 读取timestamp
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            
            timestamp = struct.unpack('d', timestamp_bytes)[0]
            base_link_lin_acc = struct.unpack('3d', f.read(24))
            base_link_ang_acc = struct.unpack('3d', f.read(24))
            head_lin_acc = struct.unpack('3d', f.read(24))
            head_ang_acc = struct.unpack('3d', f.read(24))
            
            data.append({
                'timestamp': timestamp,
                'base_link_lin_acc_x': base_link_lin_acc[0],
                'base_link_lin_acc_y': base_link_lin_acc[1],
                'base_link_lin_acc_z': base_link_lin_acc[2],
                'base_link_ang_acc_x': base_link_ang_acc[0],
                'base_link_ang_acc_y': base_link_ang_acc[1],
                'base_link_ang_acc_z': base_link_ang_acc[2],
                'head_lin_acc_x': head_lin_acc[0],
                'head_lin_acc_y': head_lin_acc[1],
                'head_lin_acc_z': head_lin_acc[2],
                'head_ang_acc_x': head_ang_acc[0],
                'head_ang_acc_y': head_ang_acc[1],
                'head_ang_acc_z': head_ang_acc[2],
            })
    
    return pd.DataFrame(data)


def read_binary_joint_state_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的joint_state_data文件
    
    二进制格式：
    - 每条记录：
      - double timestamp
      - int32_t num_joints
      - double[num_joints] joint_positions
      - int32_t num_joints
      - double[num_joints] joint_velocities
      - int32_t num_joints
      - double[num_joints] actuator_forces
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        while True:
            # 读取timestamp
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            
            timestamp = struct.unpack('d', timestamp_bytes)[0]
            
            # 读取关节数量
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack('i', num_joints_bytes)[0]
            
            # 读取关节位置
            joint_pos_bytes = f.read(num_joints * 8)
            if len(joint_pos_bytes) < num_joints * 8:
                break
            joint_positions = struct.unpack(f'{num_joints}d', joint_pos_bytes)
            
            # 读取关节数量（速度）
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack('i', num_joints_bytes)[0]
            
            # 读取关节速度
            joint_vel_bytes = f.read(num_joints * 8)
            if len(joint_vel_bytes) < num_joints * 8:
                break
            joint_velocities = struct.unpack(f'{num_joints}d', joint_vel_bytes)
            
            # 读取关节数量（力矩）
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack('i', num_joints_bytes)[0]
            
            # 读取电机力矩
            actuator_force_bytes = f.read(num_joints * 8)
            if len(actuator_force_bytes) < num_joints * 8:
                break
            actuator_forces = struct.unpack(f'{num_joints}d', actuator_force_bytes)
            
            row = {'timestamp': timestamp}
            for i in range(num_joints):
                row[f'joint_{i}_position'] = joint_positions[i]
                row[f'joint_{i}_velocity'] = joint_velocities[i]
                row[f'actuator_{i}_force'] = actuator_forces[i]
            
            data.append(row)
    
    return pd.DataFrame(data)


def read_binary_perturbation_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的perturbation_data文件
    
    二进制格式：
    - 每条记录：
      - double sim_time
      - int perturbation_id
      - int32_t body_name_len + body_name (string)
      - double start_time
      - double duration
      - double elapsed_time
      - double[3] force
      - double force_magnitude
      - double[3] torque
      - double torque_magnitude
      - double[3] std_pose
      - double[3] world_force
      - double world_force_magnitude
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        while True:
            # 读取sim_time
            sim_time_bytes = f.read(8)
            if len(sim_time_bytes) < 8:
                break
            sim_time = struct.unpack('d', sim_time_bytes)[0]
            
            # 读取perturbation_id
            perturbation_id_bytes = f.read(4)
            if len(perturbation_id_bytes) < 4:
                break
            perturbation_id = struct.unpack('i', perturbation_id_bytes)[0]
            
            # 读取body_name
            body_name_len_bytes = f.read(4)
            if len(body_name_len_bytes) < 4:
                break
            body_name_len = struct.unpack('i', body_name_len_bytes)[0]
            body_name = f.read(body_name_len).decode('utf-8')
            
            # 读取其他数据
            start_time = struct.unpack('d', f.read(8))[0]
            duration = struct.unpack('d', f.read(8))[0]
            elapsed_time = struct.unpack('d', f.read(8))[0]
            force = struct.unpack('3d', f.read(24))
            force_magnitude = struct.unpack('d', f.read(8))[0]
            torque = struct.unpack('3d', f.read(24))
            torque_magnitude = struct.unpack('d', f.read(8))[0]
            std_pose = struct.unpack('3d', f.read(24))
            world_force = struct.unpack('3d', f.read(24))
            world_force_magnitude = struct.unpack('d', f.read(8))[0]
            
            data.append({
                'timestamp': sim_time,
                'perturbation_id': perturbation_id,
                'body_name': body_name,
                'start_time': start_time,
                'duration': duration,
                'elapsed_time': elapsed_time,
                'force_x': force[0],
                'force_y': force[1],
                'force_z': force[2],
                'force_magnitude': force_magnitude,
                'torque_x': torque[0],
                'torque_y': torque[1],
                'torque_z': torque[2],
                'torque_magnitude': torque_magnitude,
                'std_pose_x': std_pose[0],
                'std_pose_y': std_pose[1],
                'std_pose_z': std_pose[2],
                'world_force_x': world_force[0],
                'world_force_y': world_force[1],
                'world_force_z': world_force[2],
                'world_force_magnitude': world_force_magnitude,
            })
    
    return pd.DataFrame(data)


def read_binary_joint_forces_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的joint_forces_data文件
    
    二进制格式：
    - 每条记录（每个关节）：
      - double timestamp
      - int joint_id
      - int32_t joint_name_len + joint_name (string)
      - int body_id
      - int32_t body_name_len + body_name (string)
      - double[3] child_M (力矩)
      - double[3] child_F (力)
      - double[3] parent_M (力矩)
      - double[3] parent_F (力)
      - double[3] axis (关节轴向量)
      - double F_axial_mag
      - double F_shear_mag
      - double M_torsion_mag
      - double M_bend_mag
      - double M_eq
      - double[3] F_axial
      - double[3] F_shear
      - double[3] M_torsion
      - double[3] M_bend
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        while True:
            # 读取timestamp
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            
            timestamp = struct.unpack('d', timestamp_bytes)[0]
            
            # 读取joint_id
            joint_id_bytes = f.read(4)
            if len(joint_id_bytes) < 4:
                break
            joint_id = struct.unpack('i', joint_id_bytes)[0]
            
            # 读取joint_name
            joint_name_len_bytes = f.read(4)
            if len(joint_name_len_bytes) < 4:
                break
            joint_name_len = struct.unpack('i', joint_name_len_bytes)[0]
            joint_name = f.read(joint_name_len).decode('utf-8')
            
            # 读取body_id
            body_id_bytes = f.read(4)
            if len(body_id_bytes) < 4:
                break
            body_id = struct.unpack('i', body_id_bytes)[0]
            
            # 读取body_name
            body_name_len_bytes = f.read(4)
            if len(body_name_len_bytes) < 4:
                break
            body_name_len = struct.unpack('i', body_name_len_bytes)[0]
            body_name = f.read(body_name_len).decode('utf-8')
            
            # 读取子body坐标系下的反力
            child_M = struct.unpack('3d', f.read(24))
            child_F = struct.unpack('3d', f.read(24))
            
            # 读取父body坐标系下的反力
            parent_M = struct.unpack('3d', f.read(24))
            parent_F = struct.unpack('3d', f.read(24))
            
            # 读取关节轴向量
            axis = struct.unpack('3d', f.read(24))
            
            # 读取载荷分解的标量值
            F_axial_mag = struct.unpack('d', f.read(8))[0]
            F_shear_mag = struct.unpack('d', f.read(8))[0]
            M_torsion_mag = struct.unpack('d', f.read(8))[0]
            M_bend_mag = struct.unpack('d', f.read(8))[0]
            M_eq = struct.unpack('d', f.read(8))[0]
            
            # 读取向量
            F_axial = struct.unpack('3d', f.read(24))
            F_shear = struct.unpack('3d', f.read(24))
            M_torsion = struct.unpack('3d', f.read(24))
            M_bend = struct.unpack('3d', f.read(24))
            
            data.append({
                'timestamp': timestamp,
                'joint_id': joint_id,
                'joint_name': joint_name,
                'body_id': body_id,
                'body_name': body_name,
                'child_Mx': child_M[0],
                'child_My': child_M[1],
                'child_Mz': child_M[2],
                'child_Fx': child_F[0],
                'child_Fy': child_F[1],
                'child_Fz': child_F[2],
                'parent_Mx': parent_M[0],
                'parent_My': parent_M[1],
                'parent_Mz': parent_M[2],
                'parent_Fx': parent_F[0],
                'parent_Fy': parent_F[1],
                'parent_Fz': parent_F[2],
                'axis_x': axis[0],
                'axis_y': axis[1],
                'axis_z': axis[2],
                'F_axial_mag': F_axial_mag,
                'F_shear_mag': F_shear_mag,
                'M_torsion_mag': M_torsion_mag,
                'M_bend_mag': M_bend_mag,
                'M_eq': M_eq,
                'F_axial_x': F_axial[0],
                'F_axial_y': F_axial[1],
                'F_axial_z': F_axial[2],
                'F_shear_x': F_shear[0],
                'F_shear_y': F_shear[1],
                'F_shear_z': F_shear[2],
                'M_torsion_x': M_torsion[0],
                'M_torsion_y': M_torsion[1],
                'M_torsion_z': M_torsion[2],
                'M_bend_x': M_bend[0],
                'M_bend_y': M_bend[1],
                'M_bend_z': M_bend[2],
            })
    
    return pd.DataFrame(data)


def load_data_file(file_path: str) -> pd.DataFrame:
    """
    根据文件扩展名自动选择CSV或二进制格式读取
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return pd.DataFrame()
    
    ext = Path(file_path).suffix.lower()
    
    if ext == '.bin':
        # 根据文件名判断数据类型
        filename = Path(file_path).stem.lower()
        if 'contact_data' in filename:
            return read_binary_contact_data(file_path)
        elif 'sensor_vibration_data' in filename:
            return read_binary_sensor_vibration_data(file_path)
        elif 'joint_state_data' in filename:
            return read_binary_joint_state_data(file_path)
        elif 'perturbation_data' in filename:
            return read_binary_perturbation_data(file_path)
        elif 'joint_forces_data' in filename:
            return read_binary_joint_forces_data(file_path)
        else:
            print(f"警告: 无法识别二进制文件类型: {file_path}")
            return pd.DataFrame()
    elif ext == '.csv':
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"错误: 读取CSV文件失败 {file_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"警告: 不支持的文件格式: {ext}")
        return pd.DataFrame()


def relu(x: np.ndarray) -> np.ndarray:
    """
    简单的ReLU函数：relu(x) = max(x, 0)
    支持numpy向量/矩阵输入，返回同形状数组。
    """
    return np.maximum(x, 0.0)


def get_joint_torque_limit(joint_name: str) -> float:
    """
    根据关节名称获取对应的扭矩上限
    
    高扭矩关节（Q90电机）：164.0 N·m
    - HIP_PITCH (左右): J00, J06
    - HIP_ROLL (左右): J01, J07
    - KNEE_PITCH (左右): J03, J09
    
    低扭矩关节（Q25电机）：52.0 N·m
    - HIP_YAW (左右): J02, J08
    - ANKLE_PITCH (左右): J04, J10
    - ANKLE_ROLL (左右): J05, J11
    - WAIST_YAW: J12
    - 所有手臂关节: J13-J22
    - HEAD_YAW: J23
    
    参数:
        joint_name: 关节名称，可能是 "0", "1", "J00", "J01" 等格式
    
    返回:
        扭矩上限（N·m），如果未找到则返回 52.0（默认低扭矩）
    """
    # 高扭矩关节索引（Q90电机：164.0 N·m）
    high_torque_joints = {0, 1, 3, 6, 7, 9}  # J00, J01, J03, J06, J07, J09
    
    # 尝试解析关节名称
    joint_idx = None
    # 移除 "J" 前缀（如果有）
    name_clean = joint_name.upper().lstrip('J')
    try:
        joint_idx = int(name_clean)
    except ValueError:
        # 如果无法解析为数字，返回默认值
        return 52.0
    
    # 根据关节索引返回对应的扭矩上限
    if joint_idx in high_torque_joints:
        return 164.0
    elif 0 <= joint_idx <= 23:
        return 52.0
    else:
        # 超出范围，返回默认值
        return 52.0


def extract_timestamp_from_filename(file_path: str) -> Optional[str]:
    """
    从文件名中提取时间戳（格式：YYYYMMDD_HHMMSS）
    
    例如：
    - joint_state_data_20260128_185303.csv -> 20260128_185303
    - contact_data_20250128_120000.bin -> 20250128_120000
    
    返回:
        时间戳字符串，如果未找到则返回 None
    """
    filename = Path(file_path).stem  # 去掉路径和扩展名
    # 匹配 YYYYMMDD_HHMMSS 格式
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def extract_collision_timestamps(
    contact_df: pd.DataFrame,
    force_threshold: float = 400.0
) -> np.ndarray:
    """
    从 contact_data 中提取 non-foot impact 且 force 大于阈值的时间戳集合
    
    参数:
        contact_df: 接触力数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - body1_name: 第一个物体名称
            - body2_name: 第二个物体名称
            - force_magnitude: 力的大小
        force_threshold: 力的阈值（N），默认 400.0
    
    返回:
        time_collision: 满足条件的时间戳数组（已排序且去重）
    """
    if contact_df.empty:
        return np.array([])
    
    # 检查必要的列
    required_cols = ['timestamp', 'body1_name', 'body2_name', 'force_magnitude']
    missing_cols = [col for col in required_cols if col not in contact_df.columns]
    if missing_cols:
        print(f"警告: contact_df 缺少必要的列: {missing_cols}")
        return np.array([])
    
    # 定义 foot 相关的 link 名称（用于判断是否为 foot impact）
    foot_keywords = ['foot', 'ankle', 'toe', 'heel', 'LINK_ANKLE', 'LINK_FOOT']
    
    # 判断是否为 non-foot impact
    def is_non_foot_impact(body1: str, body2: str) -> bool:
        """判断接触是否涉及 foot"""
        body1_lower = str(body1).lower()
        body2_lower = str(body2).lower()
        
        # 如果 body1 或 body2 包含 foot 关键词，则认为是 foot impact
        for keyword in foot_keywords:
            if keyword.lower() in body1_lower or keyword.lower() in body2_lower:
                return False
        return True
    
    # 筛选条件：non-foot impact 且 force > threshold
    mask = (
        contact_df.apply(
            lambda row: is_non_foot_impact(row['body1_name'], row['body2_name']),
            axis=1
        ) &
        (contact_df['force_magnitude'] > force_threshold)
    )
    
    # 提取时间戳
    collision_times = contact_df.loc[mask, 'timestamp'].values
    
    # 去重并排序
    if len(collision_times) > 0:
        collision_times = np.unique(collision_times)
        collision_times = np.sort(collision_times)
    
    return collision_times


def calculate_acceleration_risk(
    sensor_vibration_df: pd.DataFrame,
    time_collision: np.ndarray,
    a_thr: float = 30.0,
    dt: float = 0.002,
    window_ms: float = 20.0,
    save_frame_by_frame_path: Optional[str] = None
) -> float:
    """
    计算加速度风险指标 risk_acc（仅使用 torso 与 head）
    
    风险定义为：
    risk_acc = sum_{b in {torso, head}} ∫[0 to 20ms] ReLU(||a_b(t)|| - a_thr) dt
    
    参数:
        sensor_vibration_df: 传感器振动数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - base_link_lin_acc_x, base_link_lin_acc_y, base_link_lin_acc_z: torso 线加速度
            - head_lin_acc_x, head_lin_acc_y, head_lin_acc_z: head 线加速度
        time_collision: 碰撞时间戳数组（从 extract_collision_timestamps 获取）
        a_thr: 加速度阈值，默认 0.0（仅统计超过阈值的部分）
        dt: 采样时间间隔（秒）。如果为 None，则从 timestamp 列自动估计
        window_ms: 评估时间窗（毫秒），默认 20.0 ms
    
    返回:
        risk_acc: 加速度风险值
    """
    if sensor_vibration_df.empty or len(time_collision) == 0:
        return 0.0
    
    # 检查必要的列
    required_cols = [
        'timestamp',
        'base_link_lin_acc_x', 'base_link_lin_acc_y', 'base_link_lin_acc_z',
        'head_lin_acc_x', 'head_lin_acc_y', 'head_lin_acc_z'
    ]
    missing_cols = [col for col in required_cols if col not in sensor_vibration_df.columns]
    if missing_cols:
        print(f"警告: sensor_vibration_df 缺少必要的列: {missing_cols}")
        return 0.0
    
    # 验证 dt 值
    if dt <= 0 or not np.isfinite(dt):
        print(f"警告: 无效的 dt 值: {dt}，使用默认值 0.002")
        dt = 0.002
    
    # 转换时间窗为秒
    window_s = window_ms / 1000.0
    
    # 获取时间戳和加速度数据
    timestamps = sensor_vibration_df['timestamp'].to_numpy(dtype=float)
    
    # torso (base_link) 加速度
    torso_acc_x = sensor_vibration_df['base_link_lin_acc_x'].to_numpy(dtype=float)
    torso_acc_y = sensor_vibration_df['base_link_lin_acc_y'].to_numpy(dtype=float)
    torso_acc_z = sensor_vibration_df['base_link_lin_acc_z'].to_numpy(dtype=float)
    torso_acc_mag = np.sqrt(torso_acc_x**2 + torso_acc_y**2 + torso_acc_z**2)
    
    # head 加速度
    head_acc_x = sensor_vibration_df['head_lin_acc_x'].to_numpy(dtype=float)
    head_acc_y = sensor_vibration_df['head_lin_acc_y'].to_numpy(dtype=float)
    head_acc_z = sensor_vibration_df['head_lin_acc_z'].to_numpy(dtype=float)
    head_acc_mag = np.sqrt(head_acc_x**2 + head_acc_y**2 + head_acc_z**2)
    
    # 初始化总风险
    total_risk = 0.0
    
    # 用于保存逐帧数据的列表
    frame_data = []
    
    # 合并并排序碰撞时间，处理相邻时间间隔小于 20ms 的情况
    if len(time_collision) == 0:
        # 如果没有碰撞，仍然尝试保存空的 CSV（如果指定了路径）
        if save_frame_by_frame_path is not None:
            empty_df = pd.DataFrame(columns=['timestamp', 'link_name', 'acc_magnitude', 'risk_value', 'cumulative_risk'])
            try:
                empty_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"加速度风险逐帧数据已保存到: {save_frame_by_frame_path} (无数据)")
            except Exception as e:
                print(f"警告: 保存加速度风险逐帧数据到 '{save_frame_by_frame_path}' 失败: {e}")
        return 0.0
    
    # 对碰撞时间进行分组，如果两个碰撞时间间隔小于 window_ms，则合并为一个时间窗口
    # 如果两个碰撞时间相邻小于 20ms，窗口应该是从第一个时间到第二个时间 + 20ms
    collision_groups = []
    current_group = [time_collision[0]]
    
    for i in range(1, len(time_collision)):
        if time_collision[i] - current_group[-1] < window_s:
            # 如果间隔小于 window_ms，合并到当前组
            current_group.append(time_collision[i])
        else:
            # 否则开始新组
            collision_groups.append(current_group)
            current_group = [time_collision[i]]
    
    # 添加最后一组
    if current_group:
        collision_groups.append(current_group)
    
    # 累计风险（用于计算累计值）
    cumulative_risk = 0.0
    
    # 对每个碰撞组计算风险
    for group in collision_groups:
        # 使用组中最早的时间作为窗口起点
        t_start = min(group)
        # 如果组中有多个碰撞时间，窗口终点是最后一个时间 + window_ms
        # 否则窗口终点是第一个时间 + window_ms
        if len(group) > 1:
            t_end = max(group) + window_s
        else:
            t_end = t_start + window_s
        
        # 找到时间窗口内的数据索引
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # 提取窗口内的加速度数据和时间戳
        torso_acc_window = torso_acc_mag[indices]
        head_acc_window = head_acc_mag[indices]
        window_timestamps = timestamps[indices]
        
        # 计算 ReLU(||a|| - a_thr)
        torso_risk = relu(torso_acc_window - a_thr)
        head_risk = relu(head_acc_window - a_thr)
        
        # 保存逐帧数据
        if save_frame_by_frame_path is not None:
            for idx, t in enumerate(window_timestamps):
                # torso 数据
                if torso_risk[idx] > 0:
                    cumulative_risk += torso_risk[idx] * dt
                    frame_data.append({
                        'timestamp': float(t),
                        'link_name': 'torso',
                        'acc_magnitude': float(torso_acc_window[idx]),
                        'risk_value': float(torso_risk[idx]),
                        'cumulative_risk': float(cumulative_risk)
                    })
                
                # head 数据
                if head_risk[idx] > 0:
                    cumulative_risk += head_risk[idx] * dt
                    frame_data.append({
                        'timestamp': float(t),
                        'link_name': 'head',
                        'acc_magnitude': float(head_acc_window[idx]),
                        'risk_value': float(head_risk[idx]),
                        'cumulative_risk': float(cumulative_risk)
                    })
        
        # 数值积分（使用矩形法，乘以 dt）
        torso_integral = np.nansum(torso_risk) * dt
        head_integral = np.nansum(head_risk) * dt
        
        # 累加风险
        total_risk += torso_integral + head_integral
    
    # 确保结果不是 NaN
    if not np.isfinite(total_risk):
        total_risk = 0.0
    
    # 保存逐帧数据到 CSV
    if save_frame_by_frame_path is not None:
        try:
            if frame_data:
                frame_df = pd.DataFrame(frame_data)
                frame_df = frame_df.sort_values('timestamp').reset_index(drop=True)
                frame_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"加速度风险逐帧数据已保存到: {save_frame_by_frame_path} ({len(frame_df)} 条记录)")
            else:
                empty_df = pd.DataFrame(columns=['timestamp', 'link_name', 'acc_magnitude', 'risk_value', 'cumulative_risk'])
                empty_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"加速度风险逐帧数据已保存到: {save_frame_by_frame_path} (无超阈值数据)")
        except Exception as e:
            print(f"警告: 保存加速度风险逐帧数据到 '{save_frame_by_frame_path}' 失败: {e}")
    
    return float(total_risk)


def calculate_contact_force_risk(contact_df: pd.DataFrame) -> float:
    """
    计算接触力相关的risk
    
    待实现：根据用户提供的计算方法
    """
    if contact_df.empty:
        return 0.0
    
    # TODO: 实现具体的risk计算逻辑
    # 示例：基于force_magnitude的简单计算
    risk = 0.0
    
    return risk


def calculate_motor_torque_risk(
    joint_state_df: pd.DataFrame,
    dt: Optional[float] = 0.002,
    mode: str = "const",
    wp: float = 1.0,
    wr: float = 1.0,
    tau_max_const: float = 52.0,  # 注意：const 模式下会根据关节类型自动选择 164.0 或 52.0
    P0_const: float = 1000.0,
    q_tau: float = 0.9,
    q_p: float = 0.9,
    eps: float = 1e-8,
    save_path: Optional[str] = None,
    save_frame_by_frame_path: Optional[str] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    计算电机力矩相关的风险（episode 级别，离散时间积分）

    参数:
        joint_state_df: 包含关节状态的DataFrame，列名规则：
            - 电机扭矩:  actuator_<name>_force
            - 关节角速度: joint_<name>_velocity
        dt: 采样时间间隔，默认 0.002 s；
            若显式传入 None，则尝试从 time / timestamp 列估计
        mode: 阈值模式:
            - "const": 使用常数阈值（根据关节类型自动选择 164.0 或 52.0）, P0_const
            - "data":  基于数据分位数估计 tau_max_j, P0_j
        wp, wr: 峰值扭矩与再生功率风险的权重
        tau_max_const: const 模式下未使用（会根据关节类型自动选择）
        P0_const: const 模式下所有关节共享的再生功率阈值
        q_tau: data 模式下 |tau_j| 的分位数（默认 0.9，即 90% 分位数）
        q_p: data 模式下 P_regen_j 的分位数（默认 0.9，即 90% 分位数）
        eps: 数值下限，避免除以 0
        save_path: 若不为 None，则将 episode 级 breakdown 保存为 CSV

    返回:
        R_motor: float，总 motor 风险
        breakdown: pd.DataFrame，按关节的 episode 级风险明细，包含列:
            - joint_name
            - R_peak
            - R_regen
            - R_joint
            - tau_max
            - P0
    """
    if joint_state_df.empty:
        empty_breakdown = pd.DataFrame(
            columns=["joint_name", "R_peak", "R_regen", "R_joint", "tau_max", "P0"]
        )
        return 0.0, empty_breakdown

    # 处理 dt
    if dt is None:
        time_col = None
        for col in ["time", "timestamp"]:
            if col in joint_state_df.columns:
                time_col = col
                break
        if time_col is None:
            raise ValueError("无法自动推断 dt：请提供 dt 参数，或在 CSV 中包含 'time' 或 'timestamp' 列。")

        times = joint_state_df[time_col].to_numpy(dtype=float)
        if times.size < 2:
            raise ValueError("用于估计 dt 的时间戳数量不足（少于2个）。")

        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            raise ValueError("时间戳差分结果无效，无法估计 dt。")

        dt = float(np.median(diffs))
        
        # 调试信息
        if not np.isfinite(dt) or dt <= 0:
            print(f"警告: 估计的 dt 值异常: {dt}, 使用默认值 0.002")
            dt = 0.002

    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"无效的 dt 值: {dt}, 必须为正数。")

    mode = (mode or "const").lower()
    if mode not in ("const", "data"):
        raise ValueError(f"未知的阈值模式: {mode}, 仅支持 'const' 或 'data'。")

    # 自动解析所有 actuator_*_force 列
    torque_cols = [c for c in joint_state_df.columns if c.startswith("actuator_") and c.endswith("_force")]

    names: list[str] = []
    R_peak_list: list[float] = []
    R_regen_list: list[float] = []
    R_joint_list: list[float] = []
    tau_max_list: list[float] = []
    P0_list: list[float] = []
    timestamps_list: list[float] = []  # 记录每个关节第一个出现 risk 的时间戳

    # 获取时间戳列
    time_col = None
    for col in ["time", "timestamp"]:
        if col in joint_state_df.columns:
            time_col = col
            break
    
    timestamps = None
    if time_col:
        timestamps = joint_state_df[time_col].to_numpy(dtype=float)

    for torque_col in torque_cols:
        # 提取 <name>
        base_name = torque_col[len("actuator_") : -len("_force")]
        vel_col = f"joint_{base_name}_velocity"

        if vel_col not in joint_state_df.columns:
            print(f"警告: 未找到关节 '{base_name}' 的速度列 '{vel_col}'，将跳过该关节。")
            continue

        tau = joint_state_df[torque_col].to_numpy(dtype=float)
        qd = joint_state_df[vel_col].to_numpy(dtype=float)

        # 1) 峰值扭矩 surrogate 风险
        abs_tau = np.abs(tau)

        if mode == "const":
            # 根据关节名称获取对应的扭矩上限
            joint_torque_limit = get_joint_torque_limit(base_name)
            tau_max_j = max(float(joint_torque_limit), eps)
        else:
            # data 模式: tau_max_j = |tau_j| 的 q_tau 分位数
            valid_abs_tau = abs_tau[np.isfinite(abs_tau)]
            if valid_abs_tau.size == 0:
                # 如果数据无效，使用关节特定的默认值
                joint_torque_limit = get_joint_torque_limit(base_name)
                tau_max_j = max(float(joint_torque_limit), eps)
            else:
                # 确保 q_tau 在 [0, 1] 范围内
                q_tau_clamped = max(0.0, min(1.0, q_tau))
                tau_max_j = float(np.quantile(valid_abs_tau, q_tau_clamped))
                # 避免 tau_max_j 过小或为 0，防止除 0
                if not np.isfinite(tau_max_j) or tau_max_j <= 0:
                    joint_torque_limit = get_joint_torque_limit(base_name)
                    tau_max_j = max(float(joint_torque_limit), eps)

        r_peak = relu((abs_tau - tau_max_j) / tau_max_j)

        # 2) 再生功率 surrogate 风险
        P = tau * qd
        P_regen = np.maximum(0.0, -P)

        if mode == "const":
            P0_j = max(float(P0_const), eps)
        else:
            # data 模式: P0_j = P_regen_j 的 q_p 分位数
            valid_P = P_regen[np.isfinite(P_regen) & (P_regen > 0)]
            if valid_P.size == 0:
                # 若 P_regen_j 全 0 或极小，则使用 eps 以避免除 0
                P0_j = max(float(P0_const), eps)
            else:
                # 确保 q_p 在 [0, 1] 范围内
                q_p_clamped = max(0.0, min(1.0, q_p))
                P0_j = float(np.quantile(valid_P, q_p_clamped))
                if not np.isfinite(P0_j) or P0_j <= 0:
                    # 避免 P0_j == 0 或过小，使用 eps 替代
                    P0_j = max(float(P0_const), eps)

        r_regen = relu((P_regen - P0_j) / P0_j)
        
        # 找到第一个出现 risk 的时间戳（r_peak > 0 或 r_regen > 0）
        risk_timestamp = None
        if timestamps is not None and timestamps.size > 0:
            # 找到第一个有 risk 的索引
            has_risk = (r_peak > 0) | (r_regen > 0)
            risk_indices = np.where(has_risk)[0]
            if risk_indices.size > 0:
                risk_timestamp = float(timestamps[risk_indices[0]])
            else:
                # 如果没有 risk，使用第一个时间戳
                risk_timestamp = float(timestamps[0]) if timestamps.size > 0 else None

        # episode 级积分（必须乘采样间隔 dt）
        # 处理 NaN：如果 r_peak 或 r_regen 中有 NaN，使用 nansum 并替换为 0
        sum_r_peak = np.nansum(r_peak) if r_peak.size > 0 else 0.0
        sum_r_regen = np.nansum(r_regen) if r_regen.size > 0 else 0.0
        
        R_peak_j = float(sum_r_peak * dt)
        R_regen_j = float(sum_r_regen * dt)
        R_joint_j = float(wp * R_peak_j + wr * R_regen_j)
        
        # 确保结果不是 NaN（如果出现 NaN，替换为 0.0）
        if not np.isfinite(R_peak_j):
            R_peak_j = 0.0
        if not np.isfinite(R_regen_j):
            R_regen_j = 0.0
        if not np.isfinite(R_joint_j):
            R_joint_j = 0.0
        
        # 四舍五入到4位小数（在计算时就保留4位）
        R_peak_j = round(R_peak_j, 4)
        R_regen_j = round(R_regen_j, 4)
        R_joint_j = round(R_joint_j, 4)

        names.append(base_name)
        R_peak_list.append(R_peak_j)
        R_regen_list.append(R_regen_j)
        R_joint_list.append(R_joint_j)
        tau_max_list.append(float(tau_max_j))
        P0_list.append(float(P0_j))
        timestamps_list.append(risk_timestamp if risk_timestamp is not None else 0.0)

    if not names:
        empty_breakdown = pd.DataFrame(
            columns=["joint_name", "R_peak", "R_regen", "R_joint", "tau_max", "P0"]
        )
        return 0.0, empty_breakdown

    # 对 tau_max 和 P0 也进行四舍五入到4位小数
    tau_max_list = [round(float(x), 4) for x in tau_max_list]
    P0_list = [round(float(x), 4) for x in P0_list]
    
    breakdown = pd.DataFrame(
        {
            "joint_name": names,
            "R_peak": R_peak_list,
            "R_regen": R_regen_list,
            "R_joint": R_joint_list,
            "tau_max": tau_max_list,
            "P0": P0_list,
            "risk_timestamp": timestamps_list,  # 临时存储，稍后会在主函数中处理
        }
    )

    breakdown = breakdown.sort_values("R_joint", ascending=False).reset_index(drop=True)
    # 使用 nansum 处理可能的 NaN 值
    R_motor = float(breakdown["R_joint"].sum())
    if not np.isfinite(R_motor):
        R_motor = 0.0
    # 四舍五入到4位小数（在计算时就保留4位）
    R_motor = round(R_motor, 4)

    if save_path is not None:
        try:
            breakdown.to_csv(save_path, index=False)
            print(f"motor torque risk breakdown 已保存到: {save_path}")
        except Exception as e:
            print(f"警告: 保存 motor torque risk breakdown 到 '{save_path}' 失败: {e}")

    # 保存逐帧数据（如果指定了路径）
    if save_frame_by_frame_path is not None:
        try:
            frame_by_frame_df = save_frame_by_frame_risk_data(
                joint_state_df, torque_cols, timestamps, mode, wp, wr, 
                tau_max_const, P0_const, q_tau, q_p, eps, dt
            )
            if not frame_by_frame_df.empty:
                frame_by_frame_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"motor torque 逐帧风险数据已保存到: {save_frame_by_frame_path} ({len(frame_by_frame_df)} 条记录)")
            else:
                # 即使没有超阈值数据，也保存一个空的 CSV 文件
                empty_df = pd.DataFrame(columns=['timestamp', 'joint_name', 'r_peak', 'r_regen', 'r_joint', 'R_joint', 'tau_max', 'P0'])
                empty_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"motor torque 逐帧风险数据已保存到: {save_frame_by_frame_path} (无超阈值数据)")
        except Exception as e:
            print(f"警告: 保存逐帧风险数据到 '{save_frame_by_frame_path}' 失败: {e}")

    return R_motor, breakdown


def save_frame_by_frame_risk_data(
    joint_state_df: pd.DataFrame,
    torque_cols: list,
    timestamps: Optional[np.ndarray],
    mode: str,
    wp: float,
    wr: float,
    tau_max_const: float,
    P0_const: float,
    q_tau: float,
    q_p: float,
    eps: float,
    dt: float = 0.002,
) -> pd.DataFrame:
    """
    保存逐帧的风险数据（只记录超阈值的帧）
    
    返回:
        DataFrame，包含列：timestamp, joint_name, tau, tau_max, r_peak, r_regen, r_joint
    """
    frame_data = []
    
    # 获取时间戳
    time_col = None
    for col in ["time", "timestamp"]:
        if col in joint_state_df.columns:
            time_col = col
            break
    
    if timestamps is None and time_col:
        timestamps = joint_state_df[time_col].to_numpy(dtype=float)
    
    for torque_col in torque_cols:
        # 提取 <name>
        base_name = torque_col[len("actuator_") : -len("_force")]
        vel_col = f"joint_{base_name}_velocity"
        
        if vel_col not in joint_state_df.columns:
            continue
        
        tau = joint_state_df[torque_col].to_numpy(dtype=float)
        qd = joint_state_df[vel_col].to_numpy(dtype=float)
        
        # 计算阈值
        abs_tau = np.abs(tau)
        if mode == "const":
            joint_torque_limit = get_joint_torque_limit(base_name)
            tau_max_j = max(float(joint_torque_limit), eps)
        else:
            valid_abs_tau = abs_tau[np.isfinite(abs_tau)]
            if valid_abs_tau.size == 0:
                joint_torque_limit = get_joint_torque_limit(base_name)
                tau_max_j = max(float(joint_torque_limit), eps)
            else:
                q_tau_clamped = max(0.0, min(1.0, q_tau))
                tau_max_j = float(np.quantile(valid_abs_tau, q_tau_clamped))
                if not np.isfinite(tau_max_j) or tau_max_j <= 0:
                    joint_torque_limit = get_joint_torque_limit(base_name)
                    tau_max_j = max(float(joint_torque_limit), eps)
        
        # 计算瞬时风险（不乘dt）
        r_peak = relu((abs_tau - tau_max_j) / tau_max_j)
        
        # 再生功率风险
        P = tau * qd
        P_regen = np.maximum(0.0, -P)
        
        if mode == "const":
            P0_j = max(float(P0_const), eps)
        else:
            valid_P = P_regen[np.isfinite(P_regen) & (P_regen > 0)]
            if valid_P.size == 0:
                P0_j = max(float(P0_const), eps)
            else:
                q_p_clamped = max(0.0, min(1.0, q_p))
                P0_j = float(np.quantile(valid_P, q_p_clamped))
                if not np.isfinite(P0_j) or P0_j <= 0:
                    P0_j = max(float(P0_const), eps)
        
        r_regen = relu((P_regen - P0_j) / P0_j)
        r_joint = wp * r_peak + wr * r_regen
        
        # 只记录超阈值的帧（r_peak > 0 或 r_regen > 0）
        has_risk = (r_peak > 0) | (r_regen > 0)
        risk_indices = np.where(has_risk)[0]
        
        # 计算累计R_joint（从开始到当前帧）
        r_joint_cumsum = (r_joint * dt).cumsum()
        
        for idx in risk_indices:
            # R_joint是累计值（乘dt）
            R_joint_at_frame = float(r_joint_cumsum[idx])
            # r_joint是瞬时值（不乘dt）
            r_joint_at_frame = float(r_joint[idx])
            frame_data.append({
                'timestamp': float(timestamps[idx]) if timestamps is not None and idx < len(timestamps) else 0.0,
                'joint_name': base_name,
                'r_peak': round(float(r_peak[idx]), 4),  # 瞬时值，不乘dt
                'r_regen': round(float(r_regen[idx]), 4),  # 瞬时值，不乘dt
                'r_joint': round(r_joint_at_frame, 4),  # 瞬时值，不乘dt（用于绘图）
                'R_joint': round(R_joint_at_frame, 4),  # 累计值，乘dt
                'tau_max': round(float(tau_max_j), 4),
                'P0': round(float(P0_j), 4),
            })
    
    if not frame_data:
        return pd.DataFrame()
    
    if not frame_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(frame_data)
    # 按时间戳排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 确保列顺序：timestamp, joint_name, r_peak, r_regen, r_joint, R_joint, tau_max, P0
    # filename会在主函数中添加
    return df


def plot_risk_curves(frame_by_frame_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制风险曲线
    
    参数:
        frame_by_frame_df: 逐帧风险数据 DataFrame
        output_path: 图片保存路径（可选）
    """
    if frame_by_frame_df.empty:
        print("警告: 逐帧数据为空，无法绘图")
        return
    
    # 按关节分组
    joints = frame_by_frame_df['joint_name'].unique()
    n_joints = len(joints)
    
    if n_joints == 0:
        return
    
    # 创建子图：第一个子图显示总 risk，后面每个关节一个子图
    fig, axes = plt.subplots(n_joints + 1, 1, figsize=(12, 4 * (n_joints + 1)), sharex=True)
    if n_joints == 0:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # 第一个子图：总 risk（所有关节的 r_joint 总和）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    total_risk_by_time = frame_by_frame_df_sorted.groupby('timestamp')['r_joint'].sum().reset_index()
    total_risk_by_time = total_risk_by_time.sort_values('timestamp')
    
    ax_total = axes[0]
    ax_total.plot(total_risk_by_time['timestamp'], total_risk_by_time['r_joint'], 
                  'k-', label='Total Motor Risk (sum of r_joint)', linewidth=2)
    ax_total.set_ylabel('Total Risk', fontsize=10)
    ax_total.legend(loc='upper right')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Motor Risk Over Time', fontsize=12)
    
    # 每个关节的子图
    for i, joint_name in enumerate(joints):
        joint_data = frame_by_frame_df[frame_by_frame_df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        ax.plot(joint_data['timestamp'], joint_data['r_peak'], 'r-', label='r_peak', linewidth=1.5)
        ax.plot(joint_data['timestamp'], joint_data['r_regen'], 'b-', label='r_regen', linewidth=1.5)
        ax.plot(joint_data['timestamp'], joint_data['r_joint'], 'g-', label='r_joint', linewidth=2)
        
        ax.set_ylabel(f'Joint {joint_name} Risk', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Joint {joint_name} Risk Over Time', fontsize=12)
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"风险曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_acceleration_risk_curves(frame_by_frame_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制加速度风险曲线
    
    参数:
        frame_by_frame_df: 逐帧加速度风险数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - link_name: link 名称（torso 或 head）
            - acc_magnitude: 加速度模长
            - risk_value: 风险值
            - cumulative_risk: 累计风险
        output_path: 图片保存路径（可选）
    """
    if frame_by_frame_df.empty:
        print("警告: 加速度风险逐帧数据为空，无法绘图")
        return
    
    # 按 link 分组
    links = frame_by_frame_df['link_name'].unique()
    n_links = len(links)
    
    if n_links == 0:
        return
    
    # 创建子图：第一个子图显示总 risk，后面每个 link 一个子图
    fig, axes = plt.subplots(n_links + 1, 1, figsize=(12, 4 * (n_links + 1)), sharex=True)
    if n_links == 0:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # 第一个子图：总 risk（使用 cumulative_risk 的最大值）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    # 获取每个 link 的最大 cumulative_risk，然后求和
    max_cumulative_by_link = frame_by_frame_df_sorted.groupby('link_name')['cumulative_risk'].max()
    total_max_cumulative = max_cumulative_by_link.sum()
    
    # 绘制累计风险随时间的变化（取每个时刻的最大 cumulative_risk）
    cumulative_by_time = frame_by_frame_df_sorted.groupby('timestamp')['cumulative_risk'].max().reset_index()
    cumulative_by_time = cumulative_by_time.sort_values('timestamp')
    
    ax_total = axes[0]
    ax_total.plot(cumulative_by_time['timestamp'], cumulative_by_time['cumulative_risk'], 
                  'k-', label='Total Cumulative Risk', linewidth=2)
    ax_total.set_ylabel('Cumulative Risk', fontsize=10)
    ax_total.legend(loc='upper right')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Acceleration Risk Over Time', fontsize=12)
    
    # 每个 link 的子图
    for i, link_name in enumerate(links):
        link_data = frame_by_frame_df[frame_by_frame_df['link_name'] == link_name].copy()
        link_data = link_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        ax.plot(link_data['timestamp'], link_data['acc_magnitude'], 'b-', label='Acceleration Magnitude', linewidth=1.5)
        ax.plot(link_data['timestamp'], link_data['cumulative_risk'], 'g-', label='Cumulative Risk', linewidth=2)
        
        ax.set_ylabel(f'{link_name.capitalize()} Risk', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{link_name.capitalize()} Acceleration Risk Over Time', fontsize=12)
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"加速度风险曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_joint_wrench_risk(joint_forces_df: pd.DataFrame) -> float:
    """
    计算关节力/力矩相关的risk
    
    待实现：根据用户提供的计算方法
    """
    if joint_forces_df.empty:
        return 0.0
    
    # TODO: 实现具体的risk计算逻辑
    # 示例：基于关节力的简单计算
    risk = 0.0
    
    return risk


def calculate_perturbation_risk(perturbation_df: pd.DataFrame) -> float:
    """
    计算扰动相关的risk（可选）
    
    待实现：根据用户提供的计算方法
    """
    if perturbation_df.empty:
        return 0.0
    
    # TODO: 实现具体的risk计算逻辑
    # 示例：基于扰动力的简单计算
    risk = 0.0
    
    return risk


def calculate_total_fall_risk(
    contact_df: pd.DataFrame,
    sensor_vibration_df: pd.DataFrame,
    joint_state_df: pd.DataFrame,
    joint_forces_df: pd.DataFrame,
    perturbation_df: Optional[pd.DataFrame] = None,
    output_base: Optional[str] = None,
) -> Dict:
    """
    计算总的摔倒risk，包括各个子项
    
    参数:
        contact_df: 接触力数据
        sensor_vibration_df: 传感器振动数据
        joint_state_df: 关节状态数据
        joint_forces_df: 关节力数据
        perturbation_df: 扰动数据（可选）
        output_base: 输出文件路径前缀（可选）
        
    注意:
        - force_threshold 由 extract_collision_timestamps 函数管理（默认 400.0）
        - a_thr 由 calculate_acceleration_risk 函数管理（默认 20.0）
    
    返回:
        Dict包含：
        - 各个risk子项的标量值（contact_force_risk, acceleration_risk, ...）
        - total_risk: 总风险
        - breakdowns: Dict，包含各个risk的详细breakdown（如果有的话）
          例如: {'motor_torque': DataFrame, ...}
        - time_collision: 碰撞时间戳数组（全局可用，供其他 risk 计算使用）
    """
    # 提取碰撞时间戳（全局可用，供其他 risk 计算使用）
    # force_threshold 由 extract_collision_timestamps 函数管理默认值
    time_collision = extract_collision_timestamps(contact_df)
    
    # 准备保存路径
    acceleration_frame_by_frame_path = None
    motor_frame_by_frame_path = None
    if output_base:
        acceleration_frame_by_frame_path = f"{output_base}_acceleration_risk.csv"
        motor_frame_by_frame_path = f"{output_base}_motor_risk.csv"
    
    # 计算各个子risk
    contact_risk = calculate_contact_force_risk(contact_df)
    # a_thr 由 calculate_acceleration_risk 函数管理默认值
    acceleration_risk = calculate_acceleration_risk(
        sensor_vibration_df,
        time_collision=time_collision,
        save_frame_by_frame_path=acceleration_frame_by_frame_path
    )
    
    motor_torque_risk, motor_breakdown = calculate_motor_torque_risk(
        joint_state_df,
        save_frame_by_frame_path=motor_frame_by_frame_path
    )
    joint_wrench_risk = calculate_joint_wrench_risk(joint_forces_df)
    perturbation_risk = calculate_perturbation_risk(perturbation_df) if perturbation_df is not None else 0.0
    
    # 所有risk四舍五入到4位小数（在计算时就保留4位）
    contact_risk = round(contact_risk, 4)
    acceleration_risk = round(acceleration_risk, 4)
    motor_torque_risk = round(motor_torque_risk, 4)
    joint_wrench_risk = round(joint_wrench_risk, 4)
    perturbation_risk = round(perturbation_risk, 4)
    
    # 求和得到总risk（基于四舍五入后的值）
    total_risk = contact_risk + acceleration_risk + motor_torque_risk + joint_wrench_risk + perturbation_risk
    total_risk = round(total_risk, 4)
    
    # 收集所有breakdown
    breakdowns = {}
    if not motor_breakdown.empty:
        breakdowns['motor_torque'] = motor_breakdown
    
    return {
        'contact_force_risk': contact_risk,
        'acceleration_risk': acceleration_risk,
        'motor_torque_risk': motor_torque_risk,
        'joint_wrench_risk': joint_wrench_risk,
        'perturbation_risk': perturbation_risk,
        'total_risk': total_risk,
        'breakdowns': breakdowns,
        'time_collision': time_collision  # 全局可用，供其他 risk 计算使用
    }


def main():
    parser = argparse.ArgumentParser(
        description='计算每次摔倒的risk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 calculate_fall_risk.py \\
    --contact contact_data_20250128_120000.csv \\
    --sensor-vibration sensor_vibration_data_20250128_120000.csv \\
    --joint-state joint_state_data_20250128_120000.csv \\
    --joint-forces joint_forces_data_20250128_120000.csv \\
    --perturbation perturbation_data_20250128_120000.csv
        """
    )
    
    parser.add_argument(
        '--contact',
        type=str,
        required=True,
        help='接触力数据文件路径 (CSV或BIN格式)'
    )
    
    parser.add_argument(
        '--sensor-vibration',
        type=str,
        required=True,
        help='传感器振动数据文件路径 (CSV或BIN格式)'
    )
    
    parser.add_argument(
        '--joint-state',
        type=str,
        required=True,
        help='关节状态数据文件路径 (CSV或BIN格式)'
    )
    
    parser.add_argument(
        '--joint-forces',
        type=str,
        required=True,
        help='关节力数据文件路径 (CSV格式)'
    )
    
    parser.add_argument(
        '--perturbation',
        type=str,
        default=None,
        help='扰动数据文件路径 (CSV或BIN格式，可选)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出结果文件路径前缀 (CSV格式，可选)。将自动生成：'
             '{output}_summary.csv 和 {output}_*_breakdown.csv'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='绘制风险曲线图'
    )
    
    parser.add_argument(
        '--a-thr',
        type=float,
        default=None,
        help='加速度阈值 (m/s²)，用于 risk_acc 计算。如果不指定，使用默认值 20.0'
    )
    
    parser.add_argument(
        '--force-threshold',
        type=float,
        default=400.0,
        help='接触力阈值 (N)，用于提取碰撞时间戳，默认 400.0'
    )
    
    args = parser.parse_args()
    
    # 读取所有数据文件
    print("正在读取数据文件...")
    contact_df = load_data_file(args.contact)
    sensor_vibration_df = load_data_file(args.sensor_vibration)
    joint_state_df = load_data_file(args.joint_state)
    joint_forces_df = load_data_file(args.joint_forces)
    perturbation_df = load_data_file(args.perturbation) if args.perturbation else None
    
    print(f"  接触力数据: {len(contact_df)} 条记录")
    print(f"  传感器振动数据: {len(sensor_vibration_df)} 条记录")
    print(f"  关节状态数据: {len(joint_state_df)} 条记录")
    print(f"  关节力数据: {len(joint_forces_df)} 条记录")
    if perturbation_df is not None:
        print(f"  扰动数据: {len(perturbation_df)} 条记录")
    
    # 准备输出路径
    output_base = None
    if args.output:
        output_base = args.output
        if output_base.endswith('.csv'):
            output_base = output_base[:-4]
    
    # 计算risk
    print("\n正在计算risk...")
    # force_threshold 和 a_thr 由各自的函数管理默认值
    # 如果需要自定义，可以修改 extract_collision_timestamps 和 calculate_acceleration_risk 的调用
    risk_results = calculate_total_fall_risk(
        contact_df,
        sensor_vibration_df,
        joint_state_df,
        joint_forces_df,
        perturbation_df,
        output_base=output_base
    )
    
    # 输出结果
    print("\n=== Risk计算结果 ===")
    print(f"接触力risk: {risk_results['contact_force_risk']:.6f}")
    print(f"加速度risk: {risk_results['acceleration_risk']:.6f}")
    print(f"电机力矩risk: {risk_results['motor_torque_risk']:.6f}")
    print(f"关节力risk: {risk_results['joint_wrench_risk']:.6f}")
    if perturbation_df is not None:
        print(f"扰动risk: {risk_results['perturbation_risk']:.6f}")
    print(f"总risk: {risk_results['total_risk']:.6f}")
    
    # 保存结果到文件
    if args.output:
        # 确保输出路径有正确的扩展名（去掉.csv，我们会在后面加后缀）
        output_base = args.output
        if output_base.endswith('.csv'):
            output_base = output_base[:-4]
        
        # 从输入文件名中提取时间戳（优先使用 joint_state，因为它通常包含完整时间戳）
        filename_timestamp = None
        for file_path in [args.joint_state, args.contact, args.sensor_vibration]:
            if file_path:
                filename_timestamp = extract_timestamp_from_filename(file_path)
                if filename_timestamp:
                    break
        
        # 从原始数据中获取仿真器时间戳（找到第一个出现任何 risk 的时间戳）
        sim_timestamp = None
        breakdowns = risk_results.get('breakdowns', {})
        if 'motor_torque' in breakdowns and 'risk_timestamp' in breakdowns['motor_torque'].columns:
            # 找到第一个有 risk 的关节的时间戳（按 R_joint 排序后取第一个）
            motor_breakdown = breakdowns['motor_torque']
            # 过滤掉 R_joint > 0 的关节
            risk_joints = motor_breakdown[motor_breakdown['R_joint'] > 0]
            if not risk_joints.empty:
                # 取 R_joint 最大的关节的 risk_timestamp
                max_risk_joint = risk_joints.loc[risk_joints['R_joint'].idxmax()]
                sim_timestamp = float(max_risk_joint['risk_timestamp'])
            else:
                # 如果没有 risk，使用第一个时间戳
                if not joint_state_df.empty and 'timestamp' in joint_state_df.columns:
                    sim_timestamp = float(joint_state_df['timestamp'].iloc[0])
        elif not joint_state_df.empty and 'timestamp' in joint_state_df.columns:
            # 如果没有 breakdown，使用第一个时间戳
            sim_timestamp = float(joint_state_df['timestamp'].iloc[0])
        
        # 保存汇总结果（新格式：filename, contact_risk, vibration_risk, motor_risk, jointforce_risk, all_risk）
        summary_dict = {
            'filename': filename_timestamp if filename_timestamp else '',
            'contact_risk': round(risk_results['contact_force_risk'], 4),
            'vibration_risk': round(risk_results['acceleration_risk'], 4),
            'motor_risk': round(risk_results['motor_torque_risk'], 4),
            'jointforce_risk': round(risk_results['joint_wrench_risk'], 4),
            'all_risk': round(risk_results['total_risk'], 4),
        }
        
        summary_df = pd.DataFrame([summary_dict])
        summary_path = f"{output_base}_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\n汇总结果已保存到: {summary_path}")
        
        # motor_risk CSV已经在calculate_motor_torque_risk中保存了
        # 这里只需要添加filename列（如果还没有的话）
        motor_risk_path = f"{output_base}_motor_risk.csv"
        if os.path.exists(motor_risk_path):
            try:
                motor_risk_df = pd.read_csv(motor_risk_path)
                if 'filename' not in motor_risk_df.columns and filename_timestamp:
                    motor_risk_df.insert(0, 'filename', filename_timestamp)
                    # 重新排列列顺序：filename, timestamp, joint_name, ...
                    cols = ['filename', 'timestamp'] + [c for c in motor_risk_df.columns if c not in ['filename', 'timestamp']]
                    motor_risk_df = motor_risk_df[cols]
                    motor_risk_df.to_csv(motor_risk_path, index=False, float_format='%.4f')
                    print(f"motor_risk CSV已更新: {motor_risk_path} ({len(motor_risk_df)} 条记录)")
            except Exception as e:
                print(f"警告: 更新motor_risk CSV失败: {e}")
        
        # 绘制风险曲线
        # motor_risk 绘图（仅当有数据时）
        if args.plot and os.path.exists(motor_risk_path):
            try:
                motor_risk_df = pd.read_csv(motor_risk_path)
                if not motor_risk_df.empty:
                    plot_output_path = f"{output_base}_motor_risk_curves.png"
                    plot_risk_curves(motor_risk_df, plot_output_path)
                else:
                    print("motor_risk 数据为空，跳过绘图")
            except Exception as e:
                print(f"警告: 绘制 motor_risk 曲线失败: {e}")
        
        # acceleration_risk 绘图
        if args.plot:
            acceleration_risk_path = f"{output_base}_acceleration_risk.csv"
            if os.path.exists(acceleration_risk_path):
                try:
                    acceleration_risk_df = pd.read_csv(acceleration_risk_path)
                    plot_output_path = f"{output_base}_acceleration_risk_curves.png"
                    plot_acceleration_risk_curves(acceleration_risk_df, plot_output_path)
                except Exception as e:
                    print(f"警告: 绘制 acceleration_risk 曲线失败: {e}")


if __name__ == '__main__':
    main()
