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


def read_binary_link_kinetic_energy_data(bin_path: str) -> pd.DataFrame:
    """
    读取二进制格式的link_kinetic_energy_data文件
    
    二进制格式：
    - 每条记录：
      - double timestamp
      - int32_t num_bodies
      - 每个body: double[10] (vel_x, vel_y, vel_z, angvel_x, angvel_y, angvel_z, height, linear_KE, angular_KE, PE)
      - double total_linear_KE
      - double total_angular_KE
      - double total_KE
      - double total_PE
      - double total_energy
    """
    data = []
    
    with open(bin_path, 'rb') as f:
        # 读取第一条记录来获取body数量
        first_timestamp_bytes = f.read(8)
        if len(first_timestamp_bytes) < 8:
            return pd.DataFrame()
        first_timestamp = struct.unpack('d', first_timestamp_bytes)[0]
        
        num_bodies_bytes = f.read(4)
        if len(num_bodies_bytes) < 4:
            return pd.DataFrame()
        num_bodies = struct.unpack('i', num_bodies_bytes)[0]
        
        # 回到文件开头重新读取
        f.seek(0)
        
        while True:
            # 读取timestamp
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            timestamp = struct.unpack('d', timestamp_bytes)[0]
            
            # 读取body数量
            num_bodies_bytes = f.read(4)
            if len(num_bodies_bytes) < 4:
                break
            num_bodies = struct.unpack('i', num_bodies_bytes)[0]
            
            row = {'timestamp': timestamp}
            
            # 读取每个body的速度、高度和能量
            for i in range(num_bodies):
                body_data = f.read(80)  # 10 * 8 bytes (6 vel + 1 height + 3 energy)
                if len(body_data) < 80:
                    break
                vals = struct.unpack('10d', body_data)
                row[f'body_{i}_vel_x'] = vals[0]
                row[f'body_{i}_vel_y'] = vals[1]
                row[f'body_{i}_vel_z'] = vals[2]
                row[f'body_{i}_angvel_x'] = vals[3]
                row[f'body_{i}_angvel_y'] = vals[4]
                row[f'body_{i}_angvel_z'] = vals[5]
                row[f'body_{i}_height'] = vals[6]
                row[f'body_{i}_linear_KE'] = vals[7]
                row[f'body_{i}_angular_KE'] = vals[8]
                row[f'body_{i}_PE'] = vals[9]
            
            # 读取能量数据
            energy_bytes = f.read(40)  # 5 * 8 bytes
            if len(energy_bytes) < 40:
                break
            energies = struct.unpack('5d', energy_bytes)
            row['total_linear_KE'] = energies[0]
            row['total_angular_KE'] = energies[1]
            row['total_KE'] = energies[2]
            row['total_PE'] = energies[3]
            row['total_energy'] = energies[4]
            
            data.append(row)
    
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
        elif 'link_kinetic_energy_data' in filename:
            return read_binary_link_kinetic_energy_data(file_path)
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


def extract_collision_nearjointlimit_timestamps(
    time_collision: np.ndarray,
    joint_state_df: pd.DataFrame,
    joint_range_percentile: float = 0.01
) -> np.ndarray:
    """
    从 extract_collision_timestamps 的帧中选择关节位置接近 1% 和 99% 关节范围的时间戳
    
    参数:
        time_collision: 碰撞时间戳数组（从 extract_collision_timestamps 获取）
        joint_state_df: 关节状态数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - joint_*_position: 各关节位置（列名格式为 joint_<index>_position）
        joint_range_percentile: 关节范围百分位，默认 0.01（即 1% 和 99%）
    
    返回:
        time_collision_near_jointlimit: 接近关节限制的碰撞时间戳数组（已排序且去重）
    """
    if len(time_collision) == 0 or joint_state_df.empty:
        return np.array([])
    
    # 检查必要的列
    if 'timestamp' not in joint_state_df.columns:
        print(f"警告: joint_state_df 缺少 'timestamp' 列")
        return np.array([])
    
    # 提取所有关节位置列
    joint_pos_cols = [col for col in joint_state_df.columns if col.startswith('joint_') and col.endswith('_position')]
    
    if len(joint_pos_cols) == 0:
        print(f"警告: joint_state_df 中未找到关节位置列（格式应为 joint_*_position）")
        return np.array([])
    
    # 对于每个关节，计算整个 episode 的范围
    joint_ranges = {}
    for col in joint_pos_cols:
        joint_positions = joint_state_df[col].values
        valid_positions = joint_positions[np.isfinite(joint_positions)]
        if len(valid_positions) > 0:
            pos_min = np.min(valid_positions)
            pos_max = np.max(valid_positions)
            pos_range = pos_max - pos_min
            if pos_range > 0:
                joint_ranges[col] = {
                    'min': pos_min,
                    'max': pos_max,
                    'range': pos_range,
                    'lower_threshold': pos_min + joint_range_percentile * pos_range,  # 1% 位置
                    'upper_threshold': pos_max - joint_range_percentile * pos_range   # 99% 位置
                }
    
    if len(joint_ranges) == 0:
        print(f"警告: 无法计算关节范围")
        return np.array([])
    
    # 获取 joint_state_df 的时间戳
    timestamps = joint_state_df['timestamp'].to_numpy(dtype=float)
    
    # 对于每个碰撞时间戳，检查是否有任何关节接近限制
    time_collision_near_jointlimit = []
    tolerance = 0.001  # 1ms 容差
    
    for t_collision in time_collision:
        # 找到时间戳在容差范围内的数据
        time_mask = np.abs(timestamps - t_collision) <= tolerance
        if not np.any(time_mask):
            continue
        
        # 获取该时间戳的关节位置
        idx = np.where(time_mask)[0][0]
        
        # 检查是否有任何关节接近限制（1% 或 99%）
        near_limit = False
        for col, range_info in joint_ranges.items():
            joint_pos = joint_state_df.iloc[idx][col]
            if np.isfinite(joint_pos):
                # 检查是否接近下限（1%）或上限（99%）
                if (joint_pos <= range_info['lower_threshold']) or (joint_pos >= range_info['upper_threshold']):
                    near_limit = True
                    break
        
        if near_limit:
            time_collision_near_jointlimit.append(t_collision)
    
    # 去重并排序
    if len(time_collision_near_jointlimit) > 0:
        time_collision_near_jointlimit = np.unique(time_collision_near_jointlimit)
        time_collision_near_jointlimit = np.sort(time_collision_near_jointlimit)
    
    return np.array(time_collision_near_jointlimit)


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
    risk_acc = sum_{b in {torso, head}} ∫[0 to 20ms] ReLU((||a_b(t)|| - a_thr) / a_thr) dt
    
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
        
        # 计算 ReLU((||a|| - a_thr) / a_thr)
        # 归一化风险值，表示超过阈值的相对比例
        torso_risk = relu((torso_acc_window - a_thr) / a_thr)
        head_risk = relu((head_acc_window - a_thr) / a_thr)
        
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


def calculate_contact_force_risk(
    contact_df: pd.DataFrame,
    time_collision: np.ndarray,
    f_thr: float = 1000.0,
    f_thr_vulnerable: float = 400.0,
    dt: float = 0.002,
    save_frame_by_frame_path: Optional[str] = None
) -> Tuple[float, pd.DataFrame]:
    """
    计算接触力风险指标 risk_force
    
    风险定义为：
    仅在碰撞时间戳（time_collision）时计算（force 只在碰撞时产生，不需要时间窗口）
    每行计算 r_link = ReLU(||f(t)|| - f_thr) / f_thr
    每个 link 求和得到 R_link = sum_t r_link(t) * dt（episode 级积分，必须乘采样间隔 dt）
    所有 link 的 R_link 求和得到 R_force = sum_link R_link
    
    参数:
        contact_df: 接触力数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - body1_name: 第一个物体名称
            - body2_name: 第二个物体名称
            - force_magnitude: 力的大小
        time_collision: 碰撞时间戳数组（从 extract_collision_timestamps 获取）
        f_thr: 力阈值（N），默认 1000.0（用于一般关节）
        f_thr_vulnerable: 脆弱关节的力阈值（N），默认 400.0（用于 HEAD 和 ELBOW_END）
        dt: 采样时间间隔（秒），默认 0.002。用于 episode 级时间积分
        save_frame_by_frame_path: 如果指定，保存逐帧数据到 CSV
    
    返回:
        R_force: float，总接触力风险（已乘以 dt）
        breakdown: pd.DataFrame，按 link 的风险明细（已乘以 dt）
    """
    if contact_df.empty or len(time_collision) == 0:
        empty_breakdown = pd.DataFrame(columns=["link_name", "R_link"])
        return 0.0, empty_breakdown
    
    # 验证 dt 值
    if dt <= 0 or not np.isfinite(dt):
        print(f"警告: 无效的 dt 值: {dt}，使用默认值 0.002")
        dt = 0.002
    
    # 检查必要的列
    required_cols = ['timestamp', 'body1_name', 'body2_name', 'force_magnitude']
    missing_cols = [col for col in required_cols if col not in contact_df.columns]
    if missing_cols:
        print(f"警告: contact_df 缺少必要的列: {missing_cols}")
        empty_breakdown = pd.DataFrame(columns=["link_name", "R_link"])
        return 0.0, empty_breakdown
    
    # 判断哪个是机器人的 link（通常以 "LINK_" 开头，或者不是 "world"）
    def get_robot_link(body1: str, body2: str) -> Optional[str]:
        """判断哪个是机器人的 link，返回 link 名称，如果都不是则返回 None"""
        body1_str = str(body1).strip()
        body2_str = str(body2).strip()
        
        # 判断 body1 是否是机器人 link
        is_body1_robot = (
            body1_str.startswith("LINK_") or
            (body1_str.lower() != "world" and body1_str.lower() != "ground" and body1_str != "")
        )
        
        # 判断 body2 是否是机器人 link
        is_body2_robot = (
            body2_str.startswith("LINK_") or
            (body2_str.lower() != "world" and body2_str.lower() != "ground" and body2_str != "")
        )
        
        # 优先返回以 LINK_ 开头的
        if body1_str.startswith("LINK_"):
            return body1_str
        elif body2_str.startswith("LINK_"):
            return body2_str
        elif is_body1_robot and not is_body2_robot:
            return body1_str
        elif is_body2_robot and not is_body1_robot:
            return body2_str
        elif is_body1_robot and is_body2_robot:
            # 如果两个都是机器人 link，返回 body2（通常是碰撞的 link）
            return body2_str
        else:
            return None
    
    # 计算每行的风险值
    contact_df = contact_df.copy()
    contact_df['robot_link'] = contact_df.apply(
        lambda row: get_robot_link(row['body1_name'], row['body2_name']),
        axis=1
    )
    
    # 过滤掉没有机器人 link 的行
    contact_df = contact_df[contact_df['robot_link'].notna()].copy()
    
    if contact_df.empty:
        empty_breakdown = pd.DataFrame(columns=["link_name", "R_link"])
        return 0.0, empty_breakdown
    
    # 获取时间戳
    timestamps = contact_df['timestamp'].to_numpy(dtype=float)
    
    # 直接使用 time_collision 中的时间戳筛选数据（force 只在碰撞时产生，不需要时间窗口）
    # 由于时间戳可能有微小误差，使用一个很小的容差（1ms）来匹配
    tolerance = 0.001  # 1ms 容差
    
    # 筛选出时间戳在 time_collision 中的数据
    mask = np.zeros(len(contact_df), dtype=bool)
    for t_collision in time_collision:
        # 找到时间戳在容差范围内的数据
        time_mask = np.abs(timestamps - t_collision) <= tolerance
        mask = mask | time_mask
    
    # 只保留在碰撞时间戳的数据
    contact_df_collision = contact_df[mask].copy()
    
    if contact_df_collision.empty:
        empty_breakdown = pd.DataFrame(columns=["link_name", "R_link"])
        return 0.0, empty_breakdown
    
    # 定义不同关节的阈值（关键词 -> 阈值）
    # HEAD, ELBOW_END: 400N（最脆弱）
    # TORSO: 700N（中等脆弱）
    # 其他: 1000N（默认）
    vulnerable_thresholds = {
        'HEAD': f_thr_vulnerable,       # 400
        'ELBOW_END': f_thr_vulnerable,  # 400
        'TORSO': 700.0,                 # 700
    }
    
    def get_threshold_for_link(link_name: str) -> float:
        """根据 link 名称返回对应的阈值"""
        link_upper = str(link_name).upper()
        for keyword, threshold in vulnerable_thresholds.items():
            if keyword in link_upper:
                return threshold
        return f_thr
    
    # 为每行计算对应的阈值
    contact_df_collision['f_threshold'] = contact_df_collision['robot_link'].apply(get_threshold_for_link)
    
    # 计算每行的风险值 r_link = ReLU(||f(t)|| - f_thr) / f_thr
    # 使用每行对应的阈值
    contact_df_collision['r_link'] = contact_df_collision.apply(
        lambda row: relu((row['force_magnitude'] - row['f_threshold']) / row['f_threshold']),
        axis=1
    )
    
    # 保存逐帧数据（如果指定了路径）
    if save_frame_by_frame_path is not None:
        try:
            frame_data = []
            for _, row in contact_df_collision.iterrows():
                if row['r_link'] > 0:  # 只保存有风险的数据
                    frame_data.append({
                        'timestamp': round(float(row['timestamp']), 4),
                        'link_name': str(row['robot_link']),
                        'force_magnitude': round(float(row['force_magnitude']), 4),
                        'f_threshold': round(float(row['f_threshold']), 1),
                        'r_link': round(float(row['r_link']), 4),
                    })
            
            if frame_data:
                frame_df = pd.DataFrame(frame_data)
                frame_df = frame_df.sort_values('timestamp').reset_index(drop=True)
                frame_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"contact force 逐帧风险数据已保存到: {save_frame_by_frame_path} ({len(frame_df)} 条记录)")
            else:
                empty_df = pd.DataFrame(columns=['timestamp', 'link_name', 'force_magnitude', 'f_threshold', 'r_link'])
                empty_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"contact force 逐帧风险数据已保存到: {save_frame_by_frame_path} (无超阈值数据)")
        except Exception as e:
            print(f"警告: 保存 contact force 逐帧风险数据到 '{save_frame_by_frame_path}' 失败: {e}")
    
    # 按 link 分组求和得到 R_link（episode 级积分，必须乘采样间隔 dt）
    link_risks = contact_df_collision.groupby('robot_link')['r_link'].sum().reset_index()
    link_risks.columns = ['link_name', 'R_link']
    
    # 乘以 dt 进行时间积分
    link_risks['R_link'] = link_risks['R_link'] * dt
    
    # 四舍五入到4位小数
    link_risks['R_link'] = link_risks['R_link'].round(4)
    
    # 按 R_link 降序排序
    link_risks = link_risks.sort_values('R_link', ascending=False).reset_index(drop=True)
    
    # 计算总风险 R_force（已乘以 dt）
    R_force = float(link_risks['R_link'].sum())
    if not np.isfinite(R_force):
        R_force = 0.0
    R_force = round(R_force, 4)
    
    return R_force, link_risks


def calculate_motor_torque_risk(
    joint_state_df: pd.DataFrame,
    dt: Optional[float] = 0.002,
    mode: str = "const",
    wp: float = 1.0,
    wr: float = 1.0,
    tau_max_const: float = 52.0,  # 注意：const 模式下会根据关节类型自动选择 164.0 或 52.0
    P0_const: float = 500.0,
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


def plot_motor_torque_risk_curves(frame_by_frame_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制电机扭矩风险曲线
    
    参数:
        frame_by_frame_df: 逐帧电机扭矩风险数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - joint_name: 关节名称
            - r_peak: 峰值扭矩风险
            - r_regen: 再生功率风险
            - r_joint: 关节总风险
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
    # 使用绝对时间戳（不进行归一化）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    total_risk_by_time = frame_by_frame_df_sorted.groupby('timestamp')['r_joint'].sum().reset_index()
    total_risk_by_time = total_risk_by_time.sort_values('timestamp')
    
    # 获取所有时间戳的最小值和最大值，用于设置 x 轴范围
    min_timestamp = frame_by_frame_df_sorted['timestamp'].min()
    max_timestamp = frame_by_frame_df_sorted['timestamp'].max()
    # 添加一些边距
    timestamp_range = max_timestamp - min_timestamp
    x_margin = max(0.01, timestamp_range * 0.02)  # 2% 边距，至少 0.01 秒
    
    ax_total = axes[0]
    # 直接使用绝对时间戳，不进行任何偏移或归一化
    ax_total.plot(total_risk_by_time['timestamp'], total_risk_by_time['r_joint'], 
                  'k-', label='Total Motor Risk (sum of r_joint)', linewidth=2)
    # 设置 x 轴范围从数据的最小时间戳开始
    ax_total.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
    ax_total.set_ylabel('Total Risk', fontsize=10)
    ax_total.legend(loc='upper right')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Motor Risk Over Time', fontsize=12)
    
    # 每个关节的子图
    for i, joint_name in enumerate(joints):
        joint_data = frame_by_frame_df[frame_by_frame_df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        # 直接使用绝对时间戳，不进行任何偏移或归一化
        ax.plot(joint_data['timestamp'], joint_data['r_peak'], 'r-', label='r_peak', linewidth=1.5)
        ax.plot(joint_data['timestamp'], joint_data['r_regen'], 'b-', label='r_regen', linewidth=1.5)
        ax.plot(joint_data['timestamp'], joint_data['r_joint'], 'g-', label='r_joint', linewidth=2)
        # 设置 x 轴范围从数据的最小时间戳开始
        ax.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
        
        ax.set_ylabel(f'Joint {joint_name} Risk', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Joint {joint_name} Risk Over Time', fontsize=12)
    
    axes[-1].set_xlabel('Time (s) - Absolute Timestamp', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"电机扭矩风险曲线图已保存到: {output_path}")
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
    
    # 第一个子图：总 risk（瞬时风险和累计风险）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    
    # 计算每个时刻的瞬时风险总和（所有 link 的 risk_value 之和）
    instant_risk_by_time = frame_by_frame_df_sorted.groupby('timestamp')['risk_value'].sum().reset_index()
    instant_risk_by_time = instant_risk_by_time.sort_values('timestamp')
    
    # 绘制累计风险随时间的变化（取每个时刻的最大 cumulative_risk）
    cumulative_by_time = frame_by_frame_df_sorted.groupby('timestamp')['cumulative_risk'].max().reset_index()
    cumulative_by_time = cumulative_by_time.sort_values('timestamp')
    
    ax_total = axes[0]
    # 左 y 轴：瞬时风险
    line1, = ax_total.plot(instant_risk_by_time['timestamp'], instant_risk_by_time['risk_value'], 
                           'b-', label='Instantaneous Risk (sum of risk_value)', linewidth=1.5)
    ax_total.set_ylabel('Instantaneous Risk', fontsize=10, color='blue')
    ax_total.tick_params(axis='y', labelcolor='blue')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Acceleration Risk Over Time', fontsize=12)
    
    # 右 y 轴：累计风险
    ax_total_twin = ax_total.twinx()
    line2, = ax_total_twin.plot(cumulative_by_time['timestamp'], cumulative_by_time['cumulative_risk'], 
                                'r-', label='Cumulative Risk', linewidth=2)
    ax_total_twin.set_ylabel('Cumulative Risk', fontsize=10, color='red')
    ax_total_twin.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_total.legend(lines, labels, loc='upper left')
    
    # 每个 link 的子图（使用双 y 轴，避免数值范围差异导致风险值看起来像0）
    for i, link_name in enumerate(links):
        link_data = frame_by_frame_df[frame_by_frame_df['link_name'] == link_name].copy()
        link_data = link_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        # 左 y 轴：加速度模长
        line1, = ax.plot(link_data['timestamp'], link_data['acc_magnitude'], 'b-', 
                         label='Acceleration (m/s²)', linewidth=1.5)
        ax.set_ylabel(f'Acceleration (m/s²)', fontsize=10, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{link_name.capitalize()} Acceleration Risk Over Time', fontsize=12)
        
        # 右 y 轴：瞬时风险值（risk_value）和累计风险（cumulative_risk）
        ax_twin = ax.twinx()
        line2, = ax_twin.plot(link_data['timestamp'], link_data['risk_value'], 'r-', 
                              label='Instant Risk (risk_value)', linewidth=2)
        line3, = ax_twin.plot(link_data['timestamp'], link_data['cumulative_risk'], 'g--', 
                              label='Cumulative Risk', linewidth=1.5, alpha=0.7)
        ax_twin.set_ylabel('Risk Value', fontsize=10, color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        # 合并图例
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"加速度风险曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_acceleration_curves(
    sensor_vibration_df: pd.DataFrame,
    time_collision: Optional[np.ndarray] = None,
    window_ms: float = 20.0,
    a_thr: float = 30.0,
    output_path: Optional[str] = None
) -> None:
    """
    单独绘制加速度曲线（torso / head 模长），便于观察护具对作用时间与峰值的影响。
    
    参数:
        sensor_vibration_df: 传感器振动数据 DataFrame，应包含列：
            - timestamp
            - base_link_lin_acc_x, base_link_lin_acc_y, base_link_lin_acc_z
            - head_lin_acc_x, head_lin_acc_y, head_lin_acc_z
        time_collision: 碰撞时间戳数组（可选），若提供则画垂线并标出 20ms 窗口
        window_ms: 时间窗（ms），用于标出窗口范围，默认 20.0
        a_thr: 加速度阈值（m/s^2），画水平参考线，默认 30.0
        output_path: 图片保存路径（可选）
    """
    if sensor_vibration_df.empty:
        print("警告: sensor_vibration_df 为空，无法绘图")
        return
    
    required = [
        'timestamp',
        'base_link_lin_acc_x', 'base_link_lin_acc_y', 'base_link_lin_acc_z',
        'head_lin_acc_x', 'head_lin_acc_y', 'head_lin_acc_z'
    ]
    missing = [c for c in required if c not in sensor_vibration_df.columns]
    if missing:
        print(f"警告: 缺少列 {missing}，无法绘图")
        return
    
    ts = sensor_vibration_df['timestamp'].to_numpy(dtype=float)
    torso_acc = np.sqrt(
        sensor_vibration_df['base_link_lin_acc_x'].to_numpy(dtype=float)**2 +
        sensor_vibration_df['base_link_lin_acc_y'].to_numpy(dtype=float)**2 +
        sensor_vibration_df['base_link_lin_acc_z'].to_numpy(dtype=float)**2
    )
    head_acc = np.sqrt(
        sensor_vibration_df['head_lin_acc_x'].to_numpy(dtype=float)**2 +
        sensor_vibration_df['head_lin_acc_y'].to_numpy(dtype=float)**2 +
        sensor_vibration_df['head_lin_acc_z'].to_numpy(dtype=float)**2
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    window_s = window_ms / 1000.0
    
    for ax, acc, name in zip(axes, [torso_acc, head_acc], ['Torso (base_link)', 'Head']):
        ax.plot(ts, acc, 'b-', label=f'||a|| (m/s^2)', linewidth=1.2)
        ax.axhline(y=a_thr, color='orange', linestyle='--', linewidth=1, label=f'threshold={a_thr}')
        ax.set_ylabel(f'{name} acc (m/s^2)', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if time_collision is not None and len(time_collision) > 0:
            for t in time_collision:
                ax.axvline(x=t, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
                ax.axvspan(t, t + window_s, alpha=0.08, color='red')
            from matplotlib.lines import Line2D
            h, l = ax.get_legend_handles_labels()
            h.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5))
            l.append('collision')
            h.append(Line2D([0], [0], color='red', alpha=0.3, linewidth=8))
            l.append(f'{window_ms}ms window')
            ax.legend(handles=h, labels=l, loc='upper right')
    
    axes[0].set_title('Acceleration magnitude (for vibration risk: integral over 20ms window)', fontsize=12)
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"加速度曲线图已保存到: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_contact_force_curves(
    contact_df: pd.DataFrame,
    time_collision: Optional[np.ndarray] = None,
    f_thr: float = 1000.0,
    output_path: Optional[str] = None
) -> None:
    """
    单独绘制接触力曲线（每时刻最大接触力），便于观察护具对力峰值与时间的影响。
    
    参数:
        contact_df: 接触力数据 DataFrame，应包含列：timestamp, force_magnitude
        time_collision: 碰撞时间戳数组（可选），若提供则画垂线
        f_thr: 力阈值（N），画水平参考线，默认 1000.0
        output_path: 图片保存路径（可选）
    """
    if contact_df.empty:
        print("警告: contact_df 为空，无法绘图")
        return
    if 'timestamp' not in contact_df.columns or 'force_magnitude' not in contact_df.columns:
        print("警告: contact_df 缺少 timestamp 或 force_magnitude 列，无法绘图")
        return
    
    # 每时刻取最大接触力（同一时刻可能有多接触点）
    by_time = contact_df.groupby('timestamp')['force_magnitude'].max().reset_index()
    by_time = by_time.sort_values('timestamp')
    ts = by_time['timestamp'].to_numpy(dtype=float)
    force = by_time['force_magnitude'].to_numpy(dtype=float)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(ts, force, 'b-', label='max contact force (N)', linewidth=1.2)
    ax.axhline(y=f_thr, color='orange', linestyle='--', linewidth=1, label=f'threshold={f_thr} N')
    ax.set_ylabel('Force (N)', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Contact force magnitude (max per timestamp)', fontsize=12)
    
    if time_collision is not None and len(time_collision) > 0:
        for t in time_collision:
            ax.axvline(x=t, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5))
        l.append('collision')
        ax.legend(handles=h, labels=l, loc='upper right')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"接触力曲线图已保存到: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_contact_force_risk_curves(frame_by_frame_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制接触力风险曲线
    
    参数:
        frame_by_frame_df: 逐帧接触力风险数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - link_name: link 名称
            - force_magnitude: 力的大小
            - r_link: 风险值
        output_path: 图片保存路径（可选）
    """
    if frame_by_frame_df.empty:
        print("警告: 接触力风险逐帧数据为空，无法绘图")
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
    
    # 第一个子图：总 risk（瞬时风险和累计风险）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    total_risk_by_time = frame_by_frame_df_sorted.groupby('timestamp')['r_link'].sum().reset_index()
    total_risk_by_time = total_risk_by_time.sort_values('timestamp')
    
    # 计算累计风险
    total_risk_by_time['cumulative_risk'] = total_risk_by_time['r_link'].cumsum()
    
    # 获取所有时间戳的最小值和最大值，用于设置 x 轴范围
    min_timestamp = frame_by_frame_df_sorted['timestamp'].min()
    max_timestamp = frame_by_frame_df_sorted['timestamp'].max()
    timestamp_range = max_timestamp - min_timestamp
    x_margin = max(0.01, timestamp_range * 0.02)  # 2% 边距，至少 0.01 秒
    
    ax_total = axes[0]
    # 左 y 轴：瞬时风险
    line1, = ax_total.plot(total_risk_by_time['timestamp'], total_risk_by_time['r_link'], 
                           'b-', label='Instantaneous Risk (sum of r_link)', linewidth=1.5)
    ax_total.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
    ax_total.set_ylabel('Instantaneous Risk', fontsize=10, color='blue')
    ax_total.tick_params(axis='y', labelcolor='blue')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Contact Force Risk Over Time', fontsize=12)
    
    # 右 y 轴：累计风险
    ax_total_twin = ax_total.twinx()
    line2, = ax_total_twin.plot(total_risk_by_time['timestamp'], total_risk_by_time['cumulative_risk'], 
                                'r-', label='Cumulative Risk', linewidth=2)
    ax_total_twin.set_ylabel('Cumulative Risk', fontsize=10, color='red')
    ax_total_twin.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_total.legend(lines, labels, loc='upper left')
    
    # 每个 link 的子图（使用双 y 轴，避免数值范围差异导致风险值看起来像0）
    for i, link_name in enumerate(links):
        link_data = frame_by_frame_df[frame_by_frame_df['link_name'] == link_name].copy()
        link_data = link_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        # 左 y 轴：force_magnitude
        line1, = ax.plot(link_data['timestamp'], link_data['force_magnitude'], 'b-', 
                         label='Force Magnitude (N)', linewidth=1.5)
        ax.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
        ax.set_ylabel(f'Force (N)', fontsize=10, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{link_name} Contact Force Risk Over Time', fontsize=12)
        
        # 右 y 轴：r_link（风险值有独立的 y 轴，不会被压扁）
        ax_twin = ax.twinx()
        line2, = ax_twin.plot(link_data['timestamp'], link_data['r_link'], 'r-', 
                              label='Risk Value (r_link)', linewidth=2)
        ax_twin.set_ylabel('Risk Value', fontsize=10, color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        # 合并图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    axes[-1].set_xlabel('Time (s) - Absolute Timestamp', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"接触力风险曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_joint_wrench_risk_curves(frame_by_frame_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制关节力/力矩风险曲线
    
    参数:
        frame_by_frame_df: 逐帧关节力风险数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - joint_name: 关节名称
            - F_axial_mag: 轴向力大小
            - F_shear_mag: 剪切力大小
            - M_torsion_mag: 扭转力矩大小
            - M_bend_mag: 弯曲力矩大小
            - r1, r2, r3, r4: 各风险项
            - r_j_joint: 合成风险值
        output_path: 图片保存路径（可选）
    """
    if frame_by_frame_df.empty:
        print("警告: 关节力风险逐帧数据为空，无法绘图")
        return
    
    # 按关节分组
    joints = frame_by_frame_df['joint_name'].unique()
    n_joints = len(joints)
    
    if n_joints == 0:
        return
    
    # 创建子图：第一个子图显示总 risk，后面每个关节一个子图
    fig, axes = plt.subplots(n_joints + 1, 1, figsize=(14, 4 * (n_joints + 1)), sharex=True)
    if n_joints == 0:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # 第一个子图：总 risk（所有关节的 r_j_joint 总和）
    frame_by_frame_df_sorted = frame_by_frame_df.sort_values('timestamp')
    total_risk_by_time = frame_by_frame_df_sorted.groupby('timestamp')['r_j_joint'].sum().reset_index()
    total_risk_by_time = total_risk_by_time.sort_values('timestamp')
    
    # 获取所有时间戳的最小值和最大值，用于设置 x 轴范围
    min_timestamp = frame_by_frame_df_sorted['timestamp'].min()
    max_timestamp = frame_by_frame_df_sorted['timestamp'].max()
    timestamp_range = max_timestamp - min_timestamp
    x_margin = max(0.01, timestamp_range * 0.02)  # 2% 边距，至少 0.01 秒
    
    ax_total = axes[0]
    ax_total.plot(total_risk_by_time['timestamp'], total_risk_by_time['r_j_joint'], 
                  'k-', label='Total Joint Wrench Risk (sum of r_j_joint)', linewidth=2)
    ax_total.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
    ax_total.set_ylabel('Total Risk', fontsize=10)
    ax_total.legend(loc='upper right')
    ax_total.grid(True, alpha=0.3)
    ax_total.set_title('Total Joint Wrench Risk Over Time', fontsize=12)
    
    # 每个关节的子图
    for i, joint_name in enumerate(joints):
        joint_data = frame_by_frame_df[frame_by_frame_df['joint_name'] == joint_name].copy()
        joint_data = joint_data.sort_values('timestamp')
        
        ax = axes[i + 1]
        # 绘制各风险项（不显示 r_j_joint）
        ax.plot(joint_data['timestamp'], joint_data['r1'], 'r-', label='r1 (F_axial)', linewidth=1.5, alpha=0.7)
        ax.plot(joint_data['timestamp'], joint_data['r2'], 'b-', label='r2 (F_shear)', linewidth=1.5, alpha=0.7)
        ax.plot(joint_data['timestamp'], joint_data['r3'], 'g-', label='r3 (M_torsion)', linewidth=1.5, alpha=0.7)
        ax.plot(joint_data['timestamp'], joint_data['r4'], 'm-', label='r4 (M_bend)', linewidth=1.5, alpha=0.7)
        
        ax.set_xlim(min_timestamp - x_margin, max_timestamp + x_margin)
        ax.set_ylabel(f'{joint_name} Risk', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{joint_name} Joint Wrench Risk Over Time', fontsize=12)
    
    axes[-1].set_xlabel('Time (s) - Absolute Timestamp', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"关节力风险曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_link_group(body_name: str) -> str:
    """
    根据 body 名称返回分组名称
    分组规则：
    - Body: base, torso, head
    - Shoulder: 包含 shoulder
    - Elbow: 包含 elbow
    - Hip: 包含 hip
    - Knee: 包含 knee 或 ankle
    - Skip: foot（质量几乎为零，忽略）、world
    """
    name_lower = body_name.lower()
    
    # 忽略 foot（质量几乎为零）和 world
    if 'foot' in name_lower or name_lower == 'world':
        return None  # 返回 None 表示跳过
    
    # Body 组: base, torso, head
    if any(keyword in name_lower for keyword in ['base', 'torso', 'head']):
        return 'Body'
    
    # Shoulder 组
    if 'shoulder' in name_lower:
        return 'Shoulder'
    
    # Elbow 组
    if 'elbow' in name_lower:
        return 'Elbow'
    
    # Hip 组
    if 'hip' in name_lower:
        return 'Hip'
    
    # Knee 组（包括 ankle）
    if 'knee' in name_lower or 'ankle' in name_lower:
        return 'Knee'
    
    # 其他归入 Other
    return 'Other'


def plot_link_energy_curves(energy_df: pd.DataFrame, output_path: Optional[str] = None,
                           contact_df: Optional[pd.DataFrame] = None):
    """
    绘制link能量曲线（动能、势能、总能量）
    
    参数:
        energy_df: link动能数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - 每个body的速度列: {body_name}_vel_x/y/z, {body_name}_angvel_x/y/z
            - total_linear_KE: 总线动能
            - total_angular_KE: 总角动能
            - total_KE: 总动能
            - total_PE: 总重力势能
            - total_energy: 总机械能
        output_path: 图片保存路径（可选）
        contact_df: 接触力数据 DataFrame（可选），应包含 timestamp, force_magnitude。
                   若提供，第一个子图将绘制 Total PE + Total contact force，并从 0 时刻开始。
    """
    if energy_df.empty:
        print("警告: link能量数据为空，无法绑图")
        return
    
    # 检查必要的总能量列
    required_cols = ['timestamp', 'total_linear_KE', 'total_angular_KE', 'total_KE', 'total_PE', 'total_energy']
    missing_cols = [col for col in required_cols if col not in energy_df.columns]
    if missing_cols:
        print(f"警告: link能量数据缺少必要的列: {missing_cols}")
        return
    
    # 按时间戳排序
    energy_df = energy_df.sort_values('timestamp').reset_index(drop=True)
    timestamps = energy_df['timestamp'].values
    min_t = timestamps.min()
    max_t = timestamps.max()
    t_range = max_t - min_t
    x_margin = max(0.01, t_range * 0.02)
    # 从 0 时刻开始的相对时间，t0 为两者最早时间
    t0 = min_t
    contact_by_time = None   # 总图用：每帧 contact force 最大值
    if contact_df is not None and not contact_df.empty and 'timestamp' in contact_df.columns and 'force_magnitude' in contact_df.columns:
        # 总图：每帧取最大值
        contact_by_time = contact_df.groupby('timestamp')['force_magnitude'].max().reset_index()
        contact_by_time.columns = ['timestamp', 'max_contact_force']
        t0 = min(t0, contact_by_time['timestamp'].min())
    
    # 提取所有 body 名称（从列名中提取）
    # 列名格式: {body_name}_vel_x, {body_name}_vel_y, {body_name}_vel_z, 
    #           {body_name}_angvel_x, {body_name}_angvel_y, {body_name}_angvel_z
    body_names = set()
    for col in energy_df.columns:
        if col.endswith('_vel_x'):
            body_name = col[:-6]  # 去掉 '_vel_x'
            body_names.add(body_name)
    body_names = sorted(body_names)
    
    print(f"  找到 {len(body_names)} 个 body: {body_names}")
    
    # 按组分类 body（跳过 foot 等质量为零的 link）
    group_bodies = {}  # group_name -> [body_names]
    skipped_bodies = []
    for body_name in body_names:
        group = get_link_group(body_name)
        if group is None:
            skipped_bodies.append(body_name)
            continue
        if group not in group_bodies:
            group_bodies[group] = []
        group_bodies[group].append(body_name)
    
    if skipped_bodies:
        print(f"  跳过质量为零的 body: {skipped_bodies}")
    
    # 定义分组顺序
    group_order = ['Body', 'Shoulder', 'Elbow', 'Hip', 'Knee', 'Other']
    # 过滤掉没有数据的组
    groups = [g for g in group_order if g in group_bodies]
    
    print(f"  分组结果:")
    for g in groups:
        print(f"    {g}: {group_bodies[g]}")
    
    # 各 body 各自 contact force（每帧该 body 参与接触时的最大力），供各部位子图使用
    per_body_contact = {}
    if contact_df is not None and not contact_df.empty and 'body1_name' in contact_df.columns and 'body2_name' in contact_df.columns and 'force_magnitude' in contact_df.columns:
        for body_name in body_names:
            mask = (contact_df['body1_name'] == body_name) | (contact_df['body2_name'] == body_name)
            if mask.any():
                df_b = contact_df.loc[mask].groupby('timestamp')['force_magnitude'].max().reset_index()
                per_body_contact[body_name] = df_b
    
    # 计算每组的能量（KE 和 PE）
    # 优先使用 linear_KE、angular_KE、PE 列，如果不存在则用 v^2 近似
    group_linear_ke = {}   # group_name -> array of linear KE for each timestamp
    group_angular_ke = {}  # group_name -> array of angular KE for each timestamp
    group_pe = {}          # group_name -> array of PE for each timestamp
    use_real_energy = False    # 是否使用真正的能量列
    
    # 检查是否有能量列
    sample_body = body_names[0] if body_names else None
    if sample_body and f'{sample_body}_linear_KE' in energy_df.columns:
        use_real_energy = True
        has_pe = f'{sample_body}_PE' in energy_df.columns
        print(f"  使用真正的能量数据 (linear_KE, angular_KE" + (", PE)" if has_pe else ")"))
    else:
        has_pe = False
        print(f"  未找到能量列，使用 v² 近似")
    
    for group in groups:
        linear_ke_sum = np.zeros(len(energy_df))
        angular_ke_sum = np.zeros(len(energy_df))
        pe_sum = np.zeros(len(energy_df))
        for body_name in group_bodies[group]:
            if use_real_energy:
                # 使用真正的能量列
                linear_ke_col = f'{body_name}_linear_KE'
                angular_ke_col = f'{body_name}_angular_KE'
                pe_col = f'{body_name}_PE'
                if linear_ke_col in energy_df.columns:
                    linear_ke_sum += energy_df[linear_ke_col].values
                if angular_ke_col in energy_df.columns:
                    angular_ke_sum += energy_df[angular_ke_col].values
                if pe_col in energy_df.columns:
                    pe_sum += energy_df[pe_col].values
            else:
                # 使用 v^2 近似
                vel_x_col = f'{body_name}_vel_x'
                vel_y_col = f'{body_name}_vel_y'
                vel_z_col = f'{body_name}_vel_z'
                if all(col in energy_df.columns for col in [vel_x_col, vel_y_col, vel_z_col]):
                    vel_x = energy_df[vel_x_col].values
                    vel_y = energy_df[vel_y_col].values
                    vel_z = energy_df[vel_z_col].values
                    linear_ke_sum += vel_x**2 + vel_y**2 + vel_z**2  # 近似
        group_linear_ke[group] = linear_ke_sum
        group_angular_ke[group] = angular_ke_sum
        group_pe[group] = pe_sum
    
    # 创建子图：len(groups) 个分组图 + 1个总能量图（总图放最后以便显示 Time 坐标轴）
    n_groups = len(groups)
    n_subplots = 1 + n_groups
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 3.5 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    # 定义颜色
    colors = {
        'Body': '#E74C3C',      # 红色
        'Shoulder': '#3498DB',  # 蓝色
        'Elbow': '#2ECC71',     # 绿色
        'Hip': '#9B59B6',       # 紫色
        'Knee': '#F39C12',      # 橙色
        'Other': '#95A5A6',     # 灰色
    }
    
    # 相对时间（从 0 开始），供所有子图统一 x 轴
    t_rel = timestamps - t0
    x_max_rel = (max_t - t0) + x_margin

    # ===== 子图 1~N：各分组 PE（求和）+ contact force（各 body 各自一条曲线）=====
    import matplotlib.cm as mcm
    for i, group in enumerate(groups):
        ax = axes[i]
        color = colors.get(group, '#333333')
        pe = group_pe[group]
        # 左 Y：该组 PE 求和
        ax.plot(t_rel, pe, color='#1ABC9C', linewidth=1.5, label=f'PE (sum)', linestyle='-')
        ax.set_ylabel('PE (J)', fontsize=11, color='#1ABC9C')
        ax.tick_params(axis='y', labelcolor='#1ABC9C')
        ax.set_xlim(0, x_max_rel)
        ax.grid(True, alpha=0.3)
        # 右 Y：该组内每个 body 的 contact force 各自一条曲线（不求和）
        ax2 = ax.twinx()
        bodies_in_group = group_bodies[group]
        cmap = mcm.get_cmap('tab10')
        for j, body_name in enumerate(bodies_in_group):
            if body_name not in per_body_contact:
                continue
            df_b = per_body_contact[body_name]
            t_rel_b = df_b['timestamp'].values - t0
            ax2.plot(t_rel_b, df_b['force_magnitude'].values, color=cmap(j % 10), linewidth=1.0, label=body_name, alpha=0.9)
        ax2.set_ylabel('Contact force (N)', fontsize=11, color='#333333')
        ax2.tick_params(axis='y', labelcolor='#333333')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        body_list = ', '.join(group_bodies[group])
        ax.set_title(f'{group} Group: PE (sum) + Contact force (per link): {body_list}', fontsize=10)
    
    # ===== 最后一个子图：Total PE + Contact force，从 0 时刻开始，带 Time 坐标轴 =====
    ax_total = axes[n_groups]
    t_rel_energy = timestamps - t0
    ax_total.plot(t_rel_energy, energy_df['total_PE'], 'b-', label='Total PE', linewidth=1.5)
    ax_total.set_xlim(0, x_max_rel)
    ax_total.set_ylabel('Total PE (J)', fontsize=11)
    if contact_by_time is not None:
        t_rel_contact = contact_by_time['timestamp'].values - t0
        ax_force = ax_total.twinx()
        ax_force.plot(t_rel_contact, contact_by_time['max_contact_force'].values, 'r-', label='Contact force (max per frame)', linewidth=1.5)
        ax_force.set_ylabel('Contact force (N)', fontsize=11, color='r')
        ax_force.tick_params(axis='y', labelcolor='r')
        lines1, labels1 = ax_total.get_legend_handles_labels()
        lines2, labels2 = ax_force.get_legend_handles_labels()
        ax_total.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax_total.set_title('Total PE (all links sum) + Contact force (max per frame)', fontsize=12)
    else:
        ax_total.legend(loc='upper right')
        ax_total.set_title('Total PE (from t=0)', fontsize=12)
    ax_total.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Link能量曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_joint_wrench_risk(
    joint_forces_df: pd.DataFrame,
    time_collision: np.ndarray,
    time_collision_near_jointlimit: np.ndarray,
    dt: float = 0.002,
    F_axial_thr: float = 1000.0,
    F_shear_thr: float = 1000.0,
    M_torsion_thr: float = 100.0,
    M_bend_thr: float = 1000.0,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 0.01,
    w4: float = 0.01,
    save_frame_by_frame_path: Optional[str] = None
) -> Tuple[float, pd.DataFrame]:
    """
    计算关节力/力矩相关的risk
    
    风险定义为：
    - r1(t) = gate(collision) * ReLU((F_axial_mag(t) - F_axial_thr) / F_axial_thr) * dt
    - r2(t) = gate(collision) * ReLU((F_shear_mag(t) - F_shear_thr) / F_shear_thr) * dt
    - r3(t) = gate(collision near jointlimit) * ReLU((M_torsion_mag(t) - M_torsion_thr) / M_torsion_thr) * dt
    - r4(t) = gate(collision) * ReLU((M_bend_mag(t) - M_bend_thr) / M_bend_thr) * dt
    
    合成 joint-wrench risk：
    - r_j_joint(t) = (w1*r1(t) + w2*r2(t) + w3*r3(t) + w4*r4(t))
    - 先算每个 link 的 r_j_joint，把整个 episode 的 timestamp 求和
    - 再把所有 link 加起来，得到 r_joint
    
    参数:
        joint_forces_df: 关节力数据 DataFrame，应包含列：
            - timestamp: 时间戳
            - joint_name: 关节名称
            - F_axial_mag: 轴向力大小
            - F_shear_mag: 剪切力大小
            - M_torsion_mag: 扭转力矩大小
            - M_bend_mag: 弯曲力矩大小
        time_collision: 碰撞时间戳数组（从 extract_collision_timestamps 获取）
        time_collision_near_jointlimit: 接近关节限制的碰撞时间戳数组（从 extract_collision_nearjointlimit_timestamps 获取）
        dt: 采样时间间隔（秒），默认 0.002
        F_axial_thr: 轴向力阈值，默认 1000.0
        F_shear_thr: 剪切力阈值，默认 1000.0
        M_torsion_thr: 扭转力矩阈值，默认 100.0
        M_bend_thr: 弯曲力矩阈值，默认 100.0
        w1, w2, w3, w4: 各风险项的权重，默认均为 1.0
        save_frame_by_frame_path: 如果指定，保存逐帧数据到 CSV
    
    返回:
        r_joint: float，总关节力风险
        breakdown: pd.DataFrame，按 link/joint 的风险明细
    """
    if joint_forces_df.empty or len(time_collision) == 0:
        empty_breakdown = pd.DataFrame(columns=["joint_name", "R_j_joint"])
        return 0.0, empty_breakdown
    
    # 检查必要的列
    required_cols = ['timestamp', 'joint_name', 'F_axial_mag', 'F_shear_mag', 'M_torsion_mag', 'M_bend_mag']
    missing_cols = [col for col in required_cols if col not in joint_forces_df.columns]
    if missing_cols:
        print(f"警告: joint_forces_df 缺少必要的列: {missing_cols}")
        empty_breakdown = pd.DataFrame(columns=["joint_name", "R_j_joint"])
        return 0.0, empty_breakdown
    
    # 验证 dt 值
    if dt <= 0 or not np.isfinite(dt):
        print(f"警告: 无效的 dt 值: {dt}，使用默认值 0.002")
        dt = 0.002
    
    # 获取时间戳
    timestamps = joint_forces_df['timestamp'].to_numpy(dtype=float)
    tolerance = 0.001  # 1ms 容差
    
    # 创建碰撞时间戳的掩码
    collision_mask = np.zeros(len(joint_forces_df), dtype=bool)
    for t_collision in time_collision:
        time_mask = np.abs(timestamps - t_collision) <= tolerance
        collision_mask = collision_mask | time_mask
    
    # 创建接近关节限制的碰撞时间戳掩码
    collision_near_jointlimit_mask = np.zeros(len(joint_forces_df), dtype=bool)
    if len(time_collision_near_jointlimit) > 0:
        for t_collision_near in time_collision_near_jointlimit:
            time_mask = np.abs(timestamps - t_collision_near) <= tolerance
            collision_near_jointlimit_mask = collision_near_jointlimit_mask | time_mask
    
    # 计算各风险项
    joint_forces_df = joint_forces_df.copy()
    
    # r1(t) = gate(collision) * ReLU((F_axial_mag(t) - F_axial_thr) / F_axial_thr) * dt
    r1 = np.where(
        collision_mask,
        relu((joint_forces_df['F_axial_mag'].values - F_axial_thr) / F_axial_thr) * dt,
        0.0
    )
    
    # r2(t) = gate(collision) * ReLU((F_shear_mag(t) - F_shear_thr) / F_shear_thr) * dt
    r2 = np.where(
        collision_mask,
        relu((joint_forces_df['F_shear_mag'].values - F_shear_thr) / F_shear_thr) * dt,
        0.0
    )
    
    # r3(t) = gate(collision near jointlimit) * ReLU((M_torsion_mag(t) - M_torsion_thr) / M_torsion_thr) * dt
    r3 = np.where(
        collision_near_jointlimit_mask,
        relu((joint_forces_df['M_torsion_mag'].values - M_torsion_thr) / M_torsion_thr) * dt,
        0.0
    )
    
    # r4(t) = gate(collision) * ReLU((M_bend_mag(t) - M_bend_thr) / M_bend_thr) * dt
    r4 = np.where(
        collision_mask,
        relu((joint_forces_df['M_bend_mag'].values - M_bend_thr) / M_bend_thr) * dt,
        0.0
    )
    
    # r_j_joint(t) = (w1*r1(t) + w2*r2(t) + w3*r3(t) + w4*r4(t))
    r_j_joint = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4
    
    # 添加 r_j_joint 列到 DataFrame
    joint_forces_df['r_j_joint'] = r_j_joint
    
    # 保存逐帧数据（如果指定了路径）
    if save_frame_by_frame_path is not None:
        try:
            frame_data = []
            for idx, row in joint_forces_df.iterrows():
                if r_j_joint[idx] > 0:  # 只保存有风险的数据
                    frame_data.append({
                        'timestamp': round(float(row['timestamp']), 4),
                        'joint_name': str(row['joint_name']),
                        'F_axial_mag': round(float(row['F_axial_mag']), 4),
                        'F_shear_mag': round(float(row['F_shear_mag']), 4),
                        'M_torsion_mag': round(float(row['M_torsion_mag']), 4),
                        'M_bend_mag': round(float(row['M_bend_mag']), 4),
                        'r1': round(float(r1[idx]), 4),
                        'r2': round(float(r2[idx]), 4),
                        'r3': round(float(r3[idx]), 4),
                        'r4': round(float(r4[idx]), 4),
                        'r_j_joint': round(float(r_j_joint[idx]), 4),
                    })
            
            if frame_data:
                frame_df = pd.DataFrame(frame_data)
                frame_df = frame_df.sort_values('timestamp').reset_index(drop=True)
                frame_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"joint wrench 逐帧风险数据已保存到: {save_frame_by_frame_path} ({len(frame_df)} 条记录)")
            else:
                empty_df = pd.DataFrame(columns=['timestamp', 'joint_name', 'F_axial_mag', 'F_shear_mag', 'M_torsion_mag', 'M_bend_mag', 'r1', 'r2', 'r3', 'r4', 'r_j_joint'])
                empty_df.to_csv(save_frame_by_frame_path, index=False, float_format='%.4f')
                print(f"joint wrench 逐帧风险数据已保存到: {save_frame_by_frame_path} (无超阈值数据)")
        except Exception as e:
            print(f"警告: 保存 joint wrench 逐帧风险数据到 '{save_frame_by_frame_path}' 失败: {e}")
    
    # 按 joint_name 分组求和得到 R_j_joint（每个 link 的整个 episode 求和）
    joint_risks = joint_forces_df.groupby('joint_name')['r_j_joint'].sum().reset_index()
    joint_risks.columns = ['joint_name', 'R_j_joint']
    
    # 四舍五入到4位小数
    joint_risks['R_j_joint'] = joint_risks['R_j_joint'].round(4)
    
    # 按 R_j_joint 降序排序
    joint_risks = joint_risks.sort_values('R_j_joint', ascending=False).reset_index(drop=True)
    
    # 计算总风险 r_joint（所有 link 加起来）
    r_joint = float(joint_risks['R_j_joint'].sum())
    if not np.isfinite(r_joint):
        r_joint = 0.0
    r_joint = round(r_joint, 4)
    
    return r_joint, joint_risks


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
    w_contact: float = 1.0,
    w_acceleration: float = 0.001,
    w_motor: float = 10.0,
    w_joint_wrench: float = 1.0,
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
        w_contact: 接触力风险权重，默认 1.0
        w_acceleration: 加速度风险权重，默认 1.0
        w_motor: 电机扭矩风险权重，默认 1.0
        w_joint_wrench: 关节力风险权重，默认 1.0
        
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
    
    # 提取接近关节限制的碰撞时间戳
    time_collision_near_jointlimit = extract_collision_nearjointlimit_timestamps(
        time_collision=time_collision,
        joint_state_df=joint_state_df
    )
    
    # 准备保存路径
    acceleration_frame_by_frame_path = None
    motor_frame_by_frame_path = None
    contact_force_frame_by_frame_path = None
    joint_wrench_frame_by_frame_path = None
    if output_base:
        acceleration_frame_by_frame_path = f"{output_base}_acceleration_risk.csv"
        motor_frame_by_frame_path = f"{output_base}_motor_risk.csv"
        contact_force_frame_by_frame_path = f"{output_base}_contact_force_risk.csv"
        joint_wrench_frame_by_frame_path = f"{output_base}_joint_wrench_risk.csv"
    
    # 计算各个子risk
    # 使用 time_collision 来筛选碰撞时间窗口内的接触力
    contact_risk, contact_breakdown = calculate_contact_force_risk(
        contact_df,
        time_collision=time_collision,
        save_frame_by_frame_path=contact_force_frame_by_frame_path
    )
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
    joint_wrench_risk, joint_wrench_breakdown = calculate_joint_wrench_risk(
        joint_forces_df,
        time_collision=time_collision,
        time_collision_near_jointlimit=time_collision_near_jointlimit,
        save_frame_by_frame_path=joint_wrench_frame_by_frame_path
    )
    perturbation_risk = calculate_perturbation_risk(perturbation_df) if perturbation_df is not None else 0.0
    
    
    # 求和得到总risk（基于四舍五入后的值，使用权重）
    total_risk = (
        w_contact * contact_risk +
        w_acceleration * acceleration_risk +
        w_motor * motor_torque_risk +
        w_joint_wrench * joint_wrench_risk +
        perturbation_risk  # perturbation_risk 保持权重为 1.0
    )
    total_risk = round(total_risk, 4)
    
    # 所有risk四舍五入到4位小数（在计算时就保留4位）
    contact_risk = round(contact_risk, 4)
    acceleration_risk = round(acceleration_risk, 4)
    motor_torque_risk = round(motor_torque_risk, 4)
    joint_wrench_risk = round(joint_wrench_risk, 4)
    perturbation_risk = round(perturbation_risk, 4)

    # 收集所有breakdown
    breakdowns = {}
    if not motor_breakdown.empty:
        breakdowns['motor_torque'] = motor_breakdown
    if not contact_breakdown.empty:
        breakdowns['contact_force'] = contact_breakdown
    if not joint_wrench_breakdown.empty:
        breakdowns['joint_wrench'] = joint_wrench_breakdown
    
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
        '--link-energy',
        type=str,
        default=None,
        help='Link 动能/势能数据文件路径 (CSV或BIN格式，可选)。'
             '若不指定，将根据 contact 文件路径自动查找同目录下的 link_kinetic_energy_data_YYYYMMDD_HHMMSS.*'
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
                    plot_motor_torque_risk_curves(motor_risk_df, plot_output_path)
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
            # 单独画加速度曲线（torso/head ||a||，标碰撞与 20ms 窗口）
            try:
                time_collision = risk_results.get('time_collision', np.array([]))
                a_thr = 30.0 if args.a_thr is None else args.a_thr
                plot_acceleration_curves(
                    sensor_vibration_df,
                    time_collision=time_collision if len(time_collision) > 0 else None,
                    a_thr=a_thr,
                    output_path=f"{output_base}_acc_curves.png"
                )
            except Exception as e:
                print(f"警告: 绘制加速度曲线失败: {e}")
        
        # contact_force_risk 绘图
        if args.plot:
            contact_force_risk_path = f"{output_base}_contact_force_risk.csv"
            if os.path.exists(contact_force_risk_path):
                try:
                    contact_force_risk_df = pd.read_csv(contact_force_risk_path)
                    plot_output_path = f"{output_base}_contact_force_risk_curves.png"
                    plot_contact_force_risk_curves(contact_force_risk_df, plot_output_path)
                except Exception as e:
                    print(f"警告: 绘制 contact_force_risk 曲线失败: {e}")
            # 单独画接触力曲线（原始 force_magnitude，标碰撞时刻）
            try:
                time_collision = risk_results.get('time_collision', np.array([]))
                plot_contact_force_curves(
                    contact_df,
                    time_collision=time_collision if len(time_collision) > 0 else None,
                    f_thr=1000.0,
                    output_path=f"{output_base}_contact_force_curves.png"
                )
            except Exception as e:
                print(f"警告: 绘制接触力曲线失败: {e}")
        
        # joint_wrench_risk 绘图
        if args.plot:
            joint_wrench_risk_path = f"{output_base}_joint_wrench_risk.csv"
            if os.path.exists(joint_wrench_risk_path):
                try:
                    joint_wrench_risk_df = pd.read_csv(joint_wrench_risk_path)
                    if not joint_wrench_risk_df.empty:
                        plot_output_path = f"{output_base}_joint_wrench_risk_curves.png"
                        plot_joint_wrench_risk_curves(joint_wrench_risk_df, plot_output_path)
                    else:
                        print("joint_wrench_risk 数据为空，跳过绘图")
                except Exception as e:
                    print(f"警告: 绘制 joint_wrench_risk 曲线失败: {e}")
    
    # ==================== Link Energy 绘图（独立于 risk 计算） ====================
    if args.plot:
        link_energy_path = None
        if args.link_energy:
            link_energy_path = Path(args.link_energy)
        if link_energy_path is None or not link_energy_path.exists():
            # 根据 contact 文件路径自动查找对应的 link_kinetic_energy_data 文件
            contact_path = Path(args.contact)
            contact_dir = contact_path.parent
            contact_stem = contact_path.stem  # e.g., contact_data_20260131_183505
            import re
            timestamp_match = re.search(r'(\d{8}_\d{6})', contact_stem)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                for ext in ['.csv', '.bin']:
                    candidate = contact_dir / f"link_kinetic_energy_data_{timestamp_str}{ext}"
                    if candidate.exists():
                        link_energy_path = candidate
                        break
        if link_energy_path is not None and link_energy_path.exists():
            print(f"\n找到 Link Energy 数据文件: {link_energy_path}")
            link_energy_df = load_data_file(str(link_energy_path))
            if not link_energy_df.empty:
                print(f"  Link能量数据: {len(link_energy_df)} 条记录")
                energy_plot_path = f"{output_base}_link_energy_curves.png" if output_base else None
                try:
                    plot_link_energy_curves(link_energy_df, energy_plot_path, contact_df=contact_df)
                except Exception as e:
                    print(f"警告: 绘制 link energy 曲线失败: {e}")


if __name__ == '__main__':
    main()
