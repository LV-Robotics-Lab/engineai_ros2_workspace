#!/usr/bin/env python3
"""MuJoCo 日志：统一读取 contact / joint / perturbation 等 CSV 与 binary（与 C++ ros_interface 格式一致）。"""
from __future__ import annotations

import math
import os
import re
import struct
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# contact bin：body 名仅含字母数字下划线，用于过滤错位同步产生的伪记录
_CONTACT_BODY_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,120}$")

# 错位同步会通过 body 名校验但力/位姿为垃圾（如 1e247）；用物理合理上界剔除
_CONTACT_MAX_FORCE = 2e7  # N
_CONTACT_MAX_TORQUE = 2e6  # N·m
_CONTACT_MAX_POS = 5e3  # m
_CONTACT_MAX_LINVEL = 500.0
_CONTACT_MAX_ANGVEL = 500.0


def _contact_tail_values_plausible(
    world_forces,
    force_magnitude,
    force_normal,
    force_friction,
    world_torques,
    red_ball_pos,
    green_ball_pos,
    base_link_pos,
    base_link_quat,
    base_link_vel,
    base_link_angvel,
    collision_link_pos,
    collision_link_quat,
) -> bool:
    def ok3(v, lim):
        return all(math.isfinite(x) and abs(x) <= lim for x in v)

    def ok_quat(q):
        if not all(math.isfinite(x) for x in q):
            return False
        s = sum(x * x for x in q)
        return 0.01 <= s <= 25.0

    if not ok3(world_forces, _CONTACT_MAX_FORCE):
        return False
    if not ok3(world_torques, _CONTACT_MAX_TORQUE):
        return False
    for x in (force_magnitude, force_normal, force_friction):
        if not math.isfinite(x) or abs(x) > _CONTACT_MAX_FORCE:
            return False
    for v, lim in (
        (red_ball_pos, _CONTACT_MAX_POS),
        (green_ball_pos, _CONTACT_MAX_POS),
        (base_link_pos, _CONTACT_MAX_POS),
        (collision_link_pos, _CONTACT_MAX_POS),
    ):
        if not ok3(v, lim):
            return False
    if not ok_quat(base_link_quat) or not ok_quat(collision_link_quat):
        return False
    if not ok3(base_link_vel, _CONTACT_MAX_LINVEL):
        return False
    if not ok3(base_link_angvel, _CONTACT_MAX_ANGVEL):
        return False
    return True


def read_binary_policy_switch(bin_path: str) -> pd.DataFrame:
    """
    policy_switch.bin：8 字节魔数 MJPSW01\\0，每条记录：
    double timestamp; uint32 n1 + from_mode; uint32 n2 + to_mode; uint32 n3 + mimic_direction (UTF-8)
    """
    data = []
    with open(bin_path, "rb") as f:
        magic = f.read(8)
        if len(magic) < 8 or magic[:7] != b"MJPSW01":
            return pd.DataFrame()
        while True:
            tb = f.read(8)
            if len(tb) < 8:
                break
            t = struct.unpack("d", tb)[0]
            row = {"timestamp": t}
            for key in ("from_mode", "to_mode", "mimic_direction"):
                nb = f.read(4)
                if len(nb) < 4:
                    return pd.DataFrame(data)
                n = struct.unpack("I", nb)[0]
                sb = f.read(n)
                if len(sb) < n:
                    return pd.DataFrame(data)
                row[key] = sb.decode("utf-8", errors="replace")
            data.append(row)
    return pd.DataFrame(data)


def _contact_bin_tail_bytes(with_force_friction: bool) -> int:
    return 24 + 24 + 24 + 8 + 8 + (8 if with_force_friction else 0) + 24 + 24 + 32 + 24 + 24 + 24 + 32


def _try_parse_one_contact_row(raw: bytes, pos: int, with_force_friction: bool):
    """
    从 pos 尝试解析一条 contact 记录。成功返回 (row_dict, next_pos)；失败返回 None。
    """
    tail = _contact_bin_tail_bytes(with_force_friction)
    max_name = 128
    if pos + 16 > len(raw):
        return None
    sim_time = struct.unpack_from("d", raw, pos)[0]
    if math.isnan(sim_time) or math.isinf(sim_time) or sim_time < -1e-3 or sim_time > 1e7:
        return None
    contact_id = struct.unpack_from("i", raw, pos + 8)[0]
    if contact_id < -1 or contact_id > 50000:
        return None
    b1l = struct.unpack_from("i", raw, pos + 12)[0]
    if b1l < 1 or b1l > max_name:
        return None
    o = pos + 16
    if o + b1l + 4 > len(raw):
        return None
    try:
        body1_name = raw[o : o + b1l].decode("utf-8")
    except UnicodeDecodeError:
        return None
    o += b1l
    b2l = struct.unpack_from("i", raw, o)[0]
    if b2l < 1 or b2l > max_name:
        return None
    o += 4
    if o + b2l + tail > len(raw):
        return None
    try:
        body2_name = raw[o : o + b2l].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not (
        _CONTACT_BODY_NAME_RE.match(body1_name) and _CONTACT_BODY_NAME_RE.match(body2_name)
    ):
        return None
    n1, n2 = body1_name.lower(), body2_name.lower()
    if not (
        "link_" in body1_name
        or "link_" in body2_name
        or n1 in ("world", "terrain", "ground", "floor")
        or n2 in ("world", "terrain", "ground", "floor")
    ):
        return None
    o += b2l
    rest = raw[o : o + tail]
    off = 0
    red_ball_pos = struct.unpack("3d", rest[off : off + 24])
    off += 24
    green_ball_pos = struct.unpack("3d", rest[off : off + 24])
    off += 24
    world_forces = struct.unpack("3d", rest[off : off + 24])
    off += 24
    force_magnitude = struct.unpack("d", rest[off : off + 8])[0]
    off += 8
    force_normal = struct.unpack("d", rest[off : off + 8])[0]
    off += 8
    if with_force_friction:
        force_friction = struct.unpack("d", rest[off : off + 8])[0]
        off += 8
    else:
        force_friction = 0.0
    world_torques = struct.unpack("3d", rest[off : off + 24])
    off += 24
    base_link_pos = struct.unpack("3d", rest[off : off + 24])
    off += 24
    base_link_quat = struct.unpack("4d", rest[off : off + 32])
    off += 32
    base_link_vel = struct.unpack("3d", rest[off : off + 24])
    off += 24
    base_link_angvel = struct.unpack("3d", rest[off : off + 24])
    off += 24
    collision_link_pos = struct.unpack("3d", rest[off : off + 24])
    off += 24
    collision_link_quat = struct.unpack("4d", rest[off : off + 32])
    if not _contact_tail_values_plausible(
        world_forces,
        force_magnitude,
        force_normal,
        force_friction,
        world_torques,
        red_ball_pos,
        green_ball_pos,
        base_link_pos,
        base_link_quat,
        base_link_vel,
        base_link_angvel,
        collision_link_pos,
        collision_link_quat,
    ):
        return None
    next_pos = o + tail
    row = {
        "timestamp": sim_time,
        "contact_id": contact_id,
        "body1_name": body1_name,
        "body2_name": body2_name,
        "pos_x": red_ball_pos[0],
        "pos_y": red_ball_pos[1],
        "pos_z": red_ball_pos[2],
        "robot_frame_x": green_ball_pos[0],
        "robot_frame_y": green_ball_pos[1],
        "robot_frame_z": green_ball_pos[2],
        "force_x": world_forces[0],
        "force_y": world_forces[1],
        "force_z": world_forces[2],
        "force_magnitude": force_magnitude,
        "force_normal": force_normal,
        "force_friction": force_friction,
        "torque_x": world_torques[0],
        "torque_y": world_torques[1],
        "torque_z": world_torques[2],
        "base_link_x": base_link_pos[0],
        "base_link_y": base_link_pos[1],
        "base_link_z": base_link_pos[2],
        "base_link_qw": base_link_quat[0],
        "base_link_qx": base_link_quat[1],
        "base_link_qy": base_link_quat[2],
        "base_link_qz": base_link_quat[3],
        "base_link_vel_x": base_link_vel[0],
        "base_link_vel_y": base_link_vel[1],
        "base_link_vel_z": base_link_vel[2],
        "base_link_angvel_x": base_link_angvel[0],
        "base_link_angvel_y": base_link_angvel[1],
        "base_link_angvel_z": base_link_angvel[2],
        "collision_link_x": collision_link_pos[0],
        "collision_link_y": collision_link_pos[1],
        "collision_link_z": collision_link_pos[2],
        "collision_link_qw": collision_link_quat[0],
        "collision_link_qx": collision_link_quat[1],
        "collision_link_qy": collision_link_quat[2],
        "collision_link_qz": collision_link_quat[3],
    }
    return row, next_pos


def _read_contact_bin_records(f, with_force_friction: bool):
    """
    从当前位置读到 EOF。若中途因崩溃导致半条记录错位，会向前扫描重新对齐后继续读
    （否则 CSV 能看到的其它 link 在 bin 里会整段丢失）。
    """
    raw = f.read()
    return _read_contact_bin_from_bytes(raw, with_force_friction)


def _read_contact_bin_from_bytes(raw: bytes, with_force_friction: bool):
    sync_window = 600_000
    pos = 0
    data = []
    n = len(raw)
    while pos <= n - 50:
        parsed = _try_parse_one_contact_row(raw, pos, with_force_friction)
        if parsed:
            row, next_pos = parsed
            data.append(row)
            pos = next_pos
            continue
        found = False
        limit = min(pos + sync_window, n - 20)
        scan = pos + 1
        while scan < limit:
            parsed = _try_parse_one_contact_row(raw, scan, with_force_friction)
            if parsed:
                pos = scan
                found = True
                break
            scan += 1
        if not found:
            break
    return data


def read_binary_contact_data(bin_path: str) -> pd.DataFrame:
    """
    读取 contact_data.bin。支持新版（force_normal 后多 8 字节 force_friction）与旧版。
    旧版单条尾部比新版短 8 字节，用新格式读会得到 0 行；再按旧版读即可。
    """
    with open(bin_path, "rb") as f:
        if len(f.read(4)) < 4:
            return pd.DataFrame()
    for with_friction in (True, False):
        with open(bin_path, "rb") as f:
            f.seek(4)
            rows = _read_contact_bin_records(f, with_force_friction=with_friction)
            if rows is not None and len(rows) > 0:
                return pd.DataFrame(rows)
    return pd.DataFrame()


def write_contact_data_bin(df: pd.DataFrame, bin_path: str, num_joints: int = 0) -> None:
    """
    将接触 DataFrame 写成与仿真器一致的 contact_data.bin（与 read_binary_contact_data 互逆）。
    num_joints 仅作文件头占位（合并文件可用 0）；缺列按 0 或空字符串。
    """
    def fv(row, col: str, default: float = 0.0) -> float:
        if col not in row.index:
            return default
        v = row[col]
        if pd.isna(v):
            return default
        return float(v)

    def sv(row, col: str, default: str = "") -> str:
        if col not in row.index:
            return default
        v = row[col]
        if pd.isna(v):
            return default
        return str(v)

    with open(bin_path, "wb") as f:
        f.write(struct.pack("i", int(num_joints)))
        # iterrows 较慢但列类型混杂；大表可再优化
        for _, row in df.iterrows():
            b1 = sv(row, "body1_name", "world").encode("utf-8")
            b2 = sv(row, "body2_name", "").encode("utf-8")
            f.write(struct.pack("d", fv(row, "timestamp", 0.0)))
            f.write(struct.pack("i", int(fv(row, "contact_id", 0))))
            f.write(struct.pack("i", len(b1)))
            f.write(b1)
            f.write(struct.pack("i", len(b2)))
            f.write(b2)
            f.write(struct.pack("3d", fv(row, "pos_x"), fv(row, "pos_y"), fv(row, "pos_z")))
            f.write(
                struct.pack(
                    "3d",
                    fv(row, "robot_frame_x"),
                    fv(row, "robot_frame_y"),
                    fv(row, "robot_frame_z"),
                )
            )
            f.write(struct.pack("3d", fv(row, "force_x"), fv(row, "force_y"), fv(row, "force_z")))
            f.write(struct.pack("d", fv(row, "force_magnitude")))
            f.write(struct.pack("d", fv(row, "force_normal")))
            f.write(struct.pack("d", fv(row, "force_friction")))
            f.write(struct.pack("3d", fv(row, "torque_x"), fv(row, "torque_y"), fv(row, "torque_z")))
            f.write(struct.pack("3d", fv(row, "base_link_x"), fv(row, "base_link_y"), fv(row, "base_link_z")))
            f.write(
                struct.pack(
                    "4d",
                    fv(row, "base_link_qw", 1.0),
                    fv(row, "base_link_qx"),
                    fv(row, "base_link_qy"),
                    fv(row, "base_link_qz"),
                )
            )
            f.write(
                struct.pack(
                    "3d",
                    fv(row, "base_link_vel_x"),
                    fv(row, "base_link_vel_y"),
                    fv(row, "base_link_vel_z"),
                )
            )
            f.write(
                struct.pack(
                    "3d",
                    fv(row, "base_link_angvel_x"),
                    fv(row, "base_link_angvel_y"),
                    fv(row, "base_link_angvel_z"),
                )
            )
            f.write(
                struct.pack(
                    "3d",
                    fv(row, "collision_link_x"),
                    fv(row, "collision_link_y"),
                    fv(row, "collision_link_z"),
                )
            )
            f.write(
                struct.pack(
                    "4d",
                    fv(row, "collision_link_qw", 1.0),
                    fv(row, "collision_link_qx"),
                    fv(row, "collision_link_qy"),
                    fv(row, "collision_link_qz"),
                )
            )


def _parse_sensor_vibration_chunk(chunk: bytes, with_head_ang: bool) -> Optional[dict]:
    """单条记录：8 + 72 + (24 if with_head_ang else 0)。chunk 长度须恰好匹配。"""
    need = 80 + (24 if with_head_ang else 0)
    if len(chunk) != need:
        return None
    off = 0
    timestamp = struct.unpack("d", chunk[off : off + 8])[0]
    off += 8
    base_link_lin_acc = struct.unpack("3d", chunk[off : off + 24])
    off += 24
    base_link_ang_acc = struct.unpack("3d", chunk[off : off + 24])
    off += 24
    head_lin_acc = struct.unpack("3d", chunk[off : off + 24])
    off += 24
    if with_head_ang:
        head_ang_acc = struct.unpack("3d", chunk[off : off + 24])
    else:
        head_ang_acc = (0.0, 0.0, 0.0)
    return {
        "timestamp": timestamp,
        "base_link_lin_acc_x": base_link_lin_acc[0],
        "base_link_lin_acc_y": base_link_lin_acc[1],
        "base_link_lin_acc_z": base_link_lin_acc[2],
        "base_link_ang_acc_x": base_link_ang_acc[0],
        "base_link_ang_acc_y": base_link_ang_acc[1],
        "base_link_ang_acc_z": base_link_ang_acc[2],
        "head_lin_acc_x": head_lin_acc[0],
        "head_lin_acc_y": head_lin_acc[1],
        "head_lin_acc_z": head_lin_acc[2],
        "head_ang_acc_x": head_ang_acc[0],
        "head_ang_acc_y": head_ang_acc[1],
        "head_ang_acc_z": head_ang_acc[2],
    }


def read_binary_sensor_vibration_data(bin_path: str) -> pd.DataFrame:
    """
    sensor_vibration_data.bin：与 ros_interface 一致时为每条 104 字节（含 head 角加速度）。
    旧版或部分写入可能为 80 字节/条（无 head_ang，按 0 填）。
    """
    with open(bin_path, "rb") as f:
        raw = f.read()
    n = len(raw)
    if n < 80:
        return pd.DataFrame()

    rows_out: list = []
    # 新格式：整段为 104 的倍数
    if n % 104 == 0:
        for off in range(0, n, 104):
            row = _parse_sensor_vibration_chunk(raw[off : off + 104], True)
            if not row:
                break
            rows_out.append(row)
        if len(rows_out) == n // 104:
            return pd.DataFrame(rows_out)
        rows_out = []

    # 新格式 + 最后一条仅写到 head_lin（少 24 字节）：104*k + 80
    if n >= 80 and (n - 80) % 104 == 0 and n > 80:
        k = (n - 80) // 104
        for i in range(k):
            off = i * 104
            row = _parse_sensor_vibration_chunk(raw[off : off + 104], True)
            if row:
                rows_out.append(row)
        last = _parse_sensor_vibration_chunk(raw[n - 80 : n], False)
        if last and len(rows_out) == k:
            rows_out.append(last)
            return pd.DataFrame(rows_out)
        rows_out = []

    # 旧格式：80 字节/条
    if n % 80 == 0:
        for off in range(0, n, 80):
            row = _parse_sensor_vibration_chunk(raw[off : off + 80], False)
            if not row:
                return pd.DataFrame()
            rows_out.append(row)
        return pd.DataFrame(rows_out)

    # 单条 80 字节或截断成仅一条有效
    if n == 80 or (80 <= n < 104):
        row = _parse_sensor_vibration_chunk(raw[:80], False)
        return pd.DataFrame([row]) if row else pd.DataFrame()

    return pd.DataFrame()


def read_binary_joint_state_data(bin_path: str) -> pd.DataFrame:
    data = []
    with open(bin_path, "rb") as f:
        while True:
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            timestamp = struct.unpack("d", timestamp_bytes)[0]
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack("i", num_joints_bytes)[0]
            joint_pos_bytes = f.read(num_joints * 8)
            if len(joint_pos_bytes) < num_joints * 8:
                break
            joint_positions = struct.unpack(f"{num_joints}d", joint_pos_bytes)
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack("i", num_joints_bytes)[0]
            joint_vel_bytes = f.read(num_joints * 8)
            if len(joint_vel_bytes) < num_joints * 8:
                break
            joint_velocities = struct.unpack(f"{num_joints}d", joint_vel_bytes)
            num_joints_bytes = f.read(4)
            if len(num_joints_bytes) < 4:
                break
            num_joints = struct.unpack("i", num_joints_bytes)[0]
            actuator_force_bytes = f.read(num_joints * 8)
            if len(actuator_force_bytes) < num_joints * 8:
                break
            actuator_forces = struct.unpack(f"{num_joints}d", actuator_force_bytes)
            row = {"timestamp": timestamp}
            for i in range(num_joints):
                row[f"joint_{i}_position"] = joint_positions[i]
                row[f"joint_{i}_velocity"] = joint_velocities[i]
                row[f"actuator_{i}_force"] = actuator_forces[i]
            data.append(row)
    return pd.DataFrame(data)


def read_binary_perturbation_data(bin_path: str) -> pd.DataFrame:
    data = []
    with open(bin_path, "rb") as f:
        while True:
            sim_time_bytes = f.read(8)
            if len(sim_time_bytes) < 8:
                break
            sim_time = struct.unpack("d", sim_time_bytes)[0]
            perturbation_id_bytes = f.read(4)
            if len(perturbation_id_bytes) < 4:
                break
            perturbation_id = struct.unpack("i", perturbation_id_bytes)[0]
            body_name_len_bytes = f.read(4)
            if len(body_name_len_bytes) < 4:
                break
            body_name_len = struct.unpack("i", body_name_len_bytes)[0]
            body_name = f.read(body_name_len).decode("utf-8")
            start_time = struct.unpack("d", f.read(8))[0]
            duration = struct.unpack("d", f.read(8))[0]
            elapsed_time = struct.unpack("d", f.read(8))[0]
            force = struct.unpack("3d", f.read(24))
            force_magnitude = struct.unpack("d", f.read(8))[0]
            torque = struct.unpack("3d", f.read(24))
            torque_magnitude = struct.unpack("d", f.read(8))[0]
            std_pose = struct.unpack("3d", f.read(24))
            world_force = struct.unpack("3d", f.read(24))
            world_force_magnitude = struct.unpack("d", f.read(8))[0]
            data.append(
                {
                    "timestamp": sim_time,
                    "perturbation_id": perturbation_id,
                    "body_name": body_name,
                    "start_time": start_time,
                    "duration": duration,
                    "elapsed_time": elapsed_time,
                    "force_x": force[0],
                    "force_y": force[1],
                    "force_z": force[2],
                    "force_magnitude": force_magnitude,
                    "torque_x": torque[0],
                    "torque_y": torque[1],
                    "torque_z": torque[2],
                    "torque_magnitude": torque_magnitude,
                    "std_pose_x": std_pose[0],
                    "std_pose_y": std_pose[1],
                    "std_pose_z": std_pose[2],
                    "world_force_x": world_force[0],
                    "world_force_y": world_force[1],
                    "world_force_z": world_force[2],
                    "world_force_magnitude": world_force_magnitude,
                }
            )
    return pd.DataFrame(data)


def read_binary_joint_forces_data(bin_path: str) -> pd.DataFrame:
    data = []
    with open(bin_path, "rb") as f:
        while True:
            timestamp_bytes = f.read(8)
            if len(timestamp_bytes) < 8:
                break
            timestamp = struct.unpack("d", timestamp_bytes)[0]
            joint_id = struct.unpack("i", f.read(4))[0]
            joint_name_len = struct.unpack("i", f.read(4))[0]
            joint_name = f.read(joint_name_len).decode("utf-8")
            body_id = struct.unpack("i", f.read(4))[0]
            body_name_len = struct.unpack("i", f.read(4))[0]
            body_name = f.read(body_name_len).decode("utf-8")
            child_M = struct.unpack("3d", f.read(24))
            child_F = struct.unpack("3d", f.read(24))
            parent_M = struct.unpack("3d", f.read(24))
            parent_F = struct.unpack("3d", f.read(24))
            axis = struct.unpack("3d", f.read(24))
            F_axial_mag = struct.unpack("d", f.read(8))[0]
            F_shear_mag = struct.unpack("d", f.read(8))[0]
            M_torsion_mag = struct.unpack("d", f.read(8))[0]
            M_bend_mag = struct.unpack("d", f.read(8))[0]
            M_eq = struct.unpack("d", f.read(8))[0]
            F_axial = struct.unpack("3d", f.read(24))
            F_shear = struct.unpack("3d", f.read(24))
            M_torsion = struct.unpack("3d", f.read(24))
            M_bend = struct.unpack("3d", f.read(24))
            data.append(
                {
                    "timestamp": timestamp,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "body_id": body_id,
                    "body_name": body_name,
                    "child_Mx": child_M[0],
                    "child_My": child_M[1],
                    "child_Mz": child_M[2],
                    "child_Fx": child_F[0],
                    "child_Fy": child_F[1],
                    "child_Fz": child_F[2],
                    "parent_Mx": parent_M[0],
                    "parent_My": parent_M[1],
                    "parent_Mz": parent_M[2],
                    "parent_Fx": parent_F[0],
                    "parent_Fy": parent_F[1],
                    "parent_Fz": parent_F[2],
                    "axis_x": axis[0],
                    "axis_y": axis[1],
                    "axis_z": axis[2],
                    "F_axial_mag": F_axial_mag,
                    "F_shear_mag": F_shear_mag,
                    "M_torsion_mag": M_torsion_mag,
                    "M_bend_mag": M_bend_mag,
                    "M_eq": M_eq,
                    "F_axial_x": F_axial[0],
                    "F_axial_y": F_axial[1],
                    "F_axial_z": F_axial[2],
                    "F_shear_x": F_shear[0],
                    "F_shear_y": F_shear[1],
                    "F_shear_z": F_shear[2],
                    "M_torsion_x": M_torsion[0],
                    "M_torsion_y": M_torsion[1],
                    "M_torsion_z": M_torsion[2],
                    "M_bend_x": M_bend[0],
                    "M_bend_y": M_bend[1],
                    "M_bend_z": M_bend[2],
                }
            )
    return pd.DataFrame(data)


def _read_one_link_ke_row_from_buffer(
    raw: bytes, pos: int, body_names: list
) -> tuple[Optional[dict], int]:
    """从 raw[pos:] 读一条 link 能量记录；列名与 CSV 一致（LINK_BASE_vel_x 等）。"""
    need = 8 + 4 + len(body_names) * 80 + 40
    if pos + need > len(raw):
        return None, pos
    off = pos
    timestamp = struct.unpack_from("d", raw, off)[0]
    off += 8
    num_bodies = struct.unpack_from("i", raw, off)[0]
    off += 4
    if num_bodies != len(body_names):
        return None, pos
    row: dict = {"timestamp": timestamp}
    for i in range(num_bodies):
        prefix = body_names[i] if i < len(body_names) else f"body_{i}"
        vals = struct.unpack_from("10d", raw, off)
        off += 80
        row[f"{prefix}_vel_x"] = vals[0]
        row[f"{prefix}_vel_y"] = vals[1]
        row[f"{prefix}_vel_z"] = vals[2]
        row[f"{prefix}_angvel_x"] = vals[3]
        row[f"{prefix}_angvel_y"] = vals[4]
        row[f"{prefix}_angvel_z"] = vals[5]
        row[f"{prefix}_height"] = vals[6]
        row[f"{prefix}_linear_KE"] = vals[7]
        row[f"{prefix}_angular_KE"] = vals[8]
        row[f"{prefix}_PE"] = vals[9]
    energies = struct.unpack_from("5d", raw, off)
    off += 40
    row["total_linear_KE"] = energies[0]
    row["total_angular_KE"] = energies[1]
    row["total_KE"] = energies[2]
    row["total_PE"] = energies[3]
    row["total_energy"] = energies[4]
    return row, off


def read_binary_link_kinetic_energy_data(bin_path: str) -> pd.DataFrame:
    """
    link_kinetic_energy_data.bin：
    - 新格式：文件头 MJLKEN02 + nbody + 各 body 名字，随后每条与旧版相同结构（与 C++ ros_interface 一致）。
    - 旧格式：无头，列名为 body_0_* …（无法区分 LINK_BASE / world）。
    """
    with open(bin_path, "rb") as f:
        raw = f.read()
    if len(raw) < 12:
        return pd.DataFrame()

    data: list = []
    pos = 0
    body_names: Optional[list] = None

    if raw[:8] == b"MJLKEN02":
        pos = 8
        nb = struct.unpack_from("i", raw, pos)[0]
        pos += 4
        if nb < 0 or nb > 512:
            return pd.DataFrame()
        body_names = []
        for _ in range(nb):
            if pos + 4 > len(raw):
                return pd.DataFrame()
            slen = struct.unpack_from("i", raw, pos)[0]
            pos += 4
            if slen < 0 or slen > 1024 or pos + slen > len(raw):
                return pd.DataFrame()
            name = raw[pos : pos + slen].decode("utf-8", errors="replace")
            pos += slen
            body_names.append(name if name else f"body_{len(body_names)}")

        while pos < len(raw):
            row, new_pos = _read_one_link_ke_row_from_buffer(raw, pos, body_names)
            if row is None:
                break
            data.append(row)
            pos = new_pos
        return pd.DataFrame(data)

    # 旧版：首字节即 sim_time（double）
    while pos < len(raw):
        if pos + 8 + 4 > len(raw):
            break
        timestamp = struct.unpack_from("d", raw, pos)[0]
        num_bodies = struct.unpack_from("i", raw, pos + 8)[0]
        if num_bodies < 0 or num_bodies > 512:
            break
        legacy_names = [f"body_{i}" for i in range(num_bodies)]
        row, new_pos = _read_one_link_ke_row_from_buffer(raw, pos, legacy_names)
        if row is None:
            break
        data.append(row)
        pos = new_pos
    return pd.DataFrame(data)


def resolve_paths_from_log_dir(log_dir: str) -> Dict[str, Optional[str]]:
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return {}
    result = {}
    patterns = {
        "contact": "contact_data",
        "sensor_vibration": "sensor_vibration_data",
        "joint_state": "joint_state_data",
        "joint_forces": "joint_forces_data",
        "perturbation": "perturbation_data",
        "link_energy": "link_kinetic_energy_data",
        "policy_switch": "policy_switch",
    }
    for key, prefix in patterns.items():
        found = None
        for ext in (".csv", ".bin"):
            exact = log_path / f"{prefix}{ext}"
            if exact.exists():
                found = str(exact)
                break
            for f in log_path.glob(f"{prefix}_*{ext}"):
                found = str(f)
                break
            if found:
                break
        result[key] = found
    return result


def load_data_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return pd.DataFrame()
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).stem.lower()
    if ext == ".bin":
        if "contact_data" in filename:
            return read_binary_contact_data(file_path)
        if "sensor_vibration_data" in filename:
            return read_binary_sensor_vibration_data(file_path)
        if "joint_state_data" in filename:
            return read_binary_joint_state_data(file_path)
        if "perturbation_data" in filename:
            return read_binary_perturbation_data(file_path)
        if "joint_forces_data" in filename:
            return read_binary_joint_forces_data(file_path)
        if "link_kinetic_energy_data" in filename:
            return read_binary_link_kinetic_energy_data(file_path)
        if "policy_switch" in filename:
            return read_binary_policy_switch(file_path)
        print(f"警告: 无法识别二进制文件类型: {file_path}")
        return pd.DataFrame()
    if ext == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"错误: 读取CSV失败 {file_path}: {e}")
            return pd.DataFrame()
    print(f"警告: 不支持的文件格式: {ext}")
    return pd.DataFrame()


def load_contact_file(path: str) -> pd.DataFrame:
    """接触力：自动 CSV / bin。"""
    return load_data_file(path)


def resolve_contact_path(path_or_dir: str) -> Optional[str]:
    """若为目录则解析 contact_data.csv 或 .bin。"""
    p = Path(path_or_dir)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        return None
    for ext in (".csv", ".bin"):
        for name in (f"contact_data{ext}",):
            q = p / name
            if q.exists():
                return str(q)
        for f in p.glob(f"contact_data_*{ext}"):
            return str(f)
    return None
