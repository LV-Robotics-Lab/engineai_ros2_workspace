import ctypes.util
print("GLFW path:", ctypes.util.find_library("glfw"))

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import xml.etree.ElementTree as ET
import time

# 加载模型 - 使用脚本所在目录构建路径
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "iron_ball_drop.xml")

if not os.path.exists(xml_path):
    # 如果当前目录找不到，尝试从工作空间根目录
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    xml_path = os.path.join(workspace_root, "scripts", "FreeBallTest", "iron_ball_drop.xml")
    
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"无法找到 XML 文件: {xml_path}")

print(f"加载 XML 文件: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 获取XML文件所在目录
xml_dir = os.path.dirname(xml_path)

def calculate_ball_mass(radius, density=7860):
    """根据半径和密度计算球体质量"""
    import math
    volume = (4/3) * math.pi * (radius ** 3)
    mass = density * volume
    return mass

def read_xml_parameters(xml_path):
    """从XML文件中读取参数"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    params = {}
    
    # 读取仿真选项
    option = root.find('option')
    if option is not None:
        params['timestep'] = float(option.get('timestep', 0.001))
        params['gravity'] = option.get('gravity', '0 0 -9.81')
        params['integrator'] = option.get('integrator', 'RK4')
        params['tolerance'] = option.get('tolerance', '1e-10')
    
    # 读取铁球信息
    worldbody = root.find('worldbody')
    if worldbody is not None:
        iron_ball = worldbody.find("body[@name='iron_ball']")
        if iron_ball is not None:
            # 读取初始高度
            pos = iron_ball.get('pos', '0 0 1.0')
            pos_parts = pos.split()
            if len(pos_parts) >= 3:
                params['ball_height'] = float(pos_parts[2])
            
            # 读取球体半径和质量
            geom = iron_ball.find("geom[@name='ball_geom']")
            if geom is not None:
                radius = float(geom.get('size', '0.0534'))
                params['ball_radius'] = radius
                params['ball_mass'] = calculate_ball_mass(radius)
    
    # 读取default classes - 修正为处理嵌套的default结构
    outer_defaults = root.findall('default')
    iron_object_found = False
    ground_object_found = False
    
    print(f"找到 {len(outer_defaults)} 个外层 default 元素")
    
    for outer_default in outer_defaults:
        print(f"检查外层 default 元素")
        # 查找内层的default元素
        inner_defaults = outer_default.findall('default')
        print(f"  找到 {len(inner_defaults)} 个内层 default 元素")
        
        for inner_default in inner_defaults:
            class_name = inner_default.get('class', '')
            print(f"  检查内层 default 类: '{class_name}'")
            geom = inner_default.find('geom')
            if geom is not None:
                print(f"    找到 geom 元素，类名: '{class_name}'")
                if class_name == 'iron_object':
                    iron_object_found = True
                    print("    找到 iron_object 类")
                    # 检查必要参数是否存在
                    if geom.get('density') is None:
                        raise ValueError("iron_object 缺少 density 参数")
                    if geom.get('friction') is None:
                        raise ValueError("iron_object 缺少 friction 参数")
                    if geom.get('condim') is None:
                        raise ValueError("iron_object 缺少 condim 参数")
                    if geom.get('solimp') is None:
                        raise ValueError("iron_object 缺少 solimp 参数")
                    if geom.get('solref') is None:
                        raise ValueError("iron_object 缺少 solref 参数")
                    if geom.get('material') is None:
                        raise ValueError("iron_object 缺少 material 参数")
                    
                    params['iron_density'] = float(geom.get('density'))
                    params['iron_friction'] = geom.get('friction')
                    params['iron_condim'] = int(geom.get('condim'))
                    params['iron_solimp'] = geom.get('solimp')
                    params['iron_solref'] = geom.get('solref')
                    params['iron_material'] = geom.get('material')
                    
                elif class_name == 'ground_object':
                    ground_object_found = True
                    print("    找到 ground_object 类")
                    # 检查必要参数是否存在
                    if geom.get('friction') is None:
                        raise ValueError("ground_object 缺少 friction 参数")
                    if geom.get('condim') is None:
                        raise ValueError("ground_object 缺少 condim 参数")
                    if geom.get('solimp') is None:
                        raise ValueError("ground_object 缺少 solimp 参数")
                    if geom.get('solref') is None:
                        raise ValueError("ground_object 缺少 solref 参数")
                    if geom.get('material') is None:
                        raise ValueError("ground_object 缺少 material 参数")
                    
                    params['ground_material'] = geom.get('material')
                    params['ground_friction'] = geom.get('friction')
                    params['ground_condim'] = int(geom.get('condim'))
                    params['ground_solimp'] = geom.get('solimp')
                    params['ground_solref'] = geom.get('solref')
                    params['ground_conaffinity'] = int(geom.get('conaffinity', 7))
            else:
                print(f"    未找到 geom 元素")
    
    print(f"iron_object_found: {iron_object_found}")
    print(f"ground_object_found: {ground_object_found}")
    
    # 检查是否找到了必要的类
    if not iron_object_found:
        raise ValueError("XML文件中未找到 iron_object 类")
    if not ground_object_found:
        raise ValueError("XML文件中未找到 ground_object 类")
    
    # 读取材料
    materials = root.findall('material')
    for material in materials:
        name = material.get('name', '')
        if name == 'iron':
            params['iron_specular'] = float(material.get('specular', 0.3))
            params['iron_reflectance'] = float(material.get('reflectance', 0.1))
    
    return params

def get_contact_forces_mj_contactForce(model, data):
    """使用mj_contactForce方法获取接触力数据"""
    contact_data = []
    
    # 获取接触数量
    ncon = data.ncon
    
    if ncon == 0:
        # 返回空列表和零值
        return contact_data, np.zeros(3), np.zeros(3), 0.0
    
    # 计算所有接触的合力
    total_world_force = np.zeros(3)
    total_world_torque = np.zeros(3)
    
    for i in range(ncon):
        contact = data.contact[i]
        
        # 获取几何体名称
        geom1_name = model.names[model.name_geomadr[contact.geom[0]]:].decode('utf-8').split('\0')[0]
        geom2_name = model.names[model.name_geomadr[contact.geom[1]]:].decode('utf-8').split('\0')[0]
        
        # 使用mj_contactForce获取完整的6D接触力
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, contact_force)
        
        # 将接触坐标系下的力转换为世界坐标系
        world_force = np.zeros(3)
        world_torque = np.zeros(3)
        
        # 正确的坐标系转换
        # contact.frame 在MuJoCo中存储为转置形式，轴在行中
        # frame[0-2]: X轴 (法向量)
        # frame[3-5]: Y轴 (第一个切向量)  
        # frame[6-8]: Z轴 (第二个切向量)
        for j in range(3):
            # 转换力：world_force = frame^T * contact_force
            world_force[j] = (contact.frame[j*3 + 0] * contact_force[0] + 
                             contact.frame[j*3 + 1] * contact_force[1] + 
                             contact.frame[j*3 + 2] * contact_force[2])
            
            # 转换力矩：world_torque = frame^T * contact_torque
            world_torque[j] = (contact.frame[j*3 + 0] * contact_force[3] + 
                              contact.frame[j*3 + 1] * contact_force[4] + 
                              contact.frame[j*3 + 2] * contact_force[5])
        
        # 累加到合力
        total_world_force += world_force
        total_world_torque += world_torque
        
        # 计算单个接触点的合力大小
        contact_force_magnitude = np.linalg.norm(world_force)
        
        # 获取接触点位置
        pos = contact.pos
        
        # 获取body的ID
        body1_id = model.geom_bodyid[contact.geom[0]]
        body2_id = model.geom_bodyid[contact.geom[1]]
        
        # 存储接触数据
        contact_info = {
            'contact_id': i,
            'geom1_name': geom1_name,
            'geom2_name': geom2_name,
            'pos_x': pos[0],
            'pos_y': pos[1], 
            'pos_z': pos[2],
            'force_x': world_force[0],
            'force_y': world_force[1],
            'force_z': world_force[2],
            'force_magnitude': contact_force_magnitude,
            'torque_x': world_torque[0],
            'torque_y': world_torque[1],
            'torque_z': world_torque[2],
            'gap': contact.dist,
            'body1_id': body1_id,
            'body2_id': body2_id
        }
        
        contact_data.append(contact_info)
    
    # 计算总合力大小
    total_force_magnitude = np.linalg.norm(total_world_force)
    
    return contact_data, total_world_force, total_world_torque, total_force_magnitude

# 使用碰撞力方法，不需要传感器配置

# 模拟参数
timesteps = 200000  # 模拟总步数 (增加步数以适应更小的时间步长)
dt = model.opt.timestep
contact_force_log = []  # 接触力数据日志

# 接触力分析变量
impact_started = False
impact_start_time = None
impact_peak_force = 0.0
impact_end_time = None
impact_velocity = None  # 记录撞击时的实际速度

print("启动可视化并运行模拟...")
print(f"时间步长: {dt} 秒")
print(f"总模拟时间: {timesteps * dt} 秒")
print(f"使用碰撞力方法 (mj_contactForce)")

i = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("可视化窗口已打开，开始实时步进模拟...")
    while viewer.is_running() and i < timesteps:
        step_start = time.time()
        mujoco.mj_step(model, data)
        t = data.time
        
        # 使用mj_contactForce获取接触力数据
        contact_data, total_world_force, total_world_torque, total_contact_force_magnitude = get_contact_forces_mj_contactForce(model, data)
        
        # 调试：打印接触力数据
        if i % 1000 == 0:
            print(f"Step {i}, Time {t:.4f}s")
            print(f"接触数量: {len(contact_data)}")
            if len(contact_data) > 0:
                print(f"接触力合力: {total_contact_force_magnitude:.4f} N")
                print(f"接触力向量: ({total_world_force[0]:.4f}, {total_world_force[1]:.4f}, {total_world_force[2]:.4f})")
            
            # 检查球的位置
            ball_pos = data.xpos[1]  # body 1 是 iron_ball
            print(f"球的位置: x={ball_pos[0]:.4f}, y={ball_pos[1]:.4f}, z={ball_pos[2]:.4f}")
            print("---")

        # 记录接触力数据
        contact_force_log.append([t, len(contact_data), total_contact_force_magnitude, 
                                 total_world_force[0], total_world_force[1], total_world_force[2],
                                 total_world_torque[0], total_world_torque[1], total_world_torque[2]])

        # 检测冲击开始（碰撞力方法）
        # 通过检测接触力阈值来判断碰撞开始（>0.001N）
        if not impact_started and total_contact_force_magnitude > 0.001:
            impact_started = True
            impact_start_time = t
            impact_peak_force = total_contact_force_magnitude
            # 记录撞击时的实际速度
            ball_body_id = 1  # iron_ball 的 body ID
            if ball_body_id < model.nbody:
                # 方法1: 使用 cvel (body 速度，包含线速度和角速度)
                # cvel 是 6D 向量：前3个是线速度，后3个是角速度
                cvel_start = ball_body_id * 6
                if cvel_start + 3 <= len(data.cvel):
                    ball_lin_vel = data.cvel[cvel_start:cvel_start+3]
                    # z方向速度（向下为负，取绝对值）
                    impact_velocity = abs(ball_lin_vel[2])
                # 方法2: 如果没有cvel，尝试从qvel获取
                elif len(data.qvel) >= 6:
                    # 对于自由落体球，qvel的前3个可能是线速度
                    if len(data.qvel) >= 3:
                        impact_velocity = abs(data.qvel[2])  # z方向速度
                    else:
                        # 如果qvel只包含平移，直接使用
                        impact_velocity = abs(data.qvel[0]) if len(data.qvel) > 0 else None
                else:
                    impact_velocity = None

        # 冲击过程：记录峰值
        if impact_started:
            impact_peak_force = max(impact_peak_force, total_contact_force_magnitude)
            # 通过检测接触力阈值来判断碰撞结束（<0.1N）
            # 注意：这个阈值可以调整，如果接触力太小则认为碰撞结束
            if total_contact_force_magnitude < 0.1 and impact_end_time is None:
                impact_end_time = t

        viewer.sync()
        i += 1

        # 按仿真时间步进行节拍，避免一闪而过
        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

if i >= timesteps:
    print("模拟完成（达到设定步数）。")
else:
    print("模拟提前结束（可视化窗口已关闭）。")

# 接触力DataFrame
contact_columns = ["time", "num_contacts", "total_force_magnitude", 
                   "force_x", "force_y", "force_z",
                   "torque_x", "torque_y", "torque_z"]
df_contact = pd.DataFrame(contact_force_log, columns=contact_columns)

# 从XML文件中读取参数
xml_params = read_xml_parameters(xml_path)

# 生成包含小球重量和高度的文件名（用于后续保存文件）
ball_mass = xml_params.get('ball_mass', 5.0)
ball_height = xml_params.get('ball_height', 1.0)
ball_radius = xml_params.get('ball_radius', 0.0534)

# 格式化文件名
mass_str = f"{ball_mass:.1f}kg"
height_str = f"{ball_height:.1f}m"
radius_str = f"{ball_radius*1000:.0f}mm"  # 转换为毫米

# 输出碰撞力方法分析结果
if impact_start_time is not None and impact_end_time is not None:
    print("\n=== mj_contactForce方法冲击力分析结果 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    
    # 计算总冲击时间
    total_impact_time = impact_end_time - impact_start_time
    print(f"冲击持续时间: {total_impact_time:.4f} 秒 ({total_impact_time*1000:.1f} 毫秒)")
    print(f"  (注：从仿真数据中检测，开始：接触力>0.001N，结束：接触力<0.1N)")
    print(f"仿真最大冲击力: {impact_peak_force:.2f} N ({impact_peak_force/1000:.2f} kN)")
    
    # 与实测值对比
    measured_force_kN = 85.0  # 实测值：85kN
    measured_force_N = measured_force_kN * 1000
    force_error = impact_peak_force - measured_force_N
    force_error_percent = (force_error / measured_force_N) * 100
    
    print(f"\n=== 实测值对比与误差分析 ===")
    print(f"实测最大冲击力: {measured_force_N:.2f} N ({measured_force_kN:.2f} kN)")
    print(f"仿真最大冲击力: {impact_peak_force:.2f} N ({impact_peak_force/1000:.2f} kN)")
    print(f"绝对误差: {force_error:.2f} N ({abs(force_error)/1000:.2f} kN)")
    print(f"相对误差: {force_error_percent:.2f}%")
    
    if abs(force_error_percent) < 1.0:
        print(f"  ✓ 误差很小（<1%），仿真结果与实测值非常接近")
    elif abs(force_error_percent) < 5.0:
        print(f"  ✓ 误差较小（<5%），仿真结果与实测值接近")
    elif abs(force_error_percent) < 10.0:
        print(f"  ⚠ 误差中等（<10%），可能需要微调参数")
    else:
        print(f"  ⚠ 误差较大（>10%），建议调整碰撞参数")
    
    # 提供调整建议
    print(f"\n=== 参数调整建议 ===")
    current_solref = xml_params.get('iron_solref', '0.0005 1').split()
    current_solref_time = float(current_solref[0])
    current_solimp = xml_params.get('iron_solimp', '0.9 0.95').split()
    
    print(f"当前 solimp: ({current_solimp[0]}, {current_solimp[1]})")
    print(f"当前 solref: ({current_solref[0]}, {current_solref[1]})")
    
    if force_error > 0:
        # 仿真值 > 实测值，需要减小峰值力
        print(f"\n仿真值({impact_peak_force/1000:.2f} kN) > 实测值({measured_force_kN:.2f} kN)")
        print(f"建议：减小峰值力（使接触更软）")
        print(f"  1. 增大 solref[0]（时间常数）：从 {current_solref_time:.6f} 增大到 {current_solref_time * 1.2:.6f} 或更大")
        print(f"     例如：solref=\"{current_solref_time * 1.2:.6f} {current_solref[1]}\"")
        print(f"  2. 增大 solimp[0]（阻尼参数）：从 {current_solimp[0]} 增大到 {min(0.99, float(current_solimp[0]) * 1.05):.3f}")
        print(f"     例如：solimp=\"{min(0.99, float(current_solimp[0]) * 1.05):.3f} {current_solimp[1]}\"")
    else:
        # 仿真值 < 实测值，需要增大峰值力
        print(f"\n仿真值({impact_peak_force/1000:.2f} kN) < 实测值({measured_force_kN:.2f} kN)")
        print(f"建议：增大峰值力（使接触更硬）")
        print(f"  1. 减小 solref[0]（时间常数）：从 {current_solref_time:.6f} 减小到 {current_solref_time * 0.8:.6f} 或更小")
        print(f"     例如：solref=\"{current_solref_time * 0.8:.6f} {current_solref[1]}\"")
        print(f"  2. 减小 solimp[0]（阻尼参数）：从 {current_solimp[0]} 减小到 {max(0.1, float(current_solimp[0]) * 0.95):.3f}")
        print(f"     例如：solimp=\"{max(0.1, float(current_solimp[0]) * 0.95):.3f} {current_solimp[1]}\"")
    
    # 理论验证
    print(f"\n=== 速度验证 ===")
    import math
    # 计算理论撞击速度
    ball_height = xml_params.get('ball_height', 1.0)
    g = 9.81
    theoretical_velocity = math.sqrt(2 * g * ball_height)
    print(f"理论撞击速度: {theoretical_velocity:.2f} m/s")
    
    # 显示实际撞击速度（如果记录了）
    if impact_velocity is not None:
        actual_velocity = abs(impact_velocity)  # z方向向下为负，取绝对值
        print(f"实际撞击速度: {actual_velocity:.2f} m/s (z方向)")
        velocity_diff = abs(theoretical_velocity - actual_velocity)
        velocity_ratio = actual_velocity / theoretical_velocity
        print(f"速度差异: {velocity_diff:.3f} m/s")
        print(f"速度比例: {velocity_ratio:.3f} (接近1.0表示模拟准确)")
        if 0.95 < velocity_ratio < 1.05:
            print(f"  ✓ 实际速度与理论值非常接近，模拟准确")
        else:
            print(f"  ⚠ 实际速度与理论值有差异，可能受到数值误差或重力配置影响")
    else:
        print(f"  ⚠ 未能记录实际撞击速度")
        actual_velocity = theoretical_velocity  # 使用理论值作为备用
    
    print(f"\n=== 动量验证 ===")
    # 计算动量（使用实际速度，如果可用）
    ball_mass = xml_params.get('ball_mass', 5.0)
    if impact_velocity is not None:
        actual_momentum = ball_mass * abs(impact_velocity)
        theoretical_momentum = ball_mass * theoretical_velocity
        print(f"理论动量: {theoretical_momentum:.2f} kg·m/s")
        print(f"实际动量: {actual_momentum:.2f} kg·m/s")
        momentum = actual_momentum  # 使用实际动量
    else:
        momentum = ball_mass * theoretical_velocity
        print(f"撞击动量: {momentum:.2f} kg·m/s (使用理论速度)")
    
    # 根据碰撞力和动量估算碰撞时间
    print(f"\n=== 碰撞时间验证 ===")
    if impact_peak_force > 0:
        estimated_contact_time = momentum / impact_peak_force
        print(f"根据峰值力({impact_peak_force/1000:.2f} kN)和动量估算的碰撞时间: {estimated_contact_time*1000:.3f} ms")
        print(f"仿真中检测的碰撞时间: {total_impact_time*1000:.3f} ms")
        print(f"  (说明：从仿真数据中检测，开始阈值：接触力>0.001N，结束阈值：接触力<0.1N)")
        print(f"  (这不是外部测量值，而是从仿真接触力数据中根据阈值判断的)")
        time_ratio = total_impact_time / estimated_contact_time
        print(f"时间比例: {time_ratio:.2f} (接近1.0表示合理)")
        if 0.5 < time_ratio < 2.0:
            print(f"  ✓ 碰撞时间估算合理")
        else:
            print(f"  ⚠ 碰撞时间估算与实际值差异较大，可能受到接触刚度影响")
            print(f"  (提示：如果比例>2，说明检测的碰撞时间比理论估算长，可能是阈值设置导致)")
    
    # 验证solref参数
    solref = xml_params.get('iron_solref', '0.0005 1').split()
    solref_time = float(solref[0])
    print(f"\n=== solref参数验证 ===")
    print(f"solref[0] (接触时间常数): {solref_time:.6f} s ({solref_time*1000:.3f} ms)")
    print(f"实际碰撞时间: {total_impact_time*1000:.3f} ms")
    if total_impact_time > 0:
        solref_ratio = solref_time / total_impact_time
        print(f"solref时间常数/实际碰撞时间: {solref_ratio:.2f}")
        # solref时间常数通常可以比实际碰撞时间小一些（这是接触刚度的特征时间）
        # 只要在同一数量级（都在毫秒级别），就可以认为是合理的
        if 0.2 <= solref_ratio <= 10.0:
            print(f"  ✓ solref时间常数与实际碰撞时间在同一量级，配置合理")
            print(f"  说明: solref[0]是接触的特征时间常数，通常比实际接触时间稍短是正常的")
        else:
            print(f"  ⚠ solref时间常数与实际碰撞时间差异较大，可能需要调整")
            if solref_ratio < 0.2:
                print(f"  建议: solref[0]过小可能导致接触过度刚硬，可以适当增大")
            else:
                print(f"  建议: solref[0]过大可能导致接触过软，可以适当减小")
    
    # 保存接触力CSV
    csv_filename = f"Contact_Force_mj_contactForce_{mass_str}_{height_str}height_{radius_str}radius.csv"
    csv_path = os.path.join(xml_dir, csv_filename)
    df_contact.to_csv(csv_path, index=False)
    print(f"已保存mj_contactForce接触力数据为 {csv_path}")

# 打印XML碰撞参数
print(f"\n=== XML碰撞参数 ===")

# 从模型中读取铁球参数
ball_geom = model.geom(1)  # iron_ball的geom

print(f"铁球参数:")
print(f"  - 半径: {ball_geom.size[0]:.4f} m ({ball_geom.size[0]*1000:.1f} mm)")
print(f"  - 密度: {xml_params.get('iron_density', 7860):.0f} kg/m³")
print(f"  - 初始高度: {xml_params.get('ball_height', 1.0):.1f} m")

# 计算质量
volume = (4/3) * np.pi * ball_geom.size[0]**3
mass = xml_params['iron_density'] * volume
print(f"  - 质量: {mass:.2f} kg")

friction = xml_params['iron_friction'].split()
print(f"  - 摩擦系数: {friction[0]} (滑动), {friction[1]} (滚动), {friction[2]} (自旋)")
print(f"  - 接触维度: {xml_params['iron_condim']} (包含力矩)")

solimp = xml_params['iron_solimp'].split()
solref = xml_params['iron_solref'].split()
print(f"  - 求解器参数: solimp({solimp[0]}, {solimp[1]}), solref({solref[0]}, {solref[1]})")

print(f"  - 材料: {xml_params['iron_material']}, 镜面反射{xml_params.get('iron_specular', 0.3):.1f}, 反射率{xml_params.get('iron_reflectance', 0.1):.1f}")

print(f"地面参数:")
print(f"  - 材料: {xml_params['ground_material']}")
ground_friction = xml_params['ground_friction'].split()
print(f"  - 摩擦系数: {ground_friction[0]} (滑动), {ground_friction[1]} (滚动), {ground_friction[2]} (自旋)")
print(f"  - 接触维度: {xml_params['ground_condim']} (包含力矩)")
ground_solimp = xml_params['ground_solimp'].split()
ground_solref = xml_params['ground_solref'].split()
print(f"  - 求解器参数: solimp({ground_solimp[0]}, {ground_solimp[1]}), solref({ground_solref[0]}, {ground_solref[1]})")
print(f"  - 接触亲和性: {xml_params.get('ground_conaffinity', 7)}")

# 读取仿真参数
print(f"仿真参数:")
print(f"  - 时间步长: {xml_params.get('timestep', model.opt.timestep)} s")
print(f"  - 重力: {xml_params.get('gravity', '0 0 -9.81')} m/s²")
print(f"  - 积分器: {xml_params.get('integrator', 'RK4')}")
print(f"  - 数值容差: {xml_params.get('tolerance', '1e-12')}")

# 打印冲击时间详细信息
if impact_start_time is not None and impact_end_time is not None:
    total_impact_time = impact_end_time - impact_start_time
    print(f"\n=== mj_contactForce方法冲击时间详细信息 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time*1000:.1f} 毫秒")
    print(f"总冲击时间: {total_impact_time*1000000:.0f} 微秒")

# 绘图
plt.figure(figsize=(15, 10))

# 子图1：碰撞力方法总冲击力
plt.subplot(2, 2, 1)
plt.plot(df_contact["time"], df_contact["total_force_magnitude"], label="Contact Total Force", linewidth=2, color='blue')
if impact_start_time is not None and impact_end_time is not None:
    plt.axvline(impact_start_time, color="blue", linestyle="--", label="Contact Start")
    plt.axvline(impact_end_time, color="green", linestyle="--", label="Contact End")
plt.title("mj_contactForce Method - Total Impact Force")
plt.xlabel("Time (s)")
plt.ylabel("Total Force (N)")
plt.legend()
plt.grid(True)

# 子图2：接触力分量
plt.subplot(2, 2, 2)
plt.plot(df_contact["time"], df_contact["force_x"], label="Force X", linewidth=1, color='red')
plt.plot(df_contact["time"], df_contact["force_y"], label="Force Y", linewidth=1, color='green')
plt.plot(df_contact["time"], df_contact["force_z"], label="Force Z", linewidth=1, color='blue')
plt.title("Contact Force Components")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

# 子图3：接触数量
plt.subplot(2, 2, 3)
plt.plot(df_contact["time"], df_contact["num_contacts"], label="Number of Contacts", linewidth=2, color='purple')
plt.title("Number of Contact Points")
plt.xlabel("Time (s)")
plt.ylabel("Number of Contacts")
plt.legend()
plt.grid(True)

# 子图4：力矩分量
plt.subplot(2, 2, 4)
plt.plot(df_contact["time"], df_contact["torque_x"], label="Torque X", linewidth=1, color='red')
plt.plot(df_contact["time"], df_contact["torque_y"], label="Torque Y", linewidth=1, color='green')
plt.plot(df_contact["time"], df_contact["torque_z"], label="Torque Z", linewidth=1, color='blue')
plt.title("Contact Torque Components")
plt.xlabel("Time (s)")
plt.ylabel("Torque (N·m)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# 保存图片到XML文件所在目录
plot_filename = f"Contact_Force_mj_contactForce_{mass_str}_{height_str}height_{radius_str}radius.png"
plot_path = os.path.join(xml_dir, plot_filename)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"已保存冲击力图表为 {plot_path}")
plt.show()

# 检查是否检测到冲击
if impact_start_time is None or impact_end_time is None:
    print("\n 未检测到有效冲击，可能原因：")
    print("- 球未接触地面（位置、半径、重力设定错误）")
    print("- 受力太小，未超过设定阈值")
    print("- mj_contactForce 未检测到接触")





