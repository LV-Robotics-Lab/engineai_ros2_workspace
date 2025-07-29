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

# 加载模型
xml_path = "engineai_ros2_workspace/scripts/FreeBallTest/iron_ball_drop.xml"
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
    
    # 读取default classes
    defaults = root.findall('default')
    for default in defaults:
        class_name = default.get('class', '')
        geom = default.find('geom')
        if geom is not None:
            if class_name == 'iron_object':
                params['iron_density'] = float(geom.get('density', 7860))
                params['iron_friction'] = geom.get('friction', '0.5 0.005 0.0001')
                params['iron_condim'] = int(geom.get('condim', 3))
                params['iron_solimp'] = geom.get('solimp', '0.9 0.95')
                params['iron_solref'] = geom.get('solref', '0.001 1')
                params['iron_material'] = geom.get('material', 'iron')
            elif class_name == 'ground_object':
                params['ground_material'] = geom.get('material', 'iron')
                params['ground_friction'] = float(geom.get('friction', 1.0))
                params['ground_condim'] = int(geom.get('condim', 3))
                params['ground_conaffinity'] = int(geom.get('conaffinity', 7))
    
    # 读取材料
    materials = root.findall('material')
    for material in materials:
        name = material.get('name', '')
        if name == 'iron':
            params['iron_specular'] = float(material.get('specular', 0.3))
            params['iron_reflectance'] = float(material.get('reflectance', 0.1))
    
    return params

# 创建传感器位置映射
def create_sensor_positions():
    """创建传感器位置映射，返回位置字典和传感器名称列表"""
    positions = {}
    sensor_names = []
    
    # 中心传感器
    positions['center'] = (0, 0)
    sensor_names.append('center')
    
    # 北方向传感器 (y轴正方向)
    for i in range(1, 6):
        name = f'n{i}'
        positions[name] = (0, i * 0.01)
        sensor_names.append(name)
    
    # 南方向传感器 (y轴负方向)
    for i in range(1, 6):
        name = f's{i}'
        positions[name] = (0, -i * 0.01)
        sensor_names.append(name)
    
    # 东方向传感器 (x轴正方向)
    for i in range(1, 6):
        name = f'e{i}'
        positions[name] = (i * 0.01, 0)
        sensor_names.append(name)
    
    # 西方向传感器 (x轴负方向)
    for i in range(1, 6):
        name = f'w{i}'
        positions[name] = (-i * 0.01, 0)
        sensor_names.append(name)
    
    # 东北方向传感器
    for i in range(1, 6):
        name = f'ne{i}'
        positions[name] = (i * 0.01, i * 0.01)
        sensor_names.append(name)
    
    # 西北方向传感器
    for i in range(1, 6):
        name = f'nw{i}'
        positions[name] = (-i * 0.01, i * 0.01)
        sensor_names.append(name)
    
    # 东南方向传感器
    for i in range(1, 6):
        name = f'se{i}'
        positions[name] = (i * 0.01, -i * 0.01)
        sensor_names.append(name)
    
    # 西南方向传感器
    for i in range(1, 6):
        name = f'sw{i}'
        positions[name] = (-i * 0.01, -i * 0.01)
        sensor_names.append(name)
    
    # 填充其他位置的传感器
    for i in range(1, 65):
        name = f'{i:02d}'
        # 根据编号计算位置
        row = (i - 1) // 8
        col = (i - 1) % 8
        x = (col - 3.5) * 0.01
        y = (row - 3.5) * 0.01
        positions[name] = (x, y)
        sensor_names.append(name)
    
    return positions, sensor_names

# 创建传感器位置映射
sensor_positions, sensor_names = create_sensor_positions()
num_sensors = len(sensor_names)

# 调试信息：检查传感器配置
print(f"传感器数量: {model.nsensor}")
print(f"传感器名称: {[model.sensor(i).name for i in range(min(10, model.nsensor))]}...")
print(f"传感器类型: {[model.sensor(i).type for i in range(min(10, model.nsensor))]}...")

# 模拟参数
timesteps = 200000  # 模拟总步数 (增加步数以适应更小的时间步长)
dt = model.opt.timestep
force_log = []

# 记录每个传感器的最大力
max_forces = {name: 0.0 for name in sensor_names}

impact_started = False
impact_start_time = None
impact_peak_force = 0.0
impact_end_time = None

print("开始运行模拟...")
print(f"时间步长: {dt} 秒")
print(f"总模拟时间: {timesteps * dt} 秒")
print(f"传感器数量: {num_sensors}")

# 运行模拟
for i in range(timesteps):
    mujoco.mj_step(model, data)
    
    # 获取所有touch传感器数据
    sensor_data = data.sensordata[:num_sensors] if len(data.sensordata) >= num_sensors else [0.0] * num_sensors
    total_force = sum(sensor_data)  # 总冲击力
    max_force = max(sensor_data)    # 最大单个传感器力
    t = i * dt
    
    # 更新每个传感器的最大力
    for j, name in enumerate(sensor_names):
        if j < len(sensor_data):
            max_forces[name] = max(max_forces[name], sensor_data[j])
    
    # 调试：打印传感器数据
    if i % 1000 == 0:
        print(f"Step {i}, Time {t:.4f}s")
        print(f"传感器数据: {[f'{f:.4f}' for f in sensor_data[:10]]}...")
        print(f"总冲击力: {total_force:.4f} N")
        print(f"最大传感器力: {max_force:.4f} N")
        print(f"传感器数据形状: {data.sensordata.shape}")
        
        # 检查球的位置
        ball_pos = data.xpos[1]  # body 1 是 iron_ball
        print(f"球的位置: x={ball_pos[0]:.4f}, y={ball_pos[1]:.4f}, z={ball_pos[2]:.4f}")
        print("---")

    # 记录数据（包含所有传感器）
    force_log.append([t] + list(sensor_data) + [total_force, max_force])

    # 检测冲击开始
    if not impact_started and total_force > 0.001:  # 检测接触开始
        impact_started = True
        impact_start_time = t
        impact_peak_force = total_force

    # 冲击过程：记录峰值
    if impact_started:
        impact_peak_force = max(impact_peak_force, total_force)
        if total_force < 0.1 and impact_end_time is None:  # 调整结束条件
            impact_end_time = t

print("模拟完成，启动可视化...")
# 可视化模拟（在模拟完成后）
mujoco.viewer.launch(model, data)

# 转为 DataFrame
columns = ["time"] + [f"force_{name}" for name in sensor_names] + ["total_force", "max_force"]
df = pd.DataFrame(force_log, columns=columns)

# 输出分析结果
if impact_start_time is not None and impact_end_time is not None:

    # 从XML文件中读取参数
    xml_params = read_xml_parameters(xml_path)

    print("\n=== 冲击力分析结果 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    
    # 计算总冲击时间
    total_impact_time = impact_end_time - impact_start_time
    print(f"冲击持续时间: {total_impact_time:.4f} 秒 ({total_impact_time*1000:.1f} 毫秒)")
    print(f"最大冲击力: {impact_peak_force:.2f} N")
    
    # 生成包含小球重量和高度的文件名
    ball_mass = xml_params.get('ball_mass', 5.0)
    ball_height = xml_params.get('ball_height', 1.0)
    ball_radius = xml_params.get('ball_radius', 0.0534)
    
    # 格式化文件名
    mass_str = f"{ball_mass:.1f}kg"
    height_str = f"{ball_height:.1f}m"
    radius_str = f"{ball_radius*1000:.0f}mm"  # 转换为毫米
    
    # 保存CSV到XML文件所在目录
    csv_filename = f"Free_Fall_Impact_Force_{mass_str}_{height_str}height_{radius_str}radius.csv"
    csv_path = os.path.join(xml_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"已保存接触力数据为 {csv_path}")
    
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
    mass = xml_params.get('iron_density', 7860) * volume
    print(f"  - 质量: {mass:.2f} kg")
    
    friction = xml_params.get('iron_friction', '0.5 0.005 0.0001').split()
    print(f"  - 摩擦系数: {friction[0]} (滑动), {friction[1]} (滚动), {friction[2]} (自旋)")
    print(f"  - 接触维度: {xml_params.get('iron_condim', 6)} (包含力矩)")
    
    solimp = xml_params.get('iron_solimp', '0.9 0.95').split()
    solref = xml_params.get('iron_solref', '0.001 1').split()
    print(f"  - 求解器参数: solimp({solimp[0]}, {solimp[1]}), solref({solref[0]}, {solref[1]})")
    
    print(f"  - 材料: {xml_params.get('iron_material', 'iron')}, 镜面反射{xml_params.get('iron_specular', 0.3):.1f}, 反射率{xml_params.get('iron_reflectance', 0.1):.1f}")
    
    print(f"地面参数:")
    print(f"  - 材料: {xml_params.get('ground_material', 'iron')}")
    print(f"  - 摩擦系数: {xml_params.get('ground_friction', 1.0):.1f}")
    print(f"  - 接触维度: {xml_params.get('ground_condim', 6)} (包含力矩)")
    print(f"  - 接触亲和性: {xml_params.get('ground_conaffinity', 7)}")
    
    # 读取仿真参数
    print(f"仿真参数:")
    print(f"  - 时间步长: {xml_params.get('timestep', model.opt.timestep)} s")
    print(f"  - 重力: {xml_params.get('gravity', '0 0 -9.81')} m/s²")
    print(f"  - 积分器: {xml_params.get('integrator', 'RK4')}")
    print(f"  - 数值容差: {xml_params.get('tolerance', '1e-12')}")
    
    # 打印冲击时间详细信息
    print(f"\n=== 冲击时间详细信息 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time*1000:.1f} 毫秒")
    print(f"总冲击时间: {total_impact_time*1000000:.0f} 微秒")
    
    # 绘图
    plt.figure(figsize=(15, 6))
    
    # 子图1：总冲击力
    plt.subplot(1, 2, 1)
    plt.plot(df["time"], df["total_force"], label="Total Force", linewidth=2, color='red')
    plt.axvline(impact_start_time, color="red", linestyle="--", label="Contact Start")
    plt.axvline(impact_end_time, color="green", linestyle="--", label="Contact End")
    plt.title("Free Fall Impact Force Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Force (N)")
    plt.legend()
    plt.grid(True)
    
    # 子图2：3D图显示各传感器力的最大值
    ax3d = plt.subplot(1, 2, 2, projection='3d')
    
    # 准备3D数据
    x_coords = []
    y_coords = []
    z_coords = []
    
    for name in sensor_names:
        if name in sensor_positions:
            x, y = sensor_positions[name]
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(max_forces[name])
    
    # 创建3D散点图
    scatter = ax3d.scatter(x_coords, y_coords, z_coords, 
                           c=z_coords, cmap='viridis', s=50, alpha=0.8)
    
    ax3d.set_xlabel('X Position (m)')
    ax3d.set_ylabel('Y Position (m)')
    ax3d.set_zlabel('Max Force (N)')
    ax3d.set_title('3D Sensor Force Distribution')
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax3d, label='Max Force (N)')
    
    plt.tight_layout()
    
    
    # 保存图片到XML文件所在目录
    plot_filename = f"Free_Fall_Impact_Force_{mass_str}_{height_str}height_{radius_str}radius.png"
    plot_path = os.path.join(xml_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"已保存冲击力图表为 {plot_path}")
    plt.show()
    
    # 保存3D数据
    sensor_data_3d = {
        'sensor_name': sensor_names,
        'x_position': [sensor_positions[name][0] for name in sensor_names],
        'y_position': [sensor_positions[name][1] for name in sensor_names],
        'max_force': [max_forces[name] for name in sensor_names]
    }
    
    df_3d = pd.DataFrame(sensor_data_3d)
    csv_filename = f"sensor_3d_data_{mass_str}_{height_str}height_{radius_str}radius.csv"
    csv_3d_path = os.path.join(xml_dir, csv_filename)
    df_3d.to_csv(csv_3d_path, index=False)
    print(f"已保存3D传感器数据为 {csv_3d_path}")

else:
    print("\n 未检测到有效冲击，可能原因：")
    print("- 球未接触地面（位置、半径、重力设定错误）")
    print("- 受力太小，未超过设定阈值")
    print("- sensordata 未正确读取 site 上的受力")





