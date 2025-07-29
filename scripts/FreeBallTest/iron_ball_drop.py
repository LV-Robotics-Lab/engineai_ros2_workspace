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
contact_force_log = []  # 新增：接触力数据日志

# 记录每个传感器的最大力
max_forces = {name: 0.0 for name in sensor_names}

impact_started = False
impact_start_time = None
impact_peak_force = 0.0
impact_end_time = None

# 新增：接触力分析变量
contact_impact_started = False
contact_impact_start_time = None
contact_impact_peak_force = 0.0
contact_impact_end_time = None

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
    
    # 新增：使用mj_contactForce获取接触力数据
    contact_data, total_world_force, total_world_torque, total_contact_force_magnitude = get_contact_forces_mj_contactForce(model, data)
    
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
        
        # 新增：打印接触力数据
        print(f"接触数量: {len(contact_data)}")
        if len(contact_data) > 0:
            print(f"接触力合力: {total_contact_force_magnitude:.4f} N")
            print(f"接触力向量: ({total_world_force[0]:.4f}, {total_world_force[1]:.4f}, {total_world_force[2]:.4f})")
        
        # 检查球的位置
        ball_pos = data.xpos[1]  # body 1 是 iron_ball
        print(f"球的位置: x={ball_pos[0]:.4f}, y={ball_pos[1]:.4f}, z={ball_pos[2]:.4f}")
        print("---")

    # 记录数据（包含所有传感器）
    force_log.append([t] + list(sensor_data) + [total_force, max_force])
    
    # 新增：记录接触力数据
    contact_force_log.append([t, len(contact_data), total_contact_force_magnitude, 
                             total_world_force[0], total_world_force[1], total_world_force[2],
                             total_world_torque[0], total_world_torque[1], total_world_torque[2]])

    # 检测冲击开始（传感器方法）
    if not impact_started and total_force > 0.001:  # 检测接触开始
        impact_started = True
        impact_start_time = t
        impact_peak_force = total_force

    # 冲击过程：记录峰值（传感器方法）
    if impact_started:
        impact_peak_force = max(impact_peak_force, total_force)
        if total_force < 0.1 and impact_end_time is None:  # 调整结束条件
            impact_end_time = t
    
    # 新增：检测冲击开始（接触力方法）
    if not contact_impact_started and total_contact_force_magnitude > 0.001:
        contact_impact_started = True
        contact_impact_start_time = t
        contact_impact_peak_force = total_contact_force_magnitude

    # 冲击过程：记录峰值（接触力方法）
    if contact_impact_started:
        contact_impact_peak_force = max(contact_impact_peak_force, total_contact_force_magnitude)
        if total_contact_force_magnitude < 0.1 and contact_impact_end_time is None:
            contact_impact_end_time = t

print("模拟完成，启动可视化...")
# 可视化模拟（在模拟完成后）
mujoco.viewer.launch(model, data)

# 转为 DataFrame
columns = ["time"] + [f"force_{name}" for name in sensor_names] + ["total_force", "max_force"]
df = pd.DataFrame(force_log, columns=columns)

# 新增：接触力DataFrame
contact_columns = ["time", "num_contacts", "total_force_magnitude", 
                   "force_x", "force_y", "force_z",
                   "torque_x", "torque_y", "torque_z"]
df_contact = pd.DataFrame(contact_force_log, columns=contact_columns)

# 输出分析结果
if impact_start_time is not None and impact_end_time is not None:
    print("\n=== 传感器方法冲击力分析结果 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    
    # 计算总冲击时间
    total_impact_time = impact_end_time - impact_start_time
    print(f"冲击持续时间: {total_impact_time:.4f} 秒 ({total_impact_time*1000:.1f} 毫秒)")
    print(f"最大冲击力: {impact_peak_force:.2f} N")
    
    # 保存CSV到XML文件所在目录
    csv_path = os.path.join(xml_dir, "Free_Fall_Impact_Force_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"已保存传感器接触力数据为 {csv_path}")

# 新增：输出接触力方法分析结果
if contact_impact_start_time is not None and contact_impact_end_time is not None:
    print("\n=== mj_contactForce方法冲击力分析结果 ===")
    print(f"冲击开始时间: {contact_impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {contact_impact_end_time:.4f} 秒")
    
    # 计算总冲击时间
    contact_total_impact_time = contact_impact_end_time - contact_impact_start_time
    print(f"冲击持续时间: {contact_total_impact_time:.4f} 秒 ({contact_total_impact_time*1000:.1f} 毫秒)")
    print(f"最大冲击力: {contact_impact_peak_force:.2f} N")
    
    # 保存接触力CSV
    contact_csv_path = os.path.join(xml_dir, "Contact_Force_mj_contactForce_log.csv")
    df_contact.to_csv(contact_csv_path, index=False)
    print(f"已保存mj_contactForce接触力数据为 {contact_csv_path}")
    
    # 对比两种方法的结果
    print("\n=== 两种方法对比 ===")
    print(f"传感器方法 - 冲击时间: {total_impact_time*1000:.1f} ms, 最大力: {impact_peak_force:.2f} N")
    print(f"接触力方法 - 冲击时间: {contact_total_impact_time*1000:.1f} ms, 最大力: {contact_impact_peak_force:.2f} N")
    
    # 计算差异
    time_diff = abs(total_impact_time - contact_total_impact_time) * 1000
    force_diff = abs(impact_peak_force - contact_impact_peak_force)
    print(f"时间差异: {time_diff:.1f} ms")
    print(f"力差异: {force_diff:.2f} N")

# 打印XML碰撞参数
print(f"\n=== XML碰撞参数 ===")

# 从XML文件中读取参数
xml_params = read_xml_parameters(xml_path)

# 从模型中读取铁球参数
ball_geom = model.geom(1)  # iron_ball的geom

print(f"铁球参数:")
print(f"  - 半径: {ball_geom.size[0]:.4f} m")
print(f"  - 密度: {xml_params.get('iron_density', 7860):.0f} kg/m³")

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
if impact_start_time is not None and impact_end_time is not None:
    print(f"\n=== 传感器方法冲击时间详细信息 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time*1000:.1f} 毫秒")
    print(f"总冲击时间: {total_impact_time*1000000:.0f} 微秒")

if contact_impact_start_time is not None and contact_impact_end_time is not None:
    print(f"\n=== 接触力方法冲击时间详细信息 ===")
    print(f"冲击开始时间: {contact_impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {contact_impact_end_time:.4f} 秒")
    print(f"总冲击时间: {contact_total_impact_time:.4f} 秒")
    print(f"总冲击时间: {contact_total_impact_time*1000:.1f} 毫秒")
    print(f"总冲击时间: {contact_total_impact_time*1000000:.0f} 微秒")

# 绘图
plt.figure(figsize=(20, 10))

# 子图1：传感器方法总冲击力
plt.subplot(2, 3, 1)
plt.plot(df["time"], df["total_force"], label="Sensor Total Force", linewidth=2, color='red')
if impact_start_time is not None and impact_end_time is not None:
    plt.axvline(impact_start_time, color="red", linestyle="--", label="Contact Start")
    plt.axvline(impact_end_time, color="green", linestyle="--", label="Contact End")
plt.title("Sensor Method - Total Impact Force")
plt.xlabel("Time (s)")
plt.ylabel("Total Force (N)")
plt.legend()
plt.grid(True)

# 子图2：接触力方法总冲击力
plt.subplot(2, 3, 2)
plt.plot(df_contact["time"], df_contact["total_force_magnitude"], label="Contact Total Force", linewidth=2, color='blue')
if contact_impact_start_time is not None and contact_impact_end_time is not None:
    plt.axvline(contact_impact_start_time, color="blue", linestyle="--", label="Contact Start")
    plt.axvline(contact_impact_end_time, color="green", linestyle="--", label="Contact End")
plt.title("mj_contactForce Method - Total Impact Force")
plt.xlabel("Time (s)")
plt.ylabel("Total Force (N)")
plt.legend()
plt.grid(True)

# 子图3：两种方法对比
plt.subplot(2, 3, 3)
plt.plot(df["time"], df["total_force"], label="Sensor Method", linewidth=2, color='red')
plt.plot(df_contact["time"], df_contact["total_force_magnitude"], label="Contact Method", linewidth=2, color='blue')
plt.title("Comparison of Two Methods")
plt.xlabel("Time (s)")
plt.ylabel("Total Force (N)")
plt.legend()
plt.grid(True)

# 子图4：接触力分量
plt.subplot(2, 3, 4)
plt.plot(df_contact["time"], df_contact["force_x"], label="Force X", linewidth=1, color='red')
plt.plot(df_contact["time"], df_contact["force_y"], label="Force Y", linewidth=1, color='green')
plt.plot(df_contact["time"], df_contact["force_z"], label="Force Z", linewidth=1, color='blue')
plt.title("Contact Force Components")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

# 子图5：接触数量
plt.subplot(2, 3, 5)
plt.plot(df_contact["time"], df_contact["num_contacts"], label="Number of Contacts", linewidth=2, color='purple')
plt.title("Number of Contact Points")
plt.xlabel("Time (s)")
plt.ylabel("Number of Contacts")
plt.legend()
plt.grid(True)

# 子图6：3D图显示各传感器力的最大值
ax3d = plt.subplot(2, 3, 6, projection='3d')

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
plot_path = os.path.join(xml_dir, "Free_Fall_Impact_Force_log.png")
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
csv_3d_path = os.path.join(xml_dir, "sensor_3d_data.csv")
df_3d.to_csv(csv_3d_path, index=False)
print(f"已保存3D传感器数据为 {csv_3d_path}")

# 检查是否检测到冲击
if impact_start_time is None or impact_end_time is None:
    print("\n 未检测到有效冲击，可能原因：")
    print("- 球未接触地面（位置、半径、重力设定错误）")
    print("- 受力太小，未超过设定阈值")
    print("- sensordata 未正确读取 site 上的受力")





