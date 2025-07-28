import ctypes.util
print("GLFW path:", ctypes.util.find_library("glfw"))

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
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

# 调试信息：检查传感器配置
print(f"传感器数量: {model.nsensor}")
print(f"传感器名称: {[model.sensor(i).name for i in range(model.nsensor)]}")
print(f"传感器类型: {[model.sensor(i).type for i in range(model.nsensor)]}")

# 模拟参数
timesteps = 200000  # 模拟总步数 (增加步数以适应更小的时间步长)
dt = model.opt.timestep
force_log = []

# 传感器阵列参数
sensor_names = ["center", "n", "s", "e", "w", "ne", "nw", "se", "sw"]
num_sensors = len(sensor_names)

impact_started = False
impact_start_time = None
impact_peak_force = 0.0
impact_end_time = None

print("开始运行模拟...")
print(f"时间步长: {dt} 秒")
print(f"总模拟时间: {timesteps * dt} 秒")
# 运行模拟
for i in range(timesteps):
    mujoco.mj_step(model, data)
    
    # 获取所有touch传感器数据
    sensor_data = data.sensordata[:num_sensors] if len(data.sensordata) >= num_sensors else [0.0] * num_sensors
    total_force = sum(sensor_data)  # 总冲击力
    max_force = max(sensor_data)    # 最大单个传感器力
    t = i * dt
    
    # 调试：打印传感器数据
    if i % 100 == 0:
        print(f"Step {i}, Time {t:.4f}s")
        print(f"传感器数据: {[f'{f:.4f}' for f in sensor_data]}")
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
    print("\n=== 冲击力分析结果 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    
    # 计算总冲击时间
    total_impact_time = impact_end_time - impact_start_time
    print(f"冲击持续时间: {total_impact_time:.4f} 秒 ({total_impact_time*1000:.1f} 毫秒)")
    print(f"最大冲击力: {impact_peak_force:.2f} N")
    
    # 保存CSV到XML文件所在目录
    csv_path = os.path.join(xml_dir, "Free_Fall_Impact_Force_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"已保存接触力数据为 {csv_path}")
    
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
    print(f"\n=== 冲击时间详细信息 ===")
    print(f"冲击开始时间: {impact_start_time:.4f} 秒")
    print(f"冲击结束时间: {impact_end_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time:.4f} 秒")
    print(f"总冲击时间: {total_impact_time*1000:.1f} 毫秒")
    print(f"总冲击时间: {total_impact_time*1000000:.0f} 微秒")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 子图1：总冲击力
    plt.subplot(2, 1, 1)
    plt.plot(df["time"], df["total_force"], label="Total Force", linewidth=2, color='red')
    plt.axvline(impact_start_time, color="red", linestyle="--", label="Contact Start")
    plt.axvline(impact_end_time, color="green", linestyle="--", label="Contact End")
    plt.title("Free Fall Impact Force Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Force (N)")
    plt.legend()
    plt.grid(True)
    
    # 子图2：各传感器力
    plt.subplot(2, 1, 2)
    for name in sensor_names:
        plt.plot(df["time"], df[f"force_{name}"], label=f"Sensor {name.upper()}", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Individual Sensor Force (N)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图片到XML文件所在目录
    plot_path = os.path.join(xml_dir, "Free_Fall_Impact_Force_log.png")
    plt.savefig(plot_path)
    print(f"已保存冲击力图表为 {plot_path}")
    plt.show()

else:
    print("\n 未检测到有效冲击，可能原因：")
    print("- 球未接触地面（位置、半径、重力设定错误）")
    print("- 受力太小，未超过设定阈值")
    print("- sensordata 未正确读取 site 上的受力")





