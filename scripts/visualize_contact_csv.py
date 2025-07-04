import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time

# 读取CSV
df = pd.read_csv('logs/contact_data_20250701_171101.csv')
grouped = list(df.groupby('timestamp'))

# 加载模型
model = mujoco.MjModel.from_xml_path('src/simulation/mujoco/assets/resource/robot/pm_v2/xml/serial_pm_v2.xml')
data = mujoco.MjData(model)

# 准备数据
positions_list = []
forces_list = []

for _, group in grouped:
    positions_list.append(group[['pos_x', 'pos_y', 'pos_z']].values)
    forces_list.append(group[['force_x', 'force_y', 'force_z']].values)

# 使用基础的 launch_passive，不传 render_callback
with mujoco.viewer.launch_passive(model, data) as viewer:
    for frame_idx in range(len(positions_list)):
        positions = positions_list[frame_idx]
        forces = forces_list[frame_idx]
        
        # 清除之前的 marker
        viewer.scn.ngeom = 0
        
        # 添加新的 marker
        for pos, force in zip(positions, forces):
            # 添加球体marker显示碰撞点
            mujoco.mjv_addMarker(
                viewer.scn, model, data,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.01, 0.01, 0.01]),
                pos,
                np.zeros(3),
                np.array([1, 0, 0, 1]),
                None
            )
            # 添加箭头marker显示碰撞力
            norm = np.linalg.norm(force)
            if norm > 1e-6:
                force_dir = force / norm
                mujoco.mjv_addMarker(
                    viewer.scn, model, data,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.array([0.005, 0.005, norm * 0.05]),
                    pos,
                    force_dir,
                    np.array([0, 0, 1, 1]),
                    None
                )
        
        viewer.sync()
        time.sleep(0.01)

print("播放完毕")

help(mujoco.viewer.launch_passive)

print(mujoco.__version__)
print(hasattr(mujoco.viewer, "Viewer"))