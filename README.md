# EngineAI ROS:  Development Guide with ROS2 Interface for EngineAI Robot
<img src="docs/images/pm01.jpg" alt="EngineAI Logo" width="600">

## Overview

EngineAI ROS is a ROS2 package that provides a set of ROS2 nodes and tools for the EngineAI robot. It offers two types of development modes: high-level development and low-level development.

High-level development allows the user to use EngineAI's walking controller by publishing the body velocity command.

Low-level development allows the user to develop their own controller by publishing the joint command.


## ROS2 Interface Protocol
[Interface Protocol Description](src/interface_protocol/README.md)

## Architecture 

### Computing Units
The EngineAI ROS package is designed to run on both of the computing units, with the following architecture:
|Computer Name|Architecture|Description|
|---|---|---|
|Nezha|X86|High-frequency computer, responsible for motion control and data processing|
|Jetson Orin|ARM|Embedded AI computer, responsible for application functionality|

Both computing units are maintained as open platforms by EngineAI.

### Connect to the computer
| Device | IP | SSH Username | SSH Password |
| --- | --- | --- | --- |
| Nezha | 192.168.0.163 | user | 1 |
| Jetson Orin | 192.168.0.162 | ubuntu | ubuntu |

### Workspace Structure
```bash
└── src
    ├── interface_example # Interface examples, recommended to run on Nezha
    ├── interface_protocol # Interface protocols, common modules
    ├── third_party # Third-party libraries
    ├── simulation # Simulation environment running on host
```

## Build & Run

### Development Environment Requirements for Your Host PC
#### Basic Environment
- Ubuntu 22.04
- ROS2 Humble Desktop
- GCC >= 11
- CMake >= 3.22
- Python >= 3.10
#### Software Dependencies
```
sudo apt update
sudo apt install rsync sshpass openssh-client libglfw3-dev libxinerama-dev libxcursor-dev
sudo apt install ros-dev-tools ros-humble-rmw-cyclonedds-cpp ros-humble-ros-base
sudo apt install -y liblcm-dev
sudo apt install -y libglib2.0-dev
sudo apt install -y libgoogle-glog-dev
conda create -n engineai_ros2 python=3.10
conda activate engineai_ros2
# glog：rl_basic_example_CHR；其余：scripts/ 分析（fall_risk、plot_contact_grid、merge、mujoco_data_io、violin、ThicknessCalculate 等）
conda install -y -c conda-forge glog pandas numpy matplotlib seaborn pyyaml scikit-learn scipy psutil tqdm
pip install mujoco==3.3.6 mediapy urdfpy trimesh
# publish_zero_joint_states.py：需 source ROS；check_mnn_tensor.py：可选 MNN Python 绑定

```

<!-- #### LCM (Lightweight Communications and Marshalling) Installation
For the perturbation sampling simulator (推倒采样仿真器), you need to install LCM:
```bash
# Install LCM dependencies
sudo apt --fix-broken install
sudo apt install build-essential libglib2.0-dev cmake
sudo apt install openssl libssl-dev -y
sudo apt update && sudo apt install -y libpcre3-dev libglib2.0-dev pkg-config


# Or install from source for latest version
git clone https://github.com/lcm-proj/lcm.git
cd lcm
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

> **Note**: LCM is required for the perturbation sampling simulator that provides interactive disturbance force control for robot balance testing. -->

#### Environment Variables
Add these environment variables to your ~/.bashrc :
```bash
echo -e '\nexport ROS_DOMAIN_ID=69\nexport ROS_LOCALHOST_ONLY=0\nexport RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc && source ~/.bashrc
# 如果只运行mujoco，需要防止局域网其他机器人的干扰，ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1
```

### Finite State Machine(FSM) for the Development Mode
Normally, a default process is running. Users can control the robot with EngineAI's Joystick(default Logitech F710). This is called the Joystick Mode.

The FSM which includes Joystick Mode, High-level development and Low-level development is depicted as follows, please check through carefully to safely activate the specific mode.

<img src="docs/images/FSM.jpg" alt="EngineAI Logo" width="600">

### Joystick Control
Joystick Control is the simplest mode with which the user can enjoy the default motion offered by EngineAI. Please follow the directions below carefully.
1. get the robot out of the case, make sure the robot hangs on a portable gantry or lay on the ground
2. shortly press the battery button, and then long press it again until the four led lights of the battery are enlightened.
3. wait for 1 minute so that the auto-start of default EngineAI application is finished.
4. press LB + RB to enable the motors
5. press LB + A to turn into pd-stand mode, and now you can put the robot on the ground. Please make sure the ground is flat enough.
6. Follow the FSM flow chart to enjoy the other functions. Do not activate the Joint Bridge Mode(Low-level Development) and the High-level Development Mode.

### Connect to the robot
#### Simulator
```bash
# in host
# terminal 1 开mujoco
cd /home/wang22/engineai/engineai_ros2_workspace && conda activate engineai_ros2
./src/third_party/install.sh
./scripts/build_nodes.sh sim    # or colcon build --packages-select mujoco_simulator , or colcon build
./scripts/build_nodes_4090.sh sim
source install/setup.bash 
# ros2 launch mujoco_simulator mujoco_simulator.launch.py csv_format:=csv or binary
# 修改 pm_v2.yaml protection: enabled:true 启用护具功能（thickness:设置厚度）
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true save_perturbation_csv:=true save_joint_state_csv:=true save_sensor_vibration_csv:=true save_joint_forces_csv:=true save_link_kinetic_energy_csv:=true save_policy_switch_csv:=true csv_format:=csv csv_min_force_n:=400 csv_joint_forces_min_torque_nm:=10
# contact：接触力模长≥ csv_min_force_n 才写入。joint_forces：满足「||F||≥csv_min_force_n」或「||M||≥csv_joint_forces_min_torque_nm」则写入（OR）；后者可与 calculate_fall_risk 中 M_torsion_thr 同量级，||M|| 为子link偶矢模（非单列 M_torsion_mag）。任一侧 0 表示该条件不参与过滤。
# csv_format:=binary 时写 *.bin；policy_switch 同为 policy_switch.bin（魔数 MJPSW01）。脚本见 scripts/mujoco_data_io.py（load_data_file）
# binary 完整性：接触/扰动队列满时阻塞不丢条；物理线程结束前 Drain 并 flush 全部日志流。**Linux 上物理线程使用 64MB 栈（pthread）**，缓解退出时 stack smashing。
# link_kinetic_energy_data.bin 新格式含文件头 MJLKEN02 + 各 body 名（与 CSV 列名一致）；旧 bin 无头，分析脚本只能得到 body_0…（需重新采集）。
# mesh 模型若仅足底 box 与地面相交，摔倒段 contact_force_by_link 可能几乎只有踝部；`serial_links_mesh.xml` 已在膝/躯干增加接地球体以便记录膝、胸腹与地面接触。
# contact_data.bin：进程异常退出时可能夹一条半截记录导致后续整段无法按旧逻辑连续解析；`mujoco_data_io.read_binary_contact_data` 已支持错位后向前搜索重新对齐，避免 bin 里后半段其它 link 的接触力被整段丢弃。
# save_perturbation_csv: 推力；save_policy_switch_csv: RL 切换；留空目录则 ~/data/mujoco_logs/时间戳/

# # 推倒采样仿真器 - 支持交互式干扰力控制
# # 基本启动
# ros2 launch mujoco_simulator perturbation_simulator.launch.py save_contact_csv:=true

# terminal 2 开RL
## XZL policy from engineai, 先开RL控制器，再开mujoco
source install/setup.bash
# 修改 rl_basic_param_XZL.yaml 的 enable_falling_switch 和 falling_switch，启用摔倒时切换 damping / pdstand/ non
# 修改 rl_basic_param_XZL.yaml 的 enable_pdstand_switch 可以选择是 走路policy or PDstand
ros2 launch interface_example rl_basic_example_XZL.launch.py
ros2 launch interface_example rl_basic_example_CHR.launch.py
ros2 launch interface_example rl_basic_example_AMP.launch.py
ros2 launch interface_example rl_basic_example_DAC.launch.py
## 通过修改 pm_v2.yaml 里 perturbation 组参数来改变推力大小 和 护具map； src/simulation/mujoco/assets/config/pm_v2.yaml
## 通过修改 rl_basic_param_XZL.yaml 里 initial_velocity 组参数来改变初始速度

## 滑倒采样（如果没有设置priority属性，MuJoCo会取最大值）
## ground.xml friction="0.1"
## serial_pm_v2_mesh.xml foot 和 toe 的 friction=0.1
## pm_v2.yaml default_force_magnitude = 0.0, auto_sampling = true
## rl_basic_param_XZL.yaml linear velocity = 1.0

## 绊倒采样
## pm_v2_mesh.xml, 取消 terrain.xml的注释
## pm_v2.yaml default_force_magnitude = 0.0, auto_sampling = true
## 前进绊倒，linear: [2.0, 0.0, 0.0], geom pos="2.5 0.0 0.075" type="box" size="0.15 5.0 0.075"
## 后退绊倒，linear: [-2.0, 0.0, 0.0], geom pos="-2.5 0.0 0.075" type="box" size="0.15 5.0 0.075"
## 左移绊倒，linear: [0.0, 2.0, 0.0], geom pos="0.0 2.5 0.075" type="box" size="5.0 0.15 0.075"
## 右移绊倒，linear: [0.0, -2.0, 0.0], geom pos="0.0 -2.5 0.075" type="box" size="5.0 0.15 0.075"


# # 自定义推倒采样参数
# ros2 launch mujoco_simulator perturbation_simulator.launch.py \
#     save_contact_csv:=true \
#     perturb_force_magnitude:=30.0 \
#     perturb_torque_magnitude:=8.0 \
#     perturb_duration:=0.3 \
#     perturb_body_name:=LINK_TORSO_YAW

# 推倒采样键盘控制说明：
# Shift + W/S: 前后向干扰力
# Shift + A/D: 左右向干扰力
# Shift + Q/E/Z/C: 左前/右前/左后/右后（斜 45° 干扰力）
# Shift + U/K: 上下向干扰力
# Shift + G/J: X轴干扰力矩
# Shift + Y/H: Y轴干扰力矩
# Shift + [/]: Z轴干扰力矩
# Shift + 0: 立即停止干扰力
# Shift + +/-: 调整干扰力大小
# Shift + ,/.: 调整干扰力持续时间

# terminal 1&2 自动采集碰撞
chmod +x /home/wang22/engineai/engineai_ros2_workspace/scripts/automated_collection.sh
./scripts/automated_collection.sh
# 连续模式自动采集（MuJoCo/运控常驻，仅 reset 切换 CSV，更快）
chmod +x /home/wang22/engineai/engineai_ros2_workspace/scripts/automated_collection_continuous.sh
./scripts/automated_collection_continuous.sh
./scripts/automated_collection_continuous.sh 2>&1 | tee ~/data/mujoco_logs/run_$(date +%Y%m%d_%H%M%S).log
# 断电摔倒采样
# pm_v2.yaml default_force_magnitude = 0.0, auto_sampling = true, 然后运行下面代码
chmod +x /home/wang22/engineai/engineai_ros2_workspace/scripts/automated_collection_poweroff.sh
./scripts/automated_collection_poweroff.sh

# 加护具仿真
# 修改 sim_manager.h 中最后两行 protection_enabled_  ，  protection_thickness_
# 修改 sim_manager.cc 中的排除列表 excluded_contact_pairs_
# 说明：启用护具后，ApplyProtectionToContactForces 会：
#   1. 只对 > 800N 的接触力做防护衰减（躺着时小力不处理，仿真正常）
#   2. 缩放 efc_force 并重算 qacc/cacc（CSV 中 contact_force 与 link_acc 一致）
# 注意：修改 qacc 会影响仿真行为，可能导致弹跳等问题（约束一致性被破坏）。

# terminal 3
# choose mesh or geometry: in src/simulation/mujoco/assets/config/pm_v2.yaml
# change "use_simplified_geometry: false"
cd /home/wang22/engineai/engineai_ros2_workspace
conda activate engineai_ros2
# plot contact force max
python3 scripts/analyze_contact_forces.py logs/forward-200.0N-20251005_162754/merged_contact_data_forward-200.0N-20251005_162754_20251005_171533.csv
# plot contact point with force（mujoco/mediapy 见上文 Software Dependencies 中 pip 一行）
# 合并多次采样的contact point
# 合并特定方向的数据（生成1个文件）指定输出文件名
python3 scripts/merge_contact_data.py logs/4in1 merged_4in1.csv --pattern "all_directions_merged_*.csv"

## 第一步：分别合并 8 个方向的数据（每个方向生成 1 个文件）

适用于类似目录：`/home/wang22/data/mujoco_logs/only_active_push_1600/8dir-200.0N-0.4s-20260317_235134/`。
该目录下通常为 `forward-200.0N-1/<timestamp>/contact_data.*` 这种结构，所以需要按方向聚合：

```bash
python3 scripts/merge_contact_data.py \
  /home/linslab/data/mujoco_logs/only_passive_push_1600/8dir-200.0N-0.4s-20260321_154532 \
  --add-fall-type \
  --group-by-direction \
  --min-force-n 400
```

`--min-force-n 400`：合并前删掉 **|force_normal| &lt; 400 N** 的行（无 `force_normal` 列时用 `force_magnitude`），减小下游体积与内存；与 **`--fast-append` 不能同时使用**（第二步二次合并若需过滤，应在第一步已过滤）。

注意：这一步会为每个方向生成一个文件，所以不要指定单个 `output_file`。

如需输出二进制（更省空间/更快，且 `fig4_violin.py` 支持读 `.bin`），可加 `--bin`。

## 第二步：把 8 个方向的合并文件再合并成一个大文件（这些文件已经包含 fall_type_info 列）

```bash
python3 scripts/merge_contact_data.py \
  /home/linslab/data/mujoco_logs/only_passive_push_1600/8dir-200.0N-0.4s-20260321_154532 \
  all_directions_merged.csv \
  --pattern "merged_contact_data_*.csv" \
  --fast-append
```



# 对合并后的csv进行碰撞点显示
# 使用机器人坐标系（默认）， 后面的参数 1000 是显示点数量，1、2是 过滤掉足部的点
# python3 mujoco_xml_contact_display.py <csv_file> <xml_file> [world|robot_frame] [max_spheres] [joint_filter] [enable_clustering] [uniform_distribution]
python3 scripts/mujoco_xml_contact_display.py logs/4in1/merged_contact_data_4in1_20251026_194302.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1000 "" true true

# 排除特定关节
python3 scripts/mujoco_xml_contact_display.py csv_file xml_file robot_frame 1500 "1,2,3"

# 启用聚类
python3 scripts/mujoco_xml_contact_display.py csv_file xml_file robot_frame 1500 "" true

# 禁用聚类（默认）
python3 scripts/mujoco_xml_contact_display.py csv_file xml_file robot_frame 1500 "" false

# 均匀分配（每个link获得相同数量的点）
python3 scripts/mujoco_xml_contact_display.py csv_file xml_file robot_frame 1500 "" false true

# 比例分配（默认，接触点多的link获得更多点）
python3 scripts/mujoco_xml_contact_display.py csv_file xml_file robot_frame 1500 "" false false

# 使用世界坐标系
python3 scripts/mujoco_xml_contact_display.py logs/contact_data_20250831_125541.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml world

python3 scripts/mujoco_xml_contact_display.py logs/4in1/merged_contact_data_4in1_20251026_194302.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1000 "" true true

python3 scripts/mujoco_xml_contact_display.py logs/3in1/merged_contact_data_3in1_20251026_175503.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1000 "" true true

python3 scripts/mujoco_xml_contact_display.py logs/test_push_100/merged_contact_data_test_push_100_20251026_033834.csv src/simulation/mujoco/assets/resource/pm_v2_mesh.xml robot_frame 1000 "" true true

# 完整参数说明
# 参数1: CSV文件路径
# 参数2: XML或URDF文件路径
# 参数3: 可视化类型 (sphere|cylinder) - 可选，默认为cylinder
# 参数4: 坐标系统 (world|urdf) - 可选，默认为urdf

# 各部位碰撞力风琴图
```bash
python3 scripts/violin_link_force.py
# 指定 CSV 路径，仅绘制 force_normal 小提琴图
python3 scripts/fig4_violin.py /path/to/all_directions_merged.csv
python3 scripts/fig4_violin.py /path/to/all_directions_merged.csv -o 输出目录
```
1. Shoulder: shoulder_pitch, shoulder_roll;
2. Elbow: shoulder_yaw, elbow_yaw, elbow_pitch;
3. Torso: torso, 不分左右;
4. Hip: base（根据robot_frame_y区分，y+是左，y-是右）, hip_pitch, hip_roll, hip_yaw（robot_frame_z > 0.55m);
5. Knee: hip_yaw(robot_frame_z < 0.55m), knee_pitch # (robot_frame_z >0.23m)
# 6. Crus: knee_pitch(robot_frame_z<0.23m)

# 接触力 / 护具厚度网格图（`plot_contact_grid.py`）
# - 单位约定：
#   - `--target-force`：kN（仅 `chr` 方法）
#   - `--target-pressure`：MPa（仅 `zzq` 方法）
#   - CSV 内 `force_normal` 等一般为 N，勿混用。
# - 主要输出（默认未指定 `-o`）：
#   - 力图：`${stem}_force_grid_plot_${method}.png`
#   - 厚度图：`${stem}_thickness_grid_plot_${method}.png`
#   - 表面积图：`${stem}_surface_area_grid_plot_${method}.png`
#   - 压强图：`${stem}_pressure_grid_plot_${method}.png`
# - YZ 护具查表（TSV，仅生成厚度图且未 `--no-save-yz-tsv` 时输出）：
#   - 前/后：`yz_map_front${_sfx}.tsv` / `yz_map_back${_sfx}.tsv`
#   - `_sfx`：从 CSV 父目录名解析最后一个 `YYYYMMDD_HHMMSS`；否则对父目录名做安全化短后缀
#   - TSV 网格步长（方格大小）：`--yz-map-step`（默认 `0.05m`；文件头会写 `Pixel = xxxm`）
#   - 前后划分：`robot_frame_x >= 0` -> front；`robot_frame_x < 0` -> back
#   - PNG -> TSV 映射：TSV 先用逐点厚度（与厚度 PNG 同管道）计算厚度，再把每个 PNG hexbin 的多边形覆盖到 TSV 的 2D 矩形格上，格内取 `max`（因此 TSV 方格数/位置不一定等于 hexbin 个数）
# - 可选调试/对照：
#   - `--dump-thickness-hex-csv`：把厚度 PNG 用的 `hexbin(reduce=max)` 聚合表写成 `${thickness_png_stem}_hexbin_aggregate.csv`
#     （非像素解码、非必须；用于核对“哪些 hexbin hex 会导致 PNG 的哪些值”。）
#   - `--compare-yz-hex-tsv`：终端打印 PNG hexbin（混合前后点）与 `yz_map_front/back.tsv`（单侧）非零位置对照（可配 `--compare-yz-max-print`）。
# - 指定输出前缀 `-o` 时：
#   - 文件名后缀会变为 `${prefixStem}_force_${method}.png`、`${prefixStem}_thickness_${method}.png`、`${prefixStem}_surface_${method}.png`、`${prefixStem}_pressure_${method}.png`

```bash
# 下面用 MERGED_CSV 表示你的合并接触 CSV（需含 robot_frame_*、force_normal 等）
MERGED_CSV=/home/linslab/data/mujoco_logs/protector_passive_push_1034/8dir-200.0N-0.4s-20260320_025850/all_directions_merged.csv

# 默认：力图 + 厚度图（chr）；--target-force 1.0 即 1 kN；PNG 与 YZ TSV 默认写出
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --no-hardcode --no-force-filter  # 1.0 = 1 kN

# 同上：
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0  # 1.0 kN

# 只画力图，不画厚度 / 表面积 / 压强（不会写 TSV）
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --force-only  # 1.0 kN

# 将 YZ 护具 TSV 写到指定目录（仍走默认 chr，可省略 --method）
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 \
  --yz-tsv-dir src/interface_example/config/pm01/rl_basic/basic/protector_map  # 1.0 kN

# 画厚度图但不写 TSV
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --no-save-yz-tsv  # 1.0 kN

# 导出厚度 PNG 同源的 hexbin 聚合表（用于核对；非必须）
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --dump-thickness-hex-csv

# 终端对照：PNG hexbin 与 yz_map TSV 非零位置对照
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --compare-yz-hex-tsv

# 细分 TSV 网格（例如 0.05m -> 0.025m）
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --thickness-only --yz-map-step 0.025

# zzq 方法（须指定目标压强 MPa）；常配合关闭硬编码厚度与力滤波
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --method zzq --target-pressure 10.0 --no-hardcode --no-force-filter  # 10.0 MPa

# 指定图片输出前缀（力图/厚度图等会基于该路径生成多个 PNG）
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 -o /tmp/contact_grid_out.png  # 1.0 kN

# hexbin 分辨率与 colormap
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 -b 100 -c plasma  # 1.0 kN

# 图尺寸（cm）：全局 / 力图 / 厚度图
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --figsize 12 10  # 1.0 kN
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --force-figsize 12 8 --thickness-figsize 12 10  # 1.0 kN
python3 scripts/plot_contact_grid.py "$MERGED_CSV" --target-force 1.0 --force-figsize 12 8  # 1.0 kN
```




# 绘制 joint force
# 基本使用
python3 scripts/plot_joint_meq.py ~/data/mujoco_logs/joint_forces_data_20251202_175946.csv --figsize 20 12

# 叠加模式
python3 scripts/plot_joint_meq.py ~/data/mujoco_logs/joint_forces_data_20251202_175946.csv --overlay  --figsize 20 12





# terminal 4
# 小球撞地测试
conda activate engineai_ros2
python /home/wang22/engineai/engineai_ros2_workspace/scripts/FreeBallTest/iron_ball_drop_bySensor.py






# terminal 5
# 根据urdf和xml里的初始位置，把机器人多link的mesh合成一个整体mesh
# main中，urdf提供机器人link父子关系，xml提供初始位姿，
chmod +x ./scripts/MeshCombine/install_dependencies.sh
./scripts/MeshCombine/install_dependencies.sh
conda activate urdf_mesh_tools
python ./scripts/MeshCombine/urdf_mesh_combiner.py

git push origin bench

```

<!-- #### Perturbation Sampling Simulator (推倒采样仿真器)
For interactive robot balance testing with disturbance forces:

**Quick Start (推荐):**
```bash
cd /home/wang22/engineai/engineai_ros2_workspace
./scripts/run_perturbation_simulator.sh
```

**Manual Build and Run:**
```bash
# Build the perturbation simulator
cd /home/wang22/engineai/engineai_ros2_workspace
./scripts/build_nodes.sh sim
source install/setup.bash

# Run the perturbation simulator
ros2 run mujoco_simulator perturbation_simulator
``` -->

**Keyboard Controls for Disturbance Forces:**
- Shift + F/B: Forward/Backward force
- Shift + L/R: Left/Right force  
- Shift + U/D: Up/Down force
- Shift + G/J: X-axis torque
- Shift + Y/H: Y-axis torque
- Shift + [/]: Z-axis torque
- Shift + 0: Stop all forces immediately
- Shift + +/-: Increase/Decrease force magnitude
- Shift + ,/.: Increase/Decrease force duration

> **Note**: The perturbation simulator provides interactive disturbance force control for testing robot balance and recovery capabilities. It uses keyboard in ROS2.
> **IMPORTANT**: When running simulation, either do not connect to the physical robot or set `ROS_LOCALHOST_ONLY=1` in your environment to prevent accidental connections.

#### Real robot
Connect to the robot network by Ethernet
   - Configure your network segment to 192.168.0.0/24. You can use the following command to check the network status:
   ```bash
   # in host
   ping 192.168.0.163
   ```
> **WARNING**: Do not use USB network cards as they may cause ROS2 communication issues. Use built-in Ethernet ports for reliable communication with the robot.
### High-level Development
#### Body Velocity Control Example
1. Enter Basic Walk mode(Refer to the FSM flow chart)
2. Run the example

```
# in host
./scripts/build_nodes.sh example
source install/setup.bash
python src/interface_example/scripts/body_velocity_control_example.py
```
### Low-level Development

Low-level development allows the user to use their own RL controllers trained with EngineAI's open-source RL training framework https://github.com/engineai-robotics/engineai_gym. The user can follow the directions below step by step to realize sim2sim functionality with the Mujoco simulator and deployment. 

By default, a simple policy is loaded from the directory of ```src/interface_example/config/pm01/rl_basic/basic/policies/pm01_v2_rough_ppo_42obs.mnn ```. It can realize a simple walk. Users can replace it with their own one generated with our open-source training framework.

#### Deploy: Run the rl_basic_example on your host
1. Enter pd-stand mode
   - Ensure the robot is securely positioned on flat ground with sufficient space around it
   - Use the remote control to enter pd-stand mode
   - Make sure the robot is stable and standing properly before proceeding

2. Enter joint bridge mode
   - Joint bridge mode allows control via joint commands
   - With the robot in pd-stand mode, press the remote control [LB, CROSS_Y_LEFT] combination
   - The robot will enter joint bridge mode, which allows control via joint commands
   - Verify the robot status with the following command:
   ```bash
   # in host
   ros2 topic echo /motion/motion_state
   ```
   - Confirm that the motion state is `joint_bridge` before proceeding

3. Run the RL example
   - **SAFETY WARNING**: Ensure all personnel stand at a safe distance from the robot
   - Be prepared to stop the robot quickly if it behaves unexpectedly（pressing the emergency stop button or entering passive mode）
   - Launch the example with:
   ```bash
   # in host
   ros2 launch interface_example rl_basic_example.launch.py
   ```

> **NOTE**: If the robot falls during simulation, you may need to reset the simulator by pressing the reset button.

<img src="docs/images/sim2sim.gif" alt="Sim2Sim" width="480">

### Compile the code on the target board
If you don't have the PC development environment, you can compile the code on the target board.

#### Sync the code to Board
```bash
# in host
./scripts/sync_src.sh nezha
```

#### Build the workspace

First, ssh into the Nezha board and enter the workspace directory:
```bash
# in nezha
cd ~/source/engineai_workspace
./scripts/build_nodes.sh
```

### Run PlotJuggler for data monitoring
```bash
# in host
colcon build --packages-select interface_protocol
source install/setup.bash
ros2 run plotjuggler plotjuggler -n
```
A default layout file is in `src/interface_protocol/pm_data_layout.xml`

## Get the latest firmware of EngineAI Robot
Refer to the [EngineAI Firmware](https://github.com/engineai-robotics/engineai_firmware) repository for the latest firmware.



## git 操作
```bash
# 定期从EngineAI拉取更新
git fetch upstream

# 合并到community分支
git checkout community
git merge upstream/community

# 同步到bench分支
git checkout bench
git merge community

# 推送到您的远程仓库
git push origin community
git push origin bench
```


## webm 转 mp4
```bash
sudo apt update
sudo apt install ffmpeg -y

ffmpeg -i input.webm output.mp4
ffmpeg -i "input.webm" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 23 -preset fast -c:a aac output.mp4
```


# 碰撞点viewer
## 启动必要的节点
```bash
cd /home/wang22/engineai/engineai_ros2_workspace && source install/setup.bash && ros2 launch launch_urdf_only.launch.py
```

## 在另一个终端启动RViz
```bash
cd /home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/robot/pm_v2/urdf && rviz2 -d robot.rviz
```

# 查看urdf collision
# 使用外部URDF文件，自动启动RViz
```bash
ros2 launch launch_urdf_only.launch.py urdf_file:=/home/wang22/engineai/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/pm01/urdf/serial_pm_v2_primitive.urdf
```


# Risk 计算
## 单次采样的数据处理

### 方式一：指定日志目录（推荐，CSV 在同一文件夹时）
```bash
# 只需指定一个文件夹，自动查找 contact_data.csv、sensor_vibration_data.csv 等
python3 scripts/calculate_fall_risk.py \
  --log-dir /home/linslab/data/mujoco_logs/8dir-200.0N-0.4s-20260321_095833/forward-200.0N-1/20260321_095836 \
  --plot \
  --t-start 3.0 --t-end 5.0
# 输出默认保存到同目录 risk_results_summary.csv 等
```

### 方式二：分别指定各文件路径
```bash
# active using mimic 20260203_135754
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/contact_data_20260312_001449.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/sensor_vibration_data_20260312_001449.csv \
  --joint-state /home/wang22/data/mujoco_logs/joint_state_data_20260312_001449.csv \
  --joint-forces /home/wang22/data/mujoco_logs/joint_forces_data_20260312_001449.csv \
  --link-energy /home/wang22/data/mujoco_logs/link_kinetic_energy_data_20260312_001449.csv \
  --output /home/wang22/data/mujoco_logs/risk_results_20260312_001449 \
  --t-start 2.0 --t-end 4.0 \
  --plot


# --plot 会绘制 link energy 曲线；若不指定 --link-energy，则根据 contact 同目录自动查找 link_kinetic_energy_data_*.csv
# 仿真时需开启 save_link_kinetic_energy_csv: true 才会生成该文件
# test_passive
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/test_passive/Front/contact_data_20260305_203637.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/test_passive/Front/sensor_vibration_data_20260305_203637.csv \
  --joint-state /home/wang22/data/mujoco_logs/test_passive/Front/joint_state_data_20260305_203637.csv \
  --joint-forces /home/wang22/data/mujoco_logs/test_passive/Front/joint_forces_data_20260305_203637.csv \
  --link-energy /home/wang22/data/mujoco_logs/test_passive/Front/link_kinetic_energy_data_20260305_203637.csv \
  --output /home/wang22/data/mujoco_logs/test_passive/Front/risk_results_20260305_203637 \
  --plot

# test_active
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/test_active/Front/contact_data_20260306_141257.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/test_active/Front/sensor_vibration_data_20260306_141257.csv \
  --joint-state /home/wang22/data/mujoco_logs/test_active/Front/joint_state_data_20260306_141257.csv \
  --joint-forces /home/wang22/data/mujoco_logs/test_active/Front/joint_forces_data_20260306_141257.csv \
  --link-energy /home/wang22/data/mujoco_logs/test_active/Front/link_kinetic_energy_data_20260306_141257.csv \
  --output /home/wang22/data/mujoco_logs/test_active/Front/risk_results_20260306_141257 \
  --plot

# test_PDstand 
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/test_PDstand/Front/contact_data_20260307_151446.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/test_PDstand/Front/sensor_vibration_data_20260307_151446.csv \
  --joint-state /home/wang22/data/mujoco_logs/test_PDstand/Front/joint_state_data_20260307_151446.csv \
  --joint-forces /home/wang22/data/mujoco_logs/test_PDstand/Front/joint_forces_data_20260307_151446.csv \
  --link-energy /home/wang22/data/mujoco_logs/test_PDstand/Front/link_kinetic_energy_data_20260307_151446.csv \
  --output /home/wang22/data/mujoco_logs/test_PDstand/Front/risk_results_20260307_151446 \
  --plot

# non_stumble   20260203_140022
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/non_stumble/contact_data_20260203_125000.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/non_stumble/sensor_vibration_data_20260203_125000.csv \
  --joint-state /home/wang22/data/mujoco_logs/non_stumble/joint_state_data_20260203_125000.csv \
  --joint-forces /home/wang22/data/mujoco_logs/non_stumble/joint_forces_data_20260203_125000.csv \
  --link-energy /home/wang22/data/mujoco_logs/non_stumble/link_kinetic_energy_data_20260203_125000.csv \
  --output /home/wang22/data/mujoco_logs/non_stumble/risk_results_20260203_125000 \
  --plot

# non_slip
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/non_slip/contact_data_20260130_133543.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/non_slip/sensor_vibration_data_20260130_133543.csv \
  --joint-state /home/wang22/data/mujoco_logs/non_slip/joint_state_data_20260130_133543.csv \
  --joint-forces /home/wang22/data/mujoco_logs/non_slip/joint_forces_data_20260130_133543.csv \
  --output /home/wang22/data/mujoco_logs/non_slip/risk_results_20260130_133543 \
  --plot

# non_poweroff
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/non_poweroff/contact_data_20260130_131602.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/non_poweroff/sensor_vibration_data_20260130_131602.csv \
  --joint-state /home/wang22/data/mujoco_logs/non_poweroff/joint_state_data_20260130_131602.csv \
  --joint-forces /home/wang22/data/mujoco_logs/non_poweroff/joint_forces_data_20260130_131602.csv \
  --output /home/wang22/data/mujoco_logs/non_poweroff/risk_results_20260130_131602 \
  --plot

# non_push
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/non_push/contact_data_20260130_132708.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/non_push/sensor_vibration_data_20260130_132708.csv \
  --joint-state /home/wang22/data/mujoco_logs/non_push/joint_state_data_20260130_132708.csv \
  --joint-forces /home/wang22/data/mujoco_logs/non_push/joint_forces_data_20260130_132708.csv \
  --output /home/wang22/data/mujoco_logs/non_push/risk_results_20260130_132708 \
  --plot

# passive using risk-guided manner



# passive using heuristic manner
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/passive_heuristic/contact_data_20260129_192350.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/passive_heuristic/sensor_vibration_data_20260129_192350.csv \
  --joint-state /home/wang22/data/mujoco_logs/passive_heuristic/joint_state_data_20260129_192350.csv \
  --joint-forces /home/wang22/data/mujoco_logs/passive_heuristic/joint_forces_data_20260129_192350.csv \
  --output /home/wang22/data/mujoco_logs/passive_heuristic/risk_results_20260129_192350 \
  --plot

# passive using rigid reinforcement strategies


# active using damping mode 
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/active_damping/contact_data_20260129_192809.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/active_damping/sensor_vibration_data_20260129_192809.csv \
  --joint-state /home/wang22/data/mujoco_logs/active_damping/joint_state_data_20260129_192809.csv \
  --joint-forces /home/wang22/data/mujoco_logs/active_damping/joint_forces_data_20260129_192809.csv \
  --output /home/wang22/data/mujoco_logs/active_damping/risk_results_20260129_192809 \
  --plot

# active + passive
python3 scripts/calculate_fall_risk.py \
  --contact /home/wang22/data/mujoco_logs/active_passive/contact_data_20260129_193636.csv \
  --sensor-vibration /home/wang22/data/mujoco_logs/active_passive/sensor_vibration_data_20260129_193636.csv \
  --joint-state /home/wang22/data/mujoco_logs/active_passive/joint_state_data_20260129_193636.csv \
  --joint-forces /home/wang22/data/mujoco_logs/active_passive/joint_forces_data_20260129_193636.csv \
  --output /home/wang22/data/mujoco_logs/active_passive/risk_results_20260129_193636 \
  --plot

# 

```

## 批量采样的数据处理

- 目录可为「实验文件夹下直接放日志」，或「实验文件夹/时间戳/」下再放日志（连续采集常见后者）；脚本两种都支持。
- 每个**实际日志目录**内的 `risk_results_*.csv` 仍由 `calculate_fall_risk.py` 写在该目录（与原始 csv/bin 同文件夹）。
- **`--root` 下**：`all_runs_summary.csv`（全批次各 risk **平均值**，1 行）、`per_run_risk.csv`（每次实验一行明细）、`stats_by_direction.csv`、`stats_overall.txt`。
- 另 **`policy_switch_walking_combined.csv`**：合并各子目录 `policy_switch.csv` / `.bin` 中仅 **`from_mode == walking`** 的行；`--no-combine-policy-switch` 可关闭。
- 子进程 `calculate_fall_risk` 较慢时，stderr 会**周期性刷新剩余/已用时**（见下「长时子任务进度」）。

```bash
python3 scripts/batch_calculate_fall_risk.py \
  --root /home/linslab/data/mujoco_logs/only_passive_push_1600/8dir-200.0N-0.4s-20260321_154532
```

- **长时子任务进度**：每个目录跑 `calculate_fall_risk.py` 时，**stderr** 会每隔约 **10 秒**用同一行刷新（`\r`），显示当前子任务已用时，以及有历史平均耗时后的**本批剩余时间（不含当前子任务）**。关闭：`--no-progress-ticker` 或 `--progress-ticker-interval 0`；改间隔：`--progress-ticker-interval 5`。

同一命令还会在 `--root`（或 `--output-dir`）下生成 **`policy_switch_walking_combined.csv`**：遍历各实验子目录中的 `policy_switch.csv`（或 `.bin`），只保留 **`from_mode` 为 `walking`** 的行（例如摔倒切 damping、定时 pdstand），并附加列 `run_folder`、`direction`、`log_path`。若不需要可传 `--no-combine-policy-switch`。
打印body编号 `python3 scripts/print_pm_v2_body_ids.py`