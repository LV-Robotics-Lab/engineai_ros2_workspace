# URDF坐标修正功能指南

## 问题背景

在之前的实现中，接触点在URDF坐标系中的位置计算存在一个问题：当机器人在碰撞时关节已经偏离了初始角度，但接触点的URDF坐标计算没有考虑这种变化，导致显示的碰撞点位置不准确。

## 问题分析

### 原始问题
1. **碰撞时的关节状态**: 机器人在碰撞时，各关节已经偏离了XML文件中定义的初始角度
2. **坐标系转换**: 接触点位置需要从世界坐标系转换到URDF坐标系
3. **显示问题**: 在viewer中导入XML（初始位置）和碰撞点后，显示的碰撞点位置不准确

### 根本原因
- 原始代码只是简单地将接触点从世界坐标系转换到当前body的局部坐标系
- 没有考虑从当前关节角度到初始关节角度的变换
- 导致碰撞点在URDF坐标系中的位置计算错误

## 解决方案

### 新的坐标转换逻辑

1. **获取初始和当前状态**:
   - 从模型定义中获取body的初始pose (`m->body_pos`, `m->body_quat`)
   - 从仿真数据中获取body的当前pose (`d->xpos`, `d->xquat`)

2. **计算变换矩阵**:
   - 计算从初始pose到当前pose的变换
   - 计算从当前pose到初始pose的逆变换

3. **坐标转换**:
   - 将接触点从世界坐标系转换到当前body的局部坐标系
   - 应用逆变换，将接触点从当前局部坐标系转换到初始局部坐标系

### 代码实现

```cpp
// 获取body的初始pose（从模型定义中）
mjtNum body1_init_pos[3] = {m->body_pos[body1_id*3], m->body_pos[body1_id*3+1], m->body_pos[body1_id*3+2]};
mjtNum body1_init_quat[4] = {m->body_quat[body1_id*4], m->body_quat[body1_id*4+1], m->body_quat[body1_id*4+2], m->body_quat[body1_id*4+3]};

// 将接触点从body1的当前局部坐标系转换到初始局部坐标系
mjtNum body1_init_quat_conj[4];
mju_negQuat(body1_init_quat_conj, body1_init_quat);
mjtNum body1_transform_inv[4];
mju_mulQuat(body1_transform_inv, body1_init_quat_conj, body1_quat_conj);
mju_rotVecQuat(urdf_corrected_pos_body1, body1_current_local, body1_transform_inv);
```

## 新功能特性

### 1. 向后兼容性
- 新的坐标字段命名为 `urdf_corrected_x_body1`, `urdf_corrected_y_body1`, `urdf_corrected_z_body1`
- 保留旧的字段名称以支持向后兼容
- 可视化脚本会自动检测并使用新的字段

### 2. 自动检测
- 分析脚本会自动检测CSV文件中的坐标字段类型
- 优先使用新的修正坐标，如果没有则回退到旧坐标

### 3. 测试工具
- 提供专门的测试脚本 `test_coordinate_correction.py`
- 可以比较新旧坐标的差异
- 验证坐标转换的正确性

## 使用方法

### 1. 重新编译仿真器
```bash
cd engineai_ros2_workspace
colcon build --packages-select mujoco_simulator
source install/setup.bash
```

### 2. 运行仿真并保存CSV
```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=true \
    csv_save_frequency:=10
```

### 3. 测试坐标转换
```bash
python3 scripts/test_coordinate_correction.py logs/contact_data_YYYYMMDD_HHMMSS.csv
```

### 4. 可视化碰撞点
```bash
python3 scripts/mujoco_urdf_contact_display.py \
    logs/contact_data_YYYYMMDD_HHMMSS.csv \
    src/simulation/mujoco/assets/resource/robot/pm_v2/xml/serial_pm_v2.xml \
    sphere urdf
```

## 验证方法

### 1. 坐标差异分析
运行测试脚本，检查新旧坐标的差异：
- 如果机器人在仿真过程中有关节运动，应该看到显著的坐标差异
- 如果机器人保持静止，坐标应该相同

### 2. 可视化验证
- 使用新的修正坐标在viewer中显示碰撞点
- 碰撞点应该准确显示在机器人mesh的对应位置
- 与使用世界坐标显示的结果进行对比

### 3. 数据一致性检查
- 检查CSV文件中的新字段是否正确生成
- 验证坐标值的合理性（应该在预期的范围内）

## 技术细节

### 坐标系转换流程
1. **世界坐标系** → **当前body局部坐标系**
2. **当前body局部坐标系** → **初始body局部坐标系**
3. **初始body局部坐标系** → **URDF坐标系**

### 四元数运算
- 使用MuJoCo的四元数函数进行旋转计算
- `mju_negQuat`: 计算四元数的共轭（逆）
- `mju_mulQuat`: 四元数乘法
- `mju_rotVecQuat`: 使用四元数旋转向量

### 性能考虑
- 坐标转换计算在CSV保存时进行，不影响实时性能
- 使用MuJoCo优化的数学函数，计算效率高
- 内存使用量增加很小

## 故障排除

### 常见问题

1. **编译错误**
   - 确保使用最新版本的代码
   - 检查MuJoCo库的版本兼容性

2. **坐标字段缺失**
   - 检查CSV文件是否包含新的坐标字段
   - 确认仿真器版本是否支持新功能

3. **坐标值异常**
   - 检查模型文件中的初始pose定义
   - 验证关节角度数据的正确性

### 调试命令
```bash
# 检查CSV文件结构
head -1 logs/contact_data_*.csv

# 分析坐标差异
python3 scripts/analyze_coordinates.py logs/contact_data_*.csv

# 测试坐标转换
python3 scripts/test_coordinate_correction.py logs/contact_data_*.csv
```

## 未来改进

1. **更精确的变换计算**
   - 考虑关节链的完整变换
   - 支持更复杂的机器人结构

2. **实时坐标转换**
   - 在ROS消息中也提供修正的坐标
   - 支持实时可视化

3. **更多坐标系支持**
   - 支持其他坐标系（如基座坐标系）
   - 提供多种坐标转换选项 