# MuJoCo接触点可视化调整指南

## 问题描述

在MuJoCo仿真中，接触点的显示位置可能不够准确，起点不在机器人mesh（STL）的表面，但大致位置是正确的。这个问题通常是由于接触点计算方法的限制导致的。

## 解决方案

我们提供了一个可配置的接触点显示系统，允许用户调整接触点的位置和显示参数。

### 1. 配置文件参数

在 `src/simulation/mujoco/assets/config/pm_v2.yaml` 文件中，可以配置以下参数：

```yaml
contact_visualization:
  # 接触点位置偏移量（米）
  # 正值：向法向量方向偏移（远离表面）
  # 负值：向法向量反方向偏移（靠近表面）
  # 0.0：使用MuJoCo原始计算的接触点位置
  position_offset: 0.0
  
  # 接触点显示大小（米）
  marker_size: 0.03
  
  # 接触力显示比例
  force_scale: 0.01
  
  # 是否启用接触点可视化
  enabled: true
```

### 2. 使用调整工具

我们提供了一个Python脚本来帮助调整接触点参数：

```bash
# 显示当前设置
python3 src/interface_example/scripts/adjust_contact_points.py --show

# 设置位置偏移量
python3 src/interface_example/scripts/adjust_contact_points.py --offset -0.005

# 设置标记大小
python3 src/interface_example/scripts/adjust_contact_points.py --size 0.05

# 交互式调整
python3 src/interface_example/scripts/adjust_contact_points.py --config src/simulation/mujoco/assets/config/pm_v2.yaml --interactive
```

### 3. 参数调整建议

#### 位置偏移量 (position_offset)
- **默认值**: 0.0
- **建议范围**: -0.01 到 0.01 米
- **调整方法**:
  - 如果接触点显示在mesh内部，尝试负值（如 -0.005）
  - 如果接触点显示在mesh外部，尝试正值（如 0.005）
  - 逐步调整直到接触点准确显示在表面

#### 标记大小 (marker_size)
- **默认值**: 0.03 米
- **建议范围**: 0.01 到 0.1 米
- **调整方法**:
  - 较小的值使接触点更精确但可能不易看到
  - 较大的值使接触点更明显但可能遮挡细节

### 4. 技术原理

#### 接触点计算方法
MuJoCo使用以下方法计算接触点：

1. **原始方法**: 使用 `contact.pos`，这是MuJoCo内部计算的接触点位置
2. **偏移方法**: 在原始位置基础上添加法向量方向的偏移
3. **表面投影**: 将接触点投影到几何体表面

#### 代码实现
在 `src/simulation/mujoco/src/ros_interface.cc` 中：

```cpp
// 使用MuJoCo内部计算的接触点位置
pos[0] = contact.pos[0];
pos[1] = contact.pos[1];
pos[2] = contact.pos[2];

// 应用配置的偏移量
double position_offset = config_loader_->GetContactPositionOffset();
if (position_offset != 0.0) {
    pos[0] += normal[0] * position_offset;
    pos[1] += normal[1] * position_offset;
    pos[2] += normal[2] * position_offset;
}
```

### 5. 常见问题解决

#### 问题1: 接触点显示在mesh内部
**解决方案**: 设置负的位置偏移量
```bash
python3 src/interface_example/scripts/adjust_contact_points.py --offset -0.005
```

#### 问题2: 接触点显示在mesh外部
**解决方案**: 设置正的位置偏移量
```bash
python3 src/interface_example/scripts/adjust_contact_points.py --offset 0.005
```

#### 问题3: 接触点太小不易看到
**解决方案**: 增加标记大小
```bash
python3 src/interface_example/scripts/adjust_contact_points.py --size 0.05
```

#### 问题4: 接触点太大遮挡细节
**解决方案**: 减小标记大小
```bash
python3 src/interface_example/scripts/adjust_contact_points.py --size 0.02
```

### 6. 验证调整效果

1. 启动MuJoCo仿真器：
```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py
```

2. 启动接触点可视化：
```bash
ros2 launch interface_example contact_viz.launch.py
```

3. 在RViz中观察接触点的显示效果

4. 根据观察结果调整参数

### 7. 高级配置

#### 针对不同几何体的优化
不同的几何体类型可能需要不同的偏移量：

- **球体**: 通常偏移量较小（±0.002）
- **盒子**: 可能需要较大偏移量（±0.005）
- **复杂mesh**: 可能需要更大的偏移量（±0.01）

#### 批量调整
可以创建多个配置文件用于不同的场景：

```bash
# 创建不同配置的副本
cp pm_v2.yaml pm_v2_precise.yaml
cp pm_v2.yaml pm_v2_visible.yaml

# 调整精确模式配置
python3 adjust_contact_points.py --config pm_v2_precise.yaml --offset -0.003 --size 0.02

# 调整可见模式配置
python3 adjust_contact_points.py --config pm_v2_visible.yaml --offset 0.005 --size 0.05
```

### 8. 注意事项

1. **重启仿真器**: 修改配置文件后需要重启MuJoCo仿真器才能生效
2. **参数范围**: 避免使用过大的偏移量，可能导致接触点显示在错误位置
3. **性能影响**: 接触点可视化对性能影响很小，但大量接触点可能影响渲染性能
4. **调试模式**: 在调试时可以启用更详细的日志输出

### 9. 故障排除

如果调整后仍然无法获得满意的效果：

1. 检查配置文件是否正确加载
2. 确认仿真器已重启
3. 尝试不同的偏移量范围
4. 检查几何体的碰撞设置
5. 查看MuJoCo的日志输出

## 总结

通过使用提供的配置系统和调整工具，您可以精确控制MuJoCo中接触点的显示位置和大小。建议从小的偏移量开始，逐步调整直到获得满意的显示效果。 