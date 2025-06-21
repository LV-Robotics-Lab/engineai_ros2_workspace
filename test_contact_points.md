# MuJoCo接触点显示测试

## 修改内容

我已经对MuJoCo的接触点显示进行了以下修改：

### 1. 地面可视化设置 (ground.xml)
- **接触点颜色**: 从 `(1.0, 1.0, 0.6, 0.4)` 改为 `(1.0, 0.0, 0.0, 0.8)` - 更鲜艳的红色
- **接触点大小**: 
  - `contactwidth`: 从 `0.01` 改为 `0.005` - 更细的线条
  - `contactheight`: 从 `0.02` 改为 `0.01` - 更小的高度

### 2. 机器人可视化设置 (serial_pm_v2.xml)
- 添加了专门的接触点可视化设置
- **接触点颜色**: `(1.0, 0.0, 0.0, 0.8)` - 红色，高透明度
- **接触点大小**: 
  - `contactwidth`: `0.005` - 细线条
  - `contactheight`: `0.01` - 小高度

### 3. 代码中的可视化启用
- 在 `sim_manager.cc` 中确保启用了接触点可视化
- `opt.flags[mjVIS_CONTACTPOINT] = 1`
- `opt.flags[mjVIS_CONTACTFORCE] = 1`
- `opt.flags[mjVIS_CONTACTSPLIT] = 1`

## 测试步骤

1. **启动MuJoCo仿真器**:
```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py
```

2. **观察接触点显示**:
   - 在MuJoCo GUI中，接触点现在应该显示为红色的小点
   - 接触点应该更准确地显示在碰撞几何体的表面
   - 接触点的大小应该更合适，不会过于突出

3. **如果接触点仍然不准确**:
   - 可以尝试在MuJoCo GUI的菜单中调整可视化设置
   - 或者进一步调整XML文件中的参数

## 参数说明

### 接触点颜色 (contactpoint)
- 格式: `(红, 绿, 蓝, 透明度)`
- 当前设置: `(1.0, 0.0, 0.0, 0.8)` - 红色，80%不透明度

### 接触点大小
- `contactwidth`: 接触点的宽度（米）
- `contactheight`: 接触点的高度（米）
- 较小的值使接触点更精确，较大的值使接触点更明显

## 进一步调整

如果接触点显示仍然不理想，可以：

1. **调整颜色**:
   - 更亮的颜色: `(1.0, 1.0, 0.0, 1.0)` - 黄色
   - 更暗的颜色: `(0.5, 0.0, 0.0, 0.6)` - 暗红色

2. **调整大小**:
   - 更小的接触点: `contactwidth="0.003" contactheight="0.005"`
   - 更大的接触点: `contactwidth="0.01" contactheight="0.02"`

3. **在MuJoCo GUI中调整**:
   - 使用GUI中的可视化设置菜单
   - 实时调整接触点的显示参数 