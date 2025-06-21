# MuJoCo接触点CSV保存功能实现总结

## 功能概述

成功为MuJoCo仿真器添加了接触点CSV保存功能，可以在仿真运行时自动记录接触点的位置、力和时间信息。

## 实现的功能

### 1. 核心功能
- ✅ 自动保存接触点位置 (x, y, z)
- ✅ 自动保存接触力 (fx, fy, fz)
- ✅ 自动保存接触扭矩 (tx, ty, tz)
- ✅ 自动保存接触间隙 (gap)
- ✅ 自动保存时间戳
- ✅ 自动保存几何体名称和ID

### 2. 配置选项
- ✅ 通过launch文件参数控制
- ✅ 通过命令行参数控制
- ✅ 可自定义CSV文件路径
- ✅ 自动生成带时间戳的文件名

### 3. 文件管理
- ✅ 自动创建logs目录
- ✅ 线程安全的文件写入
- ✅ 定期刷新文件缓冲区
- ✅ 程序退出时自动保存

## 修改的文件

### 1. 核心代码文件
- `src/simulation/mujoco/include/ros_interface.h` - 添加CSV相关成员变量
- `src/simulation/mujoco/src/ros_interface.cc` - 实现CSV保存逻辑
- `src/simulation/mujoco/launch/mujoco_simulator.launch.py` - 添加CSV参数

### 2. 工具脚本
- `src/interface_example/scripts/test_contact_csv.py` - Python测试脚本
- `scripts/test_contact_csv.sh` - Bash测试脚本
- `scripts/analyze_contact_data.py` - 数据分析脚本

### 3. 文档文件
- `docs/contact_csv_saving_guide.md` - 详细使用指南
- `docs/contact_csv_implementation_summary.md` - 实现总结
- `README_contact_csv.md` - 快速启动指南

## 使用方法

### 快速启动
```bash
# 编译项目
colcon build --packages-select mujoco_simulator
source install/setup.bash

# 启动仿真器并保存CSV
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true
```

### 数据分析
```bash
# 分析最新CSV文件
python3 scripts/analyze_contact_data.py

# 分析指定文件
python3 scripts/analyze_contact_data.py logs/contact_data_20231201_143022.csv
```

## CSV文件格式

```csv
timestamp,contact_id,geom1_name,geom2_name,pos_x,pos_y,pos_z,force_x,force_y,force_z,torque_x,torque_y,torque_z,gap,body1_id,body2_id
1234567890.123456,0,"LINK_FOOT_L","ground",0.123456,0.234567,0.000000,0.000000,0.000000,98.100000,0.0,0.0,0.0,0.000000,5,0
```

## 技术特点

### 1. 性能优化
- 每100帧刷新一次文件缓冲区
- 使用线程安全的文件写入
- 实时写入，不占用大量内存

### 2. 数据准确性
- 使用MuJoCo内部计算的接触点位置
- 支持位置偏移配置
- 包含完整的接触力信息

### 3. 易用性
- 自动生成文件名
- 支持多种启动方式
- 提供完整的分析工具

## 测试验证

### 1. 编译测试
- ✅ 项目编译成功
- ✅ 无编译错误
- ✅ 所有依赖正确链接

### 2. 功能测试
- ✅ 参数解析正确
- ✅ 文件创建成功
- ✅ 数据写入正常
- ✅ 程序退出时正确保存

### 3. 工具测试
- ✅ 测试脚本可执行
- ✅ 分析脚本功能正常
- ✅ 文档完整可用

## 注意事项

1. **文件权限**: 确保logs目录有写权限
2. **磁盘空间**: 长时间仿真可能产生大文件
3. **性能影响**: CSV保存对仿真性能影响很小
4. **数据完整性**: 异常退出可能丢失部分数据

## 扩展建议

1. **数据压缩**: 可考虑添加gzip压缩
2. **实时监控**: 可添加实时数据可视化
3. **多文件分割**: 长时间仿真可自动分割文件
4. **数据过滤**: 可添加数据过滤和采样选项

## 总结

成功实现了完整的接触点CSV保存功能，包括：
- 核心保存逻辑
- 配置参数系统
- 测试和分析工具
- 完整的文档说明

该功能可以满足用户对接触点数据记录和分析的需求，为机器人仿真研究提供有力的数据支持。 