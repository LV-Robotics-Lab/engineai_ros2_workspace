# CSV时间戳问题修复说明

## 🐛 问题描述

CSV文件中的时间戳全部为0，导致无法正确记录接触点的时间信息。

## 🔍 问题原因

### 1. 仿真时间设置
仿真器配置了 `use_sim_time: True`，这意味着：
- ROS节点使用仿真时间而不是系统时间
- `node_->now()` 返回的是仿真时间
- 在仿真开始前，仿真时间为0

### 2. 时间获取方式错误
原来的代码：
```cpp
auto now = node_->now();
auto seconds = now.seconds();
```

问题：
- 在仿真开始前，`now.seconds()` 返回0
- 即使仿真运行，时间戳也可能不准确

## ✅ 修复方案

### 使用MuJoCo仿真时间
修改后的代码：
```cpp
// 使用MuJoCo仿真时间而不是ROS时间
double sim_time = d->time;
```

优势：
- `d->time` 是MuJoCo内部的仿真时间
- 从仿真开始就正确递增
- 与仿真步长同步
- 不受ROS时间设置影响

## 📊 时间戳对比

| 时间类型 | 获取方式 | 特点 | 适用场景 |
|----------|----------|------|----------|
| **系统时间** | `std::chrono::system_clock::now()` | 真实世界时间 | 日志记录 |
| **ROS时间** | `node_->now()` | 受use_sim_time影响 | ROS消息 |
| **仿真时间** | `d->time` | MuJoCo内部时间 | 仿真数据 |

## 🔧 修复验证

### 测试方法
```bash
# 运行测试脚本
./scripts/test_timestamp_fix.sh

# 或手动测试
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true
```

### 预期结果
- 时间戳从0开始递增
- 时间间隔与仿真步长一致
- 数据按时间正确排序

## 📈 时间戳格式

修复后的CSV文件时间戳格式：
```csv
timestamp,contact_id,geom1_name,geom2_name,...
0.000000,0,"LINK_FOOT_L","ground",...
0.002000,0,"LINK_FOOT_L","ground",...
0.004000,0,"LINK_FOOT_L","ground",...
...
```

## 🎯 技术细节

### MuJoCo时间特性
- `d->time` 以秒为单位
- 默认仿真步长：0.002秒
- 时间精度：双精度浮点数
- 从仿真开始就可用

### 性能考虑
- `d->time` 访问开销很小
- 不需要额外的系统调用
- 与仿真同步，无延迟

## 📝 相关文件

修改的文件：
- `src/simulation/mujoco/src/ros_interface.cc` - 时间戳获取逻辑

测试文件：
- `scripts/test_timestamp_fix.sh` - 时间戳测试脚本

## 🚀 使用建议

1. **开发阶段**：使用修复后的时间戳进行调试
2. **数据分析**：基于正确的时间戳进行时序分析
3. **性能优化**：时间戳修复不影响仿真性能

## 🔮 扩展功能

未来可以考虑：
- 添加时间戳精度配置
- 支持多种时间格式
- 时间戳同步验证 