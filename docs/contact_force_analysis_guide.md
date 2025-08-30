# 接触力曲线分析使用指南

## 🎯 功能概述

本工具专门用于分析和可视化CSV文件中的接触力数据，提供详细的接触力曲线和统计分析。

## 🚀 快速开始

### 方法1: 一键分析（推荐）
```bash
# 自动运行仿真并分析
./scripts/quick_force_analysis.sh
```

### 方法2: 手动分析
```bash
# 1. 先运行仿真生成数据
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true

# 2. 分析最新CSV文件
python3 scripts/analyze_contact_forces.py

# 3. 或分析指定文件
python3 scripts/analyze_contact_forces.py logs/contact_data_20250622_011841.csv
```

## 📊 分析内容

### 1. 接触力曲线图
- **时间序列图**: 显示Fx、Fy、Fz各分量随时间变化
- **合力曲线**: 显示合力大小随时间变化
- **力分布直方图**: 显示合力大小的分布情况
- **分量对比图**: 各分量力的分布对比

### 2. 接触点分析图
- **接触点数量**: 每帧接触点数量变化
- **几何体频率**: 各几何体的接触频率
- **力-时间散点图**: 接触力与时间的关系
- **分布热力图**: 接触力在时间-力空间的分布

### 3. 统计报告
- 基本数据统计
- 接触力统计（均值、最大值、最小值、标准差）
- 接触点统计
- 主要几何体分析

## 📈 图表说明

### 接触力分量时间序列
- **蓝线**: Fx分量（X方向力）
- **绿线**: Fy分量（Y方向力）
- **红线**: Fz分量（Z方向力，通常是重力方向）

### 合力大小时间序列
- **红线**: 合力大小 = √(Fx² + Fy² + Fz²)
- 反映接触力的总体强度

### 力分布直方图
- 显示不同力大小出现的频率
- 帮助识别典型的接触力范围

### 分量分布对比
- 箱线图显示各分量力的分布特征
- 包括中位数、四分位数、异常值

## 🔍 数据分析要点

### 1. 时间戳检查
- 确保时间戳从0开始递增
- 时间间隔应与仿真步长一致（通常0.002秒）

### 2. 接触力特征
- **静态接触**: 力相对稳定，主要是重力
- **动态接触**: 力有较大波动，反映运动状态
- **冲击接触**: 力有尖峰，反映碰撞事件

### 3. 几何体分析
- 识别主要接触的机器人部件
- 分析不同部件的接触模式

## 📁 输出文件

分析完成后会生成：
- `contact_data_YYYYMMDD_HHMMSS_force_curves.png` - 接触力曲线图
- `contact_data_YYYYMMDD_HHMMSS_analysis.png` - 接触点分析图

## 🛠️ 自定义分析

### 修改分析参数
编辑 `scripts/analyze_contact_forces.py`：
```python
# 修改图表大小
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 修改颜色方案
colors = ['blue', 'green', 'red']

# 修改统计方法
df['force_magnitude'] = np.sqrt(df['force_x']**2 + df['force_y']**2 + df['force_z']**2)
```

### 添加新的分析维度
```python
# 添加速度分析
df['velocity'] = df['force_magnitude'].diff() / df['timestamp'].diff()

# 添加频率分析
from scipy import signal
frequencies, power = signal.welch(df['force_magnitude'])
```

## 🐛 常见问题

### 1. 时间戳为0
- 检查是否使用了修复后的代码
- 确认仿真器正常运行

### 2. 力数据异常
- 检查机器人是否有接触点
- 确认接触力导出已启用

### 3. 图表显示问题
- 安装matplotlib和seaborn
- 检查中文字体设置

## 📊 示例分析结果

### 典型接触力特征
```
接触力统计:
平均合力: 98.100 N
最大合力: 150.250 N
最小合力: 45.750 N
合力标准差: 25.300 N

主要接触几何体:
LINK_FOOT_L: 1250 次 (45.2%)
LINK_FOOT_R: 1180 次 (42.7%)
LINK_HAND_L: 150 次 (5.4%)
```

### 图表解读
- **稳定期**: 力相对稳定，机器人静止或匀速运动
- **波动期**: 力有规律波动，机器人周期性运动
- **尖峰期**: 力有突然变化，机器人状态改变

## 🎯 应用场景

1. **机器人稳定性分析**: 通过接触力变化评估稳定性
2. **运动模式识别**: 分析不同运动阶段的接触特征
3. **碰撞检测**: 识别异常的力尖峰
4. **性能优化**: 优化接触力分布

## 📞 技术支持

如有问题，请查看：
- `docs/timestamp_fix_explanation.md` - 时间戳问题说明
- `docs/contact_csv_saving_guide.md` - CSV保存功能指南 