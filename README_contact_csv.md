# MuJoCo接触点CSV保存功能 - 快速启动指南

## 🚀 快速开始

### 1. 编译项目
```bash
cd /home/wang22/engineai/engineai_ros2_workspace
colcon build --packages-select mujoco_simulator
source install/setup.bash
```

### 2. 启动仿真器并保存CSV
```bash
# 方法1: 使用launch文件（推荐）
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true

# 方法2: 使用测试脚本
./scripts/test_contact_csv.sh

# 方法3: 指定CSV文件路径
ros2 launch mujoco_simulator mujoco_simulator.launch.py save_contact_csv:=true csv_file_path:=/path/to/your/contact_data.csv
```

### 3. 查看保存的数据
```bash
# 查看CSV文件
ls -la logs/contact_data_*.csv

# 查看文件内容（前几行）
head -5 logs/contact_data_*.csv
```

## 📊 CSV文件格式

文件包含以下数据：
- **时间戳**: 仿真时间
- **接触点位置**: x, y, z坐标
- **接触力**: fx, fy, fz分量
- **几何体信息**: 接触的物体名称和ID
- **接触间隙**: 物体间的距离

## 🔧 参数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `save_contact_csv` | 启用CSV保存 | false |
| `csv_file_path` | 指定文件路径 | 自动生成 |
| `export_contact` | 启用接触力导出 | true |

## 📁 文件位置

- **默认路径**: `logs/contact_data_YYYYMMDD_HHMMSS.csv`
- **配置文件**: `docs/contact_csv_saving_guide.md`
- **测试脚本**: `scripts/test_contact_csv.sh`

## 🐛 常见问题

1. **文件未生成**: 检查机器人是否有接触点
2. **数据为空**: 确认仿真正常运行
3. **权限错误**: 确保logs目录可写

## 📈 数据分析示例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('logs/contact_data_20231201_143022.csv')

# 分析接触力
plt.plot(df['timestamp'], df['force_z'])
plt.xlabel('时间 (秒)')
plt.ylabel('接触力 Z (牛顿)')
plt.show()
```

## 📞 技术支持

如有问题，请查看详细文档：`docs/contact_csv_saving_guide.md` 