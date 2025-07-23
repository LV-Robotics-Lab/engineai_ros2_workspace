# 接触力CSV保存指南

## 功能概述

MuJoCo仿真器现在支持将接触点数据自动保存到CSV文件，包括：
- 时间戳（MuJoCo仿真时间）
- 接触点ID
- 几何体名称
- 接触点位置（X, Y, Z）
- 接触力（X, Y, Z方向）
- 扭矩（X, Y, Z方向）
- 接触距离
- 几何体所属的刚体ID

## 配置参数

### 启动参数

在launch文件中添加以下参数：

```python
# 启用CSV保存
Node(
    package='mujoco_simulator',
    executable='mujoco_simulator',
    name='mujoco_simulator',
    parameters=[{
        'save_contact_csv': True,                    # 启用CSV保存
        'csv_file_path': 'logs/contact_data.csv',    # CSV文件路径（可选）
        'csv_save_frequency': 1,                     # 保存频率（1=每帧都保存）
    }]
)
```

### 命令行参数

也可以直接在命令行中指定：

```bash
ros2 run mujoco_simulator mujoco_simulator --ros-args \
    -p save_contact_csv:=true \
    -p csv_file_path:=logs/contact_data.csv \
    -p csv_save_frequency:=1
```

## 保存频率配置

- `csv_save_frequency = 1`: 每帧都保存（默认，10kHz）
- `csv_save_frequency = 10`: 每10帧保存一次（1kHz）
- `csv_save_frequency = 100`: 每100帧保存一次（100Hz）

**注意**: 由于仿真步长为0.0001秒（10kHz），每帧都保存会产生大量数据。建议根据实际需求调整保存频率。

## 文件命名

如果未指定`csv_file_path`，系统会自动生成带时间戳的文件名：
```
logs/contact_data_YYYYMMDD_HHMMSS.csv
```

## CSV文件格式

```csv
timestamp,contact_id,geom1_name,geom2_name,pos_x,pos_y,pos_z,force_x,force_y,force_z,force_magnitude,torque_x,torque_y,torque_z,distance,body1_id,body2_id
0.000000,0,"LINK_FOOT_L","ground",0.123456,0.234567,0.000000,0.000000,0.000000,10.500000,10.500000,0.000000,0.000000,0.000000,0.000000,15,0
0.000100,0,"LINK_FOOT_L","ground",0.123456,0.234567,0.000000,0.000000,0.000000,10.500000,10.500000,0.000000,0.000000,0.000000,0.000000,15,0
...
```

### 字段说明

- **force_magnitude**: 单个接触点的三轴合力大小 = √(force_x² + force_y² + force_z²)
- **注意**: CSV文件只保存单个接触点数据，不包含合力行

## 性能考虑

### 数据量估算

- 仿真时间：1秒
- 保存频率：每帧（10kHz）
- 接触点数量：平均2个
- 每行数据：约200字节
- 1秒数据量：约4MB

### 优化建议

1. **降低保存频率**: 如果不需要10kHz精度，可以设置为100Hz或1kHz
2. **定期清理**: 定期删除旧的CSV文件
3. **压缩存储**: 可以启用gzip压缩

## 使用示例

### 1. 每帧保存（最高精度）

```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=true \
    csv_save_frequency:=1
```

### 2. 1kHz保存（平衡精度和性能）

```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=true \
    csv_save_frequency:=10
```

### 3. 100Hz保存（适合长时间仿真）

```bash
ros2 launch mujoco_simulator mujoco_simulator.launch.py \
    save_contact_csv:=true \
    csv_save_frequency:=100
```

## 数据分析

保存的CSV文件可以使用提供的分析脚本进行分析：

```bash
# 快速分析
python3 scripts/quick_force_analysis.sh logs/contact_data_20250122_123456.csv

# 详细分析
python3 scripts/analyze_contact_forces.py logs/contact_data_20250122_123456.csv
```

## 故障排除

### 常见问题

1. **文件权限错误**: 确保`logs`目录存在且有写权限
2. **磁盘空间不足**: 长时间仿真会产生大量数据，注意磁盘空间
3. **时间戳为0**: 确保使用最新版本的代码（已修复时间戳问题）

### 调试命令

```bash
# 检查CSV文件是否正确生成
ls -la logs/contact_data_*.csv

# 查看文件大小
du -h logs/contact_data_*.csv

# 检查文件内容
head -10 logs/contact_data_*.csv
``` 