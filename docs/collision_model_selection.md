# 碰撞模型选择功能使用说明

## 概述

EngineAI ROS2 仿真系统现在支持两种碰撞模型选择：
- **简化几何体模型**：使用球体、圆柱体、立方体等基本几何形状进行碰撞检测
- **真实 Mesh 模型**：使用高精度的 STL 网格文件进行碰撞检测

## 功能特点

### 简化几何体模型 (`use_simplified_geometry: true`)
- **优点**：
  - 计算速度快，仿真效率高
  - 内存占用少
  - 适合实时控制和快速原型开发
- **缺点**：
  - 精度较低，可能与实际形状有差异
  - 碰撞检测可能不够精确

### 真实 Mesh 模型 (`use_simplified_geometry: false`)
- **优点**：
  - 精度高，形状准确
  - 碰撞检测更精确
  - 适合精确仿真和碰撞分析
- **缺点**：
  - 计算复杂度高，仿真速度慢
  - 内存占用较大

## 配置文件设置

### 1. 修改配置文件

编辑 `src/simulation/mujoco/assets/config/pm_v2.yaml`：

```yaml
# 碰撞模型配置
collision_model:
  use_simplified_geometry: true  # 设置为 true 使用简化几何体，false 使用真实 mesh
  
  # XML文件选择配置
  xml_files:
    simplified: serial_pm_v2_simplified.xml  # 简化几何体版本
    mesh: serial_pm_v2_mesh.xml              # 真实mesh版本
```

### 2. 配置选项说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_simplified_geometry` | bool | true | 碰撞检测几何体类型选择 |
| `xml_files.simplified` | string | serial_pm_v2_simplified.xml | 简化几何体 XML 文件名 |
| `xml_files.mesh` | string | serial_pm_v2_mesh.xml | 真实 mesh XML 文件名 |

## 使用方法

### 1. 使用简化几何体（推荐用于实时控制）

```yaml
collision_model:
  use_simplified_geometry: true
```

系统将加载 `serial_pm_v2_simplified.xml`，使用以下简化几何体：
- 头部：球形 (半径 9cm)
- 基座：球形 (半径 8.5cm)
- 髋关节：圆柱形 (半径 4.6cm，高度 6cm)
- 膝关节：球形 (半径 6.2cm)
- 肘关节：球形 (半径 4.5cm)
- 脚部：盒形 (10.2×5.4×0.1cm)

### 2. 使用真实 Mesh（推荐用于精确仿真）

```yaml
collision_model:
  use_simplified_geometry: false
```

系统将加载 `serial_pm_v2_mesh.xml`，使用真实的 STL 网格文件进行碰撞检测。

## 文件结构

```
src/simulation/mujoco/assets/
├── config/
│   └── pm_v2.yaml                    # 主配置文件
└── resource/robot/pm_v2/xml/
    ├── serial_pm_v2_simplified.xml   # 简化几何体版本
    ├── serial_pm_v2_mesh.xml         # 真实 mesh 版本
    ├── serial_links.xml              # 链接定义文件
    ├── serial_actuators.xml          # 执行器配置
    ├── serial_sensors.xml            # 传感器配置
    └── assets.xml                    # 资源文件定义
```

## 技术实现

### 1. 配置加载

系统通过 `ConfigLoader` 类加载配置文件：

```cpp
// 获取碰撞模型条件
std::string collision_condition = config_loader_->GetCollisionModelCondition();
// 根据碰撞类型选择对应的XML文件
std::string xml_filename = config_loader_->GetXmlFilenameByCollisionType();
```

### 2. XML 文件选择

根据 `use_simplified_geometry` 参数自动选择对应的 XML 文件：

- `true` → `serial_pm_v2_simplified.xml`
- `false` → `serial_pm_v2_mesh.xml`

### 3. 模型加载

系统会构建完整的 XML 文件路径并加载：

```cpp
std::string full_xml_path = config_loader_->GetResourceDir() + 
                           "/robot/pm_v2/xml/" + xml_filename;
mnew = mj_loadXML(xml_path, &vfs, mj_load_error_.data(), mj_load_error_.size());
```

## 测试验证

运行测试脚本验证功能是否正常：

```bash
cd /home/wang22/engineai/engineai_ros2_workspace
python3 scripts/test_collision_models.py
```

测试内容包括：
- 配置文件加载测试
- XML 文件存在性测试
- XML 文件内容测试
- 配置逻辑测试

## 性能对比

| 指标 | 简化几何体 | 真实 Mesh |
|------|------------|-----------|
| 仿真速度 | 快 | 慢 |
| 内存占用 | 低 | 高 |
| 碰撞精度 | 中等 | 高 |
| 适用场景 | 实时控制 | 精确仿真 |

## 注意事项

1. **切换模型时需要重新编译**：修改配置文件后需要重新编译仿真节点
2. **性能考虑**：真实 mesh 模型会显著降低仿真速度，建议根据实际需求选择
3. **文件依赖**：确保对应的 XML 文件和 STL 文件都存在
4. **兼容性**：两种模型使用相同的关节和执行器配置，确保兼容性

## 故障排除

### 1. 配置文件加载失败
- 检查 YAML 语法是否正确
- 确认文件路径是否正确

### 2. XML 文件不存在
- 确认 XML 文件是否在正确位置
- 检查文件名是否与配置一致

### 3. 仿真速度过慢
- 考虑切换到简化几何体模型
- 检查 mesh 文件是否过于复杂

### 4. 碰撞检测不准确
- 考虑切换到真实 mesh 模型
- 检查简化几何体的参数设置

## 更新日志

- **v1.0**：初始版本，支持两种碰撞模型选择
- 添加了配置文件支持
- 实现了自动 XML 文件选择
- 提供了完整的测试验证
