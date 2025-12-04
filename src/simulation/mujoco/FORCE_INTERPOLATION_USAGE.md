# 力插值计算使用指南

## 概述

`ForceInterpolation` 类提供了在C++代码中计算防护后冲击力的功能。它使用双线性插值方法，根据无防护的冲击力和防护材料厚度，从RT-FEM数据表中计算防护后的力。

## 功能特点

- ✅ 纯C++实现，无需Python依赖
- ✅ 高性能，适合实时仿真
- ✅ 支持双线性插值
- ✅ 自动数据范围检查
- ✅ 异常安全

## 基本使用

### 1. 包含头文件

```cpp
#include "force_interpolation.h"
```

### 2. 创建对象

```cpp
// 使用默认路径（自动查找RT-FEM.tsv）
ForceInterpolation interpolator;

// 或指定完整路径
ForceInterpolation interpolator("/path/to/RT-FEM.tsv");
```

### 3. 计算防护后的力

```cpp
double force_unprotected = 100.0;  // 无防护的冲击力 (kN)
double thickness = 12.0;           // 防护材料厚度 (mm)

try {
    double force_protected = interpolator.GetProtectedForce(
        force_unprotected, thickness);
    std::cout << "防护后的力: " << force_protected << " kN" << std::endl;
} catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
}
```

## 在MuJoCo仿真中使用

### 示例：在接触力计算中应用

```cpp
// 在类头文件中添加成员变量
class RosInterface {
private:
    std::unique_ptr<ForceInterpolation> force_interpolator_;
    double protection_thickness_ = 12.0;  // 防护材料厚度 (mm)
};

// 在初始化函数中创建对象
void RosInterface::Initialize() {
    // ... 其他初始化代码 ...
    
    // 初始化力插值器（只需初始化一次）
    try {
        std::string tsv_path = "/path/to/RT-FEM.tsv";
        force_interpolator_ = std::make_unique<ForceInterpolation>(tsv_path);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), 
                    "无法初始化力插值器: %s", e.what());
    }
}

// 在接触力计算函数中使用
void RosInterface::PublishContactForces(const mjModel* m, mjData* d) {
    // ... 获取接触力 ...
    
    // 计算接触力大小 (kN)
    double contact_force_magnitude = sqrt(
        world_force[0]*world_force[0] + 
        world_force[1]*world_force[1] + 
        world_force[2]*world_force[2]
    ) / 1000.0;  // 转换为kN
    
    // 计算防护后的力
    if (force_interpolator_) {
        try {
            double protected_force = force_interpolator_->GetProtectedForce(
                contact_force_magnitude, 
                protection_thickness_
            );
            
            // 使用防护后的力进行后续处理
            // ...
            
        } catch (const std::exception& e) {
            // 如果超出范围，使用原始力
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 
                                1000, "力插值失败: %s", e.what());
        }
    }
}
```

## API参考

### 构造函数

```cpp
explicit ForceInterpolation(const std::string& tsv_file_path = "");
```

- `tsv_file_path`: RT-FEM数据文件路径。如果为空，将尝试从默认位置查找。

### 主要方法

#### `GetProtectedForce`

```cpp
double GetProtectedForce(double force_unprotected, double thickness) const;
```

根据无防护的冲击力和厚度计算防护后的力。

- **参数**:
  - `force_unprotected`: 无防护的冲击力 (kN)
  - `thickness`: 防护材料厚度 (mm)
- **返回**: 防护后的冲击力 (kN)
- **异常**: 如果输入超出数据范围，抛出 `std::runtime_error`

#### `IsInputValid`

```cpp
bool IsInputValid(double force_unprotected, double thickness) const;
```

检查输入是否在有效范围内。

#### `GetForceRange`

```cpp
std::pair<double, double> GetForceRange() const;
```

获取冲击力的有效范围 `[最小值, 最大值]`。

#### `GetThicknessRange`

```cpp
std::pair<double, double> GetThicknessRange() const;
```

获取厚度的有效范围 `[最小值, 最大值]`。

## 数据文件格式

RT-FEM.tsv文件应包含以下列：
- `冲击力（kN）`: 无防护的冲击力值
- `6mm`, `12mm`, `18mm`, `24mm`: 对应厚度下的防护后力值

示例：
```
冲击头质量（kg)	冲击力（kN）	6mm	12mm	18mm	24mm
10	170	90.000	57.852	41.1822	32.1167
9	153	77.852	49.9554	36.2277	30.1538
...
```

## 编译配置

确保在 `CMakeLists.txt` 中包含源文件：

```cmake
add_executable(mujoco_simulator
  # ... 其他源文件 ...
  src/force_interpolation.cc
)
```

## 注意事项

1. **文件路径**: 如果使用默认路径，确保RT-FEM.tsv文件在可访问的位置
2. **数据范围**: 输入必须在数据表的范围内，否则会抛出异常
3. **性能**: 对象创建时会加载整个数据表，建议作为成员变量只初始化一次
4. **线程安全**: 当前实现不是线程安全的，如果多线程使用需要加锁

## 示例程序

运行示例程序：

```bash
# 编译示例（需要添加到CMakeLists.txt）
cd build
cmake ..
make

# 运行示例
./force_interpolation_example
```

## 与Python版本的对比

| 特性 | C++版本 | Python版本 |
|------|---------|-----------|
| 性能 | 高 | 中等 |
| 依赖 | 无 | pandas, numpy, scipy |
| 部署 | 简单 | 需要Python环境 |
| 实时性 | 适合 | 可能较慢 |
| 维护 | 需要同步 | 原始版本 |

## 故障排除

### 问题：找不到RT-FEM.tsv文件

**解决方案**: 指定完整路径
```cpp
ForceInterpolation interpolator("/absolute/path/to/RT-FEM.tsv");
```

### 问题：输入超出范围

**解决方案**: 检查输入值是否在数据范围内
```cpp
if (interpolator.IsInputValid(force, thickness)) {
    double protected_force = interpolator.GetProtectedForce(force, thickness);
} else {
    // 处理超出范围的情况
}
```

### 问题：编译错误

**解决方案**: 确保C++标准为C++17或更高
```cmake
set(CMAKE_CXX_STANDARD 17)
```

