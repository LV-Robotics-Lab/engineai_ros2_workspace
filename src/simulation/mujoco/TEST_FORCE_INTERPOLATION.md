# ForceInterpolation 测试程序使用说明

## 功能说明

这个测试程序用于验证 `ForceInterpolation` 类的功能，可以测试给定冲击力和防护材料厚度后，计算出的防护衰减结果。

## 编译方法

在 `engineai_ros2_workspace` 目录下执行：

```bash
cd engineai_ros2_workspace
colcon build --packages-select mujoco_simulator --cmake-args -DCMAKE_BUILD_TYPE=Release
```

或者使用项目提供的编译脚本：

```bash
./scripts/build_nodes_4090.sh sim
```

## 运行方法

编译完成后，测试程序位于 `install/mujoco_simulator/lib/mujoco_simulator/test_force_interpolation`

### 方法1: 使用默认路径（推荐）

程序会自动查找 `RT-FEM.tsv` 文件：

```bash
cd engineai_ros2_workspace
./install/mujoco_simulator/lib/mujoco_simulator/test_force_interpolation
```

### 方法2: 指定RT-FEM.tsv文件路径

```bash
./install/mujoco_simulator/lib/mujoco_simulator/test_force_interpolation /path/to/RT-FEM.tsv
```

例如：

```bash
./install/mujoco_simulator/lib/mujoco_simulator/test_force_interpolation scripts/ThicknessCalculate/RT-FEM.tsv
```

## 测试内容

测试程序会执行以下测试：

1. **测试用例1**: 使用数据表中的精确值
   - 验证插值器能正确读取和返回数据表中的值
   - 例如：170 kN, 6 mm 应该返回 90.000 kN

2. **测试用例2**: 使用插值点（不在数据表中的值）
   - 测试双线性插值功能
   - 例如：100 kN, 10 mm（需要插值计算）

3. **测试用例3**: 固定冲击力，扫描不同厚度
   - 展示不同厚度下的防护效果
   - 例如：固定 100 kN，测试 6mm, 9mm, 12mm, 15mm, 18mm, 21mm, 24mm

4. **测试用例4**: 固定厚度，扫描不同冲击力
   - 展示不同冲击力下的防护效果
   - 例如：固定 12 mm，测试 17 kN 到 170 kN

5. **测试用例5**: 边界情况测试
   - 测试最小/最大冲击力和厚度的组合

## 输出说明

程序会输出：
- 有效范围（冲击力和厚度的最小/最大值）
- 每个测试用例的：
  - 输入：冲击力（kN）和厚度（mm）
  - 输出：防护后的力（kN）
  - 衰减率（%）= (原始力 - 防护后力) / 原始力 × 100%

## 示例输出

```
========================================
ForceInterpolation 测试程序
========================================
有效范围:
  冲击力: [17, 170] kN
  厚度: [6, 24] mm
========================================

测试用例1: 使用数据表中的精确值
----------------------------------------
  冲击力:  170.0000 kN, 厚度:  6.0000 mm
  -> 防护后:    90.0000 kN, 衰减:  47.06%

  冲击力:  170.0000 kN, 厚度: 12.0000 mm
  -> 防护后:    57.8520 kN, 衰减:  65.97%
...
```

## 调试技巧

1. **验证数据表值**: 如果测试用例1的结果与RT-FEM.tsv中的数据不一致，说明数据加载有问题

2. **验证插值**: 测试用例2的结果应该在相邻数据点之间，例如：
   - 100 kN, 10 mm 的结果应该在 85 kN, 12 mm 和 102 kN, 12 mm 之间

3. **验证趋势**: 
   - 厚度越大，衰减应该越明显（防护后力越小）
   - 冲击力越大，衰减比例可能不同（取决于材料特性）

4. **边界检查**: 如果输入超出范围，程序会抛出异常

## 在代码中使用

如果你想在自己的代码中测试特定的值，可以参考以下示例：

```cpp
#include "force_interpolation.h"

// 创建插值器
ForceInterpolation interpolator("scripts/ThicknessCalculate/RT-FEM.tsv");

// 测试特定值
double force_unprotected = 100.0;  // kN
double thickness = 12.0;           // mm

if (interpolator.IsInputValid(force_unprotected, thickness)) {
    double force_protected = interpolator.GetProtectedForce(force_unprotected, thickness);
    double reduction = (force_unprotected - force_protected) / force_unprotected * 100.0;
    
    std::cout << "原始力: " << force_unprotected << " kN\n";
    std::cout << "防护后: " << force_protected << " kN\n";
    std::cout << "衰减率: " << reduction << "%\n";
} else {
    std::cout << "输入超出范围！\n";
}
```

