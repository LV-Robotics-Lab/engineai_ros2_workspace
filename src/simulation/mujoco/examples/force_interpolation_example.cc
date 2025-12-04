/**
 * @file force_interpolation_example.cc
 * @brief 力插值计算使用示例
 * 
 * 演示如何在C++代码中使用ForceInterpolation类来计算防护后的力
 */

#include "force_interpolation.h"
#include <iostream>
#include <iomanip>

int main() {
  std::cout << "=== RT-FEM数据双线性插值计算器 (C++版本) ===\n" << std::endl;
  
  try {
    // 创建力插值对象
    // 可以指定RT-FEM.tsv文件的完整路径，或使用默认路径
    ForceInterpolation interpolator;
    // 或者指定完整路径：
    // ForceInterpolation interpolator("/path/to/RT-FEM.tsv");
    
    // 获取数据范围
    auto force_range = interpolator.GetForceRange();
    auto thickness_range = interpolator.GetThicknessRange();
    
    std::cout << "数据范围:" << std::endl;
    std::cout << "  冲击力: [" << force_range.first << ", " << force_range.second << "] kN" << std::endl;
    std::cout << "  厚度: [" << thickness_range.first << ", " << thickness_range.second << "] mm" << std::endl;
    std::cout << std::endl;
    
    // 测试用例
    struct TestCase {
      double force_unprotected;  // 无防护的冲击力 (kN)
      double thickness;          // 防护材料厚度 (mm)
    };
    
    std::vector<TestCase> test_cases = {
      {100.0, 12.0},   // 100 kN, 12mm
      {85.0, 18.0},    // 85 kN, 18mm
      {150.0, 6.0},    // 150 kN, 6mm
      {50.0, 24.0},    // 50 kN, 24mm
      {120.0, 15.0},   // 120 kN, 15mm (需要插值)
      {90.0, 10.0},    // 90 kN, 10mm (需要插值)
    };
    
    std::cout << "测试用例:" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(15) << "无防护力(kN)" 
              << std::setw(12) << "厚度(mm)" 
              << std::setw(15) << "防护后力(kN)"
              << std::setw(15) << "力下降(kN)"
              << std::setw(12) << "下降率(%)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& test : test_cases) {
      try {
        double force_protected = interpolator.GetProtectedForce(
          test.force_unprotected, test.thickness);
        double force_reduction = test.force_unprotected - force_protected;
        double reduction_percentage = (force_reduction / test.force_unprotected) * 100.0;
        
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(15) << test.force_unprotected
                  << std::setw(12) << test.thickness
                  << std::setw(15) << force_protected
                  << std::setw(15) << force_reduction
                  << std::setw(12) << reduction_percentage << std::endl;
      } catch (const std::exception& e) {
        std::cout << std::setw(15) << test.force_unprotected
                  << std::setw(12) << test.thickness
                  << "  错误: " << e.what() << std::endl;
      }
    }
    
    std::cout << std::string(70, '-') << std::endl;
    
    // 示例：在仿真循环中使用
    std::cout << "\n=== 在仿真循环中使用示例 ===" << std::endl;
    std::cout << "// 在类成员变量中初始化（只需初始化一次）" << std::endl;
    std::cout << "// ForceInterpolation force_interpolator_;" << std::endl;
    std::cout << std::endl;
    std::cout << "// 在计算接触力时使用" << std::endl;
    std::cout << "// double contact_force_magnitude = ...;  // 从MuJoCo获取的接触力大小 (kN)" << std::endl;
    std::cout << "// double protection_thickness = 12.0;    // 防护材料厚度 (mm)" << std::endl;
    std::cout << "// double protected_force = force_interpolator_.GetProtectedForce(" << std::endl;
    std::cout << "//     contact_force_magnitude, protection_thickness);" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

