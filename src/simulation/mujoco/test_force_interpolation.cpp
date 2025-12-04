#include "force_interpolation.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char* argv[]) {
  try {
    // 创建ForceInterpolation对象
    // 可以指定RT-FEM.tsv的路径，或者使用默认路径
    std::string tsv_path = "";
    if (argc > 1) {
      tsv_path = argv[1];
    }
    
    ForceInterpolation interpolator(tsv_path);
    
    // 获取有效范围
    auto force_range = interpolator.GetForceRange();
    auto thickness_range = interpolator.GetThicknessRange();
    
    std::cout << "========================================\n";
    std::cout << "ForceInterpolation 测试程序\n";
    std::cout << "========================================\n";
    std::cout << "有效范围:\n";
    std::cout << "  冲击力: [" << force_range.first << ", " << force_range.second << "] kN\n";
    std::cout << "  厚度: [" << thickness_range.first << ", " << thickness_range.second << "] mm\n";
    std::cout << "========================================\n\n";
    
    // 测试用例1: 使用数据表中的精确值
    std::cout << "测试用例1: 使用数据表中的精确值\n";
    std::cout << "----------------------------------------\n";
    std::vector<std::pair<double, double>> test_cases_exact = {
      {170.0, 6.0},   // 应该返回90.000
      {170.0, 12.0},  // 应该返回57.852
      {170.0, 18.0},  // 应该返回41.1822
      {170.0, 24.0},  // 应该返回32.1167
      {17.0, 6.0},    // 应该返回13.5883
      {17.0, 24.0},   // 应该返回5.52182
    };
    
    for (const auto& test : test_cases_exact) {
      double force_unprotected = test.first;
      double thickness = test.second;
      double force_protected = interpolator.GetProtectedForce(force_unprotected, thickness);
      double reduction_ratio = (force_unprotected - force_protected) / force_unprotected * 100.0;
      
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "  冲击力: " << std::setw(8) << force_unprotected << " kN, "
                << "厚度: " << std::setw(6) << thickness << " mm\n";
      std::cout << "  -> 防护后: " << std::setw(10) << force_protected << " kN, "
                << "衰减: " << std::setw(6) << std::setprecision(2) << reduction_ratio << "%\n\n";
    }
    
    // 测试用例2: 使用插值点（不在数据表中的值）
    std::cout << "测试用例2: 使用插值点（不在数据表中的值）\n";
    std::cout << "----------------------------------------\n";
    std::vector<std::pair<double, double>> test_cases_interp = {
      {10.0, 6.0},  // 在170和17之间，厚度在6和12之间
      {5.0, 6.0},   // 在51和34之间，厚度在12和18之间
      {12.0, 6.0},  // 在119和136之间，厚度在18和24之间
    };
    
    for (const auto& test : test_cases_interp) {
      double force_unprotected = test.first;
      double thickness = test.second;
      
      if (interpolator.IsInputValid(force_unprotected, thickness)) {
        double force_protected = interpolator.GetProtectedForce(force_unprotected, thickness);
        double reduction_ratio = (force_unprotected - force_protected) / force_unprotected * 100.0;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  冲击力: " << std::setw(8) << force_unprotected << " kN, "
                  << "厚度: " << std::setw(6) << thickness << " mm\n";
        std::cout << "  -> 防护后: " << std::setw(10) << force_protected << " kN, "
                  << "衰减: " << std::setw(6) << std::setprecision(2) << reduction_ratio << "%\n\n";
      } else {
        std::cout << "  输入超出范围: 冲击力=" << force_unprotected 
                  << " kN, 厚度=" << thickness << " mm\n\n";
      }
    }
    
    // 测试用例3: 扫描不同厚度下的衰减效果
    std::cout << "测试用例3: 固定冲击力，扫描不同厚度\n";
    std::cout << "----------------------------------------\n";
    double test_force = 100.0;  // 100 kN
    std::vector<double> test_thicknesses = {6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0};
    
    if (interpolator.IsInputValid(test_force, 6.0)) {
      std::cout << "固定冲击力: " << test_force << " kN\n";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << std::setw(10) << "厚度(mm)" 
                << std::setw(15) << "防护后(kN)" 
                << std::setw(15) << "衰减率(%)\n";
      std::cout << "----------------------------------------\n";
      
      for (double thickness : test_thicknesses) {
        if (interpolator.IsInputValid(test_force, thickness)) {
          double force_protected = interpolator.GetProtectedForce(test_force, thickness);
          double reduction_ratio = (test_force - force_protected) / test_force * 100.0;
          
          std::cout << std::setw(10) << thickness
                    << std::setw(15) << force_protected
                    << std::setw(15) << std::setprecision(2) << reduction_ratio << "\n";
        }
      }
      std::cout << "\n";
    }
    
    // 测试用例4: 扫描不同冲击力下的衰减效果
    std::cout << "测试用例4: 固定厚度，扫描不同冲击力\n";
    std::cout << "----------------------------------------\n";
    double test_thickness = 12.0;  // 12 mm
    std::vector<double> test_forces = {17.0, 34.0, 51.0, 68.0, 85.0, 102.0, 119.0, 136.0, 153.0, 170.0};
    
    if (interpolator.IsInputValid(17.0, test_thickness)) {
      std::cout << "固定厚度: " << test_thickness << " mm\n";
      std::cout << std::fixed << std::setprecision(4);
      std::cout << std::setw(12) << "冲击力(kN)" 
                << std::setw(15) << "防护后(kN)" 
                << std::setw(15) << "衰减率(%)\n";
      std::cout << "----------------------------------------\n";
      
      for (double force : test_forces) {
        if (interpolator.IsInputValid(force, test_thickness)) {
          double force_protected = interpolator.GetProtectedForce(force, test_thickness);
          double reduction_ratio = (force - force_protected) / force * 100.0;
          
          std::cout << std::setw(12) << force
                    << std::setw(15) << force_protected
                    << std::setw(15) << std::setprecision(2) << reduction_ratio << "\n";
        }
      }
      std::cout << "\n";
    }
    
    // 测试用例5: 边界情况
    std::cout << "测试用例5: 边界情况测试\n";
    std::cout << "----------------------------------------\n";
    std::vector<std::pair<double, double>> boundary_cases = {
      {force_range.first, thickness_range.first},   // 最小值
      {force_range.first, thickness_range.second},  // 最小力，最大厚度
      {force_range.second, thickness_range.first},  // 最大力，最小厚度
      {force_range.second, thickness_range.second}, // 最大值
    };
    
    for (const auto& test : boundary_cases) {
      double force_unprotected = test.first;
      double thickness = test.second;
      double force_protected = interpolator.GetProtectedForce(force_unprotected, thickness);
      double reduction_ratio = (force_unprotected - force_protected) / force_unprotected * 100.0;
      
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "  冲击力: " << std::setw(8) << force_unprotected << " kN, "
                << "厚度: " << std::setw(6) << thickness << " mm\n";
      std::cout << "  -> 防护后: " << std::setw(10) << force_protected << " kN, "
                << "衰减: " << std::setw(6) << std::setprecision(2) << reduction_ratio << "%\n\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "测试完成！\n";
    std::cout << "========================================\n";
    
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }
}

