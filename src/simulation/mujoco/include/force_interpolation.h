#pragma once

#include <string>
#include <vector>
#include <stdexcept>

/**
 * @brief 力插值计算类
 * 
 * 根据无防护的冲击力和防护材料厚度，使用双线性插值方法计算防护后的力。
 * 数据来源于RT-FEM.tsv文件。
 */
class ForceInterpolation {
 public:
  /**
   * @brief 构造函数
   * @param tsv_file_path RT-FEM数据文件路径，默认为空（使用默认路径）
   */
  explicit ForceInterpolation(const std::string& tsv_file_path = "");

  /**
   * @brief 根据无防护的力和厚度，计算防护后的力
   * 
   * @param force_unprotected 无防护的冲击力 (kN)
   * @param thickness 防护材料厚度 (mm)
   * @return 防护后的冲击力 (kN)
   * @throws std::runtime_error 当数据文件加载失败或输入超出范围时
   */
  double GetProtectedForce(double force_unprotected, double thickness) const;

  /**
   * @brief 检查输入是否在有效范围内
   * 
   * @param force_unprotected 无防护的冲击力 (kN)
   * @param thickness 防护材料厚度 (mm)
   * @return true 如果输入在有效范围内
   */
  bool IsInputValid(double force_unprotected, double thickness) const;

  /**
   * @brief 获取冲击力的有效范围
   * @return std::pair<double, double> (最小值, 最大值)
   */
  std::pair<double, double> GetForceRange() const;

  /**
   * @brief 获取厚度的有效范围
   * @return std::pair<double, double> (最小值, 最大值)
   */
  std::pair<double, double> GetThicknessRange() const;

 private:
  /**
   * @brief 加载RT-FEM数据文件
   * @param file_path 数据文件路径
   * @throws std::runtime_error 当文件加载失败时
   */
  void LoadData(const std::string& file_path);

  /**
   * @brief 手动实现双线性插值
   * 
   * @param force_unprotected 无防护的冲击力 (kN)
   * @param thickness 防护材料厚度 (mm)
   * @return 防护后的冲击力 (kN)
   */
  double BilinearInterpolation(double force_unprotected, double thickness) const;

  // 数据存储
  std::vector<double> force_values_;           // 无防护的冲击力数组 (kN)
  std::vector<double> thickness_values_;       // 厚度数组 (mm): [6, 12, 18, 24]
  std::vector<std::vector<double>> force_protected_matrix_;  // 防护后的力矩阵 [force][thickness]
  
  // 数据范围
  double force_min_;
  double force_max_;
  double thickness_min_;
  double thickness_max_;
};

