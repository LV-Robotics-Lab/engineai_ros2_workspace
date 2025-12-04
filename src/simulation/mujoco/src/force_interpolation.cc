#include "force_interpolation.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sys/stat.h>

ForceInterpolation::ForceInterpolation(const std::string& tsv_file_path) {
  std::string file_path = tsv_file_path;
  
  // 如果未指定路径，使用默认路径（相对于脚本目录）
  if (file_path.empty()) {
    // 尝试从环境变量或相对路径找到RT-FEM.tsv
    // 默认假设在scripts/ThicknessCalculate/目录下
    std::string default_paths[] = {
      "scripts/ThicknessCalculate/RT-FEM.tsv",
      "../scripts/ThicknessCalculate/RT-FEM.tsv",
      "../../scripts/ThicknessCalculate/RT-FEM.tsv",
      "/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/RT-FEM.tsv"
    };
    
    // 辅助函数：检查文件是否存在
    auto file_exists = [](const std::string& path) -> bool {
      struct stat buffer;
      return (stat(path.c_str(), &buffer) == 0);
    };
    
    bool found = false;
    for (const auto& path : default_paths) {
      if (file_exists(path)) {
        file_path = path;
        found = true;
        break;
      }
    }
    
    if (!found) {
      throw std::runtime_error("无法找到RT-FEM.tsv文件，请指定完整路径");
    }
  }
  
  LoadData(file_path);
}

void ForceInterpolation::LoadData(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开RT-FEM数据文件: " + file_path);
  }

  std::string line;
  
  // 读取表头
  if (!std::getline(file, line)) {
    throw std::runtime_error("RT-FEM文件为空");
  }
  
  // 解析表头，找到各列的索引
  std::istringstream header_stream(line);
  std::vector<std::string> headers;
  std::string header;
  
  while (std::getline(header_stream, header, '\t')) {
    headers.push_back(header);
  }
  
  // 找到冲击力列和厚度列的索引
  int force_col_idx = -1;
  std::vector<int> thickness_col_indices;
  std::vector<std::string> thickness_columns = {"6mm", "12mm", "18mm", "24mm"};
  
  for (size_t i = 0; i < headers.size(); ++i) {
    if (headers[i] == "冲击力（kN）") {
      force_col_idx = static_cast<int>(i);
    }
    for (const auto& col : thickness_columns) {
      if (headers[i] == col) {
        thickness_col_indices.push_back(static_cast<int>(i));
      }
    }
  }
  
  if (force_col_idx == -1) {
    throw std::runtime_error("未找到'冲击力（kN）'列");
  }
  
  if (thickness_col_indices.size() != 4) {
    throw std::runtime_error("未找到所有厚度列（6mm, 12mm, 18mm, 24mm）");
  }
  
  // 提取厚度值
  thickness_values_ = {6.0, 12.0, 18.0, 24.0};
  
  // 读取数据行
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    
    std::istringstream line_stream(line);
    std::vector<std::string> fields;
    std::string field;
    
    while (std::getline(line_stream, field, '\t')) {
      fields.push_back(field);
    }
    
    if (fields.size() <= static_cast<size_t>(std::max(force_col_idx, 
        *std::max_element(thickness_col_indices.begin(), thickness_col_indices.end())))) {
      continue;  // 跳过不完整的行
    }
    
    // 读取冲击力值
    try {
      double force = std::stod(fields[force_col_idx]);
      force_values_.push_back(force);
      
      // 读取该冲击力下各厚度的防护后力值
      std::vector<double> protected_forces;
      for (int col_idx : thickness_col_indices) {
        double protected_force = std::stod(fields[col_idx]);
        protected_forces.push_back(protected_force);
      }
      force_protected_matrix_.push_back(protected_forces);
    } catch (const std::exception& e) {
      // 跳过无法解析的行
      continue;
    }
  }
  
  file.close();
  
  if (force_values_.empty()) {
    throw std::runtime_error("RT-FEM文件中没有有效数据");
  }
  
  // 对数据进行排序，确保force_values_是升序的（lower_bound需要升序数据）
  // 同时需要同步排序force_protected_matrix_
  std::vector<size_t> indices(force_values_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
    return force_values_[a] < force_values_[b];
  });
  
  // 创建排序后的数据
  std::vector<double> sorted_force_values;
  std::vector<std::vector<double>> sorted_force_protected_matrix;
  for (size_t idx : indices) {
    sorted_force_values.push_back(force_values_[idx]);
    sorted_force_protected_matrix.push_back(force_protected_matrix_[idx]);
  }
  
  force_values_ = std::move(sorted_force_values);
  force_protected_matrix_ = std::move(sorted_force_protected_matrix);
  
  // 计算数据范围
  force_min_ = *std::min_element(force_values_.begin(), force_values_.end());
  force_max_ = *std::max_element(force_values_.begin(), force_values_.end());
  thickness_min_ = *std::min_element(thickness_values_.begin(), thickness_values_.end());
  thickness_max_ = *std::max_element(thickness_values_.begin(), thickness_values_.end());
}

double ForceInterpolation::GetProtectedForce(double force_unprotected, double thickness) const {
  if (!IsInputValid(force_unprotected, thickness)) {
    throw std::runtime_error(
      "输入超出范围: 冲击力=" + std::to_string(force_unprotected) + 
      " kN (范围: [" + std::to_string(force_min_) + ", " + std::to_string(force_max_) + "]), "
      "厚度=" + std::to_string(thickness) + 
      " mm (范围: [" + std::to_string(thickness_min_) + ", " + std::to_string(thickness_max_) + "])"
    );
  }
  
  return BilinearInterpolation(force_unprotected, thickness);
}

bool ForceInterpolation::IsInputValid(double force_unprotected, double thickness) const {
  return (force_unprotected >= force_min_ && force_unprotected <= force_max_) &&
         (thickness >= thickness_min_ && thickness <= thickness_max_);
}

std::pair<double, double> ForceInterpolation::GetForceRange() const {
  return {force_min_, force_max_};
}

std::pair<double, double> ForceInterpolation::GetThicknessRange() const {
  return {thickness_min_, thickness_max_};
}

double ForceInterpolation::BilinearInterpolation(double force_unprotected, double thickness) const {
  // 检查是否正好匹配数据表中的值
  const double tolerance = 1e-6;
  
  for (size_t i = 0; i < force_values_.size(); ++i) {
    if (std::abs(force_values_[i] - force_unprotected) < tolerance) {
      for (size_t j = 0; j < thickness_values_.size(); ++j) {
        if (std::abs(thickness_values_[j] - thickness) < tolerance) {
          return force_protected_matrix_[i][j];
        }
      }
    }
  }
  
  // 找到冲击力维度上的两个相邻点
  size_t force_idx = std::lower_bound(force_values_.begin(), force_values_.end(), force_unprotected) 
                     - force_values_.begin();
  
  // 处理边界情况
  if (force_idx == 0) {
    force_idx = 1;
  } else if (force_idx >= force_values_.size()) {
    force_idx = force_values_.size() - 1;
  }
  
  double force_low = force_values_[force_idx - 1];
  double force_high = force_values_[force_idx];
  
  // 找到厚度维度上的两个相邻点
  size_t thickness_idx = std::lower_bound(thickness_values_.begin(), thickness_values_.end(), thickness) 
                         - thickness_values_.begin();
  
  // 处理边界情况
  if (thickness_idx == 0) {
    thickness_idx = 1;
  } else if (thickness_idx >= thickness_values_.size()) {
    thickness_idx = thickness_values_.size() - 1;
  }
  
  double thickness_low = thickness_values_[thickness_idx - 1];
  double thickness_high = thickness_values_[thickness_idx];
  
  // 获取四个角点的防护后力值
  double f11 = force_protected_matrix_[force_idx - 1][thickness_idx - 1];  // (force_low, thickness_low)
  double f12 = force_protected_matrix_[force_idx - 1][thickness_idx];      // (force_low, thickness_high)
  double f21 = force_protected_matrix_[force_idx][thickness_idx - 1];      // (force_high, thickness_low)
  double f22 = force_protected_matrix_[force_idx][thickness_idx];          // (force_high, thickness_high)
  
  // 计算插值权重
  double t_weight = (thickness_high == thickness_low) ? 0.0 : 
                    (thickness - thickness_low) / (thickness_high - thickness_low);
  double f_weight = (force_high == force_low) ? 0.0 : 
                    (force_unprotected - force_low) / (force_high - force_low);
  
  // 双线性插值
  // 先在厚度维度上插值
  double f1 = f11 * (1.0 - t_weight) + f12 * t_weight;  // 在force_low处的插值
  double f2 = f21 * (1.0 - t_weight) + f22 * t_weight;  // 在force_high处的插值
  
  // 再在冲击力维度上插值
  double force_protected = f1 * (1.0 - f_weight) + f2 * f_weight;
  
  return force_protected;
}

