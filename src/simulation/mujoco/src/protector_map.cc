/**
 * @file protector_map.cc
 * @brief 护具地图查表实现
 */

#include "protector_map.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>

ProtectorMap::ProtectorMap(const std::string& map_dir) {
  if (map_dir.empty()) {
    return;
  }
  std::string path_front = map_dir + "/yz_map_front.tsv";
  std::string path_back = map_dir + "/yz_map_back.tsv";
  if (!LoadTsv(path_front, grid_front_) || !LoadTsv(path_back, grid_back_)) {
    std::cerr << "ProtectorMap: Failed to load map from " << map_dir << std::endl;
    return;
  }
  ComputeBounds();
  loaded_ = true;
}

bool ProtectorMap::LoadTsv(const std::string& file_path,
                          std::vector<std::vector<double>>& grid) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "ProtectorMap: Cannot open " << file_path << std::endl;
    return false;
  }
  std::string line;
  // 跳过注释行
  if (!std::getline(file, line) || line.empty()) return false;
  if (line[0] == '#') {
    if (!std::getline(file, line)) return false;
  }
  // 解析表头: z\y, -0.4, -0.35, ..., 0.4
  std::istringstream header_stream(line);
  std::string cell;
  y_coords_.clear();
  std::getline(header_stream, cell, '\t');  // 跳过 "z\y"
  while (std::getline(header_stream, cell, '\t')) {
    try {
      y_coords_.push_back(std::stod(cell));
    } catch (...) {
      break;
    }
  }
  if (y_coords_.empty()) return false;
  // 读取数据行
  grid.clear();
  z_coords_.clear();
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::istringstream row_stream(line);
    std::string z_str;
    if (!std::getline(row_stream, z_str, '\t')) continue;
    try {
      z_coords_.push_back(std::stod(z_str));
    } catch (...) {
      continue;
    }
    std::vector<double> row;
    while (std::getline(row_stream, cell, '\t')) {
      try {
        row.push_back(std::stod(cell));
      } catch (...) {
        row.push_back(0.0);
      }
    }
    if (row.size() == y_coords_.size()) {
      grid.push_back(std::move(row));
    }
  }
  return !grid.empty() && grid.size() == z_coords_.size();
}

void ProtectorMap::ComputeBounds() {
  // 根据格心坐标与相邻间距计算有效范围：格心 ± 半格距
  // 半格距 = 方格间距 / 2，由 TSV 表头/行首坐标差计算，非写死
  if (y_coords_.size() < 2 || z_coords_.size() < 2) {
    return;
  }
  // Y 方向：左边界 = 首格心 - 首格半间距，右边界 = 末格心 + 末格半间距
  double half_dy_left = (y_coords_[1] - y_coords_[0]) * 0.5;
  double half_dy_right = (y_coords_.back() - y_coords_[y_coords_.size() - 2]) * 0.5;
  y_min_ = y_coords_.front() - half_dy_left;
  y_max_ = y_coords_.back() + half_dy_right;
  // Z 方向：同理
  double half_dz_bottom = (z_coords_[1] - z_coords_[0]) * 0.5;
  double half_dz_top = (z_coords_.back() - z_coords_[z_coords_.size() - 2]) * 0.5;
  z_min_ = z_coords_.front() - half_dz_bottom;
  z_max_ = z_coords_.back() + half_dz_top;
}

double ProtectorMap::LookupInGrid(const std::vector<std::vector<double>>& grid,
                                  double y, double z) const {
  if (grid.empty() || y_coords_.empty() || z_coords_.empty()) return 0.0;
  // 最近邻：找最近的 y 列和 z 行
  auto y_it = std::min_element(y_coords_.begin(), y_coords_.end(),
      [y](double a, double b) { return std::abs(a - y) < std::abs(b - y); });
  auto z_it = std::min_element(z_coords_.begin(), z_coords_.end(),
      [z](double a, double b) { return std::abs(a - z) < std::abs(b - z); });
  int col = static_cast<int>(y_it - y_coords_.begin());
  int row = static_cast<int>(z_it - z_coords_.begin());
  if (row < 0 || row >= static_cast<int>(grid.size()) ||
      col < 0 || col >= static_cast<int>(grid[0].size())) {
    return 0.0;
  }
  return grid[row][col];
}

double ProtectorMap::LookupThickness(double x, double y, double z) const {
  if (!loaded_) return 0.0;
  // 边界检查：使用 ComputeBounds() 算出的有效范围（格心 ± 半格距）
  if (y < y_min_ || y > y_max_ || z < z_min_ || z > z_max_) return 0.0;
  // x >= 0 用 front（前侧），x < 0 用 back（后侧）
  if (x >= 0) {
    return LookupInGrid(grid_front_, y, z);
  } else {
    return LookupInGrid(grid_back_, y, z);
  }
}
