/**
 * @file protector_map.h
 * @brief 护具地图查表 - 根据碰撞点站立系坐标查询护具厚度
 * @details 加载 yz_map_front.tsv (x>=0) 和 yz_map_back.tsv (x<0)，
 *          用接触点世界坐标 (x,y,z) 查表得到厚度 (mm)，最近邻插值。
 *          边界与半格距由 TSV 表头/行首的 Y、Z 坐标计算得出，非写死。
 */

#ifndef PROTECTOR_MAP_H_
#define PROTECTOR_MAP_H_

#include <string>
#include <vector>
#include <optional>

class ProtectorMap {
 public:
  /**
   * @brief 从目录加载护具地图
   * @param map_dir 包含 yz_map_front.tsv 和 yz_map_back.tsv 的目录路径
   */
  explicit ProtectorMap(const std::string& map_dir);

  /**
   * @brief 根据接触点世界坐标（=默认站立系）查表得到护具厚度
   * @param x 世界系 x（前正后负）
   * @param y 世界系 y（左正右负）
   * @param z 世界系 z（离地高度）
   * @return 厚度 (mm)，若超出范围或未加载则返回 0
   */
  double LookupThickness(double x, double y, double z) const;

  /**
   * @brief 是否已成功加载地图
   */
  bool IsLoaded() const { return loaded_; }

 private:
  bool LoadTsv(const std::string& file_path, std::vector<std::vector<double>>& grid);
  void ComputeBounds();  // 根据 y_coords_、z_coords_ 计算边界（格心 ± 半格距）
  double LookupInGrid(const std::vector<std::vector<double>>& grid, double y, double z) const;

  std::vector<std::vector<double>> grid_front_;  // x >= 0（前侧）
  std::vector<std::vector<double>> grid_back_;   // x < 0（后侧）
  std::vector<double> y_coords_;   // 列对应的 Y 格心坐标
  std::vector<double> z_coords_;  // 行对应的 Z 格心坐标
  double y_min_ = 0.0, y_max_ = 0.0;  // 有效范围（由格心 ± 半格距算出）
  double z_min_ = 0.0, z_max_ = 0.0;
  bool loaded_ = false;
};

#endif  // PROTECTOR_MAP_H_
