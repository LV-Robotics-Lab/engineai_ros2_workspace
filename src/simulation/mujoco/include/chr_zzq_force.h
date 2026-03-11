/**
 * @file chr_zzq_force.h
 * @brief CHR 力衰减公式：F_after = C * t^alpha * p^beta * F_before^gamma
 * @details 从 fitted_parameters.json 加载参数，与 force_calculator.py / thickness_selection 同公式
 */

#ifndef CHR_ZZQ_FORCE_H_
#define CHR_ZZQ_FORCE_H_

#include <string>

class ChrZzqForce {
 public:
  /**
   * @brief 从 fitted_parameters.json 加载参数
   * @param params_path 参数文件路径（可为空，使用默认查找）
   */
  explicit ChrZzqForce(const std::string& params_path = "");

  /**
   * @brief 计算衰减后的力
   * @param force_unprotected 衰减前力 (kN)
   * @param thickness 厚度 (mm)
   * @param density 密度，默认 0.4
   * @return 衰减后力 (kN)
   */
  double GetProtectedForce(double force_unprotected, double thickness,
                          double density = 0.4) const;

  /**
   * @brief 是否已成功加载参数
   */
  bool IsLoaded() const { return loaded_; }

  /**
   * @brief 输入范围检查（ZZQ 公式对任意正输入有效，但建议在合理范围内）
   */
  bool IsInputValid(double force_unprotected, double thickness) const;

  std::pair<double, double> GetForceRange() const;
  std::pair<double, double> GetThicknessRange() const;

 private:
  bool LoadParams(const std::string& path);

  double C_ = 0.02858;
  double alpha_ = -1.16;
  double beta_ = -0.5;
  double gamma_ = 1.878;
  bool loaded_ = false;
};

#endif  // CHR_ZZQ_FORCE_H_
