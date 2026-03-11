/**
 * @file chr_zzq_force.cc
 * @brief CHR 力衰减公式实现（fitted_parameters）
 */

#include "chr_zzq_force.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <sys/stat.h>

ChrZzqForce::ChrZzqForce(const std::string& params_path) {
  std::string path = params_path;
  if (path.empty()) {
    std::string candidates[] = {
      "scripts/ThicknessCalculate/fitted_parameters.json",
      "../scripts/ThicknessCalculate/fitted_parameters.json",
      "../../scripts/ThicknessCalculate/fitted_parameters.json",
      "/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json",
    };
    auto exists = [](const std::string& p) {
      struct stat buffer;
      return stat(p.c_str(), &buffer) == 0;
    };
    for (const auto& c : candidates) {
      if (exists(c)) {
        path = c;
        break;
      }
    }
  }
  if (!path.empty()) {
    loaded_ = LoadParams(path);
  }
}

bool ChrZzqForce::LoadParams(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ChrZzqForce: Cannot open " << path << std::endl;
    return false;
  }
  std::string line;
  std::string content;
  while (std::getline(f, line)) {
    content += line;
  }
  f.close();
  // 简单解析 JSON（C, alpha, beta, gamma）
  auto extract = [&content](const std::string& key) -> double {
    std::string search = "\"" + key + "\":";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return 0.0;
    pos += search.size();
    size_t end = content.find_first_of(",}", pos);
    if (end == std::string::npos) return 0.0;
    try {
      return std::stod(content.substr(pos, end - pos));
    } catch (...) {
      return 0.0;
    }
  };
  C_ = extract("C");
  alpha_ = extract("alpha");
  beta_ = extract("beta");
  gamma_ = extract("gamma");
  if (C_ <= 0 || alpha_ == 0 || beta_ == 0 || gamma_ == 0) {
    std::cerr << "ChrZzqForce: Invalid params C=" << C_ << " alpha=" << alpha_
              << " beta=" << beta_ << " gamma=" << gamma_ << std::endl;
    return false;
  }
  return true;
}

double ChrZzqForce::GetProtectedForce(double force_unprotected, double thickness,
                                     double density) const {
  if (!loaded_ || force_unprotected <= 0 || thickness <= 0 || density <= 0) {
    return force_unprotected;
  }
  return C_ * std::pow(thickness, alpha_) * std::pow(density, beta_) *
         std::pow(force_unprotected, gamma_);
}

bool ChrZzqForce::IsInputValid(double force_unprotected, double thickness) const {
  return loaded_ && force_unprotected > 0 && thickness > 0;
}

std::pair<double, double> ChrZzqForce::GetForceRange() const {
  return {0.01, 200.0};  // 公式对任意正输入有效
}

std::pair<double, double> ChrZzqForce::GetThicknessRange() const {
  return {0.1, 50.0};  // 公式对任意正厚度有效
}
