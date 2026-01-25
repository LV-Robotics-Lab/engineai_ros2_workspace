// 必须在包含 glog 之前定义这个宏
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif

// glog 需要先包含 export.h 来定义 GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#include <glog/export.h>
#include <glog/logging.h>
#include "rl_dance/csv_loader.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
// 暂时注释掉，避免 basic_param 依赖
// #include "parameter/global_config_initializer.h"
// #include "rl_dance_param/rl_dance_param.h"
// #include "tool/string_join.h"

namespace rl_dance {

// 简单csv读取和插值实现
Eigen::MatrixXd CsvLoader::LoadAndInterpolateJointTrajectory(const std::string& csv_path,
                                                             const std::vector<std::string>& joint_names, double src_dt,
                                                             double target_dt, int start_frame, int end_frame) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open csv file: " + csv_path);
  }

  std::string line;
  std::vector<std::string> headers;
  // 读取表头
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
      headers.push_back(col);
    }
  }
  // 建立关节名到列索引的映射
  std::map<std::string, int> joint_col_idx;
  for (size_t i = 0; i < joint_names.size(); ++i) {
    auto it = std::find(headers.begin(), headers.end(), joint_names[i]);
    if (it == headers.end()) {
      throw std::runtime_error("Joint name not found in csv: " + joint_names[i]);
    }
    joint_col_idx[joint_names[i]] = std::distance(headers.begin(), it);
  }

  // 读取所有帧
  std::vector<std::vector<double>> raw_traj;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string val;
    std::vector<double> row;
    int col_idx = 0;
    int joint_count = 0;
    while (std::getline(ss, val, ',')) {
      if (joint_count < (int)joint_names.size() && col_idx == joint_col_idx[joint_names[joint_count]]) {
        row.push_back(std::stod(val));
        ++joint_count;
      }
      ++col_idx;
    }
    if ((int)row.size() == (int)joint_names.size()) {
      raw_traj.push_back(row);
    }
  }
  size_t src_N = raw_traj.size();
  size_t num_joints = joint_names.size();
  if (src_N == 0) {
    throw std::runtime_error("No data rows found in csv: " + csv_path);
  }

  // 截取traj_frame区间
  int start_idx = std::max(0, start_frame);
  int end_idx;
  if (end_frame == -1) {
    // 如果end_frame为-1，则播放到最后
    end_idx = (int)src_N - 1;
  } else {
    end_idx = std::min((int)src_N - 1, end_frame);
  }
  if (start_idx > end_idx) {
    throw std::runtime_error("Invalid traj_frame range: start > end");
  }
  std::vector<std::vector<double>> clipped_traj(raw_traj.begin() + start_idx, raw_traj.begin() + end_idx + 1);

  size_t clipped_N = clipped_traj.size();
  
  // 检查是否需要插值：如果源帧率和目标帧率相同（误差<1e-6），直接返回原始数据
  const double dt_epsilon = 1e-6;
  if (std::abs(src_dt - target_dt) < dt_epsilon) {
    // 不需要插值，直接转换为 Eigen 矩阵
    Eigen::MatrixXd traj(clipped_N, num_joints);
    for (size_t i = 0; i < clipped_N; ++i) {
      for (size_t j = 0; j < num_joints; ++j) {
        traj(i, j) = clipped_traj[i][j];
      }
    }
    LOG(INFO) << "LoadAndInterpolateJointTrajectory: No interpolation needed (src_dt=" << src_dt 
              << ", target_dt=" << target_dt << "), loaded " << clipped_N << " frames";
    return traj;
  }

  // 需要插值
  double total_time = (clipped_N - 1) * src_dt;
  size_t target_N = static_cast<size_t>(std::round(total_time / target_dt)) + 1;
  Eigen::MatrixXd interp_traj(target_N, num_joints);
  for (size_t j = 0; j < num_joints; ++j) {
    std::vector<double> src_vals(clipped_N);
    for (size_t i = 0; i < clipped_N; ++i) {
      src_vals[i] = clipped_traj[i][j];
    }
    for (size_t t = 0; t < target_N; ++t) {
      double tgt_time = t * target_dt;
      double src_idx_f = tgt_time / src_dt;
      size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
      size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
      double alpha = src_idx_f - idx0;
      double v0 = src_vals[idx0];
      double v1 = src_vals[idx1];
      interp_traj(t, j) = (1 - alpha) * v0 + alpha * v1;
    }
  }
  LOG(INFO) << "LoadAndInterpolateJointTrajectory: Interpolated from " << clipped_N 
            << " frames (dt=" << src_dt << ") to " << target_N << " frames (dt=" << target_dt << ")";
  return interp_traj;
}

Eigen::MatrixXd CsvLoader::LoadAndInterpolateJointTrajectoryWithVel(const std::string& csv_path,
                                                                     const std::vector<std::string>& joint_names,
                                                                     double src_dt, double target_dt, int start_frame,
                                                                     int end_frame) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open csv file: " + csv_path);
  }

  std::string line;
  std::vector<std::string> headers;
  // 读取表头
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
      headers.push_back(col);
    }
  }

  // 建立关节名到列索引的映射（位置和速度）
  std::map<std::string, int> joint_pos_col_idx;
  std::map<std::string, int> joint_vel_col_idx;
  
  for (size_t i = 0; i < joint_names.size(); ++i) {
    // 位置列
    auto pos_it = std::find(headers.begin(), headers.end(), joint_names[i]);
    if (pos_it == headers.end()) {
      throw std::runtime_error("Joint position name not found in csv: " + joint_names[i]);
    }
    joint_pos_col_idx[joint_names[i]] = std::distance(headers.begin(), pos_it);
    
    // 速度列（添加"d"前缀）
    std::string vel_name = "d" + joint_names[i];
    auto vel_it = std::find(headers.begin(), headers.end(), vel_name);
    if (vel_it == headers.end()) {
      throw std::runtime_error("Joint velocity name not found in csv: " + vel_name);
    }
    joint_vel_col_idx[joint_names[i]] = std::distance(headers.begin(), vel_it);
  }

  // 读取所有帧
  std::vector<std::vector<double>> raw_pos_traj;
  std::vector<std::vector<double>> raw_vel_traj;
  
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string val;
    std::vector<double> pos_row;
    std::vector<double> vel_row;
    std::vector<std::string> all_vals;
    
    // 先读取所有值
    while (std::getline(ss, val, ',')) {
      all_vals.push_back(val);
    }
    
    // 按顺序提取位置和速度
    for (size_t i = 0; i < joint_names.size(); ++i) {
      int pos_idx = joint_pos_col_idx[joint_names[i]];
      int vel_idx = joint_vel_col_idx[joint_names[i]];
      
      if (pos_idx < (int)all_vals.size() && vel_idx < (int)all_vals.size()) {
        pos_row.push_back(std::stod(all_vals[pos_idx]));
        vel_row.push_back(std::stod(all_vals[vel_idx]));
      }
    }
    
    if ((int)pos_row.size() == (int)joint_names.size() && (int)vel_row.size() == (int)joint_names.size()) {
      raw_pos_traj.push_back(pos_row);
      raw_vel_traj.push_back(vel_row);
    }
  }
  
  size_t src_N = raw_pos_traj.size();
  size_t num_joints = joint_names.size();
  if (src_N == 0) {
    throw std::runtime_error("No data rows found in csv: " + csv_path);
  }

  // 截取traj_frame区间
  int start_idx = std::max(0, start_frame);
  int end_idx;
  if (end_frame == -1) {
    end_idx = (int)src_N - 1;
  } else {
    end_idx = std::min((int)src_N - 1, end_frame);
  }
  if (start_idx > end_idx) {
    throw std::runtime_error("Invalid traj_frame range: start > end");
  }
  
  std::vector<std::vector<double>> clipped_pos_traj(raw_pos_traj.begin() + start_idx, raw_pos_traj.begin() + end_idx + 1);
  std::vector<std::vector<double>> clipped_vel_traj(raw_vel_traj.begin() + start_idx, raw_vel_traj.begin() + end_idx + 1);

  size_t clipped_N = clipped_pos_traj.size();
  
  // 检查是否需要插值：如果源帧率和目标帧率相同（误差<1e-6），直接返回原始数据
  const double dt_epsilon = 1e-6;
  if (std::abs(src_dt - target_dt) < dt_epsilon) {
    // 不需要插值，直接转换为 Eigen 矩阵
    // 返回矩阵格式：[N, 2*num_joints]，每行为 [pos0, pos1, ..., posN, vel0, vel1, ..., velN]
    Eigen::MatrixXd traj(clipped_N, 2 * num_joints);
    for (size_t i = 0; i < clipped_N; ++i) {
      for (size_t j = 0; j < num_joints; ++j) {
        traj(i, j) = clipped_pos_traj[i][j];
        traj(i, num_joints + j) = clipped_vel_traj[i][j];
      }
    }
    LOG(INFO) << "LoadAndInterpolateJointTrajectoryWithVel: No interpolation needed (src_dt=" << src_dt 
              << ", target_dt=" << target_dt << "), loaded " << clipped_N << " frames";
    return traj;
  }

  // 需要插值
  double total_time = (clipped_N - 1) * src_dt;
  size_t target_N = static_cast<size_t>(std::round(total_time / target_dt)) + 1;
  
  // 返回矩阵格式：[N, 2*num_joints]，每行为 [pos0, pos1, ..., posN, vel0, vel1, ..., velN]
  Eigen::MatrixXd interp_traj(target_N, 2 * num_joints);
  
  // 插值位置
  for (size_t j = 0; j < num_joints; ++j) {
    std::vector<double> src_vals(clipped_N);
    for (size_t i = 0; i < clipped_N; ++i) {
      src_vals[i] = clipped_pos_traj[i][j];
    }
    for (size_t t = 0; t < target_N; ++t) {
      double tgt_time = t * target_dt;
      double src_idx_f = tgt_time / src_dt;
      size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
      size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
      double alpha = src_idx_f - idx0;
      double v0 = src_vals[idx0];
      double v1 = src_vals[idx1];
      interp_traj(t, j) = (1 - alpha) * v0 + alpha * v1;
    }
  }
  
  // 插值速度
  for (size_t j = 0; j < num_joints; ++j) {
    std::vector<double> src_vals(clipped_N);
    for (size_t i = 0; i < clipped_N; ++i) {
      src_vals[i] = clipped_vel_traj[i][j];
    }
    for (size_t t = 0; t < target_N; ++t) {
      double tgt_time = t * target_dt;
      double src_idx_f = tgt_time / src_dt;
      size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
      size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
      double alpha = src_idx_f - idx0;
      double v0 = src_vals[idx0];
      double v1 = src_vals[idx1];
      interp_traj(t, num_joints + j) = (1 - alpha) * v0 + alpha * v1;
    }
  }
  
  LOG(INFO) << "LoadAndInterpolateJointTrajectoryWithVel: Interpolated from " << clipped_N 
            << " frames (dt=" << src_dt << ") to " << target_N << " frames (dt=" << target_dt << ")";
  return interp_traj;
}

Eigen::MatrixXd CsvLoader::LoadAndInterpolateJointTrajectoryWithVelAndQuat(const std::string& csv_path,
                                                                            const std::vector<std::string>& joint_names,
                                                                            double src_dt, double target_dt,
                                                                            int start_frame, int end_frame) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open csv file: " + csv_path);
  }

  std::string line;
  std::vector<std::string> headers;
  // 读取表头
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
      headers.push_back(col);
    }
  }

  // 建立关节名到列索引的映射（位置和速度）
  std::map<std::string, int> joint_pos_col_idx;
  std::map<std::string, int> joint_vel_col_idx;
  
  for (size_t i = 0; i < joint_names.size(); ++i) {
    // 位置列
    auto pos_it = std::find(headers.begin(), headers.end(), joint_names[i]);
    if (pos_it == headers.end()) {
      throw std::runtime_error("Joint position name not found in csv: " + joint_names[i]);
    }
    joint_pos_col_idx[joint_names[i]] = std::distance(headers.begin(), pos_it);
    
    // 速度列（添加"d"前缀）
    std::string vel_name = "d" + joint_names[i];
    auto vel_it = std::find(headers.begin(), headers.end(), vel_name);
    if (vel_it == headers.end()) {
      throw std::runtime_error("Joint velocity name not found in csv: " + vel_name);
    }
    joint_vel_col_idx[joint_names[i]] = std::distance(headers.begin(), vel_it);
  }

  // 查找四元数列（优先查找 base_quat_x 等，如果没有则查找 base_qx 等）
  std::vector<std::string> quat_names = {"base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"};
  std::vector<std::string> quat_names_alt = {"base_qx", "base_qy", "base_qz", "base_qw"};
  std::vector<int> quat_col_idx;
  
  // 先检查第一组列名是否存在
  bool use_alt_names = false;
  auto first_quat_it = std::find(headers.begin(), headers.end(), quat_names[0]);
  if (first_quat_it == headers.end()) {
    // 第一组不存在，尝试第二组
    auto alt_quat_it = std::find(headers.begin(), headers.end(), quat_names_alt[0]);
    if (alt_quat_it == headers.end()) {
      throw std::runtime_error("Quaternion columns not found in csv. Expected either: base_quat_x/y/z/w or base_qx/y/z/w");
    }
    use_alt_names = true;
  }
  
  // 使用确定的列名组进行查找
  const auto& selected_quat_names = use_alt_names ? quat_names_alt : quat_names;
  for (const auto& quat_name : selected_quat_names) {
    auto quat_it = std::find(headers.begin(), headers.end(), quat_name);
    if (quat_it == headers.end()) {
      throw std::runtime_error("Quaternion column not found in csv: " + quat_name);
    }
    quat_col_idx.push_back(std::distance(headers.begin(), quat_it));
  }

  // 读取所有帧
  std::vector<std::vector<double>> raw_pos_traj;
  std::vector<std::vector<double>> raw_vel_traj;
  std::vector<std::vector<double>> raw_quat_traj;
  
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string val;
    std::vector<double> pos_row;
    std::vector<double> vel_row;
    std::vector<double> quat_row;
    std::vector<std::string> all_vals;
    
    // 先读取所有值
    while (std::getline(ss, val, ',')) {
      all_vals.push_back(val);
    }
    
    // 按顺序提取位置和速度
    for (size_t i = 0; i < joint_names.size(); ++i) {
      int pos_idx = joint_pos_col_idx[joint_names[i]];
      int vel_idx = joint_vel_col_idx[joint_names[i]];
      
      if (pos_idx < (int)all_vals.size() && vel_idx < (int)all_vals.size()) {
        pos_row.push_back(std::stod(all_vals[pos_idx]));
        vel_row.push_back(std::stod(all_vals[vel_idx]));
      }
    }
    
    // 提取四元数
    for (int quat_idx : quat_col_idx) {
      if (quat_idx < (int)all_vals.size()) {
        quat_row.push_back(std::stod(all_vals[quat_idx]));
      }
    }
    
    if ((int)pos_row.size() == (int)joint_names.size() && 
        (int)vel_row.size() == (int)joint_names.size() &&
        (int)quat_row.size() == 4) {
      raw_pos_traj.push_back(pos_row);
      raw_vel_traj.push_back(vel_row);
      raw_quat_traj.push_back(quat_row);
    }
  }
  
  size_t src_N = raw_pos_traj.size();
  size_t num_joints = joint_names.size();
  if (src_N == 0) {
    throw std::runtime_error("No data rows found in csv: " + csv_path);
  }

  // 截取traj_frame区间
  int start_idx = std::max(0, start_frame);
  int end_idx;
  if (end_frame == -1) {
    end_idx = (int)src_N - 1;
  } else {
    end_idx = std::min((int)src_N - 1, end_frame);
  }
  if (start_idx > end_idx) {
    throw std::runtime_error("Invalid traj_frame range: start > end");
  }
  
  std::vector<std::vector<double>> clipped_pos_traj(raw_pos_traj.begin() + start_idx, raw_pos_traj.begin() + end_idx + 1);
  std::vector<std::vector<double>> clipped_vel_traj(raw_vel_traj.begin() + start_idx, raw_vel_traj.begin() + end_idx + 1);
  std::vector<std::vector<double>> clipped_quat_traj(raw_quat_traj.begin() + start_idx, raw_quat_traj.begin() + end_idx + 1);

  size_t clipped_N = clipped_pos_traj.size();
  
  // 检查是否需要插值：如果源帧率和目标帧率相同（误差<1e-6），直接返回原始数据
  const double dt_epsilon = 1e-6;
  if (std::abs(src_dt - target_dt) < dt_epsilon) {
    // 不需要插值，直接转换为 Eigen 矩阵
    // 返回矩阵格式：[N, 52]，每行为 [pos0...pos23, vel0...vel23, qx, qy, qz, qw]
    Eigen::MatrixXd traj(clipped_N, 2 * num_joints + 4);
    for (size_t i = 0; i < clipped_N; ++i) {
      for (size_t j = 0; j < num_joints; ++j) {
        traj(i, j) = clipped_pos_traj[i][j];
        traj(i, num_joints + j) = clipped_vel_traj[i][j];
      }
      // 四元数 [qx, qy, qz, qw]
      traj(i, 2 * num_joints + 0) = clipped_quat_traj[i][0];
      traj(i, 2 * num_joints + 1) = clipped_quat_traj[i][1];
      traj(i, 2 * num_joints + 2) = clipped_quat_traj[i][2];
      traj(i, 2 * num_joints + 3) = clipped_quat_traj[i][3];
    }
    LOG(INFO) << "LoadAndInterpolateJointTrajectoryWithVelAndQuat: No interpolation needed (src_dt=" << src_dt 
              << ", target_dt=" << target_dt << "), loaded " << clipped_N << " frames with quaternions";
    return traj;
  }

  // 需要插值
  double total_time = (clipped_N - 1) * src_dt;
  size_t target_N = static_cast<size_t>(std::round(total_time / target_dt)) + 1;
  
  // 返回矩阵格式：[N, 52]，每行为 [pos0...pos23, vel0...vel23, qx, qy, qz, qw]
  Eigen::MatrixXd interp_traj(target_N, 2 * num_joints + 4);
  
  // 插值位置
  for (size_t j = 0; j < num_joints; ++j) {
    std::vector<double> src_vals(clipped_N);
    for (size_t i = 0; i < clipped_N; ++i) {
      src_vals[i] = clipped_pos_traj[i][j];
    }
    for (size_t t = 0; t < target_N; ++t) {
      double tgt_time = t * target_dt;
      double src_idx_f = tgt_time / src_dt;
      size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
      size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
      double alpha = src_idx_f - idx0;
      double v0 = src_vals[idx0];
      double v1 = src_vals[idx1];
      interp_traj(t, j) = (1 - alpha) * v0 + alpha * v1;
    }
  }
  
  // 插值速度
  for (size_t j = 0; j < num_joints; ++j) {
    std::vector<double> src_vals(clipped_N);
    for (size_t i = 0; i < clipped_N; ++i) {
      src_vals[i] = clipped_vel_traj[i][j];
    }
    for (size_t t = 0; t < target_N; ++t) {
      double tgt_time = t * target_dt;
      double src_idx_f = tgt_time / src_dt;
      size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
      size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
      double alpha = src_idx_f - idx0;
      double v0 = src_vals[idx0];
      double v1 = src_vals[idx1];
      interp_traj(t, num_joints + j) = (1 - alpha) * v0 + alpha * v1;
    }
  }
  
  // 插值四元数（使用 SLERP - 球面线性插值）
  for (size_t t = 0; t < target_N; ++t) {
    double tgt_time = t * target_dt;
    double src_idx_f = tgt_time / src_dt;
    size_t idx0 = static_cast<size_t>(std::floor(src_idx_f));
    size_t idx1 = std::min(idx0 + 1, clipped_N - 1);
    double alpha = src_idx_f - idx0;
    
    // 构造Eigen四元数 (x, y, z, w)
    Eigen::Quaterniond q0(clipped_quat_traj[idx0][3], clipped_quat_traj[idx0][0], 
                          clipped_quat_traj[idx0][1], clipped_quat_traj[idx0][2]);
    Eigen::Quaterniond q1(clipped_quat_traj[idx1][3], clipped_quat_traj[idx1][0], 
                          clipped_quat_traj[idx1][1], clipped_quat_traj[idx1][2]);
    
    // SLERP插值
    Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
    
    // 存储为 [qx, qy, qz, qw]
    interp_traj(t, 2 * num_joints + 0) = q_interp.x();
    interp_traj(t, 2 * num_joints + 1) = q_interp.y();
    interp_traj(t, 2 * num_joints + 2) = q_interp.z();
    interp_traj(t, 2 * num_joints + 3) = q_interp.w();
  }
  
  LOG(INFO) << "LoadAndInterpolateJointTrajectoryWithVelAndQuat: Interpolated from " << clipped_N 
            << " frames (dt=" << src_dt << ") to " << target_N << " frames (dt=" << target_dt 
            << ") with quaternion SLERP";
  return interp_traj;
}

// LoadProfileTrajectory 暂时注释掉，因为它需要 RlDanceParam（需要 basic_param）
// 如果需要使用，需要先解决 basic_param 依赖问题
/*
Eigen::MatrixXd CsvLoader::LoadProfileTrajectory(const data::RlDanceMotionStateProfile& profile,
                                                 const data::RlDanceParam& param, double runner_period) {
  // 拼接csv路径
  std::string csv_path = common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), profile.data_path);
  const auto& joint_names = param.active_joint_names;
  double src_dt = profile.csv_dt;    // 从profile中读取csv_dt
  double target_dt = runner_period;  // 使用runner_period替代control_dt

  // 用局部变量保存原始traj_frame区间
  int start_frame = 0, end_frame = -1;
  if (!profile.traj_frame.empty()) {
    start_frame = profile.traj_frame[0];
    end_frame = profile.traj_frame.size() > 1 ? profile.traj_frame[1] : start_frame;
  }
  
  // 对于mimic_future类型，加载位置+速度+四元数数据
  if (profile.observation_type == "mimic_future") {
    Eigen::MatrixXd interp_traj =
        LoadAndInterpolateJointTrajectoryWithVelAndQuat(csv_path, joint_names, src_dt, target_dt, start_frame, end_frame);
    return interp_traj;
  } else {
    // 其他类型只加载位置数据
    Eigen::MatrixXd interp_traj =
        LoadAndInterpolateJointTrajectory(csv_path, joint_names, src_dt, target_dt, start_frame, end_frame);
    return interp_traj;
  }
}
*/

}  // namespace rl_dance
