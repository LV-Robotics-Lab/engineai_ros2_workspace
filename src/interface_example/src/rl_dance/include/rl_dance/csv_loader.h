#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

// 前向声明，避免包含 rl_dance_param.h（它需要 basic_param）
namespace data {
  class RlDanceMotionStateProfile;
  class RlDanceParam;
}

namespace rl_dance {

class CsvLoader {
 public:
  // 读取csv并插值，返回[N, 24]的矩阵（仅位置）
  static Eigen::MatrixXd LoadAndInterpolateJointTrajectory(const std::string& csv_path,
                                                           const std::vector<std::string>& joint_names, double src_dt,
                                                           double target_dt, int start_frame, int end_frame);

  // 读取csv并插值，返回[N, 48]的矩阵（位置+速度）
  static Eigen::MatrixXd LoadAndInterpolateJointTrajectoryWithVel(const std::string& csv_path,
                                                                   const std::vector<std::string>& joint_names,
                                                                   double src_dt, double target_dt, int start_frame,
                                                                   int end_frame);

  // 读取csv并插值，返回[N, 52]的矩阵（位置+速度+四元数）
  // 格式：[pos0...pos23, vel0...vel23, qx, qy, qz, qw]
  static Eigen::MatrixXd LoadAndInterpolateJointTrajectoryWithVelAndQuat(const std::string& csv_path,
                                                                          const std::vector<std::string>& joint_names,
                                                                          double src_dt, double target_dt,
                                                                          int start_frame, int end_frame);

  // 从NPZ文件加载轨迹数据（位置+速度+四元数）
  // NPZ文件格式：
  //   - joint_pos: [N, 24] - 关节位置
  //   - joint_vel: [N, 24] - 关节速度  
  //   - body_quat_w: [N, num_bodies, 4] - 所有body的四元数(wxyz格式)，取第0个body作为base
  //   - fps: [1] - 原始数据的帧率
  // 返回格式：[N, 52] = [pos0...pos23, vel0...vel23, qx, qy, qz, qw]
  static Eigen::MatrixXd LoadAndInterpolateNpzTrajectory(const std::string& npz_path,
                                                          double target_dt, int start_frame, int end_frame);

  static Eigen::MatrixXd LoadProfileTrajectory(const data::RlDanceMotionStateProfile& profile,
                                               const data::RlDanceParam& param,
                                               double runner_period = 0.02);
};

}  // namespace rl_dance
