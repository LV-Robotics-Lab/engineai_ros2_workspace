#pragma once

#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
// glog 需要先包含 export.h 来定义 GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#include <glog/export.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

#include "basic_param/basic_param.h"
#include "parameter/parameter_loader.h"

namespace data {

class RlDanceObservationScaleConfig {
 public:
  double observation_scale_linear_vel = 1.0;
  double observation_scale_angular_vel = 1.0;
  double observation_scale_dof_pos = 1.0;
  double observation_scale_dof_vel = 1.0;
  double observation_scale_quat = 1.0;
  double observation_scale_quat_error = 1.0;
  double observation_scale_dof_pos_ref_diff = 1.0;
  double observation_scale_yaw_forward = 1.0;
  double observation_clip = 100.0;
};

class RlDanceObservationGoalTypeConfig {
 public:
  int goal_length = 0;
  int goal_history_steps = 1;     // goal的历史步数
  int num_include_obs_steps = 5;  // 本体感知的历史步数
  double goal_scale = 1.0;
};

class RlDanceObservationConfig {
 public:
  int num_single_observations = 78;                                                  // 本体感知维度
  std::map<std::string, RlDanceObservationGoalTypeConfig> observation_type;  // 各goal类型配置
  RlDanceObservationScaleConfig observation_scale;                           // 统一scale配置
};

class RlDanceMotionStateProfile {
 public:
  RlDanceMotionStateProfile() { Reset(); }
  ~RlDanceMotionStateProfile() = default;

  void Reset() {
    observation_type.clear();
    data_path.clear();
    policy_path.clear();
    traj_frame.clear();
    residual_control = false;
    cycle_time.clear();
    stride_max = 0.6;
    yaw_per_cycle_max = 0.35;
    k_angle = 0.6;
    step_phase = 0.0;
    phase_range.clear();
    csv_dt = 1.0 / 50.0;
    remote_control_key.clear();
    enable = false;
    auto_transition.clear();
    proprioception_obs_with_yaw = false;
    use_gravity_error = false;
    use_quat_error = false;
  }

  std::string observation_type;                 // "mimic" or "mimic_tj"
  bool proprioception_obs_with_yaw = false;     // Whether to include yaw vector in proprioception observation
  bool use_gravity_error = false;               // Whether to use gravity projection error (current - reference) instead of absolute gravity
  bool use_quat_error = false;                  // Whether to use quaternion error (current - reference) in proprioception observation
  std::string data_path;                        // CSV trajectory path
  std::string policy_path;                      // Policy model path
  std::vector<int> traj_frame;                  // Trajectory frame indices
  bool residual_control = false;                // Whether to use residual control
  std::vector<double> cycle_time;               // Motion cycle time range [min, max]
  double stride_max = 0.6;                      // Maximum stride length constraint
  double yaw_per_cycle_max = 0.35;              // Maximum yaw angle per cycle constraint
  double k_angle = 0.6;                         // Angle coefficient for intensity calculation
  double step_phase = 0.0;                      // Step phase for mimic_tj observation type
  std::vector<double> phase_range;              // Phase range for mimic_tj observation type [start, end]
  double csv_dt = 1.0 / 50.0;                  // CSV trajectory sampling rate
  std::vector<std::string> remote_control_key;  // 支持多按键切换
  bool enable = false;                          // 新增，支持yaml配置的enable字段，默认启用
  std::string auto_transition;  // 自动转换到下一个状态，空字符串表示无转换，false表示退出
};

class RlDanceParam : public BasicParam {
 public:
  RlDanceParam(std::string_view tag = "rl_dance");

  DEFINE_PARAM_SCOPE(scope_);

  // Motion states
  std::map<std::string, RlDanceMotionStateProfile> motion_states;
  // Observation configuration
  RlDanceObservationConfig observations;
  // 统一的observation scale向量
  Eigen::VectorXd observation_scale_vec;

  // Parameters
  int num_actions;
  float observation_clip = 100.0f;
  std::vector<std::string> active_joint_names;
  std::vector<Eigen::VectorXd> default_joint_q;
  std::vector<Eigen::VectorXd> joint_kp;
  std::vector<Eigen::VectorXd> joint_kd;
  std::vector<Eigen::VectorXd> action_scale;
  std::vector<Eigen::VectorXd> action_scale85;
  // 补充runner用到的参数
  std::optional<std::vector<Eigen::VectorXd>> qd_mask;
  float remote_command_sampling_frequency;
  float remote_command_cut_off_frequency;
  float imu_install_delta_bias;
  float linear_vel_delta_bias;
  bool enable_remote_command_lpf;
  Eigen::Vector3d command_scale;
  Eigen::Vector3d command_bias;
  float action_clip;
  float transition_time;

  // Torque limiting parameters
  bool torque_limit = false;
  std::vector<Eigen::VectorXd> max_torque_joint;
  // New: group torque budget for lower body (sum of |torque| over lower-body joints)
  double max_lower_body_torque = 500.0;

  // Stability adjustment parameters
  double ang_vel_threshold = 0.4;
  double gravity_dev_threshold = 0.15;
  double stability_smoothing_threshold = 0.6;
  int stability_history_length = 10;

  // Velocity zero threshold parameters
  double lin_vel_set_zero_threshold = 0.3;
  double ang_vel_set_zero_threshold = 0.314;

  // Gait phase adjustment parameters
  double unstable_cycle_time_scale = 0.5;

  // Default dynamic cycle time when configuration is invalid
  double default_dynamic_cycle_time = 0.8;
};

}  // namespace data

namespace YAML {
template <>
struct convert<data::RlDanceObservationScaleConfig> {
  static bool decode(const Node& node, data::RlDanceObservationScaleConfig& param) {
    if (node["observation_scale_linear_vel"])
      param.observation_scale_linear_vel = node["observation_scale_linear_vel"].as<double>();
    if (node["observation_scale_angular_vel"])
      param.observation_scale_angular_vel = node["observation_scale_angular_vel"].as<double>();
    if (node["observation_scale_dof_pos"])
      param.observation_scale_dof_pos = node["observation_scale_dof_pos"].as<double>();
    if (node["observation_scale_dof_vel"])
      param.observation_scale_dof_vel = node["observation_scale_dof_vel"].as<double>();
    if (node["observation_scale_quat"]) param.observation_scale_quat = node["observation_scale_quat"].as<double>();
    if (node["observation_scale_quat_error"]) 
      param.observation_scale_quat_error = node["observation_scale_quat_error"].as<double>();
    if (node["observation_scale_dof_pos_ref_diff"])
      param.observation_scale_dof_pos_ref_diff = node["observation_scale_dof_pos_ref_diff"].as<double>();
    if (node["observation_scale_yaw_forward"])
      param.observation_scale_yaw_forward = node["observation_scale_yaw_forward"].as<double>();
    if (node["observation_clip"]) param.observation_clip = node["observation_clip"].as<double>();
    return true;
  }
};

template <>
struct convert<data::RlDanceObservationGoalTypeConfig> {
  static bool decode(const Node& node, data::RlDanceObservationGoalTypeConfig& param) {
    if (node["goal_length"]) param.goal_length = node["goal_length"].as<int>();
    if (node["goal_history_steps"]) param.goal_history_steps = node["goal_history_steps"].as<int>();
    if (node["num_include_obs_steps"]) param.num_include_obs_steps = node["num_include_obs_steps"].as<int>();
    if (node["goal_scale"]) param.goal_scale = node["goal_scale"].as<double>();
    return true;
  }
};

template <>
struct convert<data::RlDanceObservationConfig> {
  static bool decode(const Node& node, data::RlDanceObservationConfig& param) {
    if (node["num_single_observations"]) param.num_single_observations = node["num_single_observations"].as<int>();
    if (node["observation_type"])
      param.observation_type =
          node["observation_type"].as<std::map<std::string, data::RlDanceObservationGoalTypeConfig>>();
    if (node["observation_scale"])
      param.observation_scale = node["observation_scale"].as<data::RlDanceObservationScaleConfig>();
    return true;
  }
};

template <>
struct convert<data::RlDanceMotionStateProfile> {
  static bool decode(const Node& node, data::RlDanceMotionStateProfile& param) {
    if (node["observation_type"]) param.observation_type = node["observation_type"].as<std::string>();
    if (node["data_path"]) param.data_path = node["data_path"].as<std::string>();
    if (node["policy_path"]) param.policy_path = node["policy_path"].as<std::string>();
    if (node["traj_frame"]) param.traj_frame = node["traj_frame"].as<std::vector<int>>();
    if (node["residual_control"]) param.residual_control = node["residual_control"].as<bool>();
    if (node["cycle_time"]) param.cycle_time = node["cycle_time"].as<std::vector<double>>();
    if (node["stride_max"]) param.stride_max = node["stride_max"].as<double>();
    if (node["yaw_per_cycle_max"]) param.yaw_per_cycle_max = node["yaw_per_cycle_max"].as<double>();
    if (node["k_angle"]) param.k_angle = node["k_angle"].as<double>();
    if (node["step_phase"]) param.step_phase = node["step_phase"].as<double>();
    if (node["phase_range"]) param.phase_range = node["phase_range"].as<std::vector<double>>();
    if (node["csv_dt"]) param.csv_dt = node["csv_dt"].as<double>();
    if (node["remote_control_key"])
      param.remote_control_key = node["remote_control_key"].as<std::vector<std::string>>();
    if (node["enable"]) param.enable = node["enable"].as<bool>();
    if (node["auto_transition"]) {
      if (node["auto_transition"].IsScalar()) {
        if (node["auto_transition"].as<std::string>() == "false") {
          param.auto_transition = "false";
        } else {
          param.auto_transition = node["auto_transition"].as<std::string>();
        }
      }
    }
    if (node["proprioception_obs_with_yaw"])
      param.proprioception_obs_with_yaw = node["proprioception_obs_with_yaw"].as<bool>();
    if (node["use_gravity_error"])
      param.use_gravity_error = node["use_gravity_error"].as<bool>();
    if (node["use_quat_error"])
      param.use_quat_error = node["use_quat_error"].as<bool>();
    return true;
  }
};
}  // namespace YAML