#include "rl_dance/rl_dance_runner.h"

#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <chrono>

#include "math/interpolation.h"
#include "math/roll_pitch_yaw.h"
#include "math/rotation_matrix.h"
#include "tool/concatenate_vector.h"

namespace runner {

/**
 * @brief 进入runner状态，初始化参数和状态。
 * @return true 初始化成功
 */
bool RlDanceRunner::Enter() {
  if (param_tag_ != last_param_tag_) {
    param_ = data::ParamManager::create<data::RlDanceParam>(param_tag_);
    last_param_tag_ = param_tag_;
  }
  // Set profile key before Init
  Init();

  run_reference_trajectory_ = true;
  task_completed_ = false;  // Reset task completion flag
  // Reset motor commands and initial joint states
  data_store_->joint_info.SetZeroCommand();
  data_store_->parallel_by_classic_parser.store(false);
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, initial_joint_q_);
  iter_ = 0;

  // Initialize torque logging (per profile)
  start_time_ = std::chrono::steady_clock::now();
  long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time_.time_since_epoch()).count();
  lower_body_torque_log_path_dance1_ = std::string("/tmp/rl_dance_lower_body_torque_dance_1_") + std::to_string(ns) + ".log";
  lower_body_torque_log_path_dance2_ = std::string("/tmp/rl_dance_lower_body_torque_dance_2_") + std::to_string(ns) + ".log";
  lower_body_torque_log_dance1_.open(lower_body_torque_log_path_dance1_);
  if (lower_body_torque_log_dance1_.is_open()) {
    lower_body_torque_log_dance1_ << "# rl_dance lower-body torque sum log (dance_1)" << '\n';
    lower_body_torque_log_dance1_ << "# columns: time_s sum_abs_lower" << '\n';
  }
  lower_body_torque_log_dance2_.open(lower_body_torque_log_path_dance2_);
  if (lower_body_torque_log_dance2_.is_open()) {
    lower_body_torque_log_dance2_ << "# rl_dance lower-body torque sum log (dance_2)" << '\n';
    lower_body_torque_log_dance2_ << "# columns: time_s sum_abs_lower" << '\n';
  }
  return true;
}

/**
 * @brief 主循环，每周期调用，依次更新状态、观测、动作等。
 *
 * 执行顺序：
 * 1. UpdateState() - 更新关节状态和稳定性检测
 * 2. UpdateRemoteCommand() - 处理手柄输入和状态切换
 * 3. CalculateObservation() - 计算神经网络观测数据
 * 4. CalculateMotorCommand() - 神经网络推理和电机命令计算
 * 5. SendMotorCommand() - 发送电机命令到硬件
 */
void RlDanceRunner::Run() {
  // If task is completed, do nothing and let runner exit
  if (task_completed_) {
    SetRunnerState(RunnerState::kTryExit);
    return;
  }

  UpdateState();
  UpdateRemoteCommand();
  CalculateObservation();
  CalculateMotorCommand();
  SendMotorCommand();
}

/**
 * @brief 判断是否可以退出runner状态。
 * @return TransitionState::kCompleted 表示可退出
 */
TransitionState RlDanceRunner::TryExit() { return TransitionState::kCompleted; }

/**
 * @brief 退出runner状态，清理相关状态和变量。
 * @return true 退出成功
 */
bool RlDanceRunner::Exit() {
  data_store_->parallel_by_classic_parser.store(true);
  // Clear observation, action, counter variables
  if (observation_manager_) observation_manager_->Reset();
  mlp_net_action_.setZero();
  reference_trajectory_iter_ = 0;
  iter_ = 0;
  is_first_time_ = true;
  if (lower_body_torque_log_dance1_.is_open()) {
    lower_body_torque_log_dance1_.flush();
    lower_body_torque_log_dance1_.close();
  }
  if (lower_body_torque_log_dance2_.is_open()) {
    lower_body_torque_log_dance2_.flush();
    lower_body_torque_log_dance2_.close();
  }
  return true;
}

/**
 * @brief 初始化runner的所有核心参数、观测管理器、轨迹等。
 *        只负责主流程调度，细节交由各自模块实现。
 */
void RlDanceRunner::Init() {
  // 1. Reset profile_key_ to default value on each initialization
  profile_key_ = "dance_1";

  // 2. Select current profile
  const auto& profile_map = param_->motion_states;
  if (profile_map.find(profile_key_) == profile_map.end() || !profile_map.at(profile_key_).enable) {
    std::cerr << "ERROR: Profile key " << profile_key_ << " not found or not enabled!" << std::endl;
    std::cerr << "Available enabled keys:";
    for (const auto& kv : profile_map) {
      if (kv.second.enable) std::cerr << kv.first << " ";
    }
    std::cerr << std::endl;
    throw std::runtime_error("Profile key not found or not enabled: " + profile_key_);
  }
  const auto& profile = profile_map.at(profile_key_);

  // 3. Initialize ObservationManager and set observation profile
  if (!observation_manager_)
    observation_manager_ = std::make_unique<rl_dance::ObservationManager>(*param_);
  observation_manager_->SetProfile(profile_key_, profile);
  observation_manager_->SetRunnerPeriod(runner_period_);  // Set runner period time
  std::cout << "ObservationManager initialized successfully" << std::endl;

  // 4. Pre-load all policy models
  policy_models_.clear();
  interpolated_trajs_.clear();
  for (const auto& kv : param_->motion_states) {
    const std::string& key = kv.first;
    const auto& prof = kv.second;
    if (policy_models_.find(key) == policy_models_.end()) {
      std::string policy_path =
          common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), prof.policy_path);
      std::ifstream policy_file(policy_path);
      if (!policy_file.good()) {
        LOG(ERROR) << "Policy file not found: " << policy_path;
        throw std::runtime_error(std::string("Policy file not found: ") + policy_path);
      }
      policy_file.close();
      policy_models_[key] = std::make_unique<math::MNNModel>(policy_path);
      std::cout << "Policy model loaded for key: " << key << std::endl;
    }
    // Pre-load interpolated trajectory for mimic-like types
    if (prof.observation_type == "mimic" || prof.observation_type == "control_mimic" || 
        prof.observation_type == "mimic_future") {
      interpolated_trajs_[key] =
          rl_dance::CsvLoader::LoadProfileTrajectory(prof, *param_, runner_period_);
      std::cout << "Interpolated trajectory loaded for key: " << key 
                << ", type: " << prof.observation_type << std::endl;
    }
    // mimic_tj doesn't need trajectory, uses phase to determine action end
  }
  // Set policy pointer for current profile
  mlp_net_boxing_ = policy_models_[profile_key_].get();
  // Set trajectory pointer for current profile (mimic type)
  if (interpolated_trajs_.count(profile_key_)) {
    current_traj_ = &interpolated_trajs_[profile_key_];
  } else {
    current_traj_ = nullptr;
  }
  observation_manager_->SetCurrentTrajectory(current_traj_);
  mlp_net_action_.setZero(param_->num_actions);
  saved_dance2_second_last_ = false;
  saved_torque_log_hint_ = false;
  last_logged_index_dance1_ = (size_t)(-1);
  last_logged_index_dance2_ = (size_t)(-1);

  // 5. Initialize joint parameters
  default_joint_q_ = common::ConcatenateVectors(param_->default_joint_q);
  // q_ref_ = default_joint_q_;
  q_des_ = default_joint_q_;
  q_actual_temp_ = default_joint_q_;
  joint_kp_ = common::ConcatenateVectors(param_->joint_kp);
  joint_kd_ = common::ConcatenateVectors(param_->joint_kd);
  // Select action scale by profile: dance_1 -> action_scale, dance_2 -> action_scale85
  if (profile_key_ == "dance_2") {
    action_scale_ = common::ConcatenateVectors(param_->action_scale85);
    LOG(INFO) << "Using action_scale85 for profile '" << profile_key_ << "'";
  } else {
    action_scale_ = common::ConcatenateVectors(param_->action_scale);
    LOG(INFO) << "Using action_scale for profile '" << profile_key_ << "'";
  }
  if (param_->qd_mask) {
    qd_mask_ = common::ConcatenateVectors(param_->qd_mask.value());
  }

  // 6. Initialize state
  is_first_time_ = true;
  last_remote_command_ = *data_store_->gamepad_info.Get();

  // 7. Initialize low-pass filter
  lpf_command_ = std::make_unique<math::FirstOrderLowPassFilter<Eigen::Vector3d>>(
      param_->remote_command_sampling_frequency, param_->remote_command_cut_off_frequency);
  lpf_command_->Reset();

  // 8. Load and interpolate joint trajectories, all handled by CsvLoader

  // 9. Ensure all vectors have correct size to avoid segmentation errors
  q_actual_.resize(default_joint_q_.size());
  qd_actual_.resize(default_joint_q_.size());
  qd_des_.resize(model_param_->num_total_joints);
  tau_ff_des_.resize(model_param_->num_total_joints);

  // 10. Initialize stability detection
  stability_history_.clear();
  stability_history_.resize(param_->stability_history_length, false);

  // 11. Initialize yaw calculation state
  initial_quat_set_ = false;
}

/**
 * @brief 重置参考轨迹和观测管理器，常用于切换profile或重新开始。
 */
void RlDanceRunner::ResetReferenceTrajectory() {
  const auto& profile_map = param_->motion_states;
  auto profile_it = profile_map.find(profile_key_);
  if (profile_it == profile_map.end() || !profile_it->second.enable) return;
  const auto& profile = profile_it->second;
  // Print profile switching message
  if (observation_manager_) {
    observation_manager_->SetProfile(profile_key_, profile);
    // Don't reset observation buffers - keep proprioception history for smooth switching
  }
  reference_trajectory_iter_ = 0;
  run_reference_trajectory_ = true;
  is_first_time_ = true;
  iter_ = 0;
  // Use current robot position as transition start point instead of initial_joint_q_
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, initial_joint_q_);
  default_joint_q_ = common::ConcatenateVectors(param_->default_joint_q);
  q_ref_ = default_joint_q_;
  q_des_ = default_joint_q_;
  q_actual_temp_ = default_joint_q_;
  joint_kp_ = common::ConcatenateVectors(param_->joint_kp);
  joint_kd_ = common::ConcatenateVectors(param_->joint_kd);
  // Select action scale by profile: dance_1 -> action_scale, dance_2 -> action_scale85
  if (profile_key_ == "dance_2") {
    action_scale_ = common::ConcatenateVectors(param_->action_scale85);
    LOG(INFO) << "Using action_scale85 for profile '" << profile_key_ << "' (on switch)";
  } else {
    action_scale_ = common::ConcatenateVectors(param_->action_scale);
    LOG(INFO) << "Using action_scale for profile '" << profile_key_ << "' (on switch)";
  }
  if (param_->qd_mask) {
    qd_mask_ = common::ConcatenateVectors(param_->qd_mask.value());
  }
  // Only switch policy pointer when switching profile
  if (policy_models_.count(profile_key_)) {
    mlp_net_boxing_ = policy_models_[profile_key_].get();
  } else {
    mlp_net_boxing_ = nullptr;
  }

  // Only switch trajectory pointer when switching profile
  if (interpolated_trajs_.count(profile_key_)) {
    current_traj_ = &interpolated_trajs_[profile_key_];
  } else {
    current_traj_ = nullptr;
  }
  observation_manager_->SetCurrentTrajectory(current_traj_);
  // Don't reset action to maintain continuity during policy switching
  // mlp_net_action_.setZero(param_->num_actions);

  // Reset yaw calculation state when switching profiles
  initial_quat_set_ = false;
  std::cout << "[state_switch] profile: " << profile_key_ << std::endl;
  // Reset second-last-step save flag when switching profiles
  saved_dance2_second_last_ = false;
}

/**
 * @brief 更新当前关节状态和速度，应用qd_mask（如有）。
 */
void RlDanceRunner::UpdateState() {
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_actual_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_actual_);
  if (param_->qd_mask) {
    qd_actual_ = qd_actual_.cwiseProduct(qd_mask_);
  }

  // Update stability detection
  UpdateStabilityHistory();
}

// ============================================================================
// KeyInputHandler Implementation
// ============================================================================

/**
 * @brief 检查指定按键是否被按下
 * @param now 当前手柄状态
 * @param key 按键名称
 * @return true 如果按键被按下
 *
 * 支持的按键：
 * - 按钮: A, B, X, Y, LB, RB, BACK, START
 * - 方向键: CROSS_X_UP, CROSS_X_DOWN, CROSS_Y_LEFT, CROSS_Y_RIGHT
 */
bool RlDanceRunner::KeyInputHandler::IsKeyPressed(const data::GamepadInfo& now,
                                                                 const std::string& key) const {
  if (key.empty()) {
    return false;
  }

  // Handle button keys - clear and readable
  if (key == "A") return now.A;
  if (key == "B") return now.B;
  if (key == "X") return now.X;
  if (key == "Y") return now.Y;
  if (key == "LB") return now.LB;
  if (key == "RB") return now.RB;
  if (key == "BACK") return now.BACK;
  if (key == "START") return now.START;

  // Handle directional pad keys - clear and readable
  if (key == "CROSS_X_UP") return now.CROSS_X == 1;
  if (key == "CROSS_X_DOWN") return now.CROSS_X == -1;
  if (key == "CROSS_Y_LEFT") return now.CROSS_Y == -1;
  if (key == "CROSS_Y_RIGHT") return now.CROSS_Y == 1;

  LOG(WARNING) << "Unknown key: " << key;
  return false;
}

bool RlDanceRunner::KeyInputHandler::IsKeyJustPressed(const data::GamepadInfo& now,
                                                                     const std::string& key,
                                                                     const data::GamepadInfo& last) const {
  return IsKeyPressed(now, key) && !IsKeyPressed(last, key);
}

bool RlDanceRunner::KeyInputHandler::IsKeyDoubleClicked(const data::GamepadInfo& now,
                                                                       const std::string& key,
                                                                       const data::GamepadInfo& last,
                                                                       double current_time) {
  // Initialize tracker for this key if not exists
  if (double_click_tracker_.find(key) == double_click_tracker_.end()) {
    double_click_tracker_[key] = DoubleClickInfo();
  }

  auto& info = double_click_tracker_[key];
  bool was_just_pressed = IsKeyJustPressed(now, key, last);

  if (was_just_pressed) {
    // Key was just pressed
    if (info.click_count == 0) {
      // First click
      info.click_count = 1;
      info.last_press_time = current_time;
    } else if (info.click_count == 1) {
      // Second click - check if within timeout
      if (current_time - info.last_press_time <= double_click_timeout_) {
        // Double click detected - reset for next detection
        info.click_count = 0;
        return true;
      } else {
        // Timeout, start new sequence
        info.click_count = 1;
        info.last_press_time = current_time;
      }
    }
  } else {
    // Key not just pressed - check for timeout
    if (info.click_count > 0 && current_time - info.last_press_time > double_click_timeout_) {
      info.click_count = 0;  // Reset on timeout
    }
  }

  return false;
}

bool RlDanceRunner::KeyInputHandler::IsKeyCombinationTriggered(const std::vector<std::string>& keys,
                                                                              const data::GamepadInfo& now,
                                                                              const data::GamepadInfo& last,
                                                                              double current_time) {
  // Require at least 2 keys for any combination
  if (keys.size() < 2) {
    LOG(WARNING) << "Key combination must have at least 2 keys, got " << keys.size();
    return false;
  }

  // Count occurrences of each key in the combination
  std::map<std::string, int> key_counts;
  for (const auto& key : keys) {
    if (key.empty()) {
      LOG(WARNING) << "Empty key found in combination";
      return false;
    }
    key_counts[key]++;
  }

  // Check if all required keys are triggered
  for (const auto& [key, required_count] : key_counts) {
    bool key_triggered = false;

    // For duplicate keys like [A, A], check if it's a double-click pattern
    if (required_count == 2) {
      // Check if this is a double-click pattern (same key appears twice)
      key_triggered = IsKeyDoubleClicked(now, key, last, current_time);
    } else if (required_count == 1) {
      // Regular key press - check if key is pressed
      key_triggered = IsKeyPressed(now, key);
    } else {
      // More than 2 occurrences - treat as regular key press
      LOG(WARNING) << "Key '" << key << "' appears " << required_count << " times, treating as regular key press";
      key_triggered = IsKeyPressed(now, key);
    }

    if (!key_triggered) {
      return false;  // At least one key in combination is not triggered
    }
  }

  return true;  // All keys in combination are triggered
}

void RlDanceRunner::KeyInputHandler::ResetDoubleClickTracker(const std::string& key) {
  if (key.empty()) {
    // Reset all trackers
    double_click_tracker_.clear();
  } else {
    // Reset specific key tracker
    double_click_tracker_.erase(key);
  }
}

void RlDanceRunner::UpdateRemoteCommand() {
  // 获取当前profile
  const auto& profile = param_->motion_states.at(profile_key_);
  // 1. Update command_ for modes that consume joystick command in observation
  if (profile.observation_type == "rl_locomotion" || profile.observation_type == "rl_saw_locomotion" ||
      profile.observation_type == "control_mimic") {
    double raw_x = data_store_->gamepad_info.Get()->LeftStick_X * param_->command_scale.x();
    double raw_y = data_store_->gamepad_info.Get()->LeftStick_Y * param_->command_scale.y();
    double theta = 0;  // rad, head angle + waist angle
    double rotated_x = std::cos(theta) * raw_x - std::sin(theta) * raw_y;
    double rotated_y = std::sin(theta) * raw_x + std::cos(theta) * raw_y;
    command_.x() = rotated_x;
    command_.y() = rotated_y;
    command_.z() = data_store_->gamepad_info.Get()->RightStick_Y * param_->command_scale.z();
    if (param_->enable_remote_command_lpf) {
      command_ = lpf_command_->Update(command_);
    }
    command_ += command_bias_ + param_->command_bias;

    // Stand still for small command abs - only execute in stable state
    bool current_stand_still = (std::abs(command_.x()) < param_->lin_vel_set_zero_threshold &&
                                std::abs(command_.y()) < param_->lin_vel_set_zero_threshold &&
                                std::abs(command_.z()) < param_->ang_vel_set_zero_threshold);
    if (current_stand_still) {
      command_.x() = 0.0;
      command_.y() = 0.0;
      command_.z() = 0.0;
    }
  }
  // 2. Check for motion state switching via key combinations
  const auto& gamepad = *data_store_->gamepad_info.Get();
  // Use a monotonic clock for reliable timing (independent of iter_ and transition phase)
  static const auto kStartTime = std::chrono::steady_clock::now();
  double current_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - kStartTime).count();

  for (const auto& [state_key, state_profile] : param_->motion_states) {
    if (!state_profile.enable || state_profile.remote_control_key.empty()) {
      continue;
    }

    // Check if all keys in the combination are triggered
    if (key_handler_.IsKeyCombinationTriggered(state_profile.remote_control_key, gamepad, last_remote_command_,
                                               current_time)) {
      if (profile_key_ != state_key) {
        LOG(INFO) << "Switching from " << profile_key_ << " to " << state_key;
        profile_key_ = state_key;
        ResetReferenceTrajectory();
      }
      break;  // Found and switched to a state, no need to check others
    }
  }
update_last:
  last_remote_command_ = gamepad;
}
void RlDanceRunner::CalculateObservation() {
  // Calculate base state - keep it simple and readable
  Eigen::Matrix3d R_install = math::RollPitchYawd(imu_install_bias_).ToRotationMatrix().matrix();
  Eigen::Matrix3d R_local = math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
  Eigen::Matrix3d R_real = R_local * R_install.transpose();
  Eigen::Vector3d w_real = R_real.transpose() * R_local * data_store_->imu_info.Get()->angular_velocity;
  Eigen::Vector3d projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();
  
  // Get current base quaternion for quat_error calculation
  Eigen::Quaterniond current_quat = Eigen::Quaterniond(R_real);

  // Calculate yaw forward vector if needed for proprioception observation
  const auto& profile = param_->motion_states.at(profile_key_);
  std::optional<Eigen::Vector2d> yaw_forward_vec = std::nullopt;
  if (profile.proprioception_obs_with_yaw) {
    yaw_forward_vec = CalculateBaseYawForwardVector();
  }

  // Ensure observation_manager_ is properly initialized before calling UpdateObservation
  if (!observation_manager_) {
    LOG(ERROR) << "ObservationManager is null in CalculateObservation";
    return;
  }

  // Extract reference quaternion from trajectory if use_gravity_error or use_quat_error is enabled
  std::optional<Eigen::Quaterniond> ref_quat = std::nullopt;
  if ((profile.use_gravity_error || profile.use_quat_error) && observation_manager_->current_traj_ != nullptr) {
    const auto& traj = *observation_manager_->current_traj_;
    const int num_joints = static_cast<int>(param_->active_joint_names.size());
    
    // Trajectory format: [pos0...pos23, vel0...vel23, qx, qy, qz, qw]
    // Check if trajectory has quaternion data (52 columns for mimic_future type)
    if (traj.cols() == 2 * num_joints + 4) {
      size_t traj_idx = observation_manager_->GetCurrentTrajectoryIndex();
      if (traj_idx < static_cast<size_t>(traj.rows())) {
        double qx = traj(traj_idx, 2 * num_joints + 0);
        double qy = traj(traj_idx, 2 * num_joints + 1);
        double qz = traj(traj_idx, 2 * num_joints + 2);
        double qw = traj(traj_idx, 2 * num_joints + 3);
        ref_quat = Eigen::Quaterniond(qw, qx, qy, qz);
        
        VLOG(2) << "Extracted reference quaternion at frame " << traj_idx 
                << ": [" << qw << ", " << qx << ", " << qy << ", " << qz << "]";
      } else {
        task_completed_ = true;
      }
    } else {
      LOG_EVERY_N(WARNING, 100) << "use_gravity_error or use_quat_error is enabled but trajectory doesn't have quaternion data "
                                << "(expected " << (2 * num_joints + 4) << " columns, got " << traj.cols() << ")";
    }
  }

  observation_manager_->UpdateObservation(q_actual_, qd_actual_, mlp_net_action_, w_real, projected_gravity_real,
                                          command_, default_joint_q_, is_stable_, yaw_forward_vec, ref_quat, current_quat);
}

void RlDanceRunner::CalculateMotorCommand() {
  // Get current observation
  Eigen::MatrixXd obs_matrix = observation_manager_->GetCurrentObservation();

  // Properly handle observation buffer - flatten entire matrix to 1D vector
  Eigen::VectorXd obs = Eigen::Map<const Eigen::VectorXd>(obs_matrix.data(), obs_matrix.size());

  // Inference
  mlp_net_action_ = mlp_net_boxing_->Inference(obs.cast<float>()).cast<double>();
  mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

  data_monitor_.Add("joint/action", mlp_net_action_);

  // Trajectory stepping and reference trajectory
  const auto& profile = param_->motion_states.at(profile_key_);
  observation_manager_->StepTrajectory(false);  // Always step forward, no replay

  // At dance_2 second last frame: save and print the log file path once
  if (!saved_torque_log_hint_ && current_traj_ != nullptr) {
    size_t rows = static_cast<size_t>(current_traj_->rows());
    size_t idx_now = observation_manager_->GetCurrentTrajectoryIndex();
    if (rows >= 2 && idx_now == rows - 2) {
        std::string log_path_hint;
        if (lower_body_torque_log_dance1_.is_open()) {
            lower_body_torque_log_dance1_.flush();
            lower_body_torque_log_dance1_.close();
            log_path_hint = lower_body_torque_log_path_dance1_;
        }
        if (lower_body_torque_log_dance2_.is_open()) {
            lower_body_torque_log_dance2_.flush();
            lower_body_torque_log_dance2_.close();
            if (!log_path_hint.empty()) log_path_hint += ", ";
            log_path_hint += lower_body_torque_log_path_dance2_;
        }
        LOG(INFO) << "Lower-body torque log saved to: " << log_path_hint;
        std::cout << "[rl_dance] lower-body torque log saved: " << log_path_hint << std::endl;
        saved_torque_log_hint_ = true;
    }
}

  // New logic: q_ref_ managed by ObservationManager
  // q_ref_ = observation_manager_->GetCurrentReference(); // You need to implement this interface in
  // observation_manager.h/cc

  // Only add residual action when residual_control is true
  if (profile.residual_control) {
    // q_des_ = q_ref_ + mlp_net_action_.cwiseProduct(action_scale_);
    // Assume q_ref_ is maintained internally by ObservationManager
    q_des_ = observation_manager_->GetCurrentReference();
    q_des_ += mlp_net_action_.cwiseProduct(action_scale_);
  } else {
    q_des_ = default_joint_q_ + mlp_net_action_.cwiseProduct(action_scale_);
  }

  // Interpolates joint position - only during initial startup, not during policy switching
  double ratio = param_->transition_time
                     ? std::min(1.0, static_cast<double>(iter_) * runner_period_ / param_->transition_time)
                     : 1.0;
  if (ratio < 1.0) {
    q_des_ = math::LinearInterpolate(initial_joint_q_, q_des_, ratio);
    ++iter_;
  }
  // Automatically switch mimic-like types to next policy when trajectory finishes
  // Note: trajectory stepping already handles staying at last frame (see StepTrajectory with replay=false)
  if ((profile.observation_type == "mimic" || profile.observation_type == "control_mimic" ||
       profile.observation_type == "mimic_future" || profile.observation_type == "mimic_tj") &&
      IsTrajectoryFinished()) {
    // Always delegate to TransitionToNextPolicy, which handles three cases:
    // 1) auto_transition == "false" -> mark task_completed_ = true (exit runner)
    // 2) auto_transition is a valid state -> switch profile
    // 3) auto_transition empty -> stay at last frame
    TransitionToNextPolicy();
  }
}

void RlDanceRunner::SendMotorCommand() {
  qd_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);
  tau_ff_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);

  // Apply torque limits to prevent excessive joint torques
  if (param_->torque_limit) {
    ApplyTorqueLimits(q_des_, q_actual_, qd_actual_, joint_kp_, joint_kd_);
  }

  // Logging: compute and save current lower-body absolute torque sum (based on finalized q_des_)
  if (current_traj_ != nullptr) {
    size_t rows = static_cast<size_t>(current_traj_->rows());
    size_t idx_now = observation_manager_->GetCurrentTrajectoryIndex();
    bool is_dance2 = (profile_key_ == "dance_2");
    size_t& last_idx = is_dance2 ? last_logged_index_dance2_ : last_logged_index_dance1_;

    if (idx_now < rows && idx_now != last_idx) {
      const int n = q_des_.size();
      const int lower_body_end_index = std::min(12, n);
      if (lower_body_end_index > 0) {
        // tau = Kp*(q_des - q_actual) - Kd*qd_actual
        Eigen::ArrayXd Kp = joint_kp_.array();
        Eigen::ArrayXd Kd = joint_kd_.array();
        Eigen::VectorXd tau_now = (Kp * (q_des_ - q_actual_).array() - Kd * qd_actual_.array()).matrix();
        double sum_abs_lower_now = tau_now.segment(0, lower_body_end_index).cwiseAbs().sum();
        double t_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();

        std::ofstream* log_ptr = nullptr;
        if (is_dance2) {
          if (lower_body_torque_log_dance2_.is_open()) log_ptr = &lower_body_torque_log_dance2_;
        } else {
          if (lower_body_torque_log_dance1_.is_open()) log_ptr = &lower_body_torque_log_dance1_;
        }
        if (log_ptr) {
          (*log_ptr) << t_sec << " " << sum_abs_lower_now << "\n";
          log_ptr->flush();
        }

        // Publish to DataMonitor
        Eigen::VectorXd lb_sum(1);
        lb_sum(0) = sum_abs_lower_now;
        data_monitor_.Add("metrics/lower_body_torque_sum", lb_sum);

        last_idx = idx_now;  // mark this frame as logged
      }
    }
  }

  data_store_->joint_info.SetCommand(data::JointInfoType::kPosition, q_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kVelocity, qd_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kFeedForwardTorque, tau_ff_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kStiffness, joint_kp_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kDamping, joint_kd_);
}

void RlDanceRunner::ApplyTorqueLimits(Eigen::VectorXd& q_des, const Eigen::VectorXd& q_actual,
                                                     const Eigen::VectorXd& qd_actual, const Eigen::VectorXd& joint_kp,
                                                     const Eigen::VectorXd& joint_kd) {
  const int n = q_des.size();
  auto check_same = [&](const Eigen::VectorXd& v, const char* name) {
    if (v.size() != n) {
      throw std::runtime_error(std::string("Size mismatch: ") + name);
    }
  };
  check_same(q_actual, "q_actual");
  check_same(qd_actual, "qd_actual");
  check_same(joint_kp, "joint_kp");
  check_same(joint_kd, "joint_kd");

  // ---- 构造每关节扭矩上限向量 tau_max ----
  Eigen::VectorXd tau_max = Eigen::VectorXd::Zero(n);
  {
    int idx = 0;
    for (const auto& group : param_->max_torque_joint) {
      const int gsz = static_cast<int>(group.size());
      const int remain = n - idx;
      if (gsz > remain) {
        throw std::runtime_error("max_torque_joint total size exceeds DOF count");
      }
      tau_max.segment(idx, gsz) = group;
      idx += gsz;
    }
    if (idx != n) {
      throw std::runtime_error("max_torque_joint total size does not match DOF count");
    }
    for (int i = 0; i < n; ++i) {
      if (tau_max(i) <= 0.0) tau_max(i) = std::numeric_limits<double>::infinity();
    }
  }

  // ---- 1) PD 期望扭矩 tau_des = Kp*(q_des - q_actual) - Kd*qd ----
  Eigen::VectorXd tau_des(n);
  {
    Eigen::ArrayXd Kp = joint_kp.array();
    Eigen::ArrayXd Kd = joint_kd.array();
    tau_des = (Kp * (q_des - q_actual).array() - Kd * qd_actual.array()).matrix();
  }

  // ---- 2) 逐关节硬限幅 ----
  Eigen::VectorXd tau = tau_des;
  for (int i = 0; i < n; ++i) {
    const double m = tau_max(i);
    if (std::isfinite(m)) {
      if (tau(i) > m) tau(i) = m;
      if (tau(i) < -m) tau(i) = -m;
    }
  }

  // ---- 3) 下肢总扭矩限制（示例：下标 0..11）----
  const int lower_body_end_index = 12;  // [0, 12) 为下肢
  if (n >= lower_body_end_index && param_->max_lower_body_torque > 0.0) {
    const double sum_abs_lower = tau.segment(0, lower_body_end_index).cwiseAbs().sum();
    const double limit = param_->max_lower_body_torque;
    if (sum_abs_lower > limit) {
      const double scale = limit / sum_abs_lower;
      tau.segment(0, lower_body_end_index) *= scale;
    }
  }

  // (Logging moved to SendMotorCommand after q_des is finalized)

  // ---- 4) 回推 q_des'：q_des = q_actual + (tau + Kd*qd)/Kp ----
  const double eps = 1e-6;
  for (int i = 0; i < n; ++i) {
    const double kp = joint_kp(i);
    const double kd = joint_kd(i);
    if (std::abs(kp) > eps && std::isfinite(kp)) {
      q_des(i) = q_actual(i) + (tau(i) + kd * qd_actual(i)) / kp;
    } else {
      // keep q_des(i) unchanged when Kp is too small
    }
  }
}

/**
 * @brief 检查当前机器人是否处于稳定状态
 * @return true 如果机器人稳定，false 如果不稳定
 */
bool RlDanceRunner::CheckStability() {
  // 1) 角速度条件 - 仅关心 pitch/roll (x,y分量)
  Eigen::Vector2d ang_vel_xy = base_angular_velocity_.head<2>();
  bool ang_vel_stable = ang_vel_xy.norm() < param_->ang_vel_threshold;

  // 2) 重力方向条件 - 计算与理想重力方向的偏差
  Eigen::Vector3d g_hat = projected_gravity_.normalized();
  Eigen::Vector3d g0(0.0, 0.0, -1.0);  // 理想重力方向
  double grav_dev = (g_hat - g0).norm();
  bool gravity_stable = grav_dev < param_->gravity_dev_threshold;

  // 当前时刻的稳定性
  bool current_stable = ang_vel_stable && gravity_stable;

  // 3) 历史平滑 - 多数投票
  stability_history_.push_back(current_stable);
  stability_history_.pop_front();

  int stable_count = 0;
  for (bool stable : stability_history_) {
    if (stable) stable_count++;
  }

  double stable_ratio = static_cast<double>(stable_count) / stability_history_.size();
  is_stable_ = stable_ratio > param_->stability_smoothing_threshold;

  return is_stable_;
}

/**
 * @brief 更新稳定性历史记录和当前状态
 */
void RlDanceRunner::UpdateStabilityHistory() {
  // Calculate current base angular velocity and projected gravity - keep it simple
  Eigen::Matrix3d R_install = math::RollPitchYawd(imu_install_bias_).ToRotationMatrix().matrix();
  Eigen::Matrix3d R_local = math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
  Eigen::Matrix3d R_real = R_local * R_install.transpose();

  // Update base angular velocity
  base_angular_velocity_ = R_real.transpose() * R_local * data_store_->imu_info.Get()->angular_velocity;

  // Update projected gravity
  projected_gravity_ = -R_real.transpose() * Eigen::Vector3d::UnitZ();

  // Check stability
  CheckStability();
}

bool RlDanceRunner::IsTrajectoryFinished() const {
  // For mimic type, use trajectory length to determine
  if (profile_key_ != "" && param_->motion_states.count(profile_key_)) {
    const auto& profile = param_->motion_states.at(profile_key_);
    if (profile.observation_type == "mimic") {
      return current_traj_ && observation_manager_->GetCurrentTrajectoryIndex() >= current_traj_->rows() - 1;
    } else if (profile.observation_type == "control_mimic") {
      return current_traj_ && observation_manager_->GetCurrentTrajectoryIndex() >= current_traj_->rows() - 1;
    } else if (profile.observation_type == "mimic_future") {
      return current_traj_ && observation_manager_->GetCurrentTrajectoryIndex() >= current_traj_->rows() - 1;
    } else if (profile.observation_type == "mimic_tj") {
      // For mimic_tj, use phase range to determine action end
      double step_phase = profile.step_phase;
      if (step_phase == 0.0) {
        LOG(ERROR) << "step_phase not configured for mimic_tj";
        return false;
      }

      if (profile.phase_range.empty() || profile.phase_range.size() < 2) {
        LOG(ERROR) << "phase_range not configured for mimic_tj";
        return false;
      }

      // For mimic_tj, use phase counter with consistent calculation
      double current_step = static_cast<double>(observation_manager_->GetMimicTjPhaseCounter());
      double phase_start = profile.phase_range[0];
      double phase_end = profile.phase_range[1];

      // Calculate current phase using the same logic as in CreateGoalObservation
      double current_phase = phase_start + (phase_end - phase_start) *
                                               (step_phase * (2 * 3.14159) / (phase_end - phase_start) * current_step);

      // Action ends when phase reaches or exceeds phase_range end value
      bool action_ended = current_phase >= phase_end;

      return action_ended;
    }
  }

  // Default use trajectory length to determine
  return current_traj_ && observation_manager_->GetCurrentTrajectoryIndex() >= current_traj_->rows() - 1;
}

void RlDanceRunner::TransitionToNextPolicy() {
  // Get current profile to check auto_transition
  const auto& profile = param_->motion_states.at(profile_key_);

  if (!profile.auto_transition.empty()) {
    // Check if auto_transition is "false" (exit runner)
    if (profile.auto_transition == "false") {
      LOG(INFO) << "Auto transition set to false, completing task";
      task_completed_ = true;
      return;
    }

    // Check if the target state exists and is enabled
    if (param_->motion_states.count(profile.auto_transition) &&
        param_->motion_states.at(profile.auto_transition).enable) {
      LOG(INFO) << "Auto transitioning from " << profile_key_ << " to " << profile.auto_transition;
      profile_key_ = profile.auto_transition;
      ResetReferenceTrajectory();
    } else {
      LOG(WARNING) << "Auto transition target " << profile.auto_transition << " not found or not enabled";
      LOG(INFO) << "Trajectory finished, staying at last frame";
      // Don't set task_completed_ - just stay at last frame
    }
  } else {
    // No auto_transition configured, stay at last frame instead of exiting
    LOG(INFO) << "Trajectory finished, no auto_transition configured, staying at last frame";
    // Don't set task_completed_ - keep running at last frame
  }
}

Eigen::Vector2d RlDanceRunner::CalculateBaseYawForwardVector() {
  // Get current base quaternion from IMU (w,x,y,z format)
  Eigen::Quaterniond current_quat = data_store_->imu_info.Get()->quaternion;
  Eigen::Vector4d current_base_quat(current_quat.w(), current_quat.x(), current_quat.y(), current_quat.z());

  // Initialize stored initial quaternion at episode start
  if (!initial_quat_set_ || is_first_time_) {
    initial_base_quat_ = current_base_quat;
    initial_quat_set_ = true;
    is_first_time_ = false;
  }

  // Relative quaternion: q_rel = conj(init) * current
  // Convert to Eigen quaternion for easier calculation
  Eigen::Quaterniond init_quat(initial_base_quat_[0], initial_base_quat_[1], initial_base_quat_[2],
                               initial_base_quat_[3]);

  // q_rel = conj(init) * current
  Eigen::Quaterniond q_rel = init_quat.conjugate() * current_quat;

  // Extract yaw (heading) from relative quaternion
  // yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
  double w = q_rel.w();
  double x = q_rel.x();
  double y = q_rel.y();
  double z = q_rel.z();

  double num = 2.0 * (w * z + x * y);
  double den = 1.0 - 2.0 * (y * y + z * z);
  double yaw_rad = std::atan2(num, den);

  // Convert to 2D forward vector [cos(yaw), sin(yaw)]
  Eigen::Vector2d forward_vec;
  forward_vec[0] = std::cos(yaw_rad);
  forward_vec[1] = std::sin(yaw_rad);

  return forward_vec;
}

}  // namespace runner