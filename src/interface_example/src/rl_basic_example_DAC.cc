#include <chrono>
#include <deque>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <fstream>
#include <map>
#include <sstream>
#include <limits>
#include <cmath>

#include "components/message_handler.hpp"
#include "math/concatenate_vector.h"
#include "math/mnn_model.h"
#include "math/rotation_matrix.h"
#include <yaml-cpp/yaml.h>
#include "rl_dance/csv_loader.h"
#include "rl_dance/rl_dance_motion_state_profile.h"
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

namespace example {

// DAC：基座策略仅为舞蹈（motion_states 中的 mimic policy + 单条轨迹），无 XZL 行走、无摔倒方向 mimic。
// 检测到摔倒时直接进入 damping（被动阻尼），不切换任何“摔倒策略”网络。
// 与 rl_basic_example_XZL 一致：MuJoCo reset 时 time_=0；falling_detect_after_sec 内不做摔倒/失稳->damping，避免起步瞬态误触。

class RlBasicRunnerDAC : public rclcpp::Node {
 public:
  explicit RlBasicRunnerDAC(const std::string& config_file_dir, const std::string& config_file_name = "rl_basic_param_DAC.yaml") : Node("rl_basic_runner_DAC") {
    std::string config_file = config_file_dir + "/" + config_file_name;
    RCLCPP_INFO(get_logger(), "Loading config file: %s", config_file.c_str());
    config_ = YAML::LoadFile(config_file);
    config_file_dir_ = config_file_dir;
    config_file_name_ = config_file_name;
    joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();
  }
  
  Eigen::VectorXd LoadVectorFromYaml(const YAML::Node& node) {
    std::vector<double> vec;
    for (const auto& item : node) {
      vec.push_back(item.as<double>());
    }
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
  }
  
  Eigen::VectorXi LoadIntVectorFromYaml(const YAML::Node& node) {
    std::vector<int> vec;
    for (const auto& item : node) {
      vec.push_back(item.as<int>());
    }
    return Eigen::Map<Eigen::VectorXi>(vec.data(), vec.size());
  }
  
  std::vector<Eigen::VectorXd> LoadVectorArrayFromYaml(const YAML::Node& node) {
    std::vector<Eigen::VectorXd> result;
    for (const auto& item : node) {
      result.push_back(LoadVectorFromYaml(item));
    }
    return result;
  }

  bool Initialize() {
    try {
      // Initialize message handler
      message_handler_ = std::make_shared<MessageHandler>(shared_from_this());
      message_handler_->Initialize();
      while (!message_handler_->GetLatestMotionState() ||
             message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      auto initial_state = message_handler_->GetLatestJointState();
      if (!initial_state) return false;

      initial_joint_q_ = Eigen::Map<const Eigen::VectorXd>(initial_state->position.data(), initial_state->position.size());
      InitializeDanceParam();
      LoadParametersFromYaml();

      if (config_["active_joint_idx"]) {
        active_joint_idx_ = LoadIntVectorFromYaml(config_["active_joint_idx"]);
      } else {
        active_joint_idx_ = Eigen::VectorXi::LinSpaced(active_joint_names_.size(), 0, active_joint_names_.size() - 1);
      }
      
      default_joint_q_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["default_joint_q"]));
      joint_kp_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["joint_kp"]));
      joint_kd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["joint_kd"]));
      mimic_start_from_zero_ = config_["mimic_start_from_zero"] ? config_["mimic_start_from_zero"].as<bool>() : false;
      if (mimic_start_from_zero_) {
        RCLCPP_INFO(get_logger(), "mimic_start_from_zero: true (trajectory always start from frame 0)");
      }
      action_scale_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["action_scale"]));
      command_scale_ = command_scale_vec_;

      if (config_["imu_install_delta_bias"]) {
        if (config_["imu_install_delta_bias"].IsScalar()) {
          double bias_val = config_["imu_install_delta_bias"].as<double>();
          imu_install_bias_ = Eigen::Vector3d(bias_val, bias_val, bias_val);
        } else {
          imu_install_bias_ = LoadVectorFromYaml(config_["imu_install_delta_bias"]);
        }
      } else if (config_["imu_install_bias"]) {
        imu_install_bias_ = LoadVectorFromYaml(config_["imu_install_bias"]);
      }

      std::string workspace_root = GetWorkspaceRoot(config_file_dir_);

      LoadDancePolicy(workspace_root);
      mlp_net_action_.setZero(num_actions_);
      
      const int num_joints = static_cast<int>(active_joint_names_.size());
      q_diff_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      qd_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      action_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      w_history_ = Eigen::MatrixXd::Zero(3, num_include_obs_steps_);
      gravity_history_ = Eigen::MatrixXd::Zero(3, num_include_obs_steps_);
      
      if (observation_type_ == "mimic_future") {
        goal_buffer_ = Eigen::MatrixXd::Zero(480, 1);  // 10帧 × 24关节 × 2(pos+vel) = 480
      }
      
      if (current_profile_.use_quat_error) {
        quat_error_history_ = Eigen::MatrixXd::Zero(6, num_include_obs_steps_);
        initial_quat_offset_computed_ = false;
      }
      
      LoadCsvTrajectory(workspace_root);
      trajectory_index_ = 0;
      time_ = 0.0;
      global_phase_ = 0.0;
      is_first_time_ = true;
      is_dance_mode_ = true;

      mujoco_reset_sub_ = create_subscription<std_msgs::msg::Empty>(
          "/mujoco/reset_complete", 10,
          std::bind(&RlBasicRunnerDAC::MujocoResetCallback, this, std::placeholders::_1));

      policy_switch_pub_ = create_publisher<std_msgs::msg::String>("/rl/policy_switch", 10);

      control_timer_ = create_wall_timer(std::chrono::duration<double>(control_dt_),
                                         std::bind(&RlBasicRunnerDAC::ControlCallback, this));

      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
      return false;
    }
  }

 private:
  void LoadParametersFromYaml() {
    active_joint_names_ = config_["active_joint_names"].as<std::vector<std::string>>();
    num_actions_ = active_joint_names_.size();
    
    if (config_["num_observations"]) {
      num_observations_ = config_["num_observations"].as<int>();
    } else if (config_["observations"] && config_["observations"]["num_single_observations"]) {
      num_observations_ = config_["observations"]["num_single_observations"].as<int>();
    } else {
      num_observations_ = 42;
    }
    
    if (config_["num_include_obs_steps"]) {
      num_include_obs_steps_ = config_["num_include_obs_steps"].as<int>();
    } else if (config_["observations"] && config_["observations"]["observation_type"]) {
      for (const auto& obs_type : config_["observations"]["observation_type"]) {
        if (obs_type.second["num_include_obs_steps"]) {
          num_include_obs_steps_ = obs_type.second["num_include_obs_steps"].as<int>();
          break;
        }
      }
    } else {
      num_include_obs_steps_ = 5;
    }
    
    num_commands_ = config_["num_commands"] ? config_["num_commands"].as<int>() : 3;
    num_clock_signal_ = config_["num_clock_signal"] ? config_["num_clock_signal"].as<int>() : 2;
    control_dt_ = config_["control_dt"] ? config_["control_dt"].as<double>() : 0.02;
    
    if (config_["cycle_time"]) {
      cycle_time_ = config_["cycle_time"].as<double>();
    } else if (config_["default_dynamic_cycle_time"]) {
      cycle_time_ = config_["default_dynamic_cycle_time"].as<double>();
    } else {
      cycle_time_ = 0.8;
    }
    
    transition_time_ = config_["transition_time"] ? config_["transition_time"].as<double>() : 0.5;
    action_clip_ = config_["action_clip"] ? config_["action_clip"].as<double>() : 100.0;
    observation_clip_ = config_["observation_clip"] ? config_["observation_clip"].as<double>() : 100.0;
    
    double obs_scale_linear_vel = 2.0;
    double obs_scale_angular_vel = 1.0;
    double obs_scale_dof_pos = 1.0;
    double obs_scale_dof_vel = 0.05;
    double obs_scale_quat = 1.0;
    
    if (config_["observations"] && config_["observations"]["observation_scale"]) {
      const YAML::Node& obs_scale = config_["observations"]["observation_scale"];
      if (obs_scale["observation_scale_linear_vel"]) obs_scale_linear_vel = obs_scale["observation_scale_linear_vel"].as<double>();
      if (obs_scale["observation_scale_angular_vel"]) obs_scale_angular_vel = obs_scale["observation_scale_angular_vel"].as<double>();
      if (obs_scale["observation_scale_dof_pos"]) obs_scale_dof_pos = obs_scale["observation_scale_dof_pos"].as<double>();
      if (obs_scale["observation_scale_dof_vel"]) obs_scale_dof_vel = obs_scale["observation_scale_dof_vel"].as<double>();
      if (obs_scale["observation_scale_quat"]) obs_scale_quat = obs_scale["observation_scale_quat"].as<double>();
    } else {
      if (config_["observation_scale_linear_vel"]) obs_scale_linear_vel = config_["observation_scale_linear_vel"].as<double>();
      if (config_["observation_scale_angular_vel"]) obs_scale_angular_vel = config_["observation_scale_angular_vel"].as<double>();
      if (config_["observation_scale_dof_pos"]) obs_scale_dof_pos = config_["observation_scale_dof_pos"].as<double>();
      if (config_["observation_scale_dof_vel"]) obs_scale_dof_vel = config_["observation_scale_dof_vel"].as<double>();
      if (config_["observation_scale_quat"]) obs_scale_quat = config_["observation_scale_quat"].as<double>();
    }
    
    if (IsMimicFlatObservation()) {
      observation_scale_ = Eigen::VectorXd::Ones(num_observations_);
    } else {
      observation_scale_ = Eigen::VectorXd::Zero(num_observations_);
      observation_scale_ <<
          Eigen::VectorXd::Constant(num_actions_, obs_scale_dof_pos),
          Eigen::VectorXd::Constant(num_actions_, obs_scale_dof_vel),
          Eigen::VectorXd::Ones(num_actions_),
          Eigen::Vector3d::Constant(obs_scale_angular_vel),
          Eigen::Vector3d::Constant(obs_scale_quat);
    }
    
    obs_commands_scale_ = Eigen::VectorXd::Zero(num_commands_);
    obs_commands_scale_ << Eigen::Vector2d::Constant(obs_scale_linear_vel), obs_scale_angular_vel;
    
    if (config_["command_scale"]) {
      auto cmd_scale = config_["command_scale"];
      command_scale_vec_ = Eigen::Vector3d(cmd_scale[0].as<double>(), cmd_scale[1].as<double>(), cmd_scale[2].as<double>());
    } else {
      command_scale_vec_ = Eigen::Vector3d(1.0, 1.0, 1.0);
    }
    
    if (config_["qd_mask"]) {
      qd_mask_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["qd_mask"]));
    }
    
    if (config_["command_bias"]) {
      auto cmd_bias = config_["command_bias"];
      command_bias_ = Eigen::Vector3d(cmd_bias[0].as<double>(), cmd_bias[1].as<double>(), cmd_bias[2].as<double>());
    }
    
    if (config_["action_scale85"]) {
      action_scale85_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["action_scale85"]));
    }
    
    // 与 XZL 一致：enable_falling_switch / falling_detect_after_sec
    enable_falling_switch_ = config_["enable_falling_switch"] ? config_["enable_falling_switch"].as<bool>() : true;
    falling_detect_after_sec_ = config_["falling_detect_after_sec"] ? config_["falling_detect_after_sec"].as<double>() : 5.0;

    // 加载摔倒检测参数
    if (config_["fall_detection"]) {
      const YAML::Node& fall_config = config_["fall_detection"];
      if (fall_config["tilt_threshold"]) {
        fall_tilt_threshold_ = fall_config["tilt_threshold"].as<double>();
      }
      if (fall_config["omega_threshold"]) {
        fall_omega_threshold_ = fall_config["omega_threshold"].as<double>();
      }
      if (fall_config["confirm_frames"]) {
        fall_confirm_frames_ = fall_config["confirm_frames"].as<int>();
      }
      if (fall_config["fast_fall_omega"]) {
        fast_fall_omega_ = fall_config["fast_fall_omega"].as<double>();
      }
      if (fall_config["fast_fall_confirm_frames"]) {
        fast_fall_confirm_frames_ = fall_config["fast_fall_confirm_frames"].as<int>();
      }
      // 未写顶层 falling_detect_after_sec 时，可用 fall_detection.detection_delay（与旧 DAC 配置兼容）
      if (fall_config["detection_delay"] && !config_["falling_detect_after_sec"]) {
        falling_detect_after_sec_ = fall_config["detection_delay"].as<double>();
      }
      if (fall_config["passive_damping"]) {
        passive_damping_kd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(fall_config["passive_damping"]));
        RCLCPP_INFO(get_logger(), "Loaded passive_damping: %zd joints", passive_damping_kd_.size());
      } else {
        passive_damping_kd_ = Eigen::VectorXd::Zero(24);
        passive_damping_kd_.setConstant(0.5);
      }
    }

    RCLCPP_INFO(get_logger(),
                "Fall switch: %s, detect after %.2f s (XZL-style; MuJoCo reset clears time_)",
                enable_falling_switch_ ? "enabled" : "disabled", falling_detect_after_sec_);

    // 稳定性检测（舞蹈播放中不稳定则提前切 damping，可选；启用门控与摔倒相同：time_ >= falling_detect_after_sec_）
    mimic_stability_to_damping_enable_ = config_["mimic_stability_to_damping_enable"] ?
        config_["mimic_stability_to_damping_enable"].as<bool>() : false;
    if (config_["stability_detection"]) {
      const YAML::Node& stab = config_["stability_detection"];
      if (stab["ang_vel_threshold"]) ang_vel_threshold_ = stab["ang_vel_threshold"].as<double>();
      if (stab["gravity_dev_threshold"]) gravity_dev_threshold_ = stab["gravity_dev_threshold"].as<double>();
      if (stab["stability_smoothing_threshold"]) stability_smoothing_threshold_ = stab["stability_smoothing_threshold"].as<double>();
      if (stab["stability_history_length"]) stability_history_length_ = stab["stability_history_length"].as<int>();
    }
    stability_history_.clear();
    stability_history_.resize(std::max(1, stability_history_length_), true);
    RCLCPP_INFO(get_logger(), "mimic_stability_to_damping_enable: %s (gated by falling_detect_after_sec)",
                mimic_stability_to_damping_enable_ ? "true" : "false");
    
    // 读取扭矩限制参数
    torque_limit_enabled_ = config_["torque_limit"] ? config_["torque_limit"].as<bool>() : false;
    if (config_["max_torque_joint"]) {
      max_torque_joint_ = LoadVectorArrayFromYaml(config_["max_torque_joint"]);
      RCLCPP_INFO(get_logger(), "Torque limit enabled: %s, loaded %zu groups of max_torque_joint", 
                  torque_limit_enabled_ ? "true" : "false", max_torque_joint_.size());
    } else {
      RCLCPP_WARN(get_logger(), "max_torque_joint not found in config, torque limit will be disabled");
    }
    max_lower_body_torque_ = config_["max_lower_body_torque"] ? config_["max_lower_body_torque"].as<double>() : 0.0;
    // 读取 soft_torque_limit（软扭矩限制系数，默认0.9）
    soft_torque_limit_ = config_["soft_torque_limit"] ? config_["soft_torque_limit"].as<double>() : 0.9;
    if (torque_limit_enabled_) {
      RCLCPP_INFO(get_logger(), "Max lower body torque: %.1f N·m, soft_torque_limit: %.2f", 
                  max_lower_body_torque_, soft_torque_limit_);
    }
  }
  
  std::string GetWorkspaceRoot(const std::string& config_dir) {
    std::string current = config_dir;
    size_t install_pos = current.find("/install/");
    if (install_pos != std::string::npos) {
      current = current.substr(0, install_pos);
    }
    size_t build_pos = current.find("/build/");
    if (build_pos != std::string::npos) {
      current = current.substr(0, build_pos);
    }
    size_t config_pos = current.find("/config/");
    if (config_pos != std::string::npos) {
      current = current.substr(0, config_pos);
    }
    return current;
  }
  
  void InitializeDanceParam() {
    if (!config_["motion_states"]) return;
    
    const YAML::Node& motion_states = config_["motion_states"];
    for (const auto& state : motion_states) {
      const YAML::Node& profile_node = state.second;
      if (profile_node["enable"] && profile_node["enable"].as<bool>()) {
        current_profile_.enable = true;
        if (profile_node["observation_type"]) {
          current_profile_.observation_type = profile_node["observation_type"].as<std::string>();
          observation_type_ = current_profile_.observation_type;
        }
        if (profile_node["data_path"]) {
          current_profile_.data_path = profile_node["data_path"].as<std::string>();
          csv_data_path_ = current_profile_.data_path;
        }
        if (profile_node["policy_path"]) {
          current_profile_.policy_path = profile_node["policy_path"].as<std::string>();
        }
        if (profile_node["csv_dt"]) {
          current_profile_.csv_dt = profile_node["csv_dt"].as<double>();
          csv_dt_ = current_profile_.csv_dt;
        }
        if (profile_node["traj_frame"]) {
          current_profile_.traj_frame = profile_node["traj_frame"].as<std::vector<int>>();
          traj_frame_ = current_profile_.traj_frame;
        }
        if (profile_node["use_quat_error"]) {
          current_profile_.use_quat_error = profile_node["use_quat_error"].as<bool>();
        }
        if (profile_node["use_gravity_error"]) {
          current_profile_.use_gravity_error = profile_node["use_gravity_error"].as<bool>();
        }
        return;
      }
    }
  }

  bool IsMimicFlatObservation() const {
    return observation_type_ == "mimic_flat" || observation_type_ == "tracking_flat";
  }
  
  void LoadDancePolicy(const std::string& workspace_root) {
    std::string path;
    if (!current_profile_.policy_path.empty()) {
      path = ResolvePolicyPath(current_profile_.policy_path, workspace_root);
    }
    if (path.empty() && config_["policy_file"]) {
      path = ResolvePolicyPath(config_["policy_file"].as<std::string>(), workspace_root);
    }
    if (path.empty()) {
      RCLCPP_ERROR(get_logger(),
                   "Dance policy missing: set motion_states.<name>.policy_path or top-level policy_file in YAML");
      throw std::runtime_error("DAC: no dance policy path");
    }
    mlp_net_dance_ = std::make_unique<math::MnnModel>(path);
    mlp_net_ = mlp_net_dance_.get();
    RCLCPP_INFO(get_logger(), "Loaded dance (base) MNN: %s", path.c_str());
  }

  // 解析 Policy 文件路径（支持相对路径和绝对路径）
  std::string ResolvePolicyPath(const std::string& policy_file, const std::string& workspace_root) {
    if (policy_file.empty()) {
      return "";
    }
    
    if (policy_file[0] == '/') {
      // 绝对路径
      return policy_file;
    }
    
    // 尝试相对于配置目录
    std::string config_based = config_file_dir_ + "/" + policy_file;
    std::ifstream test(config_based);
    if (test.good()) {
      test.close();
      return config_based;
    }
    
    // 尝试相对于工作空间根目录
    std::string workspace_based = workspace_root + "/" + policy_file;
    test.open(workspace_based);
    if (test.good()) {
      test.close();
      return workspace_based;
    }
    
    RCLCPP_WARN(get_logger(), "Policy file not found: %s (tried config-based: %s, workspace-based: %s)",
                policy_file.c_str(), config_based.c_str(), workspace_based.c_str());
    return "";
  }
  
  void LoadCsvTrajectory(const std::string& /* workspace_root */) {
    if (csv_data_path_.empty()) return;
    
    std::string full_csv_path;
    if (csv_data_path_[0] == '/') {
      full_csv_path = csv_data_path_;
    } else {
      full_csv_path = config_file_dir_ + "/" + csv_data_path_;
    }
    
    csv_file_path_ = full_csv_path;
    
    if (observation_type_ == "mimic_future" || IsMimicFlatObservation()) {
      Eigen::MatrixXd traj = rl_dance::CsvLoader::LoadAndInterpolateJointTrajectoryWithVelAndQuat(
          full_csv_path, active_joint_names_, csv_dt_, control_dt_,
          traj_frame_.empty() ? 0 : traj_frame_[0],
          traj_frame_.size() > 1 ? traj_frame_[1] : -1);
      interpolated_trajs_["default_profile"] = traj;
    } else {
      Eigen::MatrixXd traj = rl_dance::CsvLoader::LoadAndInterpolateJointTrajectory(
          full_csv_path, active_joint_names_, csv_dt_, control_dt_,
          traj_frame_.empty() ? 0 : traj_frame_[0],
          traj_frame_.size() > 1 ? traj_frame_[1] : -1);
      interpolated_trajs_["default_profile"] = traj;
    }
    
    current_traj_ = &interpolated_trajs_["default_profile"];
  }

  void MujocoResetCallback(const std_msgs::msg::Empty::SharedPtr msg) {
    (void)msg;
    if (is_damping_mode_) {
      PublishPolicySwitch("damping", "dance", "");
    }
    // 与 XZL 一致：重置计时，使 falling_detect_after_sec 在每次 reset 后重新生效
    time_ = 0.0;
    is_first_time_ = true;
    is_dance_mode_ = true;
    is_damping_mode_ = false;
    global_phase_ = 0.0;
    fall_stable_count_ = 0;
    trajectory_index_ = 0;
    if (mimic_stability_to_damping_enable_) {
      stability_history_.clear();
      stability_history_.resize(std::max(1, stability_history_length_), true);
    }
    mlp_net_ = mlp_net_dance_.get();
    if (interpolated_trajs_.count("default_profile") > 0) {
      current_traj_ = &interpolated_trajs_["default_profile"];
    }

    q_diff_history_.setZero();
    qd_history_.setZero();
    action_history_.setZero();
    w_history_.setZero();
    gravity_history_.setZero();
    if (quat_error_history_.size() > 0) quat_error_history_.setZero();
    initial_quat_offset_computed_ = false;
    initial_quat_offset_ = std::nullopt;
    if (mlp_net_action_.size() > 0) {
      mlp_net_action_.setZero();
    }
    if (goal_buffer_.size() > 0) {
      goal_buffer_.setZero();
    }

    RCLCPP_INFO(get_logger(), "MuJoCo reset: restart dance (base policy) at time: %.2f", time_);
  }

  // 稳定性检测（舞蹈播放中）：与 rl_dance 同逻辑，角速度+重力偏差+历史平滑
  // 返回 true 表示当前稳定，false 表示不稳定（可提前切 damping）
  bool CheckMimicStability() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu || stability_history_.empty()) return true;
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x,
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    Eigen::Vector3d w_real = R_real.transpose() * R_local *
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d projected_gravity = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    Eigen::Vector2d ang_vel_xy = w_real.head<2>();
    bool ang_vel_stable = ang_vel_xy.norm() < ang_vel_threshold_;
    Eigen::Vector3d g0(0.0, 0.0, -1.0);
    double grav_dev = (projected_gravity.normalized() - g0).norm();
    bool gravity_stable = grav_dev < gravity_dev_threshold_;
    bool current_stable = ang_vel_stable && gravity_stable;
    stability_history_.push_back(current_stable);
    stability_history_.pop_front();
    int stable_count = 0;
    for (bool s : stability_history_) if (s) stable_count++;
    double stable_ratio = static_cast<double>(stable_count) / static_cast<int>(stability_history_.size());
    return stable_ratio > stability_smoothing_threshold_;
  }
  
  // 摔倒检测：倾斜角 / 角速度 + 连续帧确认；不区分方向，触发后由上层直接进入 damping
  bool DetectFall() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      fall_stable_count_ = 0;
      return false;
    }

    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();

    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x,
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();

    Eigen::Vector3d w_real = R_real.transpose() * R_local *
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d projected_gravity = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    const double gz = projected_gravity.z();

    double tilt = std::acos(std::clamp(-gz, -1.0, 1.0));
    double omega_xy = std::hypot(w_real.x(), w_real.y());

    bool candidate = (tilt > fall_tilt_threshold_) || (omega_xy > fall_omega_threshold_);
    int N_confirm = (omega_xy > fast_fall_omega_) ? fast_fall_confirm_frames_ : fall_confirm_frames_;

    if (candidate) {
      fall_stable_count_++;
    } else {
      fall_stable_count_ = 0;
    }

    if (fall_stable_count_ >= N_confirm) {
      fall_stable_count_ = 0;
      RCLCPP_INFO(get_logger(),
                  "Fall detected -> damping (tilt=%.3f rad, omega_xy=%.3f rad/s)",
                  tilt, omega_xy);
      return true;
    }
    return false;
  }

  void ControlCallback() {
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
      time_ = 0.0;
      is_first_time_ = true;
      is_dance_mode_ = true;
      is_damping_mode_ = false;
      return;
    }

    auto joint_state = message_handler_->GetLatestJointState();
    if (!joint_state) return;

    // 舞蹈（基座）模式下：与 XZL 相同门控 time_ >= falling_detect_after_sec_ 后再检测摔倒 -> damping
    if (enable_falling_switch_ && is_dance_mode_ && !is_damping_mode_ &&
        time_ >= falling_detect_after_sec_ && DetectFall()) {
      is_damping_mode_ = true;
      if (mimic_stability_to_damping_enable_) {
        stability_history_.clear();
        stability_history_.resize(std::max(1, stability_history_length_), true);
      }
      PublishPolicySwitch("dance", "damping", "");
    }

    // 舞蹈播放中：可选稳定性检测 -> 提前 damping（同一延迟，避免 reset/起步瞬态）
    if (is_dance_mode_ && !is_damping_mode_ && mimic_stability_to_damping_enable_ &&
        time_ >= falling_detect_after_sec_) {
      if (!CheckMimicStability()) {
        is_damping_mode_ = true;
        RCLCPP_INFO(get_logger(), "[dance] 不稳定，提前进入 damping (time=%.2f)", time_);
        PublishPolicySwitch("dance", "damping", "");
      }
    }

    UpdateState(joint_state);

    // 每次 MuJoCo reset / 离开再进 joint_bridge 后首帧：用当前真机关节位姿作为 transition 起点，
    // 避免仍用启动时 initial_joint_q_ 做插值，导致 reset 后策略被拉向错误姿态、跟踪崩坏。
    if (is_first_time_) {
      if (q_real_.size() == initial_joint_q_.size()) {
        initial_joint_q_ = q_real_;
        RCLCPP_DEBUG(get_logger(), "DAC: resampled initial_joint_q_ from current state (size=%zu)", q_real_.size());
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "DAC: skip initial_joint_q_ resample, DOF mismatch q_real=%zu vs initial=%zu",
                             q_real_.size(), initial_joint_q_.size());
      }
      if (mlp_net_action_.size() > 0) {
        mlp_net_action_.setZero();
      }
      is_first_time_ = false;
    }

    if (!is_damping_mode_) {
      CalculateObservation();
    }
    CalculateMotorCommand();
    
    // 应用扭矩限制（如果启用）
    if (torque_limit_enabled_) {
      ApplyTorqueLimits();
    }
    
    SendMotorCommand();
    time_ += control_dt_;
  }

  void UpdateState(const interface_protocol::msg::JointState::SharedPtr& joint_state) {
    q_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size());
    qd_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->velocity.data(), joint_state->velocity.size());
    
    if (qd_mask_.size() > 0 && qd_mask_.size() == qd_real_.size()) {
      qd_real_ = qd_real_.cwiseProduct(qd_mask_);
    }

    if (is_dance_mode_ && !is_damping_mode_ && observation_type_ != "mimic_future") {
      auto gamepad = message_handler_->GetLatestGamepad();
      if (gamepad) {
        command_.x() = gamepad->analog_states[interface_protocol::msg::GamepadKeys::LEFT_STICK_X] * command_scale_vec_.x();
        command_.y() = gamepad->analog_states[interface_protocol::msg::GamepadKeys::LEFT_STICK_Y] * command_scale_vec_.y();
        command_.z() = gamepad->analog_states[interface_protocol::msg::GamepadKeys::RIGHT_STICK_Y] * command_scale_vec_.z();
        if (command_bias_.size() == 3) {
          command_ += command_bias_;
        }
      }
    }
  }

  void CalculateObservation() {
    if (observation_type_ == "mimic_future") {
      CalculateObservationMimicFuture();
    } else if (IsMimicFlatObservation()) {
      // mimic_flat policy is single-step: observation is assembled on demand in CalculateMotorCommand().
    }
  }
  
  Eigen::VectorXd ComputeMotionAnchorOriObservation(const Eigen::Matrix3d& R_real) {
    const int num_joints = static_cast<int>(active_joint_names_.size());
    Eigen::VectorXd motion_anchor_ori = Eigen::VectorXd::Zero(6);
    std::optional<Eigen::Quaterniond> ref_quat = std::nullopt;

    if (current_traj_ != nullptr && current_traj_->cols() >= 2 * num_joints + 4 && current_traj_->rows() > 0) {
      size_t traj_idx = std::min(trajectory_index_, static_cast<size_t>(current_traj_->rows() - 1));
      double qx = (*current_traj_)(traj_idx, 2 * num_joints + 0);
      double qy = (*current_traj_)(traj_idx, 2 * num_joints + 1);
      double qz = (*current_traj_)(traj_idx, 2 * num_joints + 2);
      double qw = (*current_traj_)(traj_idx, 2 * num_joints + 3);
      ref_quat = Eigen::Quaterniond(qw, qx, qy, qz);
    }

    if (!ref_quat.has_value()) {
      return motion_anchor_ori;
    }

    Eigen::Quaterniond current_quat(R_real);
    if (!initial_quat_offset_computed_) {
      initial_quat_offset_ = current_quat.conjugate() * ref_quat.value();
      initial_quat_offset_computed_ = true;
    }

    Eigen::Quaterniond q_ref = initial_quat_offset_.value().conjugate() * ref_quat.value();
    Eigen::Quaterniond q_error = current_quat.conjugate() * q_ref;
    q_error.normalize();

    Eigen::Matrix3d R_error = q_error.toRotationMatrix();
    motion_anchor_ori << R_error(0, 0), R_error(0, 1),
                         R_error(1, 0), R_error(1, 1),
                         R_error(2, 0), R_error(2, 1);
    return motion_anchor_ori;
  }
  
  void CalculateObservationMimicFuture() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) return;
    
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    Eigen::Vector3d w_real = R_real.transpose() * R_local * 
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d projected_gravity = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    
    const int num_joints = static_cast<int>(active_joint_names_.size());
    Eigen::VectorXd q_diff = q_real_ - default_joint_q_;
    Eigen::VectorXd qd = qd_real_;
    
    if (num_include_obs_steps_ > 1) {
      q_diff_history_.leftCols(num_include_obs_steps_ - 1) = q_diff_history_.rightCols(num_include_obs_steps_ - 1);
      qd_history_.leftCols(num_include_obs_steps_ - 1) = qd_history_.rightCols(num_include_obs_steps_ - 1);
      action_history_.leftCols(num_include_obs_steps_ - 1) = action_history_.rightCols(num_include_obs_steps_ - 1);
      w_history_.leftCols(num_include_obs_steps_ - 1) = w_history_.rightCols(num_include_obs_steps_ - 1);
      gravity_history_.leftCols(num_include_obs_steps_ - 1) = gravity_history_.rightCols(num_include_obs_steps_ - 1);
    }
    
    q_diff_history_.rightCols(1) = q_diff;
    qd_history_.rightCols(1) = qd;
    action_history_.rightCols(1) = mlp_net_action_;
    w_history_.rightCols(1) = w_real;
    gravity_history_.rightCols(1) = projected_gravity;
    
    if (current_profile_.use_quat_error) {
      if (num_include_obs_steps_ > 1) {
        quat_error_history_.leftCols(num_include_obs_steps_ - 1) = quat_error_history_.rightCols(num_include_obs_steps_ - 1);
      }
      quat_error_history_.rightCols(1) = ComputeMotionAnchorOriObservation(R_real);
    }
    
    if (current_traj_ != nullptr) {
      const int num_future_frames = 10;  // 10帧：当前帧 + 未来9帧（与 rl_dance 一致）
      const size_t max_idx = current_traj_->rows() - 1;
      Eigen::VectorXd goal_obs(480);  // 10帧 × 24关节 × 2(pos+vel) = 480
      for (int frame = 0; frame < num_future_frames; ++frame) {
        size_t traj_idx = std::min(trajectory_index_ + frame, max_idx);  // 包含当前帧（frame=0）
        int obs_offset = frame * 2 * num_joints;
        goal_obs.segment(obs_offset, num_joints) = current_traj_->row(traj_idx).head(num_joints);
        goal_obs.segment(obs_offset + num_joints, num_joints) = 
            current_traj_->row(traj_idx).segment(num_joints, num_joints);
      }
      goal_buffer_.col(0) = goal_obs;
    } else {
      goal_buffer_.col(0).setZero();
    }
  }
  
  // Step trajectory index (在 CalculateMotorCommand 之后调用，与 rl_dance 一致)
  void StepTrajectory() {
    if (current_traj_ != nullptr) {
      size_t end = current_traj_->rows() > 0 ? current_traj_->rows() - 1 : 0;
      if (trajectory_index_ < end) {
        trajectory_index_++;
      } else if (!is_damping_mode_ && trajectory_index_ >= end) {
        // 轨迹播放完成，进入 damping mode
        is_damping_mode_ = true;
        RCLCPP_INFO(get_logger(), "[mimic] 轨迹播放完成（第 %zu 帧），进入 damping mode (time=%.2f)",
                    trajectory_index_, time_);
        PublishPolicySwitch("dance", "damping", "");
      }
    }
  }

  void CalculateMotorCommand() {
    // Damping mode: mimic 轨迹播放完后，只发送阻尼命令（kp=0, kd=damping）
    if (is_damping_mode_) {
      // 保持当前关节位置，但不使用 kp 控制，只靠阻尼稳定关节
      // q_des_ 设为当前实际位置，这样在 SendJointCommands 中 position 会是当前位置
      q_des_ = q_real_;
      return;
    }
    
    Eigen::VectorXd obs;

    if (IsMimicFlatObservation()) {
      const int num_joints = static_cast<int>(active_joint_names_.size());
      const int command_dim = 2 * num_joints;
      const int motion_anchor_ori_dim = 6;
      const int obs_dim = command_dim + motion_anchor_ori_dim + 3 + num_joints + num_joints + num_joints;

      obs = Eigen::VectorXd::Zero(obs_dim);

      auto imu = message_handler_->GetLatestImu();
      if (!imu) {
        RCLCPP_WARN(get_logger(), "IMU data not available for mimic_flat observation");
        return;
      }

      Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
      Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
      Eigen::Matrix3d R_install = q_install.toRotationMatrix();
      Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x,
                                                   imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
      Eigen::Matrix3d R_real = R_local * R_install.transpose();
      Eigen::Vector3d w_real = R_real.transpose() * R_local *
          Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);

      Eigen::VectorXd command_obs = Eigen::VectorXd::Zero(command_dim);
      if (current_traj_ != nullptr && current_traj_->rows() > 0 && current_traj_->cols() >= 2 * num_joints + 4) {
        size_t traj_idx = std::min(trajectory_index_, static_cast<size_t>(current_traj_->rows() - 1));
        command_obs.head(num_joints) = current_traj_->row(traj_idx).head(num_joints);
        command_obs.tail(num_joints) = current_traj_->row(traj_idx).segment(num_joints, num_joints);
      }

      int offset = 0;
      obs.segment(offset, command_dim) = command_obs;
      offset += command_dim;
      obs.segment(offset, motion_anchor_ori_dim) = ComputeMotionAnchorOriObservation(R_real);
      offset += motion_anchor_ori_dim;
      obs.segment(offset, 3) = w_real;
      offset += 3;
      obs.segment(offset, num_joints) = q_real_ - default_joint_q_;
      offset += num_joints;
      obs.segment(offset, num_joints) = qd_real_;
      offset += num_joints;
      obs.segment(offset, num_joints) = mlp_net_action_;

      if (num_observations_ != obs_dim) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "mimic_flat observation size mismatch: yaml=%d, runtime=%d",
                             num_observations_, obs_dim);
      }
    } else if (observation_type_ == "mimic_future") {
      // 构建 mimic_future 的完整 observation（与 rl_dance 一致）
      // 结构: [proprioceptive_history (390维), quat_error_history (30维，如果启用), future_frames (480维)]
      // 如果 use_quat_error=true: 390 + 30 + 480 = 900维
      // 如果 use_quat_error=false: 390 + 0 + 480 = 870维
      // 注意：rl_dance 没有 command 部分，future_frames 包含当前帧 + 未来9帧 = 10帧
      
      const int num_joints = static_cast<int>(active_joint_names_.size());
      const int proprio_dim = num_joints + num_joints + num_joints + 3 + 3;  // q_diff + qd + action + w + gravity = 78
      const int proprio_total = proprio_dim * num_include_obs_steps_;  // 78 * 5 = 390
      const int quat_error_total = current_profile_.use_quat_error ? 6 * num_include_obs_steps_ : 0;  // 6 * 5 = 30 或 0
      const int future_frames_total = 480;  // 10帧 × 24关节 × 2(pos+vel) = 480（当前帧 + 未来9帧）
      
      obs = Eigen::VectorXd::Zero(proprio_total + quat_error_total + future_frames_total);
      
      int offset = 0;
      
      // 1. Proprioceptive history (390维): [q_diff_history, qd_history, action_history, w_history, gravity_history]
      // 应用 scaling（从配置中读取）
      double pos_scale = 1.0;
      double vel_scale = 0.05;
      double ang_vel_scale = 1.0;
      double quat_scale = 1.0;
      
      // 从配置中读取 scale 值
      if (config_["observations"] && config_["observations"]["observation_scale"]) {
        const YAML::Node& obs_scale = config_["observations"]["observation_scale"];
        if (obs_scale["observation_scale_dof_pos"]) pos_scale = obs_scale["observation_scale_dof_pos"].as<double>();
        if (obs_scale["observation_scale_dof_vel"]) vel_scale = obs_scale["observation_scale_dof_vel"].as<double>();
        if (obs_scale["observation_scale_angular_vel"]) ang_vel_scale = obs_scale["observation_scale_angular_vel"].as<double>();
        if (obs_scale["observation_scale_quat"]) quat_scale = obs_scale["observation_scale_quat"].as<double>();
      }
      
      // 与 rl_dance 一致：按 buffer 类型拼接，而不是按时间步
      // 顺序：所有 q_diff, 所有 qd, 所有 action, 所有 w, 所有 gravity
      // 注意：先构建不带 scaling 的 observation，然后统一应用 scaling（与 rl_dance 一致）
      
      // 1. q_diff (所有时间步) - 先不应用 scaling
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        obs.segment(offset, num_joints) = q_diff_history_.col(t);
        offset += num_joints;
      }
      
      // 2. qd (所有时间步) - 先不应用 scaling
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        obs.segment(offset, num_joints) = qd_history_.col(t);
        offset += num_joints;
      }
      
      // 3. action (所有时间步) - 先不应用 scaling
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        obs.segment(offset, num_joints) = action_history_.col(t);
        offset += num_joints;
      }
      
      // 4. w (所有时间步) - 先不应用 scaling
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        obs.segment(offset, 3) = w_history_.col(t);
        offset += 3;
      }
      
      // 5. gravity (所有时间步) - 先不应用 scaling
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        obs.segment(offset, 3) = gravity_history_.col(t);
        offset += 3;
      }
      
      // 应用 proprioceptive scaling（与 rl_dance 的 BuildProprioceptiveScale 一致）
      // 构建 scale 向量：对于每个 buffer 类型，对于每个时间步，应用对应的 scale
      Eigen::VectorXd proprio_scale = Eigen::VectorXd::Zero(proprio_total);
      int scale_idx = 0;
      
      // q_diff scale
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        proprio_scale.segment(scale_idx, num_joints) = Eigen::VectorXd::Constant(num_joints, pos_scale);
        scale_idx += num_joints;
      }
      
      // qd scale
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        proprio_scale.segment(scale_idx, num_joints) = Eigen::VectorXd::Constant(num_joints, vel_scale);
        scale_idx += num_joints;
      }
      
      // action scale (1.0)
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        proprio_scale.segment(scale_idx, num_joints) = Eigen::VectorXd::Constant(num_joints, 1.0);
        scale_idx += num_joints;
      }
      
      // w scale
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        proprio_scale.segment(scale_idx, 3) = Eigen::VectorXd::Constant(3, ang_vel_scale);
        scale_idx += 3;
      }
      
      // gravity scale
      for (int t = 0; t < num_include_obs_steps_; ++t) {
        proprio_scale.segment(scale_idx, 3) = Eigen::VectorXd::Constant(3, quat_scale);
        scale_idx += 3;
      }
      
      // 应用 scaling
      obs.head(proprio_total).array() *= proprio_scale.array();
      
      // 2. Quat error history (30维，如果启用)
      // 注意：quat_error scaling 在 ApplyGoalScaling 中应用（与 rl_dance 一致）
      if (quat_error_total > 0) {
        // Flatten quat_error_history_ (6 rows, time_steps cols) -> (6*time_steps, 1) vector
        Eigen::Map<const Eigen::VectorXd> quat_error_flat(quat_error_history_.data(), quat_error_total);
        obs.segment(offset, quat_error_total) = quat_error_flat;
        offset += quat_error_total;
      }
      
      // 3. Future frames (480维 = 10帧 × 24关节 × 2(pos+vel))
      // 包含当前帧 + 未来9帧，与 rl_dance 一致
      obs.segment(offset, future_frames_total) = goal_buffer_.col(0);
      offset += future_frames_total;
      
      // 应用 quat_error 和 goal scaling（与 rl_dance 的 ApplyGoalScaling 一致）
      // 先应用 quat_error scaling（如果启用）
      if (quat_error_total > 0) {
        double quat_error_scale = 1.0;
        if (config_["observations"] && config_["observations"]["observation_scale"] && 
            config_["observations"]["observation_scale"]["observation_scale_quat_error"]) {
          quat_error_scale = config_["observations"]["observation_scale"]["observation_scale_quat_error"].as<double>();
        }
        obs.segment(proprio_total, quat_error_total) *= quat_error_scale;
      }
      
      // 应用 goal scaling（对于 mimic_future: pos_scale=1.0, vel_scale=0.05）
      // 与 rl_dance 一致：10帧（当前帧 + 未来9帧）
      const int num_future_frames = 10;
      const double goal_pos_scale = 1.0;
      const double goal_vel_scale = 0.05;
      int goal_start_idx = proprio_total + quat_error_total;
      for (int frame = 0; frame < num_future_frames; ++frame) {
        int frame_offset = goal_start_idx + frame * 2 * num_joints;
        // Scale position part
        obs.segment(frame_offset, num_joints) *= goal_pos_scale;
        // Scale velocity part
        obs.segment(frame_offset + num_joints, num_joints) *= goal_vel_scale;
      }
      
      // 应用 clip（与 rl_dance 一致：最后应用）
      obs = obs.cwiseMax(-observation_clip_).cwiseMin(observation_clip_);
      
      // 正确的 observation 结构：future_frames (480维，10帧：当前帧 + 未来9帧)
      
      int expected_dim = current_profile_.use_quat_error ? 900 : 870;
      RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: mimic_future observation built, size: %ld (expected: %d, use_quat_error=%s)", 
                   obs.size(), expected_dim, current_profile_.use_quat_error ? "true" : "false");
    } else {
      // 其他类型的 observation 构建（回退逻辑）
      Eigen::Vector2d clock_signal(std::sin(2 * M_PI * global_phase_), std::cos(2 * M_PI * global_phase_));
      obs = Eigen::VectorXd::Zero(num_observations_ * num_include_obs_steps_ + num_clock_signal_ + num_commands_);
      
      int offset = 0;
      for (int i = 0; i < num_observations_; ++i) {
        obs.segment(offset, num_include_obs_steps_) = mlp_net_observation_.row(i);
        offset += num_include_obs_steps_;
      }
      
      command_.array() *= obs_commands_scale_.array();
      obs.tail(num_clock_signal_ + num_commands_) << clock_signal, command_;
    }
    
    // Get MNN output
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Preparing MNN inference");
    
    // 添加安全检查
    if (!mlp_net_) {
      RCLCPP_ERROR(get_logger(), "MNN model not initialized");
      return;
    }
    
    if (obs.size() == 0) {
      RCLCPP_ERROR(get_logger(), "Empty observation vector");
      return;
    }
    
    Eigen::MatrixXf obs_matrix = obs.cast<float>().transpose();
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Observation matrix created, size: %ld x %ld", obs_matrix.rows(), obs_matrix.cols());
    
    // 检查观察矩阵大小
    if (obs_matrix.rows() == 0 || obs_matrix.cols() == 0) {
      RCLCPP_ERROR(get_logger(), "Invalid observation matrix size: %ld x %ld", obs_matrix.rows(), obs_matrix.cols());
      return;
    }
    
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Calling MNN inference");
    try {
      mlp_net_action_ = mlp_net_->Inference(obs_matrix).cast<double>();
      RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: MNN inference completed, action size: %ld", mlp_net_action_.size());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "MNN inference failed: %s", e.what());
      return;
    }
    mlp_net_action_ = mlp_net_action_.cwiseMax(-action_clip_).cwiseMin(action_clip_);

    // Trajectory stepping (与 rl_dance 一致：在推理之后、计算 q_des 之前)
    if (observation_type_ == "mimic_future" || IsMimicFlatObservation()) {
      StepTrajectory();
    }

    // Calculate desired joint positions
    // 对于 mimic_future (residual_control=false): q_des = default_joint_q + action * action_scale
    // mlp_net_action_ 和 action_scale_ 都是对所有关节的（24维）
    if (mlp_net_action_.size() != action_scale_.size()) {
      RCLCPP_ERROR(get_logger(), "Action size mismatch: mlp_net_action=%ld, action_scale=%ld", 
                   mlp_net_action_.size(), action_scale_.size());
      return;
    }
    if (mlp_net_action_.size() != default_joint_q_.size()) {
      RCLCPP_ERROR(get_logger(), "Action size mismatch with default_joint_q: mlp_net_action=%ld, default_joint_q=%ld", 
                   mlp_net_action_.size(), default_joint_q_.size());
      return;
    }
    
    // 与 rl_dance 一致：q_des = default_joint_q + mlp_net_action * action_scale
    q_des_ = default_joint_q_ + mlp_net_action_.cwiseProduct(action_scale_);
    
    // Transition interpolation（与 rl_dance 一致）；transition_time<=0 时跳过，避免除零
    if (transition_time_ > 1e-9 && time_ < transition_time_) {
      double ratio = std::min(1.0, time_ / transition_time_);
      if (initial_joint_q_.size() == q_des_.size()) {
        q_des_ = ratio * q_des_ + (1.0 - ratio) * initial_joint_q_;
      }
    }
  }

  void ApplyTorqueLimits() {
    // 获取当前使用的 kp/kd
    Eigen::VectorXd joint_kp, joint_kd;
    if (is_damping_mode_) {
      // Damping mode: kp=0, kd=0.5
      joint_kp = Eigen::VectorXd::Zero(joint_kd_.size());
      joint_kd = Eigen::VectorXd::Constant(joint_kd_.size(), 0.5);
    } else {
      joint_kp = joint_kp_;
      joint_kd = joint_kd_;
    }
    
    const int n = q_des_.size();
    if (q_real_.size() != n || qd_real_.size() != n || joint_kp.size() != n || joint_kd.size() != n) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                          "Size mismatch in ApplyTorqueLimits, skipping torque limit");
      return;
    }
    
    // ---- 构造每关节扭矩上限向量 tau_max ----
    Eigen::VectorXd tau_max = Eigen::VectorXd::Zero(n);
    if (!max_torque_joint_.empty()) {
      int idx = 0;
      for (const auto& group : max_torque_joint_) {
        const int gsz = static_cast<int>(group.size());
        const int remain = n - idx;
        if (gsz > remain) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                              "max_torque_joint total size exceeds DOF count");
          break;
        }
        tau_max.segment(idx, gsz) = group;
        idx += gsz;
        if (idx >= n) break;
      }
      if (idx != n) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                            "max_torque_joint total size (%d) does not match DOF count (%d)", idx, n);
      }
      for (int i = 0; i < n; ++i) {
        if (tau_max(i) <= 0.0) {
          tau_max(i) = std::numeric_limits<double>::infinity();
        } else {
          // 应用 soft_torque_limit（参考XZL实现）
          tau_max(i) *= soft_torque_limit_;
        }
      }
    } else {
      // 如果没有配置，使用无穷大（不限制）
      tau_max = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                          "max_torque_joint is empty, torque limit disabled");
      return;  // 如果没有配置，直接返回
    }
    
    // ---- 1) PD 期望扭矩 tau_des = Kp*(q_des - q_actual) - Kd*qd ----
    Eigen::VectorXd tau_des(n);
    {
      Eigen::ArrayXd Kp = joint_kp.array();
      Eigen::ArrayXd Kd = joint_kd.array();
      tau_des = (Kp * (q_des_ - q_real_).array() - Kd * qd_real_.array()).matrix();
    }
    
    // ---- 2) 逐关节硬限幅（参考rl_dance_runner实现）----
    Eigen::VectorXd tau = tau_des;
    for (int i = 0; i < n; ++i) {
      const double m = tau_max(i);
      if (std::isfinite(m)) {
        if (tau(i) > m) tau(i) = m;
        if (tau(i) < -m) tau(i) = -m;
      }
    }
    
    // ---- 3) 下肢总扭矩限制（参考rl_dance_runner实现）----
    const int lower_body_end_index = 12;  // [0, 12) 为下肢
    if (n >= lower_body_end_index && max_lower_body_torque_ > 0.0) {
      const double sum_abs_lower = tau.segment(0, lower_body_end_index).cwiseAbs().sum();
      const double limit = max_lower_body_torque_;
      if (sum_abs_lower > limit) {
        const double scale = limit / sum_abs_lower;
        tau.segment(0, lower_body_end_index) *= scale;
      }
    }
    
    // ---- 4) 回推 q_des'：q_des = q_actual + (tau + Kd*qd)/Kp（参考rl_dance_runner实现）----
    const double eps = 1e-6;
    for (int i = 0; i < n; ++i) {
      const double kp = joint_kp(i);
      const double kd = joint_kd(i);
      if (std::abs(kp) > eps && std::isfinite(kp)) {
        q_des_(i) = q_real_(i) + (tau(i) + kd * qd_real_(i)) / kp;
      }
      // 如果 Kp 太小，保持 q_des(i) 不变
    }
    
    // ---- 5) 额外保护：通过限制 q_des_ 范围确保扭矩不超过限制（参考XZL实现）----
    // 这提供了额外的安全保护，即使q_actual和qd_real在下一帧变化，也能确保扭矩不超过限制
    const double kp_min_threshold = 1e-3;
    const Eigen::VectorXd joint_kp_safe = joint_kp.cwiseMax(kp_min_threshold);
    Eigen::VectorXd q_des_lb = (-tau_max.array() + (joint_kd.array() * qd_real_.array())).matrix();
    Eigen::VectorXd q_des_ub = (tau_max.array() + (joint_kd.array() * qd_real_.array())).matrix();
    q_des_lb = q_des_lb.array() / joint_kp_safe.array();
    q_des_ub = q_des_ub.array() / joint_kp_safe.array();
    q_des_lb += q_real_;
    q_des_ub += q_real_;
    q_des_ = q_des_.cwiseMax(q_des_lb).cwiseMin(q_des_ub);
  }

  void SendMotorCommand() {
    // Convert Eigen vectors to std::vector
    joint_command_->position = std::vector<double>(q_des_.data(), q_des_.data() + q_des_.size());
    joint_command_->velocity = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->torque = std::vector<double>(q_des_.size(), 0.0);
    
    // 根据模式选择正确的 kp/kd 参数
    if (is_damping_mode_) {
      // Damping mode: kp=0, kd 从 fall_detection.passive_damping 加载
      const int n = static_cast<int>(q_des_.size());
      joint_command_->stiffness = std::vector<double>(n, 0.0);
      if (n <= static_cast<int>(passive_damping_kd_.size())) {
        joint_command_->damping = std::vector<double>(passive_damping_kd_.data(), passive_damping_kd_.data() + n);
      } else {
        joint_command_->damping = std::vector<double>(n, 0.5);
      }
    } else {
      joint_command_->stiffness = std::vector<double>(joint_kp_.data(), joint_kp_.data() + joint_kp_.size());
      joint_command_->damping = std::vector<double>(joint_kd_.data(), joint_kd_.data() + joint_kd_.size());
    }
    
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    // Send command through message handler
    message_handler_->PublishJointCommand(*joint_command_);
  }

  // YAML 配置
  YAML::Node config_;
  
  // 从 YAML 读取的参数
  std::vector<std::string> active_joint_names_;
  int num_observations_ = 42;
  int num_include_obs_steps_ = 5;
  int num_actions_ = 0;
  int num_commands_ = 3;
  int num_clock_signal_ = 2;
  double control_dt_ = 0.02;
  double cycle_time_ = 0.8;
  double transition_time_ = 0.5;
  double action_clip_ = 100.0;
  double observation_clip_ = 100.0;
  Eigen::VectorXd observation_scale_;
  Eigen::VectorXd obs_commands_scale_;
  Eigen::Vector3d command_scale_vec_;
  
  // rl_dance 配置（直接从 YAML 读取，不依赖 RlDanceParam）
  data::RlDanceMotionStateProfile current_profile_;

  // Message handling
  std::shared_ptr<MessageHandler> message_handler_;

  // MNN：单一舞蹈（基座）策略
  std::unique_ptr<math::MnnModel> mlp_net_dance_;
  math::MnnModel* mlp_net_;
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::VectorXd mlp_net_action_;

  // ObservationManager 暂时不使用（需要 RlDanceParam）
  
  // CSV 轨迹数据
  Eigen::MatrixXd* current_traj_ = nullptr;
  Eigen::MatrixXd* current_traj_base_vel_ = nullptr;  // 基座速度轨迹 (N × 3: vx, vy, wz)
  std::map<std::string, Eigen::MatrixXd> interpolated_trajs_;
  std::map<std::string, Eigen::MatrixXd> interpolated_base_vel_trajs_;  // 基座速度轨迹
  size_t trajectory_index_ = 0;  // 当前轨迹索引
  double last_command_frame_print_time_ = -1.0;  // 上次打印 command 帧的时间（节流用）
  
  // Proprioceptive history buffers (5步历史)
  Eigen::MatrixXd q_diff_history_;      // q_actual - default_joint_q (24维 × 5步)
  Eigen::MatrixXd qd_history_;          // 关节速度 (24维 × 5步)
  Eigen::MatrixXd action_history_;      // 动作历史 (24维 × 5步)
  Eigen::MatrixXd w_history_;           // 角速度历史 (3维 × 5步)
  Eigen::MatrixXd gravity_history_;     // 重力历史 (3维 × 5步)
  Eigen::MatrixXd quat_error_history_; // 四元数误差历史 (6维 × 5步，当 use_quat_error 启用时)
  
  // Goal observation buffer (mimic_future: 480维 × 1步) - 10帧（当前帧 + 未来9帧，与 rl_dance 一致）
  Eigen::MatrixXd goal_buffer_;         // 目标观察缓冲区
  
  // Quaternion error calculation state
  std::optional<Eigen::Quaterniond> initial_quat_offset_;  // 初始四元数偏移
  bool initial_quat_offset_computed_ = false;  // 是否已计算初始偏移

  // State variables
  double time_;
  double global_phase_;
  bool is_first_time_;
  bool is_dance_mode_ = true;   // 基座：舞蹈策略运行中
  bool is_damping_mode_ = false;  // damping：被动阻尼（摔倒或轨迹结束）
  Eigen::Vector3d command_;

  int fall_stable_count_ = 0;               // 摔倒检测连续帧计数
  
  // Fall detection parameters
  double fall_tilt_threshold_ = 0.5;        // 倾斜角阈值 (rad)
  double fall_omega_threshold_ = 1.8;       // 角速度阈值 (rad/s)
  Eigen::VectorXd passive_damping_kd_;      // passive 模式 kd，从 fall_detection.passive_damping 加载
  int fall_confirm_frames_ = 8;             // 确认帧数
  double fast_fall_omega_ = 2.5;            // 快速摔倒角速度阈值 (rad/s)
  int fast_fall_confirm_frames_ = 2;        // 快速摔倒确认帧数
  // 与 XZL：enable_falling_switch / falling_detect_after_sec
  bool enable_falling_switch_ = true;
  double falling_detect_after_sec_ = 5.0;

  // 稳定性检测（舞蹈播放中不稳定则提前切 damping，可选）
  bool mimic_stability_to_damping_enable_ = false;
  double ang_vel_threshold_ = 0.4;
  double gravity_dev_threshold_ = 0.15;
  double stability_smoothing_threshold_ = 0.6;
  int stability_history_length_ = 10;
  std::deque<bool> stability_history_;
  
  // MuJoCo reset subscription
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr mujoco_reset_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr policy_switch_pub_;
  
  void PublishPolicySwitch(const std::string& from_mode, const std::string& to_mode,
                          const std::string& mimic_direction = "") {
    std_msgs::msg::String msg;
    msg.data = from_mode + "," + to_mode + "," + mimic_direction;
    policy_switch_pub_->publish(msg);
  }
  
  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  bool mimic_start_from_zero_ = false;  // 已从 YAML 加载；俯仰匹配已移除，此项暂无效果（保留兼容）
  Eigen::VectorXd action_scale_;
  Eigen::VectorXd action_scale85_;  // 用于特定 profile（如 dance_2）
  Eigen::VectorXd qd_mask_;  // 速度掩码（与 rl_dance 一致）
  Eigen::Vector3d command_scale_;
  Eigen::Vector3d command_bias_;  // 命令偏差（与 rl_dance 一致）
  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  
  // 扭矩限制参数
  bool torque_limit_enabled_ = false;
  std::vector<Eigen::VectorXd> max_torque_joint_;
  double max_lower_body_torque_ = 0.0;  // 下肢总扭矩限制（仅CHR/rl_dance使用）
  double soft_torque_limit_ = 0.9;  // 软扭矩限制系数，默认0.9（参考XZL实现）
  
  // CSV 相关配置（从 yaml 读取或使用默认值）
  std::string csv_data_path_;  // CSV 文件路径
  std::string csv_file_path_;  // CSV 文件的完整路径（用于读取 base_z）
  std::string observation_type_;  // observation 类型: "mimic", "mimic_tj", "rl_locomotion", etc.
  double csv_dt_ = 1.0 / 50.0;  // CSV 采样率
  std::vector<int> traj_frame_;  // 轨迹帧范围 [start, end]

  // ROS timer
  rclcpp::TimerBase::SharedPtr control_timer_;
  std::string config_file_dir_;
  std::string config_file_name_;  // 保存配置文件名
  interface_protocol::msg::JointCommand::SharedPtr joint_command_;
};

}  // namespace example

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("rl_basic_example_DAC"), "Usage: rl_basic_example_DAC <config_file_dir> [config_file_name]");
    return 1;
  }

  std::string config_file_name = "rl_basic_param_DAC.yaml";
  if (argc >= 3) {
    config_file_name = argv[2];
  }

  auto node = std::make_shared<example::RlBasicRunnerDAC>(argv[1], config_file_name);
  if (!node->Initialize()) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
