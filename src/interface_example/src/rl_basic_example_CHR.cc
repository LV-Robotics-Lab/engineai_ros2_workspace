#include <array>
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

using namespace std::chrono_literals;

namespace example {

// 8个方向的枚举定义
enum class MimicDirection {
  FORWARD = 0,       // 前 (+X)
  FORWARD_LEFT,      // 前左 (+X, +Y)
  LEFT,              // 左 (+Y)
  BACKWARD_LEFT,     // 后左 (-X, +Y)
  BACKWARD,          // 后 (-X)
  BACKWARD_RIGHT,    // 后右 (-X, -Y)
  RIGHT,             // 右 (-Y)
  FORWARD_RIGHT,     // 前右 (+X, -Y)
  COUNT = 8
};

// 方向名称（用于日志和配置）
const std::array<std::string, 8> kMimicDirectionNames = {
  "forward", "forward_left", "left", "backward_left",
  "backward", "backward_right", "right", "forward_right"
};

class RlBasicRunnerCHR : public rclcpp::Node {
 public:
  explicit RlBasicRunnerCHR(const std::string& config_file_dir, const std::string& config_file_name = "rl_basic_param_CHR.yaml") : Node("rl_basic_runner_CHR") {
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
      LoadParametersFromYaml();

      if (config_["active_joint_idx"]) {
        active_joint_idx_ = LoadIntVectorFromYaml(config_["active_joint_idx"]);
      } else {
        active_joint_idx_ = Eigen::VectorXi::LinSpaced(active_joint_names_.size(), 0, active_joint_names_.size() - 1);
      }
      
      default_joint_q_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["default_joint_q"]));
      joint_kp_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["joint_kp"]));
      joint_kd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_["joint_kd"]));
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

      InitializeDanceParam();
      std::string workspace_root = GetWorkspaceRoot(config_file_dir_);
      
      LoadWalkingParameters();
      
      std::string walking_policy_path;
      if (!walking_policy_file_.empty()) {
        if (walking_policy_file_[0] == '/') {
          walking_policy_path = walking_policy_file_;
        } else {
          std::string config_based = config_file_dir_ + "/" + walking_policy_file_;
          std::ifstream test(config_based);
          if (test.good()) {
            walking_policy_path = config_based;
            test.close();
          } else {
            walking_policy_path = workspace_root + "/" + walking_policy_file_;
          }
        }
      } else {
        return false;
      }
      
      // 加载8个方向的 mimic Policy
      LoadMimicDirectionalPolicies(workspace_root);
      
      mlp_net_walking_ = std::make_unique<math::MnnModel>(walking_policy_path);
      mlp_net_ = mlp_net_walking_.get();
      mlp_net_action_walking_.setZero(walking_num_actions_);
      mlp_net_action_.setZero(num_actions_);
      
      mlp_net_observation_walking_.setZero(walking_num_observations_, walking_num_include_obs_steps_);
      
      const int num_joints = static_cast<int>(active_joint_names_.size());
      q_diff_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      qd_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      action_history_ = Eigen::MatrixXd::Zero(num_joints, num_include_obs_steps_);
      w_history_ = Eigen::MatrixXd::Zero(3, num_include_obs_steps_);
      gravity_history_ = Eigen::MatrixXd::Zero(3, num_include_obs_steps_);
      
      if (observation_type_ == "mimic_future") {
        goal_buffer_ = Eigen::MatrixXd::Zero(432, 1);
      }
      
      if (current_profile_.use_quat_error) {
        quat_error_history_ = Eigen::MatrixXd::Zero(6, num_include_obs_steps_);
        initial_quat_offset_computed_ = false;
      }
      
      LoadCsvTrajectory(workspace_root);
      LoadMimicDirectionalTrajectories();  // 加载8个方向的 mimic 轨迹
      trajectory_index_ = 0;
      time_ = 0.0;
      global_phase_ = 0.0;
      is_first_time_ = true;
      is_walking_mode_ = true;
      walking_duration_ = 2.6;
      walking_start_time_ = -1.0;
      mujoco_reset_received_ = false;

      mujoco_reset_sub_ = create_subscription<std_msgs::msg::Empty>(
          "/mujoco/reset_complete", 10,
          std::bind(&RlBasicRunnerCHR::MujocoResetCallback, this, std::placeholders::_1));

      control_timer_ = create_wall_timer(std::chrono::duration<double>(control_dt_),
                                         std::bind(&RlBasicRunnerCHR::ControlCallback, this));

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
    
    observation_scale_ = Eigen::VectorXd::Zero(num_observations_);
    observation_scale_ <<
        Eigen::VectorXd::Constant(num_actions_, obs_scale_dof_pos),
        Eigen::VectorXd::Constant(num_actions_, obs_scale_dof_vel),
        Eigen::VectorXd::Ones(num_actions_),
        Eigen::Vector3d::Constant(obs_scale_angular_vel),
        Eigen::Vector3d::Constant(obs_scale_quat);
    
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
      if (fall_config["detection_delay"]) {
        fall_detection_delay_ = fall_config["detection_delay"].as<double>();
      }
    }

    // 稳定性检测（mimic 中不稳定则提前切 damping，可选）
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
    RCLCPP_INFO(get_logger(), "mimic_stability_to_damping_enable: %s", mimic_stability_to_damping_enable_ ? "true" : "false");
    
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
  
  void LoadWalkingParameters() {
    std::string walking_config_file = config_file_dir_ + "/rl_basic_param_XZL.yaml";
    config_walking_ = YAML::LoadFile(walking_config_file);
    
    walking_policy_file_ = config_walking_["policy_file"] ? config_walking_["policy_file"].as<std::string>() : "";
    walking_num_observations_ = config_walking_["num_observations"] ? config_walking_["num_observations"].as<int>() : 72;
    walking_num_include_obs_steps_ = config_walking_["num_include_obs_steps"] ? config_walking_["num_include_obs_steps"].as<int>() : 15;
    
    walking_num_actions_ = 0;
    if (config_walking_["active_joint_names"]) {
      walking_num_actions_ = config_walking_["active_joint_names"].size();
    } else {
      walking_num_actions_ = (walking_num_observations_ - 6) / 3;
    }
    
    if (config_walking_["active_joint_idx"]) {
      walking_active_joint_idx_ = LoadIntVectorFromYaml(config_walking_["active_joint_idx"]);
    } else {
      walking_active_joint_idx_ = Eigen::VectorXi::LinSpaced(walking_num_actions_, 0, walking_num_actions_ - 1);
    }
    
    double obs_scale_linear_vel = config_walking_["observation_scale_linear_vel"] ? config_walking_["observation_scale_linear_vel"].as<double>() : 2.0;
    double obs_scale_angular_vel = config_walking_["observation_scale_angular_vel"] ? config_walking_["observation_scale_angular_vel"].as<double>() : 1.0;
    double obs_scale_dof_pos = config_walking_["observation_scale_dof_pos"] ? config_walking_["observation_scale_dof_pos"].as<double>() : 1.0;
    double obs_scale_dof_vel = config_walking_["observation_scale_dof_vel"] ? config_walking_["observation_scale_dof_vel"].as<double>() : 0.05;
    double obs_scale_quat = config_walking_["observation_scale_quat"] ? config_walking_["observation_scale_quat"].as<double>() : 1.0;
    
    walking_observation_scale_ = Eigen::VectorXd::Zero(walking_num_observations_);
    walking_observation_scale_ <<
        Eigen::VectorXd::Constant(walking_num_actions_, obs_scale_dof_pos),
        Eigen::VectorXd::Constant(walking_num_actions_, obs_scale_dof_vel),
        Eigen::VectorXd::Ones(walking_num_actions_),
        Eigen::Vector3d::Constant(obs_scale_angular_vel),
        Eigen::Vector3d::Constant(obs_scale_quat);
    
    if (config_walking_["command_scale"]) {
      auto cmd_scale = config_walking_["command_scale"];
      walking_command_scale_ = Eigen::Vector3d(cmd_scale[0].as<double>(), cmd_scale[1].as<double>(), cmd_scale[2].as<double>());
    } else {
      walking_command_scale_ = Eigen::Vector3d(obs_scale_linear_vel, obs_scale_linear_vel, obs_scale_angular_vel);
    }
    
    if (config_walking_["cycle_time"]) {
      cycle_time_ = config_walking_["cycle_time"].as<double>();
    }
    
    if (config_walking_["control_dt"]) {
      control_dt_ = config_walking_["control_dt"].as<double>();
    }
    
    if (config_walking_["action_scale"]) {
      walking_action_scale_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["action_scale"]));
    } else {
      walking_action_scale_ = Eigen::VectorXd::Ones(walking_num_actions_);
    }
    
    if (config_walking_["default_joint_q"]) {
      walking_default_joint_q_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["default_joint_q"]));
    }
    
    walking_action_clip_ = config_walking_["action_clip"] ? config_walking_["action_clip"].as<double>() : 100.0;
    walking_observation_clip_ = config_walking_["observation_clip"] ? config_walking_["observation_clip"].as<double>() : 100.0;
    walking_transition_time_ = config_walking_["transition_time"] ? config_walking_["transition_time"].as<double>() : 0.5;
    
    // 加载 walking 模式的 joint_kp 和 joint_kd（从 XZL 配置）
    if (config_walking_["joint_kp"]) {
      walking_joint_kp_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["joint_kp"]));
    } else {
      walking_joint_kp_ = joint_kp_;  // 回退到 CHR 配置
    }
    
    if (config_walking_["joint_kd"]) {
      walking_joint_kd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["joint_kd"]));
    } else {
      walking_joint_kd_ = joint_kd_;  // 回退到 CHR 配置
    }
    
    if (config_walking_["imu_install_delta_bias"]) {
      if (config_walking_["imu_install_delta_bias"].IsScalar()) {
        double bias_val = config_walking_["imu_install_delta_bias"].as<double>();
        walking_imu_install_bias_ = Eigen::Vector3d(bias_val, bias_val, bias_val);
      } else {
        walking_imu_install_bias_ = LoadVectorFromYaml(config_walking_["imu_install_delta_bias"]);
      }
    } else if (config_walking_["imu_install_bias"]) {
      walking_imu_install_bias_ = LoadVectorFromYaml(config_walking_["imu_install_bias"]);
    } else {
      walking_imu_install_bias_ = Eigen::Vector3d::Zero();
    }
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
  
  // 加载8个方向的 mimic Policy
  void LoadMimicDirectionalPolicies(const std::string& workspace_root) {
    // 首先从 motion_states 下的启用 profile 中查找 mimic_policies
    YAML::Node mimic_policies;
    bool found_mimic_policies = false;
    
    if (config_["motion_states"]) {
      const YAML::Node& motion_states = config_["motion_states"];
      for (const auto& state : motion_states) {
        const YAML::Node& profile_node = state.second;
        if (profile_node["enable"] && profile_node["enable"].as<bool>()) {
          if (profile_node["mimic_policies"]) {
            mimic_policies = profile_node["mimic_policies"];
            found_mimic_policies = true;
            RCLCPP_INFO(get_logger(), "Found mimic_policies in motion_states profile");
          }
          break;
        }
      }
    }
    
    // 回退：检查根节点是否有 mimic_policies
    if (!found_mimic_policies && config_["mimic_policies"]) {
      mimic_policies = config_["mimic_policies"];
      found_mimic_policies = true;
    }
    
    if (found_mimic_policies) {
      for (size_t i = 0; i < static_cast<size_t>(MimicDirection::COUNT); ++i) {
        const std::string& dir_name = kMimicDirectionNames[i];
        if (mimic_policies[dir_name]) {
          std::string policy_file = mimic_policies[dir_name].as<std::string>();
          std::string policy_path = ResolvePolicyPath(policy_file, workspace_root);
          
          if (!policy_path.empty()) {
            try {
              mlp_net_mimic_directions_[i] = std::make_unique<math::MnnModel>(policy_path);
              RCLCPP_INFO(get_logger(), "Loaded mimic policy for direction %s: %s", 
                          dir_name.c_str(), policy_path.c_str());
            } catch (const std::exception& e) {
              RCLCPP_ERROR(get_logger(), "Failed to load mimic policy for direction %s: %s", 
                           dir_name.c_str(), e.what());
            }
          }
        }
      }
    } else {
      // 回退：尝试从 motion_states 中的 policy_path 加载单个 Policy（用于所有方向）
      std::string mimic_policy_path;
      if (!current_profile_.policy_path.empty()) {
        mimic_policy_path = ResolvePolicyPath(current_profile_.policy_path, workspace_root);
      } else if (config_["policy_file"]) {
        mimic_policy_path = config_file_dir_ + "/" + config_["policy_file"].as<std::string>();
      }
      
      if (!mimic_policy_path.empty()) {
        RCLCPP_INFO(get_logger(), "No mimic_policies config found, loading single policy for all directions: %s", 
                    mimic_policy_path.c_str());
        try {
          // 加载一个 Policy，并将其复制到所有方向
          auto single_policy = std::make_unique<math::MnnModel>(mimic_policy_path);
          // 第一个方向使用原始 Policy
          mlp_net_mimic_directions_[0] = std::move(single_policy);
          // 其他方向也加载相同的 Policy（因为 MLP 结构相同）
          for (size_t i = 1; i < static_cast<size_t>(MimicDirection::COUNT); ++i) {
            mlp_net_mimic_directions_[i] = std::make_unique<math::MnnModel>(mimic_policy_path);
          }
        } catch (const std::exception& e) {
          RCLCPP_ERROR(get_logger(), "Failed to load single mimic policy: %s", e.what());
        }
      }
    }
    
    // 确保至少有一个 Policy 被加载
    bool any_loaded = false;
    for (size_t i = 0; i < static_cast<size_t>(MimicDirection::COUNT); ++i) {
      if (mlp_net_mimic_directions_[i]) {
        any_loaded = true;
        break;
      }
    }
    
    if (!any_loaded) {
      RCLCPP_WARN(get_logger(), "No mimic policies loaded! Mimic mode may not work correctly.");
    }
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
    
    if (observation_type_ == "mimic_future") {
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
  
  // 加载8个方向的 mimic 轨迹
  void LoadMimicDirectionalTrajectories() {
    // 从 motion_states 下的启用 profile 中查找 mimic_trajectories
    YAML::Node mimic_trajectories;
    bool found = false;
    
    if (config_["motion_states"]) {
      const YAML::Node& motion_states = config_["motion_states"];
      for (const auto& state : motion_states) {
        const YAML::Node& profile_node = state.second;
        if (profile_node["enable"] && profile_node["enable"].as<bool>()) {
          if (profile_node["mimic_trajectories"]) {
            mimic_trajectories = profile_node["mimic_trajectories"];
            found = true;
            RCLCPP_INFO(get_logger(), "Found mimic_trajectories in motion_states profile");
          }
          break;
        }
      }
    }
    
    if (!found) {
      RCLCPP_INFO(get_logger(), "No mimic_trajectories config found, using default trajectory for all directions");
      return;
    }
    
    for (size_t i = 0; i < static_cast<size_t>(MimicDirection::COUNT); ++i) {
      const std::string& dir_name = kMimicDirectionNames[i];
      if (mimic_trajectories[dir_name]) {
        std::string traj_file = mimic_trajectories[dir_name].as<std::string>();
        std::string full_path;
        
        if (traj_file[0] == '/') {
          full_path = traj_file;
        } else {
          full_path = config_file_dir_ + "/" + traj_file;
        }
        
        std::ifstream test(full_path);
        if (!test.good()) {
          RCLCPP_WARN(get_logger(), "Trajectory file not found for direction %s: %s", 
                      dir_name.c_str(), full_path.c_str());
          continue;
        }
        test.close();
        
        try {
          Eigen::MatrixXd traj = rl_dance::CsvLoader::LoadAndInterpolateJointTrajectoryWithVelAndQuat(
              full_path, active_joint_names_, csv_dt_, control_dt_,
              traj_frame_.empty() ? 0 : traj_frame_[0],
              traj_frame_.size() > 1 ? traj_frame_[1] : -1);
          mimic_trajs_directions_[i] = traj;
          mimic_trajs_loaded_[i] = true;
          RCLCPP_INFO(get_logger(), "Loaded mimic trajectory for direction %s: %s (%ld frames)", 
                      dir_name.c_str(), full_path.c_str(), traj.rows());
        } catch (const std::exception& e) {
          RCLCPP_ERROR(get_logger(), "Failed to load trajectory for direction %s: %s", 
                       dir_name.c_str(), e.what());
        }
      }
    }
    
    // 默认使用 forward 方向的轨迹作为 current_traj_
    if (mimic_trajs_loaded_[static_cast<int>(MimicDirection::FORWARD)]) {
      current_traj_ = &mimic_trajs_directions_[static_cast<int>(MimicDirection::FORWARD)];
      RCLCPP_INFO(get_logger(), "Set default trajectory to forward direction");
    }
  }
  
  // 切换到指定方向的轨迹
  void SwitchToDirectionalTrajectory(MimicDirection direction) {
    int dir_idx = static_cast<int>(direction);
    if (mimic_trajs_loaded_[dir_idx]) {
      current_traj_ = &mimic_trajs_directions_[dir_idx];
      RCLCPP_INFO(get_logger(), "Switched to trajectory for direction %s (%ld frames)", 
                  kMimicDirectionNames[dir_idx].c_str(), current_traj_->rows());
    } else {
      RCLCPP_WARN(get_logger(), "Trajectory for direction %s not loaded, keeping current trajectory", 
                  kMimicDirectionNames[dir_idx].c_str());
    }
  }
  
  // 从 CSV 文件中读取四元数并计算俯仰角度（考虑 traj_frame_ 范围）
  std::vector<double> LoadPitchFromCsv(const std::string& csv_path) {
    std::vector<double> pitch_values;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(), "Failed to open CSV file for pitch reading: %s", csv_path.c_str());
      return pitch_values;
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
    
    // 查找四元数列的索引（优先查找 base_quat_x/y/z/w，如果没有则查找 base_qx/y/z/w）
    std::vector<std::string> quat_names = {"base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"};
    std::vector<std::string> quat_names_alt = {"base_qx", "base_qy", "base_qz", "base_qw"};
    std::vector<int> quat_col_idx;
    
    bool use_alt_names = false;
    auto first_quat_it = std::find(headers.begin(), headers.end(), quat_names[0]);
    if (first_quat_it == headers.end()) {
      auto alt_quat_it = std::find(headers.begin(), headers.end(), quat_names_alt[0]);
      if (alt_quat_it == headers.end()) {
        RCLCPP_WARN(get_logger(), "Quaternion columns not found in CSV file: %s", csv_path.c_str());
        return pitch_values;
      }
      use_alt_names = true;
    }
    
    const auto& selected_quat_names = use_alt_names ? quat_names_alt : quat_names;
    for (const auto& quat_name : selected_quat_names) {
      auto quat_it = std::find(headers.begin(), headers.end(), quat_name);
      if (quat_it == headers.end()) {
        RCLCPP_WARN(get_logger(), "Quaternion column not found: %s", quat_name.c_str());
        return pitch_values;
      }
      quat_col_idx.push_back(std::distance(headers.begin(), quat_it));
    }
    
    // 读取所有行的四元数并计算俯仰角度
    std::vector<double> all_pitch;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string val;
      std::vector<std::string> all_vals;
      
      while (std::getline(ss, val, ',')) {
        all_vals.push_back(val);
      }
      
      if (quat_col_idx[0] < static_cast<int>(all_vals.size()) &&
          quat_col_idx[1] < static_cast<int>(all_vals.size()) &&
          quat_col_idx[2] < static_cast<int>(all_vals.size()) &&
          quat_col_idx[3] < static_cast<int>(all_vals.size())) {
        try {
          // 读取四元数 (x, y, z, w)
          double qx = std::stod(all_vals[quat_col_idx[0]]);
          double qy = std::stod(all_vals[quat_col_idx[1]]);
          double qz = std::stod(all_vals[quat_col_idx[2]]);
          double qw = std::stod(all_vals[quat_col_idx[3]]);
          
          // 转换为旋转矩阵并计算俯仰角度
          Eigen::Quaterniond quat(qw, qx, qy, qz);
          Eigen::Matrix3d R = quat.toRotationMatrix();
          Eigen::Vector3d rpy = math::CalcRollPitchYawFromRotationMatrix(R);
          double pitch = rpy[1];  // pitch 是第二个元素
          
          all_pitch.push_back(pitch);
        } catch (const std::exception& e) {
          RCLCPP_WARN(get_logger(), "Failed to parse quaternion or calculate pitch: %s", e.what());
        }
      }
    }
    
    // 应用 traj_frame_ 范围（与 LoadCsvTrajectory 一致）
    int start_frame = traj_frame_.empty() ? 0 : traj_frame_[0];
    int end_frame = traj_frame_.size() > 1 ? traj_frame_[1] : -1;
    
    int start_idx = std::max(0, start_frame);
    int end_idx;
    if (end_frame == -1) {
      end_idx = static_cast<int>(all_pitch.size()) - 1;
    } else {
      end_idx = std::min(static_cast<int>(all_pitch.size()) - 1, end_frame);
    }
    
    if (start_idx > end_idx) {
      RCLCPP_WARN(get_logger(), "Invalid traj_frame range for pitch: start=%d, end=%d", start_idx, end_idx);
      return pitch_values;
    }
    
    // 提取范围内的 pitch 值
    pitch_values.assign(all_pitch.begin() + start_idx, all_pitch.begin() + end_idx + 1);
    
    return pitch_values;
  }
  
  // 获取当前 IMU 俯仰角度（使用 mimic 模式的 IMU bias）
  double GetCurrentPitchAngle() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      RCLCPP_WARN(get_logger(), "IMU data not available, using default pitch angle 0.0");
      return 0.0;
    }
    
    // 应用 IMU 安装偏差校正
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    
    // 从 IMU 四元数获取旋转矩阵
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    
    // 计算校正后的旋转矩阵
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    
    // 计算 RPY 角度
    Eigen::Vector3d rpy = math::CalcRollPitchYawFromRotationMatrix(R_real);
    double pitch = rpy[1];  // pitch 是第二个元素
    
    return pitch;
  }
  
  // 在插值后的轨迹中查找最匹配的俯仰角度 timestep
  size_t FindMatchingTrajectoryIndex(double current_pitch) {
    if (current_traj_ == nullptr) {
      RCLCPP_WARN(get_logger(), "Cannot find matching trajectory index: current_traj_ is null");
      return 0;
    }
    
    size_t traj_size = current_traj_->rows();
    const int num_joints = static_cast<int>(active_joint_names_.size());
    
    // 检查轨迹矩阵的列数是否正确（应该包含四元数）
    if (current_traj_->cols() != 2 * num_joints + 4) {
      RCLCPP_WARN(get_logger(), "Trajectory matrix has incorrect columns: expected %d, got %ld", 
                  2 * num_joints + 4, current_traj_->cols());
      return 0;
    }
    
    // 直接从插值后的轨迹矩阵中提取四元数并计算 pitch
    std::vector<double> traj_pitch;
    traj_pitch.reserve(traj_size);
    
    for (size_t i = 0; i < traj_size; ++i) {
      // 从轨迹矩阵中提取四元数 [qx, qy, qz, qw]
      // 四元数在最后4列：索引为 2*num_joints + 0, 1, 2, 3
      double qx = (*current_traj_)(i, 2 * num_joints + 0);
      double qy = (*current_traj_)(i, 2 * num_joints + 1);
      double qz = (*current_traj_)(i, 2 * num_joints + 2);
      double qw = (*current_traj_)(i, 2 * num_joints + 3);
      
      // 转换为旋转矩阵并计算俯仰角度
      Eigen::Quaterniond quat(qw, qx, qy, qz);
      Eigen::Matrix3d R = quat.toRotationMatrix();
      Eigen::Vector3d rpy = math::CalcRollPitchYawFromRotationMatrix(R);
      double pitch = rpy[1];  // pitch 是第二个元素
      
      traj_pitch.push_back(pitch);
    }
    
    // 从前往后查找最匹配的 timestep
    double min_diff = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    
    for (size_t i = 0; i < traj_pitch.size(); ++i) {
      // 计算角度差（考虑角度环绕）
      double diff = traj_pitch[i] - current_pitch;
      if (diff > M_PI) diff -= 2 * M_PI;
      if (diff < -M_PI) diff += 2 * M_PI;
      double abs_diff = std::abs(diff);
      
      if (abs_diff < min_diff) {
        min_diff = abs_diff;
        best_idx = i;
      }
    }
    
    RCLCPP_INFO(get_logger(), "Found matching trajectory index: %zu (pitch: %.4f rad, current: %.4f rad, diff: %.4f rad)",
                best_idx, traj_pitch[best_idx], current_pitch, min_diff);
    
    return best_idx;
  }
  
  void MujocoResetCallback(const std_msgs::msg::Empty::SharedPtr msg) {
    (void)msg;
    // 每次收到 reset 信号都重新初始化状态
    mujoco_reset_received_ = true;
    walking_start_time_ = time_;
    reset_time_ = time_;  // 记录 reset 时间，用于摔倒检测延迟
    is_walking_mode_ = true;
    is_damping_mode_ = false;
    global_phase_ = 0.0;
    fall_stable_count_ = 0;
    trajectory_index_ = 0;
    
    // 重置 mlp_net_ 指向 walking policy
    mlp_net_ = mlp_net_walking_.get();
    
    // 重置 current_traj_ 到默认（forward）轨迹
    if (mimic_trajs_loaded_[static_cast<int>(MimicDirection::FORWARD)]) {
      current_traj_ = &mimic_trajs_directions_[static_cast<int>(MimicDirection::FORWARD)];
    }
    
    RCLCPP_INFO(get_logger(), "Received MuJoCo reset signal, restarting walking from time: %.2f", time_);
  }
  
  // 根据重力投影方向选择 mimic Policy
  // projected_gravity 是在机体坐标系下的重力向量（归一化后约为 [gx, gy, gz]）
  // gx > 0 表示机器人前倾（重力向前投影），gx < 0 表示后倾
  // gy > 0 表示机器人左倾（重力向左投影），gy < 0 表示右倾
  MimicDirection SelectMimicDirectionFromGravity(const Eigen::Vector3d& projected_gravity) {
    double gx = projected_gravity.x();  // 前后方向
    double gy = projected_gravity.y();  // 左右方向
    
    // 计算在XY平面的角度（以+X轴为0度，逆时针为正）
    double angle = std::atan2(gy, gx);  // 范围 [-π, π]
    
    // 将角度转换为0-360度
    double angle_deg = angle * 180.0 / M_PI;
    if (angle_deg < 0) angle_deg += 360.0;
    
    // 8个方向，每个方向占45度
    // 前 (Forward): -22.5° ~ 22.5° (或 337.5° ~ 360° 和 0° ~ 22.5°)
    // 前左: 22.5° ~ 67.5°
    // 左: 67.5° ~ 112.5°
    // 后左: 112.5° ~ 157.5°
    // 后: 157.5° ~ 202.5°
    // 后右: 202.5° ~ 247.5°
    // 右: 247.5° ~ 292.5°
    // 前右: 292.5° ~ 337.5°
    
    MimicDirection dir;
    if (angle_deg >= 337.5 || angle_deg < 22.5) {
      dir = MimicDirection::FORWARD;
    } else if (angle_deg >= 22.5 && angle_deg < 67.5) {
      dir = MimicDirection::FORWARD_LEFT;
    } else if (angle_deg >= 67.5 && angle_deg < 112.5) {
      dir = MimicDirection::LEFT;
    } else if (angle_deg >= 112.5 && angle_deg < 157.5) {
      dir = MimicDirection::BACKWARD_LEFT;
    } else if (angle_deg >= 157.5 && angle_deg < 202.5) {
      dir = MimicDirection::BACKWARD;
    } else if (angle_deg >= 202.5 && angle_deg < 247.5) {
      dir = MimicDirection::BACKWARD_RIGHT;
    } else if (angle_deg >= 247.5 && angle_deg < 292.5) {
      dir = MimicDirection::RIGHT;
    } else {  // 292.5° ~ 337.5°
      dir = MimicDirection::FORWARD_RIGHT;
    }
    
    return dir;
  }
  
  // 获取当前 IMU 重力投影（使用 mimic 模式的 IMU bias）
  Eigen::Vector3d GetCurrentProjectedGravity() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      RCLCPP_WARN(get_logger(), "IMU data not available, using default gravity [0, 0, -1]");
      return Eigen::Vector3d(0, 0, -1);
    }
    
    // 应用 IMU 安装偏差校正
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    
    // 从 IMU 四元数获取旋转矩阵
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    
    // 计算校正后的旋转矩阵
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    
    // 计算重力投影（机体坐标系下的重力向量）
    Eigen::Vector3d projected_gravity = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    
    return projected_gravity;
  }
  
  // 更新 mimic Policy 选择（基于当前重力投影方向）
  void UpdateMimicPolicySelection() {
    Eigen::Vector3d projected_gravity = GetCurrentProjectedGravity();
    MimicDirection new_direction = SelectMimicDirectionFromGravity(projected_gravity);
    
    if (new_direction != current_mimic_direction_) {
      int dir_idx = static_cast<int>(new_direction);
      if (mlp_net_mimic_directions_[dir_idx]) {
        current_mimic_direction_ = new_direction;
        mlp_net_ = mlp_net_mimic_directions_[dir_idx].get();
        RCLCPP_INFO(get_logger(), "Switched mimic policy to direction: %s (gravity: [%.3f, %.3f, %.3f])",
                    kMimicDirectionNames[dir_idx].c_str(),
                    projected_gravity.x(), projected_gravity.y(), projected_gravity.z());
      } else {
        RCLCPP_WARN(get_logger(), "Mimic policy for direction %s not loaded, keeping current policy",
                    kMimicDirectionNames[dir_idx].c_str());
      }
    }
  }

  // 稳定性检测（mimic 播放中）：与 rl_dance 同逻辑，角速度+重力偏差+历史平滑
  // 返回 true 表示当前稳定，false 表示不稳定（可提前切 damping）
  bool CheckMimicStability() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu || stability_history_.empty()) return true;
    Eigen::AngleAxisd rollAngle(walking_imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(walking_imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(walking_imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
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
  
  // 摔倒检测：基于倾斜角度和角速度，带防抖机制
  // 返回 true 表示检测到摔倒，同时设置 detected_direction
  bool DetectFall(MimicDirection& detected_direction) {
    // 使用 walking 模式的 IMU bias 计算重力投影和角速度
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      fall_stable_count_ = 0;
      return false;
    }
    
    // 应用 walking 模式的 IMU 安装偏差校正
    Eigen::AngleAxisd rollAngle(walking_imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(walking_imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(walking_imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    
    // 计算角速度（机体坐标系）
    Eigen::Vector3d w_real = R_real.transpose() * R_local * 
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    
    // 计算重力投影
    Eigen::Vector3d projected_gravity = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    
    // 缓存值供后续使用
    w_real_cached_ = w_real;
    projected_gravity_cached_ = projected_gravity;
    
    const double gx = projected_gravity.x();
    const double gy = projected_gravity.y();
    const double gz = projected_gravity.z();
    const double gxy = std::hypot(gx, gy);
    
    // 如果重力投影在 XY 平面的分量太小，说明机器人接近直立，方向不可靠
    if (gxy < 1e-3) {
      fall_stable_count_ = 0;
      return false;
    }
    
    // 8方向分类：theta: 0=前(+X), 90°=左(+Y)
    double theta = std::atan2(gy, gx);  // [-π, π]
    int bin8 = static_cast<int>(std::llround(theta / (M_PI / 4.0)));
    bin8 = ((bin8 % 8) + 8) % 8;  // 归一化到 0..7
    
    // 计算倾斜角和 XY 平面角速度
    double tilt = std::acos(std::clamp(-gz, -1.0, 1.0));
    double omega_xy = std::hypot(w_real.x(), w_real.y());
    
    // 摔倒候选条件：倾斜角超阈值 或 角速度超阈值
    bool candidate = (tilt > fall_tilt_threshold_) || (omega_xy > fall_omega_threshold_);
    
    // 自适应确认帧数：快速摔倒时减少确认帧数
    int N_confirm = (omega_xy > fast_fall_omega_) ? fast_fall_confirm_frames_ : fall_confirm_frames_;
    
    // 防抖机制：需要连续 N 帧检测到相同方向
    if (candidate) {
      if (bin8 == fall_last_bin_) {
        fall_stable_count_++;
      } else {
        fall_last_bin_ = bin8;
        fall_stable_count_ = 1;
      }
    } else {
      fall_stable_count_ = 0;
    }
    
    // 检测到摔倒
    if (fall_stable_count_ >= N_confirm) {
      detected_direction = static_cast<MimicDirection>(bin8);
      
      RCLCPP_INFO(get_logger(), "Fall detected! Direction: %s (bin=%d, tilt=%.3f rad, omega_xy=%.3f rad/s)",
                  kMimicDirectionNames[bin8].c_str(), bin8, tilt, omega_xy);
      
      fall_stable_count_ = 0;  // 避免重复触发
      return true;
    }
    
    return false;
  }

  void ControlCallback() {
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
      time_ = 0.0;
      is_first_time_ = true;
      is_walking_mode_ = true;
      is_damping_mode_ = false;
      walking_start_time_ = -1.0;
      mujoco_reset_received_ = false;
      return;
    }
    
    auto joint_state = message_handler_->GetLatestJointState();
    if (!joint_state) return;

    // 仅当处于 damping 模式（mimic 轨迹已播完）且收到 reset 时，在本周期内恢复走路；
    // mimic 播放期间不强制恢复，否则会每帧被拉回 walking 再触发摔倒，轨迹永远播不完、进不了 damping
    if (mujoco_reset_received_ && !is_walking_mode_ && is_damping_mode_) {
      is_walking_mode_ = true;
      is_damping_mode_ = false;
      global_phase_ = 0.0;
      fall_stable_count_ = 0;
      trajectory_index_ = 0;
      mlp_net_ = mlp_net_walking_.get();
      if (mimic_trajs_loaded_[static_cast<int>(MimicDirection::FORWARD)]) {
        current_traj_ = &mimic_trajs_directions_[static_cast<int>(MimicDirection::FORWARD)];
      }
    }

    // 摔倒检测：在 walking 模式下，使用倾斜角+角速度+防抖机制检测摔倒
    // Reset 后需要等待 fall_detection_delay_ 秒才开始检测，让机器人先稳定
    if (is_walking_mode_ && mujoco_reset_received_ && (time_ - reset_time_) > fall_detection_delay_) {
      MimicDirection fall_direction;
      if (DetectFall(fall_direction)) {
        is_walking_mode_ = false;
        
        // 根据检测到的摔倒方向选择 mimic Policy
        current_mimic_direction_ = fall_direction;
        int dir_idx = static_cast<int>(fall_direction);
        
        if (mlp_net_mimic_directions_[dir_idx]) {
          mlp_net_ = mlp_net_mimic_directions_[dir_idx].get();
          RCLCPP_INFO(get_logger(), "Switching from walking to mimic mode at time: %.2f", time_);
          RCLCPP_INFO(get_logger(), "Selected mimic direction: %s (gravity: [%.3f, %.3f, %.3f])",
                      kMimicDirectionNames[dir_idx].c_str(),
                      projected_gravity_cached_.x(), projected_gravity_cached_.y(), projected_gravity_cached_.z());
        } else {
          // 如果对应方向的 Policy 未加载，尝试找到任何可用的 Policy
          for (size_t i = 0; i < static_cast<size_t>(MimicDirection::COUNT); ++i) {
            if (mlp_net_mimic_directions_[i]) {
              mlp_net_ = mlp_net_mimic_directions_[i].get();
              current_mimic_direction_ = static_cast<MimicDirection>(i);
              RCLCPP_WARN(get_logger(), "Policy for direction %s not available, using %s instead",
                          kMimicDirectionNames[dir_idx].c_str(), kMimicDirectionNames[i].c_str());
              break;
            }
          }
        }
        
        // 切换到对应方向的轨迹
        SwitchToDirectionalTrajectory(current_mimic_direction_);
        
        // 匹配 IMU 俯仰角度并设置 trajectory_index_
        if (observation_type_ == "mimic_future" && current_traj_ != nullptr) {
          // 切换到 mimic 模式后，使用 mimic 模式的 IMU bias 重新计算俯仰角
          double mimic_pitch = GetCurrentPitchAngle();
          size_t matched_idx = FindMatchingTrajectoryIndex(mimic_pitch);
          trajectory_index_ = matched_idx;
          RCLCPP_INFO(get_logger(), "[摔倒检测] mimic 选取第 %zu 帧 (pitch=%.4f rad, 轨迹总帧数=%ld)", 
                      trajectory_index_, mimic_pitch, current_traj_ ? static_cast<long>(current_traj_->rows()) : 0);
        } else {
          RCLCPP_WARN(get_logger(), "Cannot match pitch: observation_type=%s, current_traj_=%s", 
                      observation_type_.c_str(), current_traj_ ? "valid" : "null");
          trajectory_index_ = 0;
        }
        // 若开启了 mimic 中不稳定切 damping，进入 mimic 时重置稳定性历史，避免刚切就触发
        if (mimic_stability_to_damping_enable_) {
          stability_history_.clear();
          stability_history_.resize(std::max(1, stability_history_length_), true);
        }
      }
    }

    // mimic 播放过程中：若开启稳定性检测且检测到不稳定，提前切到 damping
    if (!is_walking_mode_ && !is_damping_mode_ && mimic_stability_to_damping_enable_) {
      if (!CheckMimicStability()) {
        is_damping_mode_ = true;
        mujoco_reset_received_ = false;
        RCLCPP_INFO(get_logger(), "[mimic] 检测到不稳定，提前进入 damping mode (time=%.2f)", time_);
      }
    }
    
    // 在 mimic 模式下，根据重力投影方向动态切换 Policy
    if (!is_walking_mode_) {
      UpdateMimicPolicySelection();
    }

    UpdateState(joint_state);
    if (is_walking_mode_) {
      CalculateObservationWalking();
    } else {
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

    if (!is_walking_mode_ && observation_type_ != "mimic_future") {
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
    }
  }
  
  void CalculateObservationWalking() {
    global_phase_ += control_dt_ / cycle_time_;
    global_phase_ -= static_cast<int>(global_phase_);

    auto imu = message_handler_->GetLatestImu();
    if (!imu) return;
    
    Eigen::AngleAxisd rollAngle(walking_imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(walking_imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(walking_imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    Eigen::Vector3d w_real = R_real.transpose() * R_local * 
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();

    Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(walking_num_observations_);
    mlp_net_observation_single <<
        (q_real_ - walking_default_joint_q_)(walking_active_joint_idx_),
        qd_real_(walking_active_joint_idx_),
        mlp_net_action_walking_,
        w_real,
        projected_gravity_real;

    mlp_net_observation_single.array() *= walking_observation_scale_.array();
    mlp_net_observation_single = mlp_net_observation_single.cwiseMax(-walking_observation_clip_).cwiseMin(walking_observation_clip_);

    if (is_first_time_) {
      is_first_time_ = false;
      mlp_net_observation_walking_.setZero(walking_num_observations_, walking_num_include_obs_steps_);
      mlp_net_action_walking_.setZero(walking_num_actions_);
      mlp_net_observation_walking_.colwise() = mlp_net_observation_single;
    } else {
      mlp_net_observation_walking_.leftCols(walking_num_include_obs_steps_ - 1) =
          mlp_net_observation_walking_.rightCols(walking_num_include_obs_steps_ - 1);
      mlp_net_observation_walking_.rightCols(1) = mlp_net_observation_single;
    }
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
      
      Eigen::VectorXd quat_error_to_store = Eigen::VectorXd::Zero(6);
      std::optional<Eigen::Quaterniond> ref_quat = std::nullopt;
      if (current_traj_ != nullptr && current_traj_->cols() == 2 * num_joints + 4) {
        size_t traj_idx = trajectory_index_;
        if (traj_idx < static_cast<size_t>(current_traj_->rows())) {
          double qx = (*current_traj_)(traj_idx, 2 * num_joints + 0);
          double qy = (*current_traj_)(traj_idx, 2 * num_joints + 1);
          double qz = (*current_traj_)(traj_idx, 2 * num_joints + 2);
          double qw = (*current_traj_)(traj_idx, 2 * num_joints + 3);
          ref_quat = Eigen::Quaterniond(qw, qx, qy, qz);
        }
      }
      
      Eigen::Quaterniond current_quat = Eigen::Quaterniond(R_real);
      if (ref_quat.has_value()) {
        if (!initial_quat_offset_computed_) {
          initial_quat_offset_ = current_quat.conjugate() * ref_quat.value();
          initial_quat_offset_computed_ = true;
        }
        Eigen::Quaterniond q_ref = initial_quat_offset_.value().conjugate() * ref_quat.value();
        Eigen::Quaterniond q_error = current_quat.conjugate() * q_ref;
        q_error.normalize();
        Eigen::Matrix3d R_error = q_error.toRotationMatrix();
        quat_error_to_store << R_error(0, 0), R_error(0, 1),
                               R_error(1, 0), R_error(1, 1),
                               R_error(2, 0), R_error(2, 1);
      }
      quat_error_history_.rightCols(1) = quat_error_to_store;
    }
    
    if (current_traj_ != nullptr) {
      const int num_future_frames = 9;
      const size_t max_idx = current_traj_->rows() - 1;
      Eigen::VectorXd goal_obs(432);
      for (int frame = 0; frame < num_future_frames; ++frame) {
        size_t traj_idx = std::min(trajectory_index_ + frame + 1, max_idx);
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
        // 轨迹播放完成，进入 damping mode；清除 reset 标志，避免本周期内误判为“刚收到 reset”而恢复走路
        is_damping_mode_ = true;
        mujoco_reset_received_ = false;
        RCLCPP_INFO(get_logger(), "[mimic] 轨迹播放完成（第 %zu 帧），进入 damping mode (time=%.2f)",
                    trajectory_index_, time_);
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
    
    if (is_walking_mode_) {
      obs = Eigen::VectorXd::Zero(walking_num_observations_ * walking_num_include_obs_steps_ + 3);
      obs.head(walking_num_observations_ * walking_num_include_obs_steps_) =
          Eigen::Map<Eigen::VectorXd>(mlp_net_observation_walking_.transpose().data(), mlp_net_observation_walking_.size());
      
      if (config_walking_["initial_velocity"]) {
        auto init_vel = config_walking_["initial_velocity"];
        if (init_vel["linear"]) {
          auto linear = init_vel["linear"];
          command_.x() = linear[0].as<double>();
          command_.y() = linear[1].as<double>();
          command_.z() = linear[2].as<double>();
        }
      }
      obs.tail(3) = command_.cwiseProduct(walking_command_scale_);
      
      if (!mlp_net_) {
        RCLCPP_ERROR(get_logger(), "MNN model not initialized");
        return;
      }
      
      try {
        mlp_net_action_walking_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "MNN inference failed: %s", e.what());
        return;
      }
      mlp_net_action_walking_ = mlp_net_action_walking_.cwiseMax(-walking_action_clip_).cwiseMin(walking_action_clip_);

      q_des_ = walking_default_joint_q_;
      q_des_(walking_active_joint_idx_) += mlp_net_action_walking_.cwiseProduct(walking_action_scale_);
      
      if (time_ < walking_transition_time_) {
        double ratio = time_ / walking_transition_time_;
        q_des_ = ratio * q_des_ + (1.0 - ratio) * initial_joint_q_;
      }
      return;
    }
    
    if (observation_type_ == "mimic_future") {
      // 构建 mimic_future 的完整 observation（根据正确的 observation 结构）
      // 结构: [proprioceptive_history (390维), quat_error_history (30维，如果启用), command (48维), future_frames (432维)]
      // 如果 use_quat_error=true: 390 + 30 + 48 + 432 = 900维
      // 如果 use_quat_error=false: 390 + 0 + 48 + 432 = 870维
      
      const int num_joints = static_cast<int>(active_joint_names_.size());
      const int proprio_dim = num_joints + num_joints + num_joints + 3 + 3;  // q_diff + qd + action + w + gravity = 78
      const int proprio_total = proprio_dim * num_include_obs_steps_;  // 78 * 5 = 390
      const int quat_error_total = current_profile_.use_quat_error ? 6 * num_include_obs_steps_ : 0;  // 6 * 5 = 30 或 0
      const int command_total = 48;  // 16步历史 × 3维命令 = 48
      const int future_frames_total = 432;  // 9帧 × 24关节 × 2(pos+vel) = 432
      
      obs = Eigen::VectorXd::Zero(proprio_total + quat_error_total + command_total + future_frames_total);
      
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
      
      // 3. Command (48维 = joint_pos 24维 + joint_vel 24维)
      // 从当前轨迹帧提取：command = [joint_pos, joint_vel]
      if (current_traj_ != nullptr && trajectory_index_ < static_cast<size_t>(current_traj_->rows())) {
        const int num_joints = static_cast<int>(active_joint_names_.size());
        // 提取当前帧的关节位置和速度
        obs.segment(offset, num_joints) = current_traj_->row(trajectory_index_).head(num_joints);  // joint_pos
        obs.segment(offset + num_joints, num_joints) = current_traj_->row(trajectory_index_).segment(num_joints, num_joints);  // joint_vel
        // 实时输出 command 里填的帧（节流：每 0.1s 打印一次，避免刷屏）
        if (time_ - last_command_frame_print_time_ >= 0.1) {
          RCLCPP_INFO(get_logger(), "[mimic] command 使用轨迹第 %zu 帧 (总帧数=%ld, time=%.2f)", 
                      trajectory_index_, static_cast<long>(current_traj_->rows()), time_);
          last_command_frame_print_time_ = time_;
        }
      } else {
        // 如果轨迹不可用，使用零向量
        obs.segment(offset, command_total).setZero();
      }
      offset += command_total;
      
      // 4. Future frames (432维 = 9帧 × 24关节 × 2(pos+vel))
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
      
      // 应用 command scaling（如果配置了）
      double command_scale = 1.0;
      if (config_["observations"] && config_["observations"]["observation_scale"] && 
          config_["observations"]["observation_scale"]["observation_scale_command"]) {
        command_scale = config_["observations"]["observation_scale"]["observation_scale_command"].as<double>();
      }
      obs.segment(proprio_total + quat_error_total, command_total) *= command_scale;
      
      // 应用 goal scaling（对于 mimic_future: pos_scale=1.0, vel_scale=0.05）
      const int num_future_frames = 9;  // 9帧，不是10帧
      const double goal_pos_scale = 1.0;
      const double goal_vel_scale = 0.05;
      int goal_start_idx = proprio_total + quat_error_total + command_total;
      for (int frame = 0; frame < num_future_frames; ++frame) {
        int frame_offset = goal_start_idx + frame * 2 * num_joints;
        // Scale position part
        obs.segment(frame_offset, num_joints) *= goal_pos_scale;
        // Scale velocity part
        obs.segment(frame_offset + num_joints, num_joints) *= goal_vel_scale;
      }
      
      // 应用 clip（与 rl_dance 一致：最后应用）
      obs = obs.cwiseMax(-observation_clip_).cwiseMin(observation_clip_);
      
      // 正确的 observation 结构：包含 command (48维) 和 future_frames (432维，9帧)
      
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
    if (observation_type_ == "mimic_future") {
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
    
    // Transition interpolation (与 rl_dance 一致)
    if (time_ < transition_time_) {
      double ratio = std::min(1.0, time_ / transition_time_);
      q_des_ = ratio * q_des_ + (1.0 - ratio) * initial_joint_q_;
    }
  }

  void ApplyTorqueLimits() {
    // 获取当前使用的 kp/kd
    Eigen::VectorXd joint_kp, joint_kd;
    if (is_damping_mode_) {
      // Damping mode: kp=0, kd=0.5
      joint_kp = Eigen::VectorXd::Zero(joint_kd_.size());
      joint_kd = Eigen::VectorXd::Constant(joint_kd_.size(), 0.5);
    } else if (is_walking_mode_) {
      joint_kp = walking_joint_kp_;
      joint_kd = walking_joint_kd_;
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
      // Damping mode: kp=0, kd=0.5
      joint_command_->stiffness = std::vector<double>(q_des_.size(), 0.0);
      joint_command_->damping = std::vector<double>(q_des_.size(), 0.5);
    } else if (is_walking_mode_) {
      joint_command_->stiffness = std::vector<double>(walking_joint_kp_.data(), walking_joint_kp_.data() + walking_joint_kp_.size());
      joint_command_->damping = std::vector<double>(walking_joint_kd_.data(), walking_joint_kd_.data() + walking_joint_kd_.size());
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

  // MNN model
  std::unique_ptr<math::MnnModel> mlp_net_walking_;
  std::array<std::unique_ptr<math::MnnModel>, 8> mlp_net_mimic_directions_;  // 8个方向的 mimic Policy
  math::MnnModel* mlp_net_;
  MimicDirection current_mimic_direction_ = MimicDirection::FORWARD;  // 当前选择的 mimic 方向
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::MatrixXd mlp_net_observation_walking_;
  Eigen::VectorXd mlp_net_action_;

  // ObservationManager 暂时不使用（需要 RlDanceParam）
  
  // CSV 轨迹数据
  Eigen::MatrixXd* current_traj_ = nullptr;
  Eigen::MatrixXd* current_traj_base_vel_ = nullptr;  // 基座速度轨迹 (N × 3: vx, vy, wz)
  std::map<std::string, Eigen::MatrixXd> interpolated_trajs_;
  std::map<std::string, Eigen::MatrixXd> interpolated_base_vel_trajs_;  // 基座速度轨迹
  std::array<Eigen::MatrixXd, 8> mimic_trajs_directions_;  // 8个方向的轨迹
  std::array<bool, 8> mimic_trajs_loaded_ = {false};  // 轨迹是否已加载
  size_t trajectory_index_ = 0;  // 当前轨迹索引
  double last_command_frame_print_time_ = -1.0;  // 上次打印 command 帧的时间（节流用）
  
  // Proprioceptive history buffers (5步历史)
  Eigen::MatrixXd q_diff_history_;      // q_actual - default_joint_q (24维 × 5步)
  Eigen::MatrixXd qd_history_;          // 关节速度 (24维 × 5步)
  Eigen::MatrixXd action_history_;      // 动作历史 (24维 × 5步)
  Eigen::MatrixXd w_history_;           // 角速度历史 (3维 × 5步)
  Eigen::MatrixXd gravity_history_;     // 重力历史 (3维 × 5步)
  Eigen::MatrixXd quat_error_history_; // 四元数误差历史 (6维 × 5步，当 use_quat_error 启用时)
  
  // Goal observation buffer (mimic_future: 432维 × 1步) - 9帧，不是10帧
  Eigen::MatrixXd goal_buffer_;         // 目标观察缓冲区
  
  // Quaternion error calculation state
  std::optional<Eigen::Quaterniond> initial_quat_offset_;  // 初始四元数偏移
  bool initial_quat_offset_computed_ = false;  // 是否已计算初始偏移

  // State variables
  double time_;
  double global_phase_;
  bool is_first_time_;
  bool is_walking_mode_;
  bool is_damping_mode_ = false;  // damping 模式：mimic 播放完后进入
  double walking_duration_;
  double walking_start_time_;
  bool mujoco_reset_received_;
  Eigen::Vector3d command_;
  
  // Fall detection state (cached from CalculateObservation for reuse)
  Eigen::Vector3d w_real_cached_;           // 缓存的角速度（机体坐标系）
  Eigen::Vector3d projected_gravity_cached_; // 缓存的重力投影
  int fall_last_bin_ = 0;                   // 上一帧的方向 bin
  int fall_stable_count_ = 0;               // 连续稳定帧计数
  
  // Fall detection parameters
  double fall_tilt_threshold_ = 0.5;        // 倾斜角阈值 (rad)
  double fall_omega_threshold_ = 1.8;       // 角速度阈值 (rad/s)
  int fall_confirm_frames_ = 8;             // 确认帧数
  double fast_fall_omega_ = 2.5;            // 快速摔倒角速度阈值 (rad/s)
  int fast_fall_confirm_frames_ = 2;        // 快速摔倒确认帧数
  double fall_detection_delay_ = 1.0;       // Reset 后摔倒检测延迟时间 (秒)
  double reset_time_ = -100.0;              // 上次 reset 时间

  // 稳定性检测（mimic 中不稳定则提前切 damping，可选）
  bool mimic_stability_to_damping_enable_ = false;
  double ang_vel_threshold_ = 0.4;
  double gravity_dev_threshold_ = 0.15;
  double stability_smoothing_threshold_ = 0.6;
  int stability_history_length_ = 10;
  std::deque<bool> stability_history_;
  
  // MuJoCo reset subscription
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr mujoco_reset_sub_;
  
  // Walking mode parameters
  YAML::Node config_walking_;
  std::string walking_policy_file_;
  int walking_num_observations_;
  int walking_num_include_obs_steps_;
  int walking_num_actions_;
  Eigen::VectorXi walking_active_joint_idx_;
  Eigen::VectorXd walking_observation_scale_;
  Eigen::Vector3d walking_command_scale_;
  Eigen::VectorXd walking_action_scale_;
  Eigen::VectorXd walking_default_joint_q_;
  double walking_action_clip_;
  double walking_observation_clip_;
  double walking_transition_time_;
  Eigen::Vector3d walking_imu_install_bias_;
  Eigen::VectorXd walking_joint_kp_;   // walking 模式的 kp（从 XZL 配置加载）
  Eigen::VectorXd walking_joint_kd_;   // walking 模式的 kd（从 XZL 配置加载）
  Eigen::VectorXd mlp_net_action_walking_;
  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
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
    RCLCPP_ERROR(rclcpp::get_logger("rl_basic_example_CHR"), "Usage: rl_basic_example_CHR <config_file_dir> [config_file_name]");
    return 1;
  }

  std::string config_file_name = "rl_basic_param_CHR.yaml";
  if (argc >= 3) {
    config_file_name = argv[2];
  }

  auto node = std::make_shared<example::RlBasicRunnerCHR>(argv[1], config_file_name);
  if (!node->Initialize()) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
