#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <limits>
#include <yaml-cpp/yaml.h>

#include "components/message_handler.hpp"
#include "math/concatenate_vector.h"
#include "math/mnn_model.h"
#include "math/rotation_matrix.h"
#include "parameter/rl_basic_param.h"

using namespace std::chrono_literals;

namespace example {

class RlBasicRunnerAMP : public rclcpp::Node {
 public:
  explicit RlBasicRunnerAMP(const std::string& config_file_dir, const std::string& config_file_name = "rl_basic_param.yaml") : Node("rl_basic_runner_AMP") {
    std::string config_file = config_file_dir + "/" + config_file_name;
    RCLCPP_INFO(get_logger(), "Loading config file: %s", config_file.c_str());
    param_ = std::make_shared<RlBasicParam>(config_file);
    config_file_dir_ = config_file_dir;
    config_file_ = config_file;
    joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();

    // 加载扭矩限制参数
    LoadTorqueLimitParameters();
  }
  
  void LoadTorqueLimitParameters() {
    try {
      YAML::Node config = YAML::LoadFile(config_file_);
      
      // 读取扭矩限制参数
      torque_limit_enabled_ = config["torque_limit"] ? config["torque_limit"].as<bool>() : false;
      if (config["max_torque_joint"]) {
        max_torque_joint_ = LoadVectorArrayFromYaml(config["max_torque_joint"]);
        RCLCPP_INFO(get_logger(), "Torque limit enabled: %s, loaded %zu groups of max_torque_joint", 
                    torque_limit_enabled_ ? "true" : "false", max_torque_joint_.size());
      } else {
        RCLCPP_WARN(get_logger(), "max_torque_joint not found in config, torque limit will be disabled");
      }
      // 读取 soft_torque_limit（软扭矩限制系数，默认0.9）
      soft_torque_limit_ = config["soft_torque_limit"] ? config["soft_torque_limit"].as<double>() : 0.9;
      if (torque_limit_enabled_) {
        RCLCPP_INFO(get_logger(), "Soft torque limit: %.2f", soft_torque_limit_);
      }
      // 读取摔倒策略相关
      enable_falling_switch_ = config["enable_falling_switch"] ? config["enable_falling_switch"].as<bool>() : true;
      falling_detect_after_sec_ = config["falling_detect_after_sec"] ? config["falling_detect_after_sec"].as<double>() : 5.0;
      falling_switch_ = config["falling_switch"] ? config["falling_switch"].as<std::string>() : "damping";
      if (falling_switch_ != "none" && falling_switch_ != "damping" && falling_switch_ != "pdstand") {
        RCLCPP_WARN(get_logger(), "falling_switch '%s' unknown, use damping", falling_switch_.c_str());
        falling_switch_ = "damping";
      }
      RCLCPP_INFO(get_logger(), "Enable falling switch: %s, detect after %.1fs, strategy: %s",
                  enable_falling_switch_ ? "enabled" : "disabled", falling_detect_after_sec_, falling_switch_.c_str());
      // 摔倒检测参数（与 CHR 一致：tilt/omega）
      if (config["fall_detection"]) {
        const YAML::Node& fc = config["fall_detection"];
        if (fc["tilt_threshold"]) fall_tilt_threshold_ = fc["tilt_threshold"].as<double>();
        if (fc["omega_threshold"]) fall_omega_threshold_ = fc["omega_threshold"].as<double>();
        if (fc["confirm_frames"]) fall_confirm_frames_ = fc["confirm_frames"].as<int>();
        if (fc["fast_fall_omega"]) fast_fall_omega_ = fc["fast_fall_omega"].as<double>();
        if (fc["fast_fall_confirm_frames"]) fast_fall_confirm_frames_ = fc["fast_fall_confirm_frames"].as<int>();
        RCLCPP_INFO(get_logger(), "Fall detection: tilt=%.2f rad, omega=%.2f rad/s, confirm=%d",
                    fall_tilt_threshold_, fall_omega_threshold_, fall_confirm_frames_);
      }
      // passive 模式 kd（fall_detection.passive_damping）
      if (config["fall_detection"] && config["fall_detection"]["passive_damping"]) {
        passive_damping_kd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(config["fall_detection"]["passive_damping"]));
        RCLCPP_INFO(get_logger(), "Loaded passive_damping: %zd joints", passive_damping_kd_.size());
      } else {
        passive_damping_kd_ = Eigen::VectorXd::Zero(24);
        passive_damping_kd_.setConstant(0.5);  // 默认 0.5
      }
      // 定时 pdstand 或 摔倒 pdstand 时加载 falling_pd_stand 参数
      enable_pdstand_switch_ = config["enable_pdstand_switch"] ? config["enable_pdstand_switch"].as<bool>() : false;
      pdstand_after_sec_ = config["pdstand_after_sec"] ? config["pdstand_after_sec"].as<double>() : 1.0;
      RCLCPP_INFO(get_logger(), "Enable pdstand switch: %s, after %.1fs", enable_pdstand_switch_ ? "enabled" : "disabled", pdstand_after_sec_);
      if ((enable_pdstand_switch_ || falling_switch_ == "pdstand") && config["falling_pd_stand"]) {
        const YAML::Node& pd = config["falling_pd_stand"];
        if (pd["desired_joint_position"]) {
          q_pd_des_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(pd["desired_joint_position"]));
          kp_pd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(pd["stiffness"]));
          kd_pd_ = math::ConcatenateVectors(LoadVectorArrayFromYaml(pd["damping"]));
          pd_stand_duration_ = pd["duration"] ? pd["duration"].as<double>() : 3.0;
          pd_stand_loaded_ = true;
          RCLCPP_INFO(get_logger(), "Loaded falling_pd_stand: %zd joints, duration %.2fs", q_pd_des_.size(), pd_stand_duration_);
        } else {
          RCLCPP_WARN(get_logger(), "falling_pd_stand.desired_joint_position not found, pdstand disabled");
          pd_stand_loaded_ = false;
        }
      } else {
        pd_stand_loaded_ = false;
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Failed to load torque limit parameters: %s", e.what());
      torque_limit_enabled_ = false;
    }
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

  void LoadWalkingParameters() {
    const std::string walking_config_file = config_file_dir_ + "/rl_basic_param_XZL.yaml";
    config_walking_ = YAML::LoadFile(walking_config_file);

    walking_policy_file_ = config_walking_["policy_file"] ? config_walking_["policy_file"].as<std::string>() : "";
    walking_num_observations_ = config_walking_["num_observations"] ? config_walking_["num_observations"].as<int>() : 72;
    walking_num_include_obs_steps_ = config_walking_["num_include_obs_steps"] ? config_walking_["num_include_obs_steps"].as<int>() : 15;
    walking_num_actions_ = config_walking_["active_joint_names"] ? config_walking_["active_joint_names"].size() : 22;

    if (config_walking_["active_joint_idx"]) {
      walking_active_joint_idx_ = LoadIntVectorFromYaml(config_walking_["active_joint_idx"]);
    } else {
      walking_active_joint_idx_ = Eigen::VectorXi::LinSpaced(walking_num_actions_, 0, walking_num_actions_ - 1);
    }

    const double obs_scale_linear_vel =
        config_walking_["observation_scale_linear_vel"] ? config_walking_["observation_scale_linear_vel"].as<double>() : 2.0;
    const double obs_scale_angular_vel =
        config_walking_["observation_scale_angular_vel"] ? config_walking_["observation_scale_angular_vel"].as<double>() : 1.0;
    const double obs_scale_dof_pos =
        config_walking_["observation_scale_dof_pos"] ? config_walking_["observation_scale_dof_pos"].as<double>() : 1.0;
    const double obs_scale_dof_vel =
        config_walking_["observation_scale_dof_vel"] ? config_walking_["observation_scale_dof_vel"].as<double>() : 0.05;
    const double obs_scale_quat =
        config_walking_["observation_scale_quat"] ? config_walking_["observation_scale_quat"].as<double>() : 1.0;
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

    walking_action_scale_ = config_walking_["action_scale"] ?
        math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["action_scale"])) :
        Eigen::VectorXd::Ones(walking_num_actions_);
    walking_default_joint_q_ = config_walking_["default_joint_q"] ?
        math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["default_joint_q"])) :
        default_joint_q_;
    walking_joint_kp_ = config_walking_["joint_kp"] ?
        math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["joint_kp"])) :
        joint_kp_;
    walking_joint_kd_ = config_walking_["joint_kd"] ?
        math::ConcatenateVectors(LoadVectorArrayFromYaml(config_walking_["joint_kd"])) :
        joint_kd_;
    walking_action_clip_ = config_walking_["action_clip"] ? config_walking_["action_clip"].as<double>() : 100.0;
    walking_observation_clip_ = config_walking_["observation_clip"] ? config_walking_["observation_clip"].as<double>() : 100.0;
    walking_transition_time_ = config_walking_["transition_time"] ? config_walking_["transition_time"].as<double>() : 0.5;
    walking_cycle_time_ = config_walking_["cycle_time"] ? config_walking_["cycle_time"].as<double>() : 0.8;

    if (config_walking_["imu_install_bias"]) {
      walking_imu_install_bias_ = LoadVectorFromYaml(config_walking_["imu_install_bias"]);
    } else {
      walking_imu_install_bias_ = imu_install_bias_;
    }

    walking_initial_command_.setZero();
    if (config_walking_["initial_velocity"] && config_walking_["initial_velocity"]["linear"]) {
      auto linear = config_walking_["initial_velocity"]["linear"];
      walking_initial_command_ = Eigen::Vector3d(linear[0].as<double>(), linear[1].as<double>(), linear[2].as<double>());
    }
  }

  void MujocoResetCallback(const std_msgs::msg::Empty::SharedPtr) {
    if (!is_walking_mode_) {
      PublishPolicySwitch("amp", "walking");
    }
    is_walking_mode_ = true;
    mujoco_reset_received_ = true;
    // 与 CHR/XZL 一致：episode 时间清零，否则 time_ 已很大时 walking 过渡 (walking_transition_time_) 不再生效
    time_ = 0.0;
    reset_time_ = 0.0;
    fall_stable_count_ = 0;
    global_phase_ = 0.0;
    walking_is_first_time_ = true;
    amp_is_first_time_ = true;
    mlp_net_ = mlp_net_walking_.get();
    mlp_net_action_walking_.setZero(walking_num_actions_);
    mlp_net_action_.setZero(param_->num_actions);
  }

  bool Initialize() {
    try {
      // Initialize message handler
      message_handler_ = std::make_shared<MessageHandler>(shared_from_this());
      message_handler_->Initialize();
      // Wait for first motion state
      while (!message_handler_->GetLatestMotionState() ||
             message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for joint bridge state...");
      }
      RCLCPP_INFO(get_logger(), "Already in joint bridge state");
      // Get initial joint positions
      auto initial_state = message_handler_->GetLatestJointState();
      if (!initial_state) {
        RCLCPP_ERROR(get_logger(), "Failed to get initial joint state");
        return false;
      }

      initial_joint_q_ =
          Eigen::Map<const Eigen::VectorXd>(initial_state->position.data(), initial_state->position.size());

      RCLCPP_INFO_STREAM(get_logger(), "Initial joint positions: " << initial_joint_q_.transpose());

      // Initialize active joint indices
      active_joint_idx_ = param_->active_joint_idx;
      // Concatenate joint parameters from yaml
      default_joint_q_ = math::ConcatenateVectors(param_->default_joint_q);
      joint_kp_ = math::ConcatenateVectors(param_->joint_kp);
      joint_kd_ = math::ConcatenateVectors(param_->joint_kd);
      action_scale_ = math::ConcatenateVectors(param_->action_scale);
      imu_install_bias_ = param_->imu_install_bias;
      LoadWalkingParameters();
      
      // Initialize MNN models
      mlp_net_amp_ = std::make_unique<math::MnnModel>(config_file_dir_ + "/" + param_->policy_file);
      mlp_net_walking_ = std::make_unique<math::MnnModel>(config_file_dir_ + "/" + walking_policy_file_);
      mlp_net_ = mlp_net_walking_.get();
      mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
      mlp_net_observation_walking_.setZero(walking_num_observations_, walking_num_include_obs_steps_);
      mlp_net_action_.setZero(param_->num_actions);
      mlp_net_action_walking_.setZero(walking_num_actions_);
      observation_scale_ = BuildPolicyObservationScale();

      // Initialize control variables
      time_ = 0.0;
      global_phase_ = 0.0;
      walking_is_first_time_ = true;
      amp_is_first_time_ = true;
      is_walking_mode_ = true;
      mujoco_reset_received_ = false;
      reset_time_ = -100.0;

      mujoco_reset_sub_ = create_subscription<std_msgs::msg::Empty>(
          "/mujoco/reset_complete", 10,
          std::bind(&RlBasicRunnerAMP::MujocoResetCallback, this, std::placeholders::_1));
      policy_switch_pub_ = create_publisher<std_msgs::msg::String>("/rl/policy_switch", 10);

      RCLCPP_INFO(get_logger(), "Starting control loop");
      
      // Create control timer
      RCLCPP_INFO(get_logger(), "Creating control timer with dt: %f", param_->control_dt);
      control_timer_ = create_wall_timer(std::chrono::duration<double>(param_->control_dt),
                                         std::bind(&RlBasicRunnerAMP::ControlCallback, this));
      RCLCPP_INFO(get_logger(), "Control timer created successfully");

      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
      return false;
    }
  }

 private:
  void ControlCallback() {
    RCLCPP_DEBUG(get_logger(), "ControlCallback called");
    
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
      time_ = 0.0;
      walking_is_first_time_ = true;
      amp_is_first_time_ = true;
      is_walking_mode_ = true;
      mlp_net_ = mlp_net_walking_.get();
      return;
    }
    
    auto joint_state = message_handler_->GetLatestJointState();
    if (!joint_state) {
      RCLCPP_DEBUG(get_logger(), "No joint state received yet, skipping");
      return;  // Skip if no joint state received yet
    }

    RCLCPP_DEBUG(get_logger(), "Updating state");
    UpdateState(joint_state);

    // MuJoCo reset 后 walking_is_first_time_=true：下一帧用当前 keyframe 关节作为过渡起点（initial_joint_q_ 仅 Initialize 时采一次会偏）
    if (walking_is_first_time_) {
      if (q_real_.size() == initial_joint_q_.size()) {
        initial_joint_q_ = q_real_;
      }
    }

    if (is_walking_mode_ && enable_falling_switch_ && mujoco_reset_received_ &&
        (time_ - reset_time_) >= falling_detect_after_sec_ && DetectFall()) {
      is_walking_mode_ = false;
      amp_is_first_time_ = true;
      mlp_net_action_.setZero(param_->num_actions);
      mlp_net_ = mlp_net_amp_.get();
      PublishPolicySwitch("walking", "amp");
      RCLCPP_INFO(get_logger(), "Switching from walking to AMP fall policy at %.2fs", time_);
    }
    
    RCLCPP_DEBUG(get_logger(), "Calculating observation");
    if (is_walking_mode_) {
      CalculateObservationWalking();
    } else {
      CalculateObservationAmp();
    }
    
    RCLCPP_DEBUG(get_logger(), "Calculating motor command");
    CalculateMotorCommand();
    
    // 应用扭矩限制（如果启用）
    if (torque_limit_enabled_) {
      ApplyTorqueLimits();
    }
    
    RCLCPP_DEBUG(get_logger(), "Sending motor command");
    SendMotorCommand();

    time_ += param_->control_dt;
  }

  void UpdateState(const interface_protocol::msg::JointState::SharedPtr& joint_state) {
    // 添加边界检查
    if (!joint_state || joint_state->position.empty() || joint_state->velocity.empty()) {
      RCLCPP_WARN(get_logger(), "Invalid joint state received, skipping update");
      return;
    }
    
    // 检查数据大小是否匹配
    if (joint_state->position.size() != joint_state->velocity.size()) {
      RCLCPP_ERROR(get_logger(), "Position and velocity size mismatch: %zu vs %zu", 
                   joint_state->position.size(), joint_state->velocity.size());
      return;
    }
    
    // Update joint states
    q_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size());
    qd_real_ = Eigen::Map<const Eigen::VectorXd>(joint_state->velocity.data(), joint_state->velocity.size());

  }

  bool ComputeBaseState(const Eigen::Vector3d& install_bias, Eigen::Vector3d& w_real,
                        Eigen::Vector3d& projected_gravity_real) {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) return false;

    Eigen::AngleAxisd rollAngle(install_bias.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(install_bias.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(install_bias.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();

    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    w_real = R_real.transpose() * R_local *
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();
    return true;
  }

  void CalculateObservationWalking() {
    global_phase_ += param_->control_dt / walking_cycle_time_;
    global_phase_ -= static_cast<int>(global_phase_);

    Eigen::Vector3d w_real;
    Eigen::Vector3d projected_gravity_real;
    if (!ComputeBaseState(walking_imu_install_bias_, w_real, projected_gravity_real)) {
      return;
    }

    Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(walking_num_observations_);
    mlp_net_observation_single <<
        (q_real_ - walking_default_joint_q_)(walking_active_joint_idx_),
        qd_real_(walking_active_joint_idx_),
        mlp_net_action_walking_,
        w_real,
        projected_gravity_real;

    mlp_net_observation_single.array() *= walking_observation_scale_.array();
    mlp_net_observation_single =
        mlp_net_observation_single.cwiseMax(-walking_observation_clip_).cwiseMin(walking_observation_clip_);

    if (walking_is_first_time_) {
      walking_is_first_time_ = false;
      mlp_net_observation_walking_.setZero(walking_num_observations_, walking_num_include_obs_steps_);
      mlp_net_action_walking_.setZero(walking_num_actions_);
      mlp_net_observation_walking_.colwise() = mlp_net_observation_single;
    } else {
      mlp_net_observation_walking_.leftCols(walking_num_include_obs_steps_ - 1) =
          mlp_net_observation_walking_.rightCols(walking_num_include_obs_steps_ - 1);
      mlp_net_observation_walking_.rightCols(1) = mlp_net_observation_single;
    }
  }

  void CalculateObservationAmp() {
    Eigen::Vector3d w_real;
    Eigen::Vector3d projected_gravity_real;
    if (!ComputeBaseState(imu_install_bias_, w_real, projected_gravity_real)) {
      return;
    }

    // Fall policy observation order must match mjlab/tasks/fall/fall_env_cfg.py:
    // [base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action]
    Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(param_->num_observations);
    mlp_net_observation_single <<
        w_real,                                           //  base angular velocity in base frame: 3
        projected_gravity_real,                           //  projected gravity in base frame: 3
        (q_real_ - default_joint_q_)(active_joint_idx_),  //  joint position - joint default position: kDoFs
        qd_real_(active_joint_idx_),                      //  joint velocity: kDoFs
        mlp_net_action_;                                  //  previous raw policy action: kDoFs

    // Scales and clips the observation
    mlp_net_observation_single.array() *= observation_scale_.array();
    mlp_net_observation_single =
        mlp_net_observation_single.cwiseMax(-param_->observation_clip).cwiseMin(param_->observation_clip);

    // Update the observation buffer
    if (amp_is_first_time_) {
      amp_is_first_time_ = false;
      mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
      mlp_net_action_.setZero(param_->num_actions);
      mlp_net_observation_.colwise() = mlp_net_observation_single;
    } else {
      mlp_net_observation_.leftCols(param_->num_include_obs_steps - 1) =
          mlp_net_observation_.rightCols(param_->num_include_obs_steps - 1);
      mlp_net_observation_.rightCols(1) = mlp_net_observation_single;
    }
  }

  void CalculateMotorCommand() {
    if (!mlp_net_) {
      RCLCPP_ERROR(get_logger(), "MNN model not initialized");
      return;
    }

    if (is_walking_mode_) {
      Eigen::VectorXd obs = Eigen::VectorXd::Zero(walking_num_observations_ * walking_num_include_obs_steps_ + 3);
      obs.head(walking_num_observations_ * walking_num_include_obs_steps_) =
          Eigen::Map<Eigen::VectorXd>(mlp_net_observation_walking_.transpose().data(), mlp_net_observation_walking_.size());
      obs.tail(3) = walking_initial_command_.cwiseProduct(walking_command_scale_);

      try {
        mlp_net_action_walking_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Walking MNN inference failed: %s", e.what());
        return;
      }
      mlp_net_action_walking_ =
          mlp_net_action_walking_.cwiseMax(-walking_action_clip_).cwiseMin(walking_action_clip_);

      q_des_ = walking_default_joint_q_;
      q_des_(walking_active_joint_idx_) += mlp_net_action_walking_.cwiseProduct(walking_action_scale_);
      if (time_ < walking_transition_time_) {
        double ratio = time_ / walking_transition_time_;
        q_des_ = ratio * q_des_ + (1.0 - ratio) * initial_joint_q_;
      }
      return;
    }

    Eigen::VectorXd obs = Eigen::VectorXd::Zero(param_->num_observations * param_->num_include_obs_steps);
    obs.head(param_->num_observations * param_->num_include_obs_steps) =
        Eigen::Map<Eigen::VectorXd>(mlp_net_observation_.transpose().data(), mlp_net_observation_.size());

    try {
      mlp_net_action_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "AMP MNN inference failed: %s", e.what());
      return;
    }
    mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

    q_des_ = default_joint_q_;
    q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);
  }

  Eigen::VectorXd BuildPolicyObservationScale() const {
    Eigen::VectorXd scale(param_->num_observations);
    scale <<
        Eigen::Vector3d::Constant(param_->observation_scale_angular_vel),
        Eigen::Vector3d::Constant(param_->observation_scale_quat),
        Eigen::VectorXd::Constant(param_->num_actions, param_->observation_scale_dof_pos),
        Eigen::VectorXd::Constant(param_->num_actions, param_->observation_scale_dof_vel),
        Eigen::VectorXd::Ones(param_->num_actions);
    return scale;
  }

  // 摔倒检测（与 CHR 一致：tilt + omega + 防抖）
  bool DetectFall() {
    Eigen::Vector3d w_real;
    Eigen::Vector3d projected_gravity;
    const Eigen::Vector3d& install_bias = is_walking_mode_ ? walking_imu_install_bias_ : imu_install_bias_;
    if (!ComputeBaseState(install_bias, w_real, projected_gravity)) return false;
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
      RCLCPP_INFO(get_logger(), "Fall detected! tilt=%.3f rad, omega_xy=%.3f rad/s", tilt, omega_xy);
      fall_stable_count_ = 0;
      return true;
    }
    return false;
  }

  // 获取当前 IMU 俯仰角度（弧度），保留供其他用途
  double GetCurrentPitchAngle() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      RCLCPP_WARN(get_logger(), "IMU data not available, using default pitch angle 0.0");
      return 0.0;
    }
    const Eigen::Vector3d& install_bias = is_walking_mode_ ? walking_imu_install_bias_ : imu_install_bias_;
    Eigen::AngleAxisd rollAngle(install_bias.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(install_bias.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(install_bias.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x,
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    Eigen::Vector3d rpy = math::CalcRollPitchYawFromRotationMatrix(R_real);
    return rpy[1];  // pitch
  }

  void ApplyTorqueLimits() {
    const int n = q_des_.size();
    const Eigen::VectorXd& joint_kp = is_walking_mode_ ? walking_joint_kp_ : joint_kp_;
    const Eigen::VectorXd& joint_kd = is_walking_mode_ ? walking_joint_kd_ : joint_kd_;
    if (q_real_.size() != n || qd_real_.size() != n || joint_kp.size() != n || joint_kd.size() != n) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                          "Size mismatch in ApplyTorqueLimits, skipping torque limit");
      return;
    }
    
    // ---- 构造每关节扭矩上限向量 tau_limit_（参考XZL真实实现）----
    // 在真实部署代码中，tau_limit_ 从 data_store_->joint_info.GetTorqueLimit() 获取（来自 URDF）
    // 在这个代码库中，我们使用 max_torque_joint 作为基础（相当于从 URDF 获取的值）
    Eigen::VectorXd tau_limit_ = Eigen::VectorXd::Zero(n);
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
        tau_limit_.segment(idx, gsz) = group;
        idx += gsz;
        if (idx >= n) break;
      }
      if (idx != n) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                            "max_torque_joint total size (%d) does not match DOF count (%d)", idx, n);
      }
      for (int i = 0; i < n; ++i) {
        if (tau_limit_(i) <= 0.0) {
          tau_limit_(i) = std::numeric_limits<double>::infinity();
        }
      }
    } else {
      // 如果没有配置，使用无穷大（不限制）
      tau_limit_ = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                          "max_torque_joint is empty, torque limit disabled");
      return;  // 如果没有配置，直接返回
    }
    
    // ---- 应用 soft_torque_limit（参考XZL真实实现）----
    // 在真实部署代码中：tau_limit_ = tau_limit_.cwiseProduct(soft_torque_limit)
    // 这里 soft_torque_limit 是标量，所以直接相乘
    tau_limit_ = tau_limit_ * soft_torque_limit_;
    
    // ---- 通过限制 q_des_ 范围确保扭矩不超过限制（参考XZL真实实现）----
    // 这是 XZL 的核心实现方式：不计算 PD 扭矩，直接通过限制 q_des_ 范围来限制扭矩
    const double kp_min_threshold = 1e-3;
    const Eigen::VectorXd joint_kp_safe = joint_kp.cwiseMax(kp_min_threshold);
    Eigen::VectorXd q_des_lb = (-tau_limit_.array() + (joint_kd.array() * qd_real_.array())).matrix();
    Eigen::VectorXd q_des_ub = (tau_limit_.array() + (joint_kd.array() * qd_real_.array())).matrix();
    q_des_lb = q_des_lb.array() / joint_kp_safe.array();
    q_des_ub = q_des_ub.array() / joint_kp_safe.array();
    q_des_lb += q_real_;
    q_des_ub += q_real_;
    q_des_ = q_des_.cwiseMax(q_des_lb).cwiseMin(q_des_ub);
  }

  // 五阶插值：phase in [0, duration]，输出 q_cmd, qd_cmd（与 pd_stand_runner 一致）
  void QuinticInterpolate(const Eigen::VectorXd& q_init, const Eigen::VectorXd& q_des,
                          double duration, double phase, Eigen::VectorXd& q_cmd, Eigen::VectorXd& qd_cmd) {
    const int n = static_cast<int>(q_init.size());
    if (q_des.size() != n || duration <= 0) {
      q_cmd = q_init;
      qd_cmd.setZero(n);
      return;
    }
    double s = std::min(phase / duration, 1.0);
    double f = 10.0 * std::pow(s, 3) - 15.0 * std::pow(s, 4) + 6.0 * std::pow(s, 5);
    double df_ds = 30.0 * s * s - 60.0 * std::pow(s, 3) + 30.0 * std::pow(s, 4);
    q_cmd = q_init + (q_des - q_init) * f;
    qd_cmd = (q_des - q_init) * (df_ds / duration);
  }

  void SendMotorCommand() {
    const int n = static_cast<int>(q_des_.size());
    joint_command_->position = std::vector<double>(q_des_.data(), q_des_.data() + n);
    joint_command_->velocity = std::vector<double>(n, 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(n, 0.0);
    joint_command_->torque = std::vector<double>(n, 0.0);
    if (is_walking_mode_) {
      joint_command_->stiffness = std::vector<double>(walking_joint_kp_.data(), walking_joint_kp_.data() + walking_joint_kp_.size());
      joint_command_->damping = std::vector<double>(walking_joint_kd_.data(), walking_joint_kd_.data() + walking_joint_kd_.size());
    } else {
      joint_command_->stiffness = std::vector<double>(joint_kp_.data(), joint_kp_.data() + joint_kp_.size());
      joint_command_->damping = std::vector<double>(joint_kd_.data(), joint_kd_.data() + joint_kd_.size());
    }

    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    message_handler_->PublishJointCommand(*joint_command_);
  }

  // Parameters
  std::shared_ptr<RlBasicParam> param_;

  // Message handling
  std::shared_ptr<MessageHandler> message_handler_;

  // MNN model
  std::unique_ptr<math::MnnModel> mlp_net_amp_;
  std::unique_ptr<math::MnnModel> mlp_net_walking_;
  math::MnnModel* mlp_net_ = nullptr;
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::MatrixXd mlp_net_observation_walking_;
  Eigen::VectorXd mlp_net_action_;
  Eigen::VectorXd mlp_net_action_walking_;
  Eigen::VectorXd observation_scale_;

  // State variables
  double time_;
  double global_phase_;
  bool amp_is_first_time_ = true;
  bool walking_is_first_time_ = true;
  bool is_walking_mode_ = true;
  bool mujoco_reset_received_ = false;
  double reset_time_ = -100.0;
  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;
  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d walking_imu_install_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d walking_initial_command_ = Eigen::Vector3d::Zero();

  // ROS timer
  rclcpp::TimerBase::SharedPtr control_timer_;
  std::string config_file_dir_;
  std::string config_file_;  // 保存配置文件路径，用于读取扭矩限制参数
  interface_protocol::msg::JointCommand::SharedPtr joint_command_;

  // Walking policy parameters
  YAML::Node config_walking_;
  std::string walking_policy_file_;
  int walking_num_observations_ = 72;
  int walking_num_include_obs_steps_ = 15;
  int walking_num_actions_ = 22;
  double walking_action_clip_ = 100.0;
  double walking_observation_clip_ = 100.0;
  double walking_transition_time_ = 0.5;
  double walking_cycle_time_ = 0.8;
  Eigen::VectorXi walking_active_joint_idx_;
  Eigen::VectorXd walking_observation_scale_;
  Eigen::VectorXd walking_action_scale_;
  Eigen::VectorXd walking_default_joint_q_;
  Eigen::VectorXd walking_joint_kp_;
  Eigen::VectorXd walking_joint_kd_;
  Eigen::Vector3d walking_command_scale_ = Eigen::Vector3d::Zero();
  
  // 扭矩限制参数
  bool torque_limit_enabled_ = false;
  std::vector<Eigen::VectorXd> max_torque_joint_;
  double soft_torque_limit_ = 0.9;  // 软扭矩限制系数，默认0.9（参考XZL实现）

  // 摔倒策略（YAML enable_falling_switch, falling_detect_after_sec, falling_switch, fall_detection）
  bool enable_falling_switch_ = true;
  double falling_detect_after_sec_ = 5.0;
  std::string falling_switch_ = "damping";  // none / damping / pdstand
  // 摔倒检测（与 CHR 一致）
  double fall_tilt_threshold_ = 0.5;
  double fall_omega_threshold_ = 1.8;
  int fall_confirm_frames_ = 8;
  double fast_fall_omega_ = 2.5;
  int fast_fall_confirm_frames_ = 2;
  int fall_stable_count_ = 0;
  bool in_damping_fall_ = false;          // 已进入 damping 模式，保持直到 reset
  Eigen::VectorXd passive_damping_kd_;    // passive 模式 kd，从 fall_detection.passive_damping 加载

  // pdstand 策略：从 YAML falling_pd_stand 加载
  bool pd_stand_loaded_ = false;
  Eigen::VectorXd q_pd_des_;
  Eigen::VectorXd kp_pd_;
  Eigen::VectorXd kd_pd_;
  double pd_stand_duration_ = 3.0;
  bool in_pd_stand_fall_ = false;
  Eigen::VectorXd q_pd_fall_init_;
  double pd_stand_phase_ = 0.0;
  bool fall_switch_entered_logged_ = false;

  // 定时 pdstand：XZL walk 后按时间切换到 pdstand
  bool enable_pdstand_switch_ = false;
  double pdstand_after_sec_ = 1.0;
  bool in_pdstand_timed_ = false;
  Eigen::VectorXd q_pd_timed_init_;
  double pd_stand_timed_phase_ = 0.0;

  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr mujoco_reset_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr policy_switch_pub_;

  void PublishPolicySwitch(const std::string& from_mode, const std::string& to_mode) {
    if (!policy_switch_pub_) return;
    std_msgs::msg::String msg;
    msg.data = from_mode + "," + to_mode;
    policy_switch_pub_->publish(msg);
  }
};

}  // namespace example

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("rl_basic_example_AMP"), "Usage: rl_basic_example_AMP <config_file_dir> [config_file_name]");
    return 1;
  }

  std::string config_file_name = "rl_basic_param.yaml";
  if (argc >= 3) {
    config_file_name = argv[2];
  }

  auto node = std::make_shared<example::RlBasicRunnerAMP>(argv[1], config_file_name);
  if (!node->Initialize()) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
