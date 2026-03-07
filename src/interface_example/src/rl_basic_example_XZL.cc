#include <chrono>
#include <cmath>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
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

class RlBasicRunnerXZL : public rclcpp::Node {
 public:
  explicit RlBasicRunnerXZL(const std::string& config_file_dir, const std::string& config_file_name = "rl_basic_param.yaml") : Node("rl_basic_runner_XZL") {
    std::string config_file = config_file_dir + "/" + config_file_name;
    RCLCPP_INFO(get_logger(), "Loading config file: %s", config_file.c_str());
    param_ = std::make_shared<RlBasicParam>(config_file);
    config_file_dir_ = config_file_dir;
    config_file_ = config_file;
    joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();

    // 订阅 mujoco reset，reset 后下次摔倒时重新打印
    mujoco_reset_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/mujoco/reset_complete", 10, [this](const std_msgs::msg::Empty::SharedPtr) {
          fall_switch_entered_logged_ = false;
          in_pd_stand_fall_ = false;
          in_pdstand_timed_ = false;
        });

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
      
      // Initialize command scale - from rl_basic_runner.cc
      command_scale_ << param_->observation_scale_linear_vel, param_->observation_scale_linear_vel,
          param_->observation_scale_angular_vel;

      // Initialize MNN model
      mlp_net_ = std::make_unique<math::MnnModel>(config_file_dir_ + "/" + param_->policy_file);
      mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
      mlp_net_action_.setZero(param_->num_actions);

      // Initialize control variables
      time_ = 0.0;
      global_phase_ = 0.0;
      is_first_time_ = true;

      // Load IMU install bias from param (for GetCurrentPitchAngle)
      imu_install_bias_ = param_->imu_install_bias;

      RCLCPP_INFO(get_logger(), "Starting control loop");
      
      // Create control timer
      RCLCPP_INFO(get_logger(), "Creating control timer with dt: %f", param_->control_dt);
      control_timer_ = create_wall_timer(std::chrono::duration<double>(param_->control_dt),
                                         std::bind(&RlBasicRunnerXZL::ControlCallback, this));
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
      is_first_time_ = true;
      return;
    }
    
    auto joint_state = message_handler_->GetLatestJointState();
    if (!joint_state) {
      RCLCPP_DEBUG(get_logger(), "No joint state received yet, skipping");
      return;  // Skip if no joint state received yet
    }

    RCLCPP_DEBUG(get_logger(), "Updating state");
    UpdateState(joint_state);
    
    RCLCPP_DEBUG(get_logger(), "Calculating observation");
    CalculateObservation();
    
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

    // Update command from gamepad
    auto gamepad = message_handler_->GetLatestGamepad();
    if (gamepad) {
      command_.x() =
          gamepad->analog_states[interface_protocol::msg::GamepadKeys::LEFT_STICK_X] * param_->command_scale.x();
      command_.y() =
          gamepad->analog_states[interface_protocol::msg::GamepadKeys::LEFT_STICK_Y] * param_->command_scale.y();
      command_.z() =
          gamepad->analog_states[interface_protocol::msg::GamepadKeys::RIGHT_STICK_Y] * param_->command_scale.z();
    }
  }

  void CalculateObservation() {
    // Calculate phase info
    global_phase_ += param_->control_dt / param_->cycle_time;
    global_phase_ -= static_cast<int>(global_phase_);

    // Get IMU data - using advanced algorithm from rl_basic_runner.cc
    auto imu = message_handler_->GetLatestImu();
    
    // Calculate base state with IMU installation bias correction
    // Create rotation matrix from IMU installation bias (Euler angles)
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_install = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R_install = q_install.toRotationMatrix();
    
    // Get local rotation matrix from IMU quaternion
    Eigen::Matrix3d R_local = Eigen::Quaterniond(imu->quaternion.w, imu->quaternion.x, 
                                                 imu->quaternion.y, imu->quaternion.z).toRotationMatrix();
    
    // Calculate real rotation matrix with bias correction
    Eigen::Matrix3d R_real = R_local * R_install.transpose();
    Eigen::Vector3d w_real = R_real.transpose() * R_local * 
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();

    // Stack the observation - using the same structure as rl_basic_runner.cc
    Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(param_->num_observations);
    mlp_net_observation_single <<                         //  command
        (q_real_ - default_joint_q_)(active_joint_idx_),  //  joint position - joint default position: kDoFs
        qd_real_(active_joint_idx_),                      //  joint velocity: kDoFs
        mlp_net_action_,                                  //  last joint action: kDoFs
        w_real,                                           //  base angular velocity w.r.t base frame: 3
        projected_gravity_real;                           //  projected gravity w.r.t base frame: 3

    // Scales and clips the observation
    mlp_net_observation_single.array() *= param_->observation_scale.array();
    mlp_net_observation_single =
        mlp_net_observation_single.cwiseMax(-param_->observation_clip).cwiseMin(param_->observation_clip);

    // Update the observation buffer
    if (is_first_time_) {
      is_first_time_ = false;
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
    // Using simplified but effective algorithm from rl_basic_runner.cc
    Eigen::VectorXd obs;
    obs = Eigen::VectorXd::Zero(param_->num_observations * param_->num_include_obs_steps + 3);
    obs.head(param_->num_observations * param_->num_include_obs_steps) =
        Eigen::Map<Eigen::VectorXd>(mlp_net_observation_.transpose().data(), mlp_net_observation_.size());
    // 从YAML配置加载初始速度指令
    command_ = param_->initial_linear_velocity;
    static bool debug_printed = false;
    if (!debug_printed) {
      RCLCPP_INFO(get_logger(), "Using initial velocity from YAML: [%.3f, %.3f, %.3f]", 
                  command_[0], command_[1], command_[2]);
      debug_printed = true;
    }
    obs.tail(3) = command_.cwiseProduct(command_scale_);
    
    // Get MNN output
    if (!mlp_net_) {
      RCLCPP_ERROR(get_logger(), "MNN model not initialized");
      return;
    }
    
    try {
      mlp_net_action_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "MNN inference failed: %s", e.what());
      return;
    }
    mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

    // Calculate desired joint positions
    q_des_ = default_joint_q_;
    q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);
    
    if (time_ < param_->transition_time) {
      float ratio = time_ / param_->transition_time;
      q_des_ = ratio * q_des_ + (1 - ratio) * initial_joint_q_;
    }
  }

  // 获取当前 IMU 俯仰角度（弧度），用于 damping mode 判断
  double GetCurrentPitchAngle() {
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      RCLCPP_WARN(get_logger(), "IMU data not available, using default pitch angle 0.0");
      return 0.0;
    }
    Eigen::AngleAxisd rollAngle(imu_install_bias_.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(imu_install_bias_.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(imu_install_bias_.z(), Eigen::Vector3d::UnitZ());
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
    if (q_real_.size() != n || qd_real_.size() != n || joint_kp_.size() != n || joint_kd_.size() != n) {
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
    const Eigen::VectorXd joint_kp_safe = joint_kp_.cwiseMax(kp_min_threshold);
    Eigen::VectorXd q_des_lb = (-tau_limit_.array() + (joint_kd_.array() * qd_real_.array())).matrix();
    Eigen::VectorXd q_des_ub = (tau_limit_.array() + (joint_kd_.array() * qd_real_.array())).matrix();
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

    const double kPitchDampingThresholdRad = 50.0 * M_PI / 180.0;  // 50°
    bool fall_detected = enable_falling_switch_ && time_ >= falling_detect_after_sec_;
    if (fall_detected) {
      double current_pitch = GetCurrentPitchAngle();
      if (current_pitch <= kPitchDampingThresholdRad && current_pitch >= -kPitchDampingThresholdRad) {
        fall_detected = false;
      }
    }

    // 优先级：摔倒检测 > 定时 pdstand > 正常 RL
    bool use_fall_command = false;
    bool use_pdstand_timed = false;
    if (falling_switch_ == "none") {
      use_fall_command = false;
    } else if (falling_switch_ == "damping" && fall_detected) {
      use_fall_command = true;
      if (!fall_switch_entered_logged_) {
        RCLCPP_INFO(get_logger(), "Fall detected, switching to damping mode");
        fall_switch_entered_logged_ = true;
      }
      joint_command_->stiffness = std::vector<double>(n, 0.0);
      joint_command_->damping = std::vector<double>(n, 0.5);
    } else if (falling_switch_ == "pdstand" && pd_stand_loaded_) {
      if (fall_detected && !in_pd_stand_fall_) {
        in_pd_stand_fall_ = true;
        q_pd_fall_init_ = q_real_;
        pd_stand_phase_ = 0.0;
        if (q_pd_fall_init_.size() != q_pd_des_.size()) {
          RCLCPP_WARN(get_logger(), "pd_stand size mismatch: q_real %zd vs q_pd_des %zd, disable pdstand",
                      q_pd_fall_init_.size(), q_pd_des_.size());
          in_pd_stand_fall_ = false;
        }
      }
      if (in_pd_stand_fall_ && q_pd_fall_init_.size() == q_pd_des_.size()) {
        if (!fall_switch_entered_logged_) {
          RCLCPP_INFO(get_logger(), "Fall detected, switching to pdstand mode");
          fall_switch_entered_logged_ = true;
        }
        use_fall_command = true;
        pd_stand_phase_ += param_->control_dt;
        const int np = static_cast<int>(q_pd_des_.size());
        if (pd_stand_phase_ < pd_stand_duration_) {
          Eigen::VectorXd q_cmd(np), qd_cmd(np);
          QuinticInterpolate(q_pd_fall_init_, q_pd_des_, pd_stand_duration_, pd_stand_phase_, q_cmd, qd_cmd);
          joint_command_->position = std::vector<double>(q_cmd.data(), q_cmd.data() + np);
          joint_command_->velocity = std::vector<double>(qd_cmd.data(), qd_cmd.data() + np);
          joint_command_->stiffness = std::vector<double>(kp_pd_.data(), kp_pd_.data() + np);
          joint_command_->damping = std::vector<double>(kd_pd_.data(), kd_pd_.data() + np);
        } else {
          joint_command_->position = std::vector<double>(q_pd_des_.data(), q_pd_des_.data() + np);
          joint_command_->velocity = std::vector<double>(np, 0.0);
          joint_command_->stiffness = std::vector<double>(kp_pd_.data(), kp_pd_.data() + np);
          joint_command_->damping = std::vector<double>(kd_pd_.data(), kd_pd_.data() + np);
        }
      }
    }

    // 定时 pdstand：enable_pdstand_switch 且 time >= pdstand_after_sec，且未摔倒
    if (!use_fall_command && enable_pdstand_switch_ && pd_stand_loaded_ && time_ >= pdstand_after_sec_) {
      if (!in_pdstand_timed_) {
        in_pdstand_timed_ = true;
        q_pd_timed_init_ = q_real_;
        pd_stand_timed_phase_ = 0.0;
        if (q_pd_timed_init_.size() != q_pd_des_.size()) {
          RCLCPP_WARN(get_logger(), "pd_stand timed size mismatch: q_real %zd vs q_pd_des %zd", q_pd_timed_init_.size(), q_pd_des_.size());
          in_pdstand_timed_ = false;
        } else {
          RCLCPP_INFO(get_logger(), "Switching to pdstand (timed) at %.1fs", time_);
        }
      }
      if (in_pdstand_timed_ && q_pd_timed_init_.size() == q_pd_des_.size()) {
        use_pdstand_timed = true;
        pd_stand_timed_phase_ += param_->control_dt;
        const int np = static_cast<int>(q_pd_des_.size());
        if (pd_stand_timed_phase_ < pd_stand_duration_) {
          Eigen::VectorXd q_cmd(np), qd_cmd(np);
          QuinticInterpolate(q_pd_timed_init_, q_pd_des_, pd_stand_duration_, pd_stand_timed_phase_, q_cmd, qd_cmd);
          joint_command_->position = std::vector<double>(q_cmd.data(), q_cmd.data() + np);
          joint_command_->velocity = std::vector<double>(qd_cmd.data(), qd_cmd.data() + np);
          joint_command_->stiffness = std::vector<double>(kp_pd_.data(), kp_pd_.data() + np);
          joint_command_->damping = std::vector<double>(kd_pd_.data(), kd_pd_.data() + np);
        } else {
          joint_command_->position = std::vector<double>(q_pd_des_.data(), q_pd_des_.data() + np);
          joint_command_->velocity = std::vector<double>(np, 0.0);
          joint_command_->stiffness = std::vector<double>(kp_pd_.data(), kp_pd_.data() + np);
          joint_command_->damping = std::vector<double>(kd_pd_.data(), kd_pd_.data() + np);
        }
      }
    }

    if (!use_fall_command && !use_pdstand_timed) {
      fall_switch_entered_logged_ = false;  // 恢复后下次摔倒再打印
      in_pdstand_timed_ = false;  // 回到 RL 时重置
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
  std::unique_ptr<math::MnnModel> mlp_net_;
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::VectorXd mlp_net_action_;

  // State variables
  double time_;
  double global_phase_;
  bool is_first_time_;
  Eigen::Vector3d command_;
  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;
  Eigen::Vector3d command_scale_;
  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();

  // ROS timer
  rclcpp::TimerBase::SharedPtr control_timer_;
  std::string config_file_dir_;
  std::string config_file_;  // 保存配置文件路径，用于读取扭矩限制参数
  interface_protocol::msg::JointCommand::SharedPtr joint_command_;
  
  // 扭矩限制参数
  bool torque_limit_enabled_ = false;
  std::vector<Eigen::VectorXd> max_torque_joint_;
  double soft_torque_limit_ = 0.9;  // 软扭矩限制系数，默认0.9（参考XZL实现）

  // 摔倒策略（YAML enable_falling_switch, falling_detect_after_sec, falling_switch）
  bool enable_falling_switch_ = true;
  double falling_detect_after_sec_ = 5.0;
  std::string falling_switch_ = "damping";  // none / damping / pdstand

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
};

}  // namespace example

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("rl_basic_example_XZL"), "Usage: rl_basic_example_XZL <config_file_dir> [config_file_name]");
    return 1;
  }

  std::string config_file_name = "rl_basic_param.yaml";
  if (argc >= 3) {
    config_file_name = argv[2];
  }

  auto node = std::make_shared<example::RlBasicRunnerXZL>(argv[1], config_file_name);
  if (!node->Initialize()) {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
