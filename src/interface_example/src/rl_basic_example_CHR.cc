#include <chrono>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "components/message_handler.hpp"
#include "math/concatenate_vector.h"
#include "math/mnn_model.h"
#include "math/rotation_matrix.h"
#include "parameter/rl_basic_param.h"

using namespace std::chrono_literals;

namespace example {

class RlBasicRunnerCHR : public rclcpp::Node {
 public:
  explicit RlBasicRunnerCHR(const std::string& config_file_dir, const std::string& config_file_name = "rl_basic_param_CHR.yaml") : Node("rl_basic_runner_CHR") {
    std::string config_file = config_file_dir + "/" + config_file_name;
    RCLCPP_INFO(get_logger(), "Loading config file: %s", config_file.c_str());
    param_ = std::make_shared<RlBasicParam>(config_file);
    config_file_dir_ = config_file_dir;
    joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();
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

      // Initialize IMU install bias from parameters
      imu_install_bias_ = param_->imu_install_bias;

      // Initialize MNN model
      mlp_net_ = std::make_unique<math::MnnModel>(config_file_dir_ + "/" + param_->policy_file);
      mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
      mlp_net_action_.setZero(param_->num_actions);

      // Initialize control variables
      time_ = 0.0;
      global_phase_ = 0.0;
      is_first_time_ = true;

      RCLCPP_INFO(get_logger(), "Starting control loop");
      
      // Create control timer
      RCLCPP_INFO(get_logger(), "Creating control timer with dt: %f", param_->control_dt);
      control_timer_ = create_wall_timer(std::chrono::duration<double>(param_->control_dt),
                                         std::bind(&RlBasicRunnerCHR::ControlCallback, this));
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

    // Get IMU data
    auto imu = message_handler_->GetLatestImu();
    if (!imu) {
      RCLCPP_DEBUG(get_logger(), "No IMU data received yet, skipping observation calculation");
      return;
    }
    
    // Calculate base state with IMU installation bias correction
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
    Eigen::Vector3d base_ang_vel = R_real.transpose() * R_local * 
        Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    
    // Base linear velocity (from IMU or zero if not available)
    // TODO: Get from odometry or other source if available
    Eigen::Vector3d base_lin_vel = Eigen::Vector3d::Zero();
    
    // Motion anchor position and orientation (relative to base frame)
    // TODO: Calculate from motion reference if available, otherwise set to zero
    Eigen::Vector3d motion_anchor_pos_b = Eigen::Vector3d::Zero();
    Eigen::Matrix3d motion_anchor_ori_b_matrix = Eigen::Matrix3d::Identity();
    // Extract first two rows of rotation matrix (6 elements)
    Eigen::VectorXd motion_anchor_ori_b(6);
    motion_anchor_ori_b << motion_anchor_ori_b_matrix.row(0), motion_anchor_ori_b_matrix.row(1);
    
    // Command: joint_pos (24) + joint_vel (24) = 48 dimensions
    // Use all 24 joints, not just active joints
    Eigen::VectorXd command_joint_pos = q_real_ - default_joint_q_;
    Eigen::VectorXd command_joint_vel = qd_real_;
    Eigen::VectorXd command(48);
    command << command_joint_pos, command_joint_vel;
    
    // Joint positions (24 dimensions) - relative to default
    Eigen::VectorXd joint_pos_rel = q_real_ - default_joint_q_;
    
    // Joint velocities (24 dimensions)
    Eigen::VectorXd joint_vel = qd_real_;
    
    // Actions (24 dimensions) - map from active joints to all joints
    // mlp_net_action_ corresponds to active_joint_idx, need to map to all 24 joints
    Eigen::VectorXd actions_24 = Eigen::VectorXd::Zero(24);
    for (int i = 0; i < active_joint_idx_.size() && i < mlp_net_action_.size(); ++i) {
      int joint_idx = active_joint_idx_[i];
      if (joint_idx >= 0 && joint_idx < 24) {
        actions_24[joint_idx] = mlp_net_action_[i];
      }
    }

    // Stack the observation: 135 dimensions total
    // command (48) + motion_anchor_pos_b (3) + motion_anchor_ori_b (6) + 
    // base_lin_vel (3) + base_ang_vel (3) + joint_pos (24) + joint_vel (24) + actions (24)
    Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(param_->num_observations);
    mlp_net_observation_single <<
        command,                    // 48: command (joint_pos 24 + joint_vel 24)
        motion_anchor_pos_b,        // 3: motion anchor position
        motion_anchor_ori_b,       // 6: motion anchor orientation (first 2 rows of rotation matrix)
        base_lin_vel,               // 3: base linear velocity
        base_ang_vel,               // 3: base angular velocity
        joint_pos_rel,              // 24: joint positions (relative to default)
        joint_vel,                  // 24: joint velocities
        actions_24;                 // 24: last actions

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
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Starting");
    
    // Calculate clock signal for gait phase
    Eigen::Vector2d clock_signal(std::sin(2 * M_PI * global_phase_), std::cos(2 * M_PI * global_phase_));
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Clock signal calculated");
    
    // Build observation vector: obs_history + clock_signal + commands
    Eigen::VectorXd obs = Eigen::VectorXd::Zero(
      param_->num_observations * param_->num_include_obs_steps + 
      param_->num_clock_signal + 
      param_->num_commands
    );
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Observation vector created, size: %ld", obs.size());
    
    // Fill observation history (flattened from observation buffer)
    // The model expects observation flattened by dimension: [obs_dim0_t0, obs_dim0_t1, ..., obs_dim0_t14, obs_dim1_t0, ...]
    // This means: for each dimension, all time steps are stored consecutively
    // Since mlp_net_observation_ is stored column-major (Eigen default), we need to copy row by row
    int offset = 0;
    for (int i = 0; i < param_->num_observations; ++i) {
      obs.segment(offset, param_->num_include_obs_steps) = mlp_net_observation_.row(i);
      offset += param_->num_include_obs_steps;
    }
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Observation data filled, size: %ld", 
                 param_->num_observations * param_->num_include_obs_steps);
    
    // Scale commands and append clock signal and commands
    command_.array() *= param_->obs_commands_scale.array();
    obs.tail(param_->num_clock_signal + param_->num_commands) << clock_signal, command_;
    RCLCPP_DEBUG(get_logger(), "CalculateMotorCommand: Command data filled");
    
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
    mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

    // Calculate desired joint positions
    q_des_ = default_joint_q_;
    
    // 添加维度检查
    if (mlp_net_action_.size() != active_joint_idx_.size()) {
      RCLCPP_ERROR(get_logger(), "Action size mismatch: mlp_net_action=%ld, active_joint_idx=%ld", 
                   mlp_net_action_.size(), active_joint_idx_.size());
      return;
    }
    if (action_scale_.size() != active_joint_idx_.size()) {
      RCLCPP_ERROR(get_logger(), "Action scale size mismatch: action_scale=%ld, active_joint_idx=%ld", 
                   action_scale_.size(), active_joint_idx_.size());
      return;
    }
    
    q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);
    
    if (time_ < param_->transition_time) {
      float ratio = time_ / param_->transition_time;
      q_des_ = ratio * q_des_ + (1 - ratio) * initial_joint_q_;
    }
  }

  void SendMotorCommand() {
    // Convert Eigen vectors to std::vector
    joint_command_->position = std::vector<double>(q_des_.data(), q_des_.data() + q_des_.size());
    joint_command_->velocity = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->torque = std::vector<double>(q_des_.size(), 0.0);
    joint_command_->stiffness = std::vector<double>(joint_kp_.data(), joint_kp_.data() + joint_kp_.size());
    joint_command_->damping = std::vector<double>(joint_kd_.data(), joint_kd_.data() + joint_kd_.size());
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    // Send command through message handler
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
