#include "rl_basic/rl_basic_runner.h"

#include <glog/logging.h>
#include <iostream>

#include "math/rotation_matrix.h"
#include "tool/concatenate_vector.h"

namespace runner {

RlBasicRunner::RlBasicRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
    : BasicRunner(name, data_store) {
  // Creates param ptr
  param_ = data::ParamManager::create<data::RlBasicParam>();
  if (!param_tag_.empty()) {
    param_ = data::ParamManager::create<data::RlBasicParam>(param_tag_);
  }

  // Creates mlp net model
  mlp_net_ = std::make_unique<math::MNNModel>(
      common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), param_->policy_file));
  mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
  mlp_net_action_.setZero(param_->num_actions);

  // Initializes some low pass filters
  lpf_command_ = std::make_unique<math::FirstOrderLowPassFilter<Eigen::Vector3d>>(
      param_->remote_command_sampling_frequency, param_->remote_command_cut_off_frequency);

  // Initializes active joint num
  active_joint_idx_.setZero(param_->num_actions);
  int i = 0;
  for (const std::string& name : param_->active_joint_names) {
    active_joint_idx_(i++) = model_param_->joint_id_in_total_limb.at(name);
  }

  default_joint_q_ = common::ConcatenateVectors(param_->default_joint_q);
  joint_kp_ = common::ConcatenateVectors(param_->joint_kp);
  joint_kd_ = common::ConcatenateVectors(param_->joint_kd);
  action_scale_ = common::ConcatenateVectors(param_->action_scale);
  command_scale_ << param_->observation_scale_linear_vel, param_->observation_scale_linear_vel,
      param_->observation_scale_angular_vel;
}

bool RlBasicRunner::Enter() {
  time_ = 0.0;
  is_first_time_ = true;
  last_remote_command_ = *data_store_->gamepad_info.Get();
  imu_install_bias_ = param_->imu_install_bias;

  // Initializes some low pass filters
  command_.setZero();
  lpf_command_->Reset();

  mlp_net_action_.setZero();

  // Sets all motor command to zero
  data_store_->joint_info.SetZeroCommand();

  data_store_->parallel_by_classic_parser.store(false);
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_real_);
  initial_joint_q_ = q_real_;
  return true;
}

void RlBasicRunner::Run() {
  // Gets current joint pos and vel
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_real_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_real_);
  ready_to_walk_ = time_ > param_->keep_stand_time;
  UpdateIMUInstalBias();
  UpdateRemoteCommandBias();
  UpdateRemoteCommand();
  last_remote_command_ = *data_store_->gamepad_info.Get();
  CalculateObservation();
  CalculateMotorCommand();
  SendMotorCommand();
  time_ += param_->control_dt;
}

TransitionState RlBasicRunner::TryExit() { return TransitionState::kCompleted; }

bool RlBasicRunner::Exit() {
  data_store_->parallel_by_classic_parser.store(true);
  return true;
}

void RlBasicRunner::End() {}

void RlBasicRunner::UpdateIMUInstalBias() {
  bool imu_flag = false;

  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_Y == -1 &&
      data_store_->gamepad_info.Get()->CROSS_Y < last_remote_command_.CROSS_Y) {
    imu_install_bias_.x() += param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_Y == 1 &&
      data_store_->gamepad_info.Get()->CROSS_Y > last_remote_command_.CROSS_Y) {
    imu_install_bias_.x() -= param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_X == 1 &&
      data_store_->gamepad_info.Get()->CROSS_X > last_remote_command_.CROSS_X) {
    imu_install_bias_.y() += param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_X == -1 &&
      data_store_->gamepad_info.Get()->CROSS_X < last_remote_command_.CROSS_X) {
    imu_install_bias_.y() -= param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (imu_flag) {
    LOG(INFO) << "imu_install_bias: " << imu_install_bias_;
  }
}

void RlBasicRunner::UpdateRemoteCommandBias() {
  bool command_flag = false;
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_Y == -1 &&
      data_store_->gamepad_info.Get()->CROSS_Y < last_remote_command_.CROSS_Y) {
    command_bias_.y() -= param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_Y == 1 &&
      data_store_->gamepad_info.Get()->CROSS_Y > last_remote_command_.CROSS_Y) {
    command_bias_.y() += param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_X == 1 &&
      data_store_->gamepad_info.Get()->CROSS_X > last_remote_command_.CROSS_X) {
    command_bias_.x() += param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_X == -1 &&
      data_store_->gamepad_info.Get()->CROSS_X < last_remote_command_.CROSS_X) {
    command_bias_.x() -= param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (command_flag) {
    LOG(INFO) << "command_bias: " << command_bias_;
  }
}

void RlBasicRunner::UpdateRemoteCommand() {
  // if (data_store_->gamepad_info.Get()->A == 1 && data_store_->gamepad_info.Get()->A > last_remote_command_.A) {
  //   still_pressed_ = true;
  // }

  bool is_decresing = data_store_->gamepad_info.Get()->LeftStick_X * param_->command_scale.x() < command_.x();

  if (data_store_->gamepad_info.Get()->LeftStick_X > 0) {
    command_.x() = data_store_->gamepad_info.Get()->LeftStick_X * param_->command_scale.x();
  } else {
    command_.x() = data_store_->gamepad_info.Get()->LeftStick_X * std::abs(param_->command_scale_negative.x());
  }
  if (data_store_->gamepad_info.Get()->LeftStick_Y > 0) {
    command_.y() = data_store_->gamepad_info.Get()->LeftStick_Y * param_->command_scale.y();
  } else {
    command_.y() = data_store_->gamepad_info.Get()->LeftStick_Y * std::abs(param_->command_scale_negative.y());
  }
  if (data_store_->gamepad_info.Get()->RightStick_Y > 0) {
    command_.z() = data_store_->gamepad_info.Get()->RightStick_Y * param_->command_scale.z();
  } else {
    command_.z() = data_store_->gamepad_info.Get()->RightStick_Y * std::abs(param_->command_scale_negative.z());
  }
  // command_.x() *= (command_.x() > 0);
  if (param_->enable_remote_command_lpf || is_decresing) {
    command_ = lpf_command_->Update(command_);
  }

  command_ *= ready_to_walk_;
  command_ += command_bias_ + param_->command_bias;
}

void RlBasicRunner::CalculateObservation() {
  // Calculates base state
  Eigen::Matrix3d R_install = math::RotationMatrixd(math::RollPitchYawd(imu_install_bias_)).matrix();
  Eigen::Matrix3d R_local = math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
  Eigen::Matrix3d R_real = R_local * R_install.transpose();
  Eigen::Vector3d w_real = R_real.transpose() * R_local * data_store_->imu_info.Get()->angular_velocity;
  Eigen::Vector3d euler_xyz = math::RollPitchYawd(math::RotationMatrixd(R_real)).vector();
  Eigen::Vector3d projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();

  // Stacks the observation
  Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(param_->num_observations);
  mlp_net_observation_single <<                         //  command
      (q_real_ - default_joint_q_)(active_joint_idx_),  //  joint position - joint default position: kDoFs
      qd_real_(active_joint_idx_),                      //  joint velocity: kDoFs
      mlp_net_action_,                                  //  last joint action: kDoFs
      w_real,                                           //  base angular velocity w.r.t base frame: 3
      projected_gravity_real;                           //  base euler angle rpy w.r.t base frame: 3

  // Scales and clips the observation
  mlp_net_observation_single.array() *= param_->observation_scale.array();
  mlp_net_observation_single =
      mlp_net_observation_single.cwiseMax(-param_->observation_clip).cwiseMin(param_->observation_clip);

  // Updates the observation buffer
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

void RlBasicRunner::CalculateMotorCommand() {
  Eigen::VectorXd obs;
  obs = Eigen::VectorXd::Zero(param_->num_observations * param_->num_include_obs_steps + 3);
  obs.head(param_->num_observations * param_->num_include_obs_steps) =
      Eigen::Map<Eigen::VectorXd>(mlp_net_observation_.transpose().data(), mlp_net_observation_.size());
  obs.tail(3) = command_.cwiseProduct(command_scale_);
  mlp_net_action_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
  mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

  q_des_ = default_joint_q_;
  q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);
  if (time_ < param_->transition_time) {
    float ratio = time_ / param_->transition_time;
    q_des_ = ratio * q_des_ + (1 - ratio) * initial_joint_q_;
  }
}

void RlBasicRunner::SendMotorCommand() {
  qd_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);
  tau_ff_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);
  ReplaceCommand(q_des_, qd_des_, tau_ff_des_, joint_kp_, joint_kd_);

  data_store_->joint_info.SetCommand(data::JointInfoType::kPosition, q_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kVelocity, qd_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kFeedForwardTorque, tau_ff_des_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kStiffness, joint_kp_);
  data_store_->joint_info.SetCommand(data::JointInfoType::kDamping, joint_kd_);
}

void RlBasicRunner::ReplaceCommand(Eigen::VectorXd& q, Eigen::VectorXd& qd, Eigen::VectorXd& tau_ff,
                                   Eigen::VectorXd& kp, Eigen::VectorXd& kd) {
  if (param_->replaced_limbs) {
    int start_index = 0;
    data::JointPoint joint_point = data_store_->planned_motion_info.joint_reference();
    std::vector<std::string> replaced_limbs = param_->replaced_limbs.value();
    for (const std::string& limb : model_param_->limb_names) {
      data::LimbId limb_id = model_param_->limb_id.at(limb);
      int num_joints = model_param_->num_joints_per_limb.at(limb_id);
      auto it = std::find(replaced_limbs.begin(), replaced_limbs.end(), limb);
      if (it != replaced_limbs.end()) {
        q.segment(start_index, num_joints) = joint_point.position.segment(start_index, num_joints);
        qd.segment(start_index, num_joints) = joint_point.velocity.segment(start_index, num_joints);
        tau_ff.segment(start_index, num_joints) = joint_point.feed_forward_torque.segment(start_index, num_joints);
        kp.segment(start_index, num_joints) = joint_point.stiffness.segment(start_index, num_joints);
        kd.segment(start_index, num_joints) = joint_point.damping.segment(start_index, num_joints);
      }
      start_index += num_joints;
    }
  }
}

}  // namespace runner
