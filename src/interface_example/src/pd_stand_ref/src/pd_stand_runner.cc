#include "pd_stand/pd_stand_runner.h"

#include <iostream>
#include <string>
#include "math/interpolation.h"
#include "tool/concatenate_vector.h"

namespace runner {

bool PdStandRunner::Enter() {
  if (!param_tag_.empty()) {
    param_ = data::ParamManager::create<data::PdStandParam>(param_tag_);
  }
  std::cout << "Current tag: " << param_->GetTag() << std::endl;
  auto_transition_ = param_->auto_transition.value_or(false);
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_init_);
  q_des_ = common::ConcatenateVectors(param_->desired_joint_position);
  kp_ = common::ConcatenateVectors(param_->stiffness);
  kd_ = common::ConcatenateVectors(param_->damping);
  duration_ = param_->duration;

  iter_ = 0;
  tau_ff_cmd_ = Eigen::VectorXd::Zero(q_init_.size());

  if (!common::IsInMujoco() && !CheckJointPositionBias()) {
    auto gamepad_info = data_store_->gamepad_info.Get();
    constexpr double kGamepadThreshold = 0.8;
    if (global_options_param_->strict_motion_check == false && gamepad_info->LT > kGamepadThreshold) {
      LOG(INFO) << "Strict motion check is disabled, force stand";
      return true;
    }
    return false;
  }

  return true;
}

bool PdStandRunner::CheckJointPositionBias() {
  static constexpr double kJointPositionBiasThreshold = 1.0;
  double threshold = param_->initial_joint_position_bias_threshold.value_or(kJointPositionBiasThreshold);

  Eigen::VectorXd q_init_bias = q_init_ - q_des_;
  auto large_bias = (q_init_bias.array().abs() > threshold).matrix();
  if (large_bias.any()) {
    LOG(WARNING) << "Joint position bias exceeds threshold (" << threshold << ") at joints: ";
    for (int i = 0; i < model_param_->num_total_joints; ++i) {
      if (large_bias(i)) {
        LOG(WARNING) << "Joint [" << model_param_->GetJointNameByJointId(i) << "]: bias = " << q_init_bias[i];
      }
    }
    return false;
  }

  return true;
}

void PdStandRunner::Run() {
  data_store_->reference_contact_signal_info->SetAllLegContactSignal();
  data_store_->reference_contact_signal_info->SetAllArmNonContactSignal();

  double phase = std::min(static_cast<double>(iter_) * runner_period_, duration_);
  math::QuinticInterpolate(q_init_, q_des_, duration_, phase, q_cmd_, qd_cmd_);

  data_store_->joint_info.SetCommand(q_cmd_, qd_cmd_, tau_ff_cmd_, kp_, kd_);
  ++iter_;
  if (static_cast<double>(iter_) * runner_period_ > duration_ && auto_transition_) {
    LOG(INFO) << "PdStandRunner: Transitioning to next runner after " << duration_ << " seconds.";
    SetRunnerState(RunnerState::kTryExit);
  }
}

TransitionState PdStandRunner::TryExit() {
  return TransitionState::kCompleted;
  // if (static_cast<double>(iter_) * runner_period_ > duration_) {
  //   return TransitionState::kCompleted;
  // } else {
  //   Run();
  //   return TransitionState::kTrying;
  // }
}

bool PdStandRunner::Exit() { return true; }

void PdStandRunner::End() {}
}  // namespace runner
