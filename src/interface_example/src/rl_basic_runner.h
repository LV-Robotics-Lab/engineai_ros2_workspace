#pragma once

#include "basic/basic_runner.h"
#include "math/first_order_low_pass_filter.h"
#include "math/mnn_model.h"
#include "monitor/data_monitor.h"
#include "parameter/global_config_initializer.h"
#include "rl_basic_param/rl_basic_param.h"
#include "tool/string_join.h"

namespace runner {

class RlBasicRunner : public BasicRunner {
 public:
  RlBasicRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store);
  ~RlBasicRunner() = default;

  bool Enter() override;
  void Run() override;
  TransitionState TryExit() override;
  bool Exit() override;
  void End() override;

  void Log();

 private:
  void UpdateIMUInstalBias();
  void UpdateRemoteCommandBias();
  void UpdateRemoteCommand();
  void CalculateObservation();
  void CalculateMotorCommand();
  void SendMotorCommand();
  void ReplaceCommand(Eigen::VectorXd& q, Eigen::VectorXd& qd, Eigen::VectorXd& tau_ff, Eigen::VectorXd& kp,
                      Eigen::VectorXd& kd);

  std::shared_ptr<data::RlBasicParam> param_;
  float time_ = 0.0;
  bool is_first_time_ = true;

  std::unique_ptr<math::MnnModel> mlp_net_;
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::VectorXd mlp_net_action_;

  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXd qd_des_;
  Eigen::VectorXd tau_ff_des_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXi active_joint_idx_;

  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;
  Eigen::Vector3d command_scale_;

  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d command_bias_ = Eigen::Vector3d::Zero();
  data::GamepadInfo last_remote_command_;
  Eigen::Vector3d command_ = Eigen::Vector3d::Zero();
  std::unique_ptr<math::FirstOrderLowPassFilter<Eigen::Vector3d>> lpf_command_;
  bool ready_to_walk_ = false;

  common::DataMonitor& data_monitor_ = common::DataMonitor::GetInstance();
};
}  // namespace runner
