#pragma once

#include <deque>
#include <fstream>
#include <chrono>
#include "basic/basic_runner.h"
#include "basic/runner_registry.h"
#include "math/first_order_low_pass_filter.h"
#include "math/mnn_model.h"
#include "monitor/data_monitor.h"
#include "parameter/global_config_initializer.h"
#include "rl_dance/csv_loader.h"
#include "rl_dance/observation_manager.h"
#include "rl_dance_param/rl_dance_param.h"

namespace runner {

class RlDanceRunner : public BasicRunner {
 public:
  RlDanceRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
      : BasicRunner(name, data_store) {
    param_ = data::ParamManager::create<data::RlDanceParam>();
  }
  ~RlDanceRunner() = default;

  bool Enter() override;
  void Run() override;
  TransitionState TryExit() override;
  bool Exit() override;

 private:
  void Init();
  void UpdateState();
  void UpdateRemoteCommand();
  void CalculateObservation();
  void ResetReferenceTrajectory();
  void CalculateMotorCommand();
  void SendMotorCommand();
  void CalculateBoxingCommand();

  // Helper method for torque limiting
  void ApplyTorqueLimits(Eigen::VectorXd& q_des, const Eigen::VectorXd& q_actual, const Eigen::VectorXd& qd_actual,
                         const Eigen::VectorXd& joint_kp, const Eigen::VectorXd& joint_kd);

  // Stability detection methods
  bool CheckStability();
  void UpdateStabilityHistory();

  // Trajectory management helpers
  bool IsTrajectoryFinished() const;
  void TransitionToNextPolicy();

  // Yaw calculation helper
  Eigen::Vector2d CalculateBaseYawForwardVector();

  enum class PolicyMode { kBoxing, kLocomotion };
  PolicyMode policy_mode_ = PolicyMode::kBoxing;

  std::shared_ptr<data::RlDanceParam> param_;
  std::string last_param_tag_ = "";
  bool is_first_time_ = true;
  bool run_reference_trajectory_ = false;
  bool task_completed_ = false;  // Flag to indicate task completion
  int iter_ = 0;
  size_t reference_trajectory_iter_ = 0;

  std::map<std::string, std::unique_ptr<math::MNNModel>> policy_models_;
  math::MNNModel* mlp_net_boxing_ = nullptr;  // 只做指针，不负责所有权
  Eigen::VectorXd mlp_net_action_;

  Eigen::VectorXd q_actual_;
  Eigen::VectorXd q_actual_temp_;
  Eigen::VectorXd qd_actual_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXd q_ref_;
  Eigen::VectorXd qd_des_;
  Eigen::VectorXd tau_ff_des_;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;
  Eigen::VectorXd qd_mask_;

  float gait_phase_;

  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d command_bias_ = Eigen::Vector3d::Zero();
  data::GamepadInfo last_remote_command_;
  Eigen::Vector3d command_ = Eigen::Vector3d::Zero();
  std::unique_ptr<math::FirstOrderLowPassFilter<Eigen::Vector3d>> lpf_command_;

  Eigen::Matrix3d initial_heading_rotation_;
  data::SE3Frame base_state_in_world_;

  // Yaw calculation state
  Eigen::Vector4d initial_base_quat_;  // Initial base quaternion (w,x,y,z)
  bool initial_quat_set_ = false;

  // 新增：observation自动管理
  std::unique_ptr<rl_dance::ObservationManager> observation_manager_;

  // 新增：插值后的关节轨迹 [N, 24]
  std::map<std::string, Eigen::MatrixXd> interpolated_trajs_;
  Eigen::MatrixXd* current_traj_ = nullptr;

  // 新增：当前使用的profile key
  std::string profile_key_ = "switch_to_boxing_idle";
  // rl_locomotion
  //  新增：支持多种motion类型切换
  std::vector<std::string> motion_types_;
  size_t motion_types_index_ = 0;

  common::DataMonitor& data_monitor_ = common::DataMonitor::GetInstance();

  // Stability detection members
  bool is_stable_ = false;
  bool is_stand_still_ = false;
  std::deque<bool> stability_history_;
  Eigen::Vector3d base_angular_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d projected_gravity_ = Eigen::Vector3d::Zero();

  // Key input handler
  class KeyInputHandler {
   public:
    struct DoubleClickInfo {
      double last_press_time = 0.0;
      int click_count = 0;
    };

    KeyInputHandler(double timeout = 0.5) : double_click_timeout_(timeout) {}

    // Check if key is currently pressed
    bool IsKeyPressed(const data::GamepadInfo& now, const std::string& key) const;

    // Check if key was just pressed (rising edge)
    bool IsKeyJustPressed(const data::GamepadInfo& now, const std::string& key, const data::GamepadInfo& last) const;

    // Check if key was double-clicked
    bool IsKeyDoubleClicked(const data::GamepadInfo& now, const std::string& key, const data::GamepadInfo& last,
                            double current_time);

    // Check if key combination is triggered (supports both regular and double-click keys)
    bool IsKeyCombinationTriggered(const std::vector<std::string>& keys, const data::GamepadInfo& now,
                                   const data::GamepadInfo& last, double current_time);

    // Reset double-click tracker for a specific key or all keys
    void ResetDoubleClickTracker(const std::string& key = "");

   private:
    std::map<std::string, DoubleClickInfo> double_click_tracker_;
    double double_click_timeout_;
  };

  KeyInputHandler key_handler_;

  // Logging: lower-body torque sum over time, per profile
  std::ofstream lower_body_torque_log_dance1_;
  std::string lower_body_torque_log_path_dance1_;
  std::ofstream lower_body_torque_log_dance2_;
  std::string lower_body_torque_log_path_dance2_;
  std::chrono::steady_clock::time_point start_time_;

  // Flag to mark that we have saved/printed at dance_2 second last step
  bool saved_dance2_second_last_ = false;
  // New: generic once-only torque log save hint for any profile
  bool saved_torque_log_hint_ = false;

  // Per-profile last logged trajectory index (to ensure one line per frame)
  size_t last_logged_index_dance1_ = (size_t)(-1);
  size_t last_logged_index_dance2_ = (size_t)(-1);
};
}  // namespace runner

REGISTER_RUNNER(RlDanceRunner, "rl_dance_runner", kMotion)