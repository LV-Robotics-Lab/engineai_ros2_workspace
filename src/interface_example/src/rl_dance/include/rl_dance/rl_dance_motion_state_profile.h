#pragma once

#include <string>
#include <vector>

namespace data {

// 简化的 RlDanceMotionStateProfile 定义，不依赖 basic_param
class RlDanceMotionStateProfile {
 public:
  RlDanceMotionStateProfile() { Reset(); }
  ~RlDanceMotionStateProfile() = default;

  void Reset() {
    observation_type.clear();
    data_path.clear();
    policy_path.clear();
    traj_frame.clear();
    residual_control = false;
    cycle_time.clear();
    stride_max = 0.6;
    yaw_per_cycle_max = 0.35;
    k_angle = 0.6;
    step_phase = 0.0;
    phase_range.clear();
    csv_dt = 1.0 / 50.0;
    remote_control_key.clear();
    enable = false;
    auto_transition.clear();
    proprioception_obs_with_yaw = false;
    use_gravity_error = false;
    use_quat_error = false;
  }

  std::string observation_type;                 // "mimic" or "mimic_tj"
  bool proprioception_obs_with_yaw = false;     // Whether to include yaw vector in proprioception observation
  bool use_gravity_error = false;               // Whether to use gravity projection error (current - reference) instead of absolute gravity
  bool use_quat_error = false;                  // Whether to use quaternion error (current - reference) in proprioception observation
  std::string data_path;                        // CSV trajectory path
  std::string policy_path;                      // Policy model path
  std::vector<int> traj_frame;                  // Trajectory frame indices
  bool residual_control = false;                // Whether to use residual control
  std::vector<double> cycle_time;               // Motion cycle time range [min, max]
  double stride_max = 0.6;                      // Maximum stride length constraint
  double yaw_per_cycle_max = 0.35;              // Maximum yaw angle per cycle constraint
  double k_angle = 0.6;                         // Angle coefficient for intensity calculation
  double step_phase = 0.0;                      // Step phase for mimic_tj observation type
  std::vector<double> phase_range;              // Phase range for mimic_tj observation type [start, end]
  double csv_dt = 1.0 / 50.0;                  // CSV trajectory sampling rate
  std::vector<std::string> remote_control_key;  // 支持多按键切换
  bool enable = false;                          // 新增，支持yaml配置的enable字段，默认启用
  std::string auto_transition;  // 自动转换到下一个状态，空字符串表示无转换，false表示退出
};

}  // namespace data

