#include "rl_dance_param.h"

namespace data {

RlDanceParam::RlDanceParam(std::string_view tag) : BasicParam(tag) {
  LOAD_PARAM(observations);
  LOAD_PARAM(active_joint_names);
  LOAD_PARAM(default_joint_q);
  LOAD_PARAM(joint_kp);
  LOAD_PARAM(joint_kd);
  LOAD_PARAM(action_scale);
  LOAD_PARAM(action_scale85);
  LOAD_PARAM(qd_mask);
  LOAD_PARAM(remote_command_sampling_frequency);
  LOAD_PARAM(remote_command_cut_off_frequency);
  LOAD_PARAM(imu_install_delta_bias);
  LOAD_PARAM(linear_vel_delta_bias);
  LOAD_PARAM(enable_remote_command_lpf);
  LOAD_PARAM(command_scale);
  LOAD_PARAM(command_bias);
  LOAD_PARAM(action_clip);
  LOAD_PARAM(transition_time);
  LOAD_PARAM(torque_limit);
  LOAD_PARAM(max_torque_joint);
  LOAD_PARAM(max_lower_body_torque);
  LOAD_PARAM(motion_states);
  // 加载稳定性参数
  LOAD_PARAM(ang_vel_threshold);
  LOAD_PARAM(gravity_dev_threshold);
  LOAD_PARAM(stability_smoothing_threshold);
  LOAD_PARAM(stability_history_length);
  // 加载速度阈值参数
  LOAD_PARAM(lin_vel_set_zero_threshold);
  LOAD_PARAM(ang_vel_set_zero_threshold);
  // 加载步态相位调整参数
  LOAD_PARAM(unstable_cycle_time_scale);
  LOAD_PARAM(default_dynamic_cycle_time);

  // 构建统一的observation scale向量
  const auto& scale_config = observations.observation_scale;

  // 本体感知各项配置：维度和scale值
  struct ProprioItem {
    int dim;
    double scale;
  };
  std::vector<ProprioItem> proprio_items = {
      {static_cast<int>(active_joint_names.size()), scale_config.observation_scale_dof_pos},
      {static_cast<int>(active_joint_names.size()), scale_config.observation_scale_dof_vel},
      {static_cast<int>(active_joint_names.size()), 1.0},  // action
      {3, scale_config.observation_scale_angular_vel},     // angular vel
      {3, scale_config.observation_scale_quat}             // gravity
  };

  // 计算单步本体感知维度
  int proprio_single_dim = 0;
  for (const auto& item : proprio_items) proprio_single_dim += item.dim;

  // 找到最大的time steps来确定scale向量的最大可能大小
  int max_proprio_time_steps = 0;
  for (const auto& [goal_type, goal_cfg] : observations.observation_type) {
    max_proprio_time_steps = std::max(max_proprio_time_steps, goal_cfg.num_include_obs_steps);
  }

  // 不再需要预构建scale向量，改为动态构建
  observation_scale_vec = Eigen::VectorXd::Zero(1);  // 占位符

  LOG(INFO) << "Scale configuration loaded: single_step_dim=" << proprio_single_dim
            << ", max_time_steps=" << max_proprio_time_steps;

  num_actions = active_joint_names.size();
}

}  // namespace data