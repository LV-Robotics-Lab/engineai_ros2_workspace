#include "rl_dance/observation_manager.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "math/constants.h"
#include "math/interpolation.h"
#include "math/rotation_matrix.h"
#include "tool/concatenate_vector.h"

namespace rl_dance {

ObservationManager::ObservationManager(const data::RlDanceParam& param) : param_(param) {
  // 默认不初始化，需SetProfile后Init
}

void ObservationManager::SetCurrentTrajectory(Eigen::MatrixXd* traj) { current_traj_ = traj; }

void ObservationManager::SetProfile(const std::string& profile_key,
                                    const data::RlDanceMotionStateProfile& profile) {
  profile_key_ = profile_key;
  current_profile_ = &profile;
  observation_type_ = profile.observation_type;
  // 使用统一的observation配置
  current_obs_cfg_ = &param_.observations;
  // 使用统一的observation scale向量
  current_observation_scale_ = param_.observation_scale_vec;

  // Initialize buffers: 1 proprioceptive buffer + 3 goal buffers
  // Proprioceptive buffer: stores robot state data (dynamic sizing, determined on first use)
  obs_buffer_ = Eigen::MatrixXd::Zero(1, 1);  // Temporary size, will be adjusted later

  // Goal buffers: obtain configuration for each goal buffer from config
  goal_buffers_.clear();
  goal_time_steps_.clear();
  proprioceptive_time_steps_.clear();

  for (const auto& [goal_type, goal_cfg] : param_.observations.observation_type) {
    // Each goal uses its own history steps and proprioceptive time steps
    int goal_steps = goal_cfg.goal_history_steps;
    goal_buffers_[goal_type] = Eigen::MatrixXd::Zero(goal_cfg.goal_length, goal_steps);
    goal_time_steps_[goal_type] = goal_steps;
    proprioceptive_time_steps_[goal_type] = goal_cfg.num_include_obs_steps;
  }
  obs_buffer_initialized_ = false;  // Mark for initialization on first UpdateObservation

  // 设置当前goal buffer，检查是否存在对应的配置
  auto goal_it = goal_buffers_.find(observation_type_);
  if (goal_it != goal_buffers_.end()) {
    current_goal_buffer_ = &goal_it->second;
  } else {
    throw std::runtime_error("No goal buffer configuration found for observation_type: " + observation_type_);
  }
  // Reset indices only when profile actually switches - use member variables to avoid static variables
  if (last_profile_key_ != profile_key) {
    trajectory_index_ = 0;
    mimic_tj_phase_counter_ = 0;  // Reset mimic_tj phase counter
    gait_phase_ = 0.0;            // Reset gait phase
    last_profile_key_ = profile_key;
    // Reset initial quaternion offset when switching profiles
    initial_quat_offset_computed_ = false;
    initial_quat_offset_ = std::nullopt;
    // Don't reset obs_buffer_initialized_ to preserve history buffers during policy switching
    // obs_buffer_initialized_ = false;  // Commented out to preserve history
  }
}

void ObservationManager::UpdateObservation(const Eigen::VectorXd& q_actual, const Eigen::VectorXd& qd_actual,
                                           const Eigen::VectorXd& mlp_net_action, const Eigen::Vector3d& w_real,
                                           const Eigen::Vector3d& projected_gravity_real,
                                           const Eigen::Vector3d& command, const Eigen::VectorXd& default_joint_q,
                                           bool is_stable, const std::optional<Eigen::Vector2d>& yaw_forward_vec,
                                           const std::optional<Eigen::Quaterniond>& ref_quat,
                                           const std::optional<Eigen::Quaterniond>& current_quat) {
  if (!current_obs_cfg_ || !current_goal_buffer_ || !current_profile_) {
    LOG(ERROR) << "Invalid state: current_obs_cfg_=" << (current_obs_cfg_ ? "OK" : "NULL")
               << ", current_goal_buffer_=" << (current_goal_buffer_ ? "OK" : "NULL")
               << ", current_profile_=" << (current_profile_ ? "OK" : "NULL");
    return;
  }

  // 1. Update proprioceptive buffer (using current observation_type's time steps)
  int q_size = q_actual.size();
  int qd_size = qd_actual.size();
  int action_size = mlp_net_action.size();

  // Safely get time steps for current observation_type
  auto time_steps_it = proprioceptive_time_steps_.find(observation_type_);
  if (time_steps_it == proprioceptive_time_steps_.end()) {
    LOG(ERROR) << "No proprioceptive time steps found for observation_type: " << observation_type_;
    return;
  }
  int time_steps = time_steps_it->second;

  // First-time initialization or dimension change detection
  bool needs_reinit = !obs_buffer_initialized_;

  // Check if we need to reinitialize due to dimension changes
  if (obs_buffer_initialized_) {
    const int w_dim = w_real.size();
    const int gravity_dim = projected_gravity_real.size();

    // Calculate expected total dimension (yaw is NOT stored in obs_buffer_)
    int expected_total_dim = (q_size + qd_size + action_size + w_dim + gravity_dim) * time_steps;

    if (obs_buffer_.rows() != expected_total_dim) {
      needs_reinit = true;
      VLOG(1) << "Dimension mismatch detected, reinitializing obs_buffer: current=" << obs_buffer_.rows()
              << ", expected=" << expected_total_dim;
    }
  }

  if (needs_reinit) {
    // Initialize all history buffers with dynamic dimensions
    const int w_dim = w_real.size();                        // Angular velocity dimension (3)
    const int gravity_dim = projected_gravity_real.size();  // Gravity dimension (3)

    // Note: Yaw forward vector is NOT stored in obs_buffer_
    // It will be handled separately if needed by the policy
    
    // Initialize history_buffers_ for proprioception (WITHOUT quat_error)
    // quat_error will have its own separate history buffer for goal observation
    history_buffers_.clear();
    
    // Proprioception buffers
    history_buffers_.push_back({&q_history_, q_size});
    history_buffers_.push_back({&qd_history_, qd_size});
    history_buffers_.push_back({&action_history_, action_size});
    history_buffers_.push_back({&w_history_, w_dim});
    history_buffers_.push_back({&gravity_history_, gravity_dim});
    
    // Initialize quat_error_history_ separately (if enabled)
    if (current_profile_ && current_profile_->use_quat_error) {
      const int quat_error_dim = 6;
      quat_error_history_ = Eigen::MatrixXd::Zero(quat_error_dim, time_steps);
    }

    int total_dim = 0;
    for (auto& [buffer, dim] : history_buffers_) {
      *buffer = Eigen::MatrixXd::Zero(dim, time_steps);
      total_dim += dim * time_steps;
    }
    obs_buffer_ = Eigen::MatrixXd::Zero(total_dim, 1);
    obs_buffer_initialized_ = true;

    VLOG(1) << "Initialized observation buffer for " << observation_type_ << ": time_steps=" << time_steps
            << ", total_dim=" << total_dim << ", q_size=" << q_size << ", qd_size=" << qd_size
            << ", action_size=" << action_size 
            << (current_profile_ && current_profile_->use_quat_error ? ", quat_error_dim=6" : "")
            << " (yaw NOT included in obs_buffer)";
  }

  // Update history buffers (new data on the right) - safe buffer access
  const bool needs_shift = time_steps > 1;

  // Update buffers by finding them in history_buffers_ (safer than hardcoded indices)
  for (auto& [buffer, dim] : history_buffers_) {
    if (buffer == &q_history_) {
      if (needs_shift) {
        buffer->leftCols(time_steps - 1) = buffer->rightCols(time_steps - 1);
      }
      buffer->rightCols(1) = q_actual - default_joint_q;
    } else if (buffer == &qd_history_) {
      if (needs_shift) {
        buffer->leftCols(time_steps - 1) = buffer->rightCols(time_steps - 1);
      }
      buffer->rightCols(1) = qd_actual;
    } else if (buffer == &action_history_) {
      if (needs_shift) {
        buffer->leftCols(time_steps - 1) = buffer->rightCols(time_steps - 1);
      }
      buffer->rightCols(1) = mlp_net_action;
    } else if (buffer == &w_history_) {
      if (needs_shift) {
        buffer->leftCols(time_steps - 1) = buffer->rightCols(time_steps - 1);
      }
      buffer->rightCols(1) = w_real;
    } else if (buffer == &gravity_history_) {
      if (needs_shift) {
        buffer->leftCols(time_steps - 1) = buffer->rightCols(time_steps - 1);
      }
      
      // 根据配置决定是否使用重力投影误差
      Eigen::Vector3d gravity_to_store = projected_gravity_real;
      if (current_profile_ && current_profile_->use_gravity_error && ref_quat.has_value()) {
        // 从参考四元数计算参考投影重力
        Eigen::Matrix3d R_ref = ref_quat.value().toRotationMatrix();
        Eigen::Vector3d projected_gravity_ref = -R_ref.transpose() * Eigen::Vector3d::UnitZ();
        
        // 存储差值：当前投影重力 - 参考投影重力
        gravity_to_store = projected_gravity_real - projected_gravity_ref;
        
        VLOG(2) << "Using gravity error: current=" << projected_gravity_real.transpose() 
                << ", ref=" << projected_gravity_ref.transpose() 
                << ", error=" << gravity_to_store.transpose();
      }
      
      buffer->rightCols(1) = gravity_to_store;
    }
  }
  
  // Update quat_error_history_ separately (with 5-frame history for goal observation)
  if (current_profile_ && current_profile_->use_quat_error) {
    // Shift history if needed
    if (needs_shift && quat_error_history_.cols() > 1) {
      quat_error_history_.leftCols(time_steps - 1) = quat_error_history_.rightCols(time_steps - 1);
    }
    
    // Compute current quat_error
    Eigen::VectorXd quat_error_to_store = Eigen::VectorXd::Zero(6);
    if (!ref_quat.has_value() || !current_quat.has_value()) {
      LOG_EVERY_N(WARNING, 100) << "use_quat_error is enabled but ref_quat or current_quat is not provided! "
                                << "ref_quat=" << ref_quat.has_value() << ", current_quat=" << current_quat.has_value();
      // Keep zero vector as fallback
    } else {
      // Compute initial quaternion offset on first valid observation
      // This corrects for the heading difference between robot initial pose and reference initial pose
      if (!initial_quat_offset_computed_) {
        // q_offset = q_robot_init^{-1} * q_ref_init
        // This offset will be applied to all subsequent reference quaternions
        Eigen::Quaterniond q_ref_init = ref_quat.value();
        Eigen::Quaterniond q_robot_init = current_quat.value();
        initial_quat_offset_ = q_robot_init.conjugate() * q_ref_init;
        initial_quat_offset_computed_ = true;
        
        LOG(INFO) << "Initial quaternion offset computed:";
        LOG(INFO) << "  Robot initial quat: [" << q_robot_init.w() << ", " << q_robot_init.x() 
                  << ", " << q_robot_init.y() << ", " << q_robot_init.z() << "]";
        LOG(INFO) << "  Reference initial quat: [" << q_ref_init.w() << ", " << q_ref_init.x() 
                  << ", " << q_ref_init.y() << ", " << q_ref_init.z() << "]";
        LOG(INFO) << "  Computed offset: [" << initial_quat_offset_.value().w() << ", " 
                  << initial_quat_offset_.value().x() << ", " << initial_quat_offset_.value().y() 
                  << ", " << initial_quat_offset_.value().z() << "]";
      }
      
      // Python逻辑 (observations.py:motion_anchor_ori_b):
      // _, ori = subtract_frame_transforms(robot_quat, ref_quat)
      // 其中 subtract_frame_transforms 计算: q_error = q_robot^{-1} * q_ref
      // 然后提取旋转矩阵的前两列: mat[..., :2].reshape(-1)
      
      // Apply initial offset to reference quaternion to correct for initial heading difference
      // q_ref_corrected = q_offset^{-1} * q_ref = q_ref_init^{-1} * q_robot_init * q_ref
      Eigen::Quaterniond q_ref_original = ref_quat.value();
      Eigen::Quaterniond q_ref = initial_quat_offset_.value().conjugate() * q_ref_original;
      Eigen::Quaterniond q_robot = current_quat.value();
      
      // ========== 调试日志：打开文件用于记录详细计算过程 ==========
      static std::ofstream debug_file;
      static int step_counter = 0;
      static bool file_initialized = false;
      const int DEBUG_FRAMES = 20;  // 记录前20帧
      
      if (!file_initialized) {
        std::string log_path = "/tmp/quat_error_debug.log";
        debug_file.open(log_path, std::ios::out);
        if (debug_file.is_open()) {
          debug_file << "=================================================\n";
          debug_file << "Quaternion Error Calculation Debug Log (First " << DEBUG_FRAMES << " Frames)\n";
          debug_file << "=================================================\n\n";
          LOG(INFO) << "Quaternion error debug log created at: " << log_path;
        }
        file_initialized = true;
      }
      
      // 计算相对四元数: q_error = q_robot^{-1} * q_ref (与Python完全一致)
      Eigen::Quaterniond q_robot_conj = q_robot.conjugate();
      Eigen::Quaterniond q_error = q_robot_conj * q_ref;
      Eigen::Quaterniond q_error_before_norm = q_error;  // 保存归一化前的值
      
      // 归一化四元数
      q_error.normalize();
      
      // 转换为旋转矩阵，提取前两列（6维）
      // Python: mat[..., :2].reshape(-1) 按行优先展开
      // 结果顺序: [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
      // 这与numpy/torch的默认reshape行为一致（C-contiguous, row-major）
      Eigen::Matrix3d R_error = q_error.toRotationMatrix();
      quat_error_to_store << R_error(0, 0), R_error(0, 1),  // 第0行的前两个元素
                             R_error(1, 0), R_error(1, 1),  // 第1行的前两个元素
                             R_error(2, 0), R_error(2, 1);  // 第2行的前两个元素
      
      // ========== 详细调试信息写入文件 ==========
      if (debug_file.is_open() && step_counter < DEBUG_FRAMES) {
        debug_file << "========== Step " << step_counter << " ==========\n";
        size_t traj_idx = GetCurrentTrajectoryIndex();
        debug_file << "Trajectory Index: " << traj_idx << "\n\n";
        
        // 从轨迹中提取参考位置和速度
        if (current_traj_ != nullptr && traj_idx < static_cast<size_t>(current_traj_->rows())) {
          const auto& traj = *current_traj_;
          const int num_joints = static_cast<int>(traj.cols() >= 48 ? 24 : traj.cols() / 2);
          
          debug_file << "[0] Reference Trajectory Data:\n";
          if (traj.cols() >= 48) {
            debug_file << "  Reference Joint Positions (first 6 joints): [";
            for (int i = 0; i < std::min(6, num_joints); ++i) {
              debug_file << traj(traj_idx, i);
              if (i < 5) debug_file << ", ";
            }
            debug_file << "]\n";
            
            debug_file << "  Reference Joint Velocities (first 6 joints): [";
            for (int i = 0; i < std::min(6, num_joints); ++i) {
              debug_file << traj(traj_idx, num_joints + i);
              if (i < 5) debug_file << ", ";
            }
            debug_file << "]\n";
            
            debug_file << "  All Reference Joint Positions: [";
            for (int i = 0; i < num_joints; ++i) {
              debug_file << traj(traj_idx, i);
              if (i < num_joints - 1) debug_file << ", ";
            }
            debug_file << "]\n";
            
            debug_file << "  All Reference Joint Velocities: [";
            for (int i = 0; i < num_joints; ++i) {
              debug_file << traj(traj_idx, num_joints + i);
              if (i < num_joints - 1) debug_file << ", ";
            }
            debug_file << "]\n";
          }
        }
        
        debug_file << "\n[1] Initial Quaternion Offset:\n";
        if (initial_quat_offset_.has_value()) {
          debug_file << "  Offset Quaternion (q_offset = q_robot_init^{-1} * q_ref_init):\n";
          debug_file << "    w=" << std::setprecision(10) << initial_quat_offset_.value().w()
                     << ", x=" << initial_quat_offset_.value().x()
                     << ", y=" << initial_quat_offset_.value().y()
                     << ", z=" << initial_quat_offset_.value().z() << "\n";
          debug_file << "    Norm: " << initial_quat_offset_.value().norm() << "\n";
        } else {
          debug_file << "  Offset not computed yet\n";
        }
        
        debug_file << "\n[2] Input Quaternions:\n";
        debug_file << "  Reference Quaternion Original (q_ref_original):\n";
        debug_file << "    w=" << std::setprecision(10) << q_ref_original.w() 
                   << ", x=" << q_ref_original.x() 
                   << ", y=" << q_ref_original.y() 
                   << ", z=" << q_ref_original.z() << "\n";
        debug_file << "    Norm: " << q_ref_original.norm() << "\n";
        
        debug_file << "\n  Reference Quaternion Corrected (q_ref = q_offset^{-1} * q_ref_original):\n";
        debug_file << "    w=" << std::setprecision(10) << q_ref.w() 
                   << ", x=" << q_ref.x() 
                   << ", y=" << q_ref.y() 
                   << ", z=" << q_ref.z() << "\n";
        debug_file << "    Norm: " << q_ref.norm() << "\n";
        
        debug_file << "\n  Robot Quaternion (q_robot):\n";
        debug_file << "    w=" << q_robot.w() 
                   << ", x=" << q_robot.x() 
                   << ", y=" << q_robot.y() 
                   << ", z=" << q_robot.z() << "\n";
        debug_file << "    Norm: " << q_robot.norm() << "\n";
        
        debug_file << "\n[3] Computation Steps:\n";
        debug_file << "  Step 3.1: Conjugate of q_robot (q_robot^{-1}):\n";
        debug_file << "    w=" << q_robot_conj.w() 
                   << ", x=" << q_robot_conj.x() 
                   << ", y=" << q_robot_conj.y() 
                   << ", z=" << q_robot_conj.z() << "\n";
        
        debug_file << "  Step 3.2: Quaternion multiplication (q_robot^{-1} * q_ref_corrected):\n";
        debug_file << "    Before normalization:\n";
        debug_file << "      w=" << q_error_before_norm.w() 
                   << ", x=" << q_error_before_norm.x() 
                   << ", y=" << q_error_before_norm.y() 
                   << ", z=" << q_error_before_norm.z() << "\n";
        debug_file << "      Norm: " << q_error_before_norm.norm() << "\n";
        
        debug_file << "    After normalization:\n";
        debug_file << "      w=" << q_error.w() 
                   << ", x=" << q_error.x() 
                   << ", y=" << q_error.y() 
                   << ", z=" << q_error.z() << "\n";
        debug_file << "      Norm: " << q_error.norm() << "\n";
        
        debug_file << "\n[4] Rotation Matrix (from q_error):\n";
        debug_file << "    R_error =\n";
        debug_file << "      [" << R_error(0,0) << ", " << R_error(0,1) << ", " << R_error(0,2) << "]\n";
        debug_file << "      [" << R_error(1,0) << ", " << R_error(1,1) << ", " << R_error(1,2) << "]\n";
        debug_file << "      [" << R_error(2,0) << ", " << R_error(2,1) << ", " << R_error(2,2) << "]\n";
        
        debug_file << "\n[5] Extract First Two Columns (6D observation):\n";
        debug_file << "    Column 0: [" << R_error(0,0) << ", " << R_error(1,0) << ", " << R_error(2,0) << "]\n";
        debug_file << "    Column 1: [" << R_error(0,1) << ", " << R_error(1,1) << ", " << R_error(2,1) << "]\n";
        
        debug_file << "\n[6] Final 6D Observation Vector (row-major order):\n";
        debug_file << "    [" << quat_error_to_store(0) << ", "  // R[0,0]
                         << quat_error_to_store(1) << ", "  // R[0,1]
                         << quat_error_to_store(2) << ", "  // R[1,0]
                         << quat_error_to_store(3) << ", "  // R[1,1]
                         << quat_error_to_store(4) << ", "  // R[2,0]
                         << quat_error_to_store(5) << "]\n";  // R[2,1]
        
        debug_file << "\n[7] History Buffer State (all " << time_steps << " time steps):\n";
        debug_file << "    Before update:\n";
        for (int t = 0; t < time_steps; ++t) {
          debug_file << "      t-" << (time_steps - 1 - t) << ": [";
          for (int d = 0; d < 6; ++d) {
            debug_file << quat_error_history_(d, t);
            if (d < 5) debug_file << ", ";
          }
          debug_file << "]\n";
        }
        
        debug_file << "\n";
        debug_file.flush();
        
        step_counter++;
        
        if (step_counter == DEBUG_FRAMES) {
          debug_file << "\n=================================================\n";
          debug_file << "First " << DEBUG_FRAMES << " steps recorded. Closing debug log.\n";
          debug_file << "=================================================\n";
          debug_file.close();
          LOG(INFO) << "Quaternion error debug log completed (" << DEBUG_FRAMES << " steps recorded)";
        }
      }
      
      VLOG(2) << "Quaternion error: robot=[" << q_robot.w() << "," << q_robot.x() << "," << q_robot.y() << "," << q_robot.z() 
              << "], ref_original=[" << q_ref_original.w() << "," << q_ref_original.x() << "," << q_ref_original.y() << "," << q_ref_original.z()
              << "], ref_corrected=[" << q_ref.w() << "," << q_ref.x() << "," << q_ref.y() << "," << q_ref.z()
              << "], error_quat=[" << q_error.w() << "," << q_error.x() << "," << q_error.y() << "," << q_error.z() << "]"
              << ", mat_cols=[" << quat_error_to_store.transpose() << "]";
    }
    
    // Store in history buffer
    quat_error_history_.rightCols(1) = quat_error_to_store;
  }

  // Concatenate all history data to obs_buffer_ - optimized version
  int idx = 0;
  for (const auto& [buffer, dim] : history_buffers_) {
    // Use Map for efficient memory copying
    Eigen::Map<Eigen::VectorXd> target(obs_buffer_.data() + idx, dim * time_steps);
    Eigen::Map<const Eigen::VectorXd> source(buffer->data(), dim * time_steps);
    target = source;
    idx += dim * time_steps;
  }

  // 2. Update goal buffer
  Eigen::VectorXd goal_obs = CreateGoalObservation(q_actual, command, default_joint_q);

  // Validate goal observation length against config; auto-correct by truncate/pad
  auto goal_cfg_it = param_.observations.observation_type.find(observation_type_);
  if (goal_cfg_it == param_.observations.observation_type.end()) {
    LOG(ERROR) << "No goal cfg found for observation_type: " << observation_type_;
    return;
  }
  int expected_goal_len = goal_cfg_it->second.goal_length;
  if (goal_obs.size() != expected_goal_len) {
    LOG(ERROR) << "Goal length mismatch for observation_type=" << observation_type_ << ": got=" << goal_obs.size()
               << ", expected=" << expected_goal_len << ", joints_dim=" << q_actual.size()
               << ", command_dim=" << command.size();
    if (goal_obs.size() > expected_goal_len) {
      goal_obs = goal_obs.head(expected_goal_len);
    } else {
      Eigen::VectorXd fixed = Eigen::VectorXd::Zero(expected_goal_len);
      fixed.head(goal_obs.size()) = goal_obs;
      goal_obs = fixed;
    }
  }

  // Update goal buffer (using its own time step)
  auto goal_steps_it = goal_time_steps_.find(observation_type_);
  if (goal_steps_it == goal_time_steps_.end()) {
    LOG(ERROR) << "No goal time steps found for observation_type: " << observation_type_;
    return;
  }
  int goal_steps = goal_steps_it->second;
  if (goal_steps > 1) {
    current_goal_buffer_->leftCols(goal_steps - 1) = current_goal_buffer_->rightCols(goal_steps - 1);
  }
  current_goal_buffer_->col(goal_steps - 1) = goal_obs;
}

void ObservationManager::StepTrajectory(bool replay) {
  // For mimic_tj, always increment phase counter
  if (observation_type_ == "mimic_tj") {
    ++mimic_tj_phase_counter_;
    return;
  }

  // For other observation types, use trajectory-based stepping
  if (!current_traj_) {
    LOG(INFO) << "StepTrajectory: obs_type=" << observation_type_ << ", current_traj_ is NULL (no trajectory set)";
    return;
  }
  size_t end = current_traj_->rows() > 0 ? current_traj_->rows() - 1 : 0;
  if (trajectory_index_ < end) {
    ++trajectory_index_;
  } else {
    if (replay) {
      trajectory_index_ = 0;
    } else {
      trajectory_index_ = end;
    }
  }

  // Debug: log trajectory index occasionally
}

Eigen::MatrixXd ObservationManager::GetCurrentObservation() const {
  if (!current_goal_buffer_) {
    LOG(ERROR) << "Current goal buffer is null";
    return Eigen::MatrixXd::Zero(1, 1);
  }

  const int obs_rows = obs_buffer_.rows();
  const int goal_rows = current_goal_buffer_->rows();

  // 安全获取goal_cols
  auto goal_cols_it = goal_time_steps_.find(observation_type_);
  if (goal_cols_it == goal_time_steps_.end()) {
    LOG(ERROR) << "No goal time steps found for observation_type: " << observation_type_;
    return Eigen::MatrixXd::Zero(1, 1);
  }
  const int goal_cols = goal_cols_it->second;

  // Check if quat_error is enabled and get time_steps
  auto time_steps_it = proprioceptive_time_steps_.find(observation_type_);
  int time_steps = (time_steps_it != proprioceptive_time_steps_.end()) ? time_steps_it->second : 1;
  
  const int quat_error_dim = (current_profile_ && current_profile_->use_quat_error) ? 6 : 0;
  const int quat_error_total = quat_error_dim * time_steps;  // quat_error with history: 6 * 5 = 30

  // Calculate total dimension: proprioceptive (obs_buffer_) + quat_error_history + expanded goal data
  // Note: obs_buffer_ does NOT contain yaw vector (yaw is handled separately if needed)
  const int total_rows = obs_rows + quat_error_total + goal_rows * goal_cols;

  // Create single-column observation vector (simplified to 1D output)
  Eigen::VectorXd observation = Eigen::VectorXd::Zero(total_rows);

  // Fill proprioceptive data from obs_buffer_ (does NOT contain yaw)
  observation.head(obs_rows) = obs_buffer_.col(0);

  int idx = obs_rows;
  
  // Fill quat_error history BEFORE goal data (if enabled)
  // Flatten quat_error_history_: all time steps concatenated
  if (quat_error_total > 0) {
    // Flatten quat_error_history_ (6 rows, time_steps cols) -> (6*time_steps, 1) vector
    Eigen::Map<const Eigen::VectorXd> quat_error_flat(quat_error_history_.data(), quat_error_total);
    observation.segment(idx, quat_error_total) = quat_error_flat;
    idx += quat_error_total;
  }

  // Fill expanded goal data
  for (int t = 0; t < goal_cols; ++t) {
    observation.segment(idx, goal_rows) = current_goal_buffer_->col(t);
    idx += goal_rows;
  }

  // Apply scale and clip
  // Apply proprioceptive scaling
  if (time_steps_it != proprioceptive_time_steps_.end()) {
    int current_time_steps = time_steps_it->second;
    int proprio_dim = obs_rows;

    Eigen::VectorXd current_scale = BuildProprioceptiveScale(proprio_dim, current_time_steps);

    // Apply scaling to all proprioceptive data (yaw not included in obs_buffer_)
    observation.head(obs_rows).array() *= current_scale.array();
  } else {
    LOG(WARNING) << "No time steps found for observation_type: " << observation_type_;
  }

  // Apply quat_error and goal part scaling
  ApplyGoalScaling(observation, obs_rows, quat_error_total, goal_rows, goal_cols);

  // Apply clip
  observation = observation.cwiseMax(-param_.observation_clip).cwiseMin(param_.observation_clip);

  // Return single-column matrix to maintain interface consistency
  return observation.reshaped(total_rows, 1);
}

void ObservationManager::Reset() {
  obs_buffer_.setZero();
  // Reset all history buffers
  if (obs_buffer_initialized_) {
    for (auto& [buffer, _] : history_buffers_) {
      buffer->setZero();
    }
    // Also reset quat_error_history_ (not in history_buffers_ but needs reset)
    if (quat_error_history_.size() > 0) {
      quat_error_history_.setZero();
    }
  }
  // Reset goal buffers
  for (auto& [_, buffer] : goal_buffers_) {
    buffer.setZero();
  }
  trajectory_index_ = 0;
  mimic_tj_phase_counter_ = 0;      // Reset mimic_tj phase counter
  obs_buffer_initialized_ = false;  // Force reinitialization on next UpdateObservation
  // Reset initial quaternion offset
  initial_quat_offset_computed_ = false;
  initial_quat_offset_ = std::nullopt;
}

size_t ObservationManager::GetCurrentTrajectoryIndex() const { return trajectory_index_; }

size_t ObservationManager::GetMimicTjPhaseCounter() const { return mimic_tj_phase_counter_; }

// Get current reference joints (trajectory frame for mimic/residual_control, otherwise default_joint_q)
Eigen::VectorXd ObservationManager::GetCurrentReference() const {
  if ((observation_type_ == "mimic" || observation_type_ == "control_mimic") && current_profile_ &&
      !current_profile_->data_path.empty() && current_traj_) {
    if (trajectory_index_ < current_traj_->rows()) {
      const auto ref = current_traj_->row(trajectory_index_);

      return ref;
    } else {
      const auto ref = current_traj_->row(current_traj_->rows() - 1);

      return ref;
    }
  } else if (observation_type_ == "mimic_future" && current_profile_ &&
             !current_profile_->data_path.empty() && current_traj_) {
    // For mimic_future, return only position part (first half of columns)
    const int num_joints = current_traj_->cols() / 2;
    if (trajectory_index_ < current_traj_->rows()) {
      return current_traj_->row(trajectory_index_).head(num_joints);
    } else {
      return current_traj_->row(current_traj_->rows() - 1).head(num_joints);
    }
  } else {
    // For mimic_tj and other types, return default_joint_q

    return common::ConcatenateVectors(param_.default_joint_q);
  }
}

Eigen::VectorXd ObservationManager::CreateGoalObservation(const Eigen::VectorXd& q_actual,
                                                          const Eigen::Vector3d& command,
                                                          const Eigen::VectorXd& default_joint_q) {
  if (observation_type_ == "mimic") {
    return q_actual - GetCurrentReference();
  } else if (observation_type_ == "control_mimic") {
    // control_mimic: mimic goal plus appended 3D velocity command
    Eigen::VectorXd diff = q_actual - GetCurrentReference();

    Eigen::VectorXd goal_obs(diff.size() + 3);
    goal_obs.head(diff.size()) = diff;
    goal_obs.tail(3) = command;
    return goal_obs;
  } else if (observation_type_ == "mimic_future") {
    // mimic_future: 10 frames of reference data stacked as (pos0, vel0, pos1, vel1, ..., pos9, vel9)
    // This provides the current frame + 9 future frames of reference joint positions and velocities
    const int num_joints = q_actual.size();
    const int num_future_frames = 10;
    const int total_dim = num_future_frames * 2 * num_joints;  // pos + vel for each frame
    
    if (!current_traj_) {
      LOG(ERROR) << "current_traj_ is null for mimic_future, returning zero observation with dim=" << total_dim;
      return Eigen::VectorXd::Zero(total_dim);
    }
    
    Eigen::VectorXd goal_obs(total_dim);
    
    // current_traj_ format loaded by LoadAndInterpolateJointTrajectoryWithVelAndQuat:
    // [N rows, 52 columns] where columns = [pos0...pos23, vel0...vel23, qx, qy, qz, qw]
    // We only extract the position and velocity parts (first 48 columns)
    const size_t max_idx = current_traj_->rows() - 1;
    
    // Extract 10 frames starting from current trajectory index
    for (int frame = 0; frame < num_future_frames; ++frame) {
      size_t traj_idx = std::min(trajectory_index_ + frame, max_idx);
      
      // Extract position and velocity for this frame and stack as (pos, vel)
      // Output format: frame0_pos, frame0_vel, frame1_pos, frame1_vel, ..., frame9_pos, frame9_vel
      int obs_offset = frame * 2 * num_joints;
      
      // Position data (first num_joints columns in current_traj_)
      goal_obs.segment(obs_offset, num_joints) = current_traj_->row(traj_idx).head(num_joints);
      
      // Velocity data (next num_joints columns in current_traj_, after position)
      goal_obs.segment(obs_offset + num_joints, num_joints) = current_traj_->row(traj_idx).segment(num_joints, num_joints);
      
      // Note: Quaternion data (columns 48-51) is extracted separately in rl_dance_runner.cc
      // and used to compute gravity projection difference in UpdateObservation
    }
    
    return goal_obs;
  } else if (observation_type_ == "rl_locomotion") {
    // Use member variables instead of static variables to avoid thread safety issues
    double cycle_time = CalculateDynamicCycleTime(command);
    gait_phase_ += runner_period_ / cycle_time;
    gait_phase_ = gait_phase_ - std::floor(gait_phase_);  // More precise modulo operation

    const int phase_dim = 2;                 // sin/cos phase dimension
    const int command_dim = command.size();  // Command dimension (3)
    Eigen::VectorXd goal_obs(phase_dim + command_dim);
    const double phase_2pi = math::k2Pi * gait_phase_;
    goal_obs.segment(0, phase_dim) = Eigen::Vector2d(std::sin(phase_2pi), std::cos(phase_2pi));
    goal_obs.segment(phase_dim, command_dim) = command;
    return goal_obs;
  } else if (observation_type_ == "rl_saw_locomotion") {
    return command;
  } else if (observation_type_ == "mimic_tj") {
    // mimic_tj uses step_phase to calculate sin and cos
    if (!current_profile_) {
      LOG(ERROR) << "Current profile is null for mimic_tj";
      const int mimic_tj_dim = 2;  // sin/cos dimension for mimic_tj
      return Eigen::VectorXd::Zero(mimic_tj_dim);
    }

    // Check if step_phase is configured
    if (current_profile_->step_phase == 0.0) {
      LOG(ERROR) << "step_phase is not configured for mimic_tj observation_type in profile: " << profile_key_;
      throw std::runtime_error("step_phase is required for mimic_tj observation_type but not configured");
    }

    // Check if phase_range is configured
    if (current_profile_->phase_range.empty() || current_profile_->phase_range.size() < 2) {
      LOG(ERROR) << "phase_range is not configured for mimic_tj observation_type in profile: " << profile_key_;
      throw std::runtime_error("phase_range is required for mimic_tj observation_type but not configured");
    }

    // Get configuration parameters
    double step_phase = current_profile_->step_phase;
    double phase_start = current_profile_->phase_range[0];
    double phase_end = current_profile_->phase_range[1];

    // Calculate current step (using mimic_tj_phase_counter_ as step)
    double step = static_cast<double>(mimic_tj_phase_counter_);

    // Calculate normalized phase within the specified range
    // phase = phase_start + (phase_end - phase_start) * (step_phase * step)
    double phase =
        phase_start + (phase_end - phase_start) * (step_phase * (2 * 3.14159) / (phase_end - phase_start) * step);

    // For sin/cos calculation, we can use the phase directly without fmod
    // since sin and cos are periodic functions that work with any phase value

    const int mimic_tj_dim = 2;  // sin/cos dimension for mimic_tj
    Eigen::VectorXd goal_obs(mimic_tj_dim);
    goal_obs(0) = std::sin(phase);
    goal_obs(1) = std::cos(phase);

    return goal_obs;
  } else {
    throw std::runtime_error("Unknown observation_type: " + observation_type_);
  }
}

// Calculate dynamic cycle time based on command velocity
// Implement user-provided algorithm: dynamically adjust gait cycle time based on command velocity
double ObservationManager::CalculateDynamicCycleTime(const Eigen::Vector3d& command) const {
  // Check if configuration is valid
  if (!current_profile_ || current_profile_->cycle_time.size() < 2) {
    // Return default value if cycle_time range is not configured
    return param_.default_dynamic_cycle_time;
  }

  // Get cycle time range [min, max]
  double cycT_min = current_profile_->cycle_time[0];  // Minimum cycle time
  double cycT_max = current_profile_->cycle_time[1];  // Maximum cycle time

  // Calculate linear and angular velocities
  double lin_speed = std::sqrt(command.x() * command.x() + command.y() * command.y());  // Horizontal velocity
  double ang_speed = std::abs(command.z());  // Angular velocity (absolute value)

  // Set velocity range based on zero_threshold and command_scale in config file
  // Linear velocity range: from zero_threshold to command_scale * 1.0 (assuming max input is 1.0)
  double v_min = param_.lin_vel_set_zero_threshold;
  double v_max = param_.command_scale.x() * 1.0;  // Assuming max joystick input is 1.0

  // Angular velocity range: from zero_threshold to command_scale * 1.0
  double w_min = param_.ang_vel_set_zero_threshold;
  double w_max = param_.command_scale.z() * 1.0;  // Assuming max joystick input is 1.0

  // 1) Velocity normalization → combined intensity
  double s_lin = std::max(0.0, std::min(1.0, (lin_speed - v_min) / (v_max - v_min)));
  double s_ang = std::max(0.0, std::min(1.0, (ang_speed - w_min) / (w_max - w_min)));
  double intensity = std::max(0.0, std::min(1.0, std::sqrt(s_lin * s_lin + (current_profile_->k_angle * s_ang) *
                                                                               (current_profile_->k_angle * s_ang))));

  // 2) Initial cycle calculation: higher intensity → shorter cycle
  double cycle_time = cycT_max + (cycT_min - cycT_max) * intensity;

  // 3) Stride constraints
  const double eps = 1e-6;  // Avoid division by zero
  double stride_len = lin_speed * cycle_time / 2.0;
  bool need_shorten_stride = stride_len > current_profile_->stride_max;
  double cycle_time_stride = 2.0 * current_profile_->stride_max / (lin_speed + eps);
  // Use torch.where-like logic: if need_shorten_stride is true, use cycle_time_stride, otherwise keep cycle_time
  cycle_time = need_shorten_stride ? cycle_time_stride : cycle_time;

  // 4) Single cycle yaw constraints
  double yaw_per_cycle = ang_speed * cycle_time;
  bool need_shorten_yaw = yaw_per_cycle > current_profile_->yaw_per_cycle_max;
  double cycle_time_yaw = current_profile_->yaw_per_cycle_max / (ang_speed + eps);
  // Use torch.where-like logic: if need_shorten_yaw is true, use cycle_time_yaw, otherwise keep cycle_time
  cycle_time = need_shorten_yaw ? cycle_time_yaw : cycle_time;

  // 5) Clamp to global range
  cycle_time = std::max(cycT_min, std::min(cycT_max, cycle_time));
  return cycle_time;
}

Eigen::VectorXd ObservationManager::BuildProprioceptiveScale(int proprio_dim, int time_steps) const {
  Eigen::VectorXd current_scale = Eigen::VectorXd::Zero(proprio_dim);
  int idx = 0;

  // NOTE: Yaw forward vector is NOT stored in obs_buffer_, so it's not included here
  // The obs_buffer_ only contains: q_diff, qd, action, angular_vel, gravity
  std::vector<double> scales = {
      param_.observations.observation_scale.observation_scale_dof_pos,      // q_diff
      param_.observations.observation_scale.observation_scale_dof_vel,      // qd
      1.0,                                                                  // action
      param_.observations.observation_scale.observation_scale_angular_vel,  // w
      param_.observations.observation_scale.observation_scale_quat          // gravity
  };

  // Dimensions correspond to: q_diff, qd, action, angular_vel, gravity
  const int joint_dim = static_cast<int>(param_.active_joint_names.size());
  const int angular_vel_dim = 3;  // Angular velocity dimension
  const int gravity_dim = 3;      // Gravity dimension

  std::vector<int> dims = {
      joint_dim,        // q_diff dimension
      joint_dim,        // qd dimension
      joint_dim,        // action dimension
      angular_vel_dim,  // angular velocity dimension
      gravity_dim       // gravity dimension
  };

  for (size_t buffer_idx = 0; buffer_idx < scales.size(); ++buffer_idx) {
    for (int t = 0; t < time_steps; ++t) {
      int segment_size = dims[buffer_idx];
      current_scale.segment(idx, segment_size) = Eigen::VectorXd::Constant(segment_size, scales[buffer_idx]);
      idx += segment_size;
    }
  }

  return current_scale;
}

void ObservationManager::ApplyGoalScaling(Eigen::VectorXd& observation, int obs_rows, int quat_error_total,
                                          int goal_rows, int goal_cols) const {
  // Apply quat_error scaling to all history frames (if present)
  // quat_error_total = 6 * time_steps (e.g., 6 * 5 = 30)
  if (quat_error_total > 0) {
    double quat_error_scale = param_.observations.observation_scale.observation_scale_quat_error;
    observation.segment(obs_rows, quat_error_total) *= quat_error_scale;
  }
  
  // Starting index for goal data (after proprioception and quat_error_history)
  int goal_start_idx = obs_rows + quat_error_total;
  
  auto goal_cfg_it = param_.observations.observation_type.find(observation_type_);
  if (goal_cfg_it != param_.observations.observation_type.end()) {
    double goal_scale = goal_cfg_it->second.goal_scale;
    
    // Special handling for mimic_future: different scales for pos and vel
    if (observation_type_ == "mimic_future") {
      // goal_rows contains data for all 10 frames in format: frame0_pos, frame0_vel, frame1_pos, frame1_vel, ...
      // Each goal history step (goal_cols) contains this full sequence
      const int num_joints = static_cast<int>(param_.active_joint_names.size());
      const int num_future_frames = 10;
      const double pos_scale = 1.0;  // Scale for position
      const double vel_scale = 0.05; // Scale for velocity
      
      // Apply scaling to each history step
      for (int t = 0; t < goal_cols; ++t) {
        int base_idx = goal_start_idx + t * goal_rows;
        
        // For each of the 10 future frames within this history step
        for (int frame = 0; frame < num_future_frames; ++frame) {
          int frame_offset = base_idx + frame * 2 * num_joints;
          // Scale position part (first num_joints elements of this frame)
          observation.segment(frame_offset, num_joints) *= pos_scale;
          // Scale velocity part (next num_joints elements of this frame)
          observation.segment(frame_offset + num_joints, num_joints) *= vel_scale;
        }
      }
    } else {
      // Apply scaling to goal part: each time step goal uses same scale
      for (int t = 0; t < goal_cols; ++t) {
        int start_idx = goal_start_idx + t * goal_rows;
        observation.segment(start_idx, goal_rows) *= goal_scale;
      }
    }
  }
}

}  // namespace rl_dance