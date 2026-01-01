#pragma once

#include <mujoco/mujoco.h>
#include <array>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <string_view>
#include <thread>
#include <vector>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include "config_loader.h"
#include "ros_interface.h"
#include "simulate/simulate.h"
#include "simulate/glfw_adapter.h"
#include "joint_forces_eigen.hpp"
#include "force_interpolation.h"

// Forward declaration
class CustomGlfwAdapter;

class SimManager {
 public:
  // Get the singleton instance
  static SimManager& GetInstance();

  // Delete copy constructor and assignment operator
  SimManager(const SimManager&) = delete;
  SimManager& operator=(const SimManager&) = delete;

  // Initialize the simulation
  bool Initialize();

  // Run the simulation
  void Run();

  // Controller callback used by MuJoCo
  void TorqueController(const mjModel* m, mjData* d);

  // Perturbation control functions
  void SetPerturbationForce(const Eigen::Vector3d& force);
  void SetPerturbationTorque(const Eigen::Vector3d& torque);
  void SetPerturbationBody(const std::string& body_name);
  void ApplyPerturbation(bool apply);
  void StopPerturbation();
  
  // Perturbation parameter getters and setters
  double GetPerturbationForceMagnitude() const { return perturb_force_magnitude_; }
  void SetPerturbationForceMagnitude(double magnitude) { perturb_force_magnitude_ = magnitude; }
  double GetPerturbationTorqueMagnitude() const { return perturb_torque_magnitude_; }
  void SetPerturbationTorqueMagnitude(double magnitude) { perturb_torque_magnitude_ = magnitude; }
  double GetPerturbationDuration() const { return perturb_duration_; }
  void SetPerturbationDuration(double duration) { perturb_duration_ = duration; }
  const std::string& GetPerturbationBodyName() const { return perturb_body_name_; }
  
  // Get current perturbation status for visualization
  bool IsPerturbationActive() const { return apply_perturb_; }
  const Eigen::Vector3d& GetCurrentPerturbationForce() const { return perturb_force_; }
  const Eigen::Vector3d& GetCurrentPerturbationTorque() const { return perturb_torque_; }
  
  // Multi-perturbation support
  struct PerturbationData {
    int id;
    std::string body_name;
    Eigen::Vector3d force;
    Eigen::Vector3d torque;
    double start_time;
    double duration;
    bool is_active;
  };

// 全局函数声明
void ApplyPerturbationForcesFromSimManager();
bool IsContactVisualizationEnabled();
  
  // Get all active perturbations for CSV recording
  std::vector<PerturbationData> GetActivePerturbations() const;
  int GetNextPerturbationId() { return next_perturbation_id_++; }
  
  // Configuration access methods
  double GetDefaultForceMagnitude() const;
  double GetDefaultTorqueMagnitude() const;
  double GetForceStep() const;
  double GetTorqueStep() const;
  const std::vector<double>& GetPerturbationForce(const std::string& direction) const;
  const std::vector<double>& GetPerturbationForce(const std::string& direction, double magnitude) const;
  const std::vector<double>& GetPerturbationTorque(const std::string& direction) const;
  const std::vector<double>& GetPerturbationTorque(const std::string& direction, double magnitude) const;

  // 推力施加函数 - 在UI系统清空外力后调用
  void ApplyPerturbationForces();
  
  // 重置auto_sampling状态，用于MuJoCo重置后重新触发推力
  void ResetAutoSampling();
  
  // 获取配置加载器
  std::shared_ptr<ConfigLoader> GetConfigLoader() const { return config_loader_; }
  
  // 关节反力计算相关方法
  // 计算并更新关节反力数据（在mj_step或mj_forward之后调用）
  void ComputeJointForces();
  
  // 获取子body坐标系下的关节反力
  const std::vector<WrenchEigen>& GetJointWrenchesChild() const { return joint_wrenches_child_; }
  
  // 获取父body坐标系下的关节反力
  const std::vector<WrenchEigen>& GetJointWrenchesParent() const { return joint_wrenches_parent_; }
  
  // 获取每个link两端的关节受力
  const std::vector<LinkEndForcesEigen>& GetLinkEndWrenches() const { return link_end_wrenches_; }
  
  // 获取指定关节的载荷分解（轴向力、剪切力、扭矩、弯矩）
  // @param joint_name 关节名称
  // @return 载荷分解结果，如果关节不存在则返回nullptr
  std::unique_ptr<DecomposedWrenchEigen> GetJointDecomposedWrench(const std::string& joint_name) const;

  // 防护护具相关方法
  // 设置防护材料厚度
  void SetProtectionThickness(double thickness) { protection_thickness_ = thickness; }
  double GetProtectionThickness() const { return protection_thickness_; }
  
  // 启用/禁用防护功能
  void EnableProtection(bool enable) { protection_enabled_ = enable; }
  bool IsProtectionEnabled() const { return protection_enabled_; }

 private:
  // Private constructor for singleton
  SimManager();
  ~SimManager();

  // Private member functions
  void PhysicsThread(std::string_view filename);
  void PhysicsLoop();
  mjModel* LoadModel(std::string_view file);
  const char* Diverged(int disableflags, const mjData* d);
  void HandleDropLoad();
  void HandleUILoad();
  
  // 应用防护到接触力（在mj_fwdConstraint之后调用）
  // 该函数会修改d->efc_force来减少接触力
  void ApplyProtectionToContactForces();

  // Private member variables
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<ConfigLoader> config_loader_;
  std::unique_ptr<mujoco::RosInterface> ros_interface_;
  std::unique_ptr<mujoco::Simulate> sim_;
  std::thread physics_thread_;

  // MuJoCo model and data
  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;

  std::array<char, 1024> mj_load_error_;

  // Perturbation control variables
  bool apply_perturb_ = false;
  double perturb_force_magnitude_ = 20.0;  // 将在初始化时从YAML加载
  double perturb_torque_magnitude_ = 5.0;
  Eigen::Vector3d perturb_force_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d perturb_torque_ = Eigen::Vector3d::Zero();
  std::string perturb_body_name_ = "LINK_TORSO_YAW";
  double perturb_duration_ = 0.2;
  double perturb_start_time_ = -1.0;
  
  // Multi-perturbation support
  std::vector<PerturbationData> active_perturbations_;
  int next_perturbation_id_ = 1;
  mutable std::mutex perturbation_mutex_;
  
  // Auto-sampling state variables
  bool auto_triggered_ = false;
  double auto_start_time_ = -1.0;
  bool auto_completed_ = false;
  
  // 关节反力数据存储
  std::vector<WrenchEigen> joint_wrenches_child_;   // 子body坐标系下的关节反力
  std::vector<WrenchEigen> joint_wrenches_parent_;  // 父body坐标系下的关节反力
  std::vector<LinkEndForcesEigen> link_end_wrenches_;  // 每个link两端的关节受力
  mutable std::mutex joint_forces_mutex_;  // 保护关节反力数据的互斥锁
  
  // 防护护具相关成员变量
  std::unique_ptr<ForceInterpolation> force_interpolator_;  // 力插值计算器
  double protection_thickness_ = 12.0;  // 防护材料厚度 (mm)，默认12mm
  bool protection_enabled_ = false;  // 是否启用防护功能，默认启用
  
  // 排除防护的接触对列表（body1_name, body2_name）
  // 例如：排除 "LINK_ANKLE_PITCH_L" 和 "world" 之间的接触
  // 使用 std::pair<std::string, std::string> 存储，顺序无关
  static const std::vector<std::pair<std::string, std::string>> excluded_contact_pairs_;

};