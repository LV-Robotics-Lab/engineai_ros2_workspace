#pragma once

#include <mujoco/mujoco.h>
#include <array>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <string_view>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include "config_loader.h"
#include "ros_interface.h"
#include "simulate/simulate.h"
#include "simulate/glfw_adapter.h"

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
  
  // Get all active perturbations for CSV recording
  std::vector<PerturbationData> GetActivePerturbations() const;
  int GetNextPerturbationId() { return next_perturbation_id_++; }

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
  double perturb_force_magnitude_ = 20.0;
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

};