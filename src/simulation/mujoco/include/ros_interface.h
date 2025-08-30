#ifndef MUJOCO_ROS_INTERFACE_H_
#define MUJOCO_ROS_INTERFACE_H_

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>

#include "interface_protocol/msg/imu_info.hpp"
#include "interface_protocol/msg/joint_command.hpp"
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/motion_state.hpp"
#include "interface_protocol/msg/contact_force.hpp"
#include "rclcpp/rclcpp.hpp"
#include <mujoco/mujoco.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Forward declarations
namespace mujoco {
class Simulate;
}
class ConfigLoader;

namespace mujoco {

class RosInterface {
 public:
  RosInterface(const std::shared_ptr<rclcpp::Node>& node, std::shared_ptr<ConfigLoader> config_loader);
  ~RosInterface();

  // Initialize the MuJoCo interface
  bool Initialize();

  // Callback for joint command messages
  void JointCommandCallback(const interface_protocol::msg::JointCommand::SharedPtr msg);

  // Update the simulation state to publish to ROS
  void UpdateSimState(const mjModel* m, mjData* d);

  // Get joint command values (thread-safe)
  interface_protocol::msg::JointCommand GetCommandedSafe();

  // Set the current mjModel and mjData
  void SetModelAndData(mjModel* model, mjData* data);

  void PublishContacts(const mjModel* m, const mjData* d);

  // Get the ROS node
  std::shared_ptr<rclcpp::Node> GetNode() const { return node_; }

  // Publish contact forces
  void PublishContactForces(const mjModel* m, mjData* d);

 private:
  // ROS2 node
  std::shared_ptr<rclcpp::Node> node_;

  // Publishers
  rclcpp::Publisher<interface_protocol::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<interface_protocol::msg::ImuInfo>::SharedPtr imu_pub_;
  rclcpp::Publisher<interface_protocol::msg::MotionState>::SharedPtr motion_state_pub_;
  rclcpp::Publisher<interface_protocol::msg::ContactForce>::SharedPtr contact_force_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr contact_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr contact_marker_pub_;
  
  // Subscribers
  rclcpp::Subscription<interface_protocol::msg::JointCommand>::SharedPtr joint_cmd_sub_;

  // Config loader
  std::shared_ptr<ConfigLoader> config_loader_;

  // Number of joints
  int num_total_joints_ = 0;

  // Current joint command
  interface_protocol::msg::JointCommand joint_command_;

  // MuJoCo model and data
  mjModel* model_;
  mjData* data_;

  // Timer for publishing motion state
  rclcpp::TimerBase::SharedPtr motion_state_timer_;
  
  // Motion state timer callback
  void MotionStateTimerCallback();

  // Contact force publishing parameters
  bool export_contact_ = false;
  std::string contact_topic_ = "/mujoco/contact_forces";

  // CSV logging parameters
  bool save_contact_csv_ = false;
  std::string csv_file_path_;
  std::ofstream csv_file_;
  std::mutex csv_mutex_;
  int csv_save_frequency_ = 1;  // 保存频率，1表示每帧都保存

  // Mutex for thread safety
  mutable std::mutex mtx_;

  // Flag indicating if we have a floating base robot
  bool is_floating_base_;
};

}  // namespace mujoco

#endif  // MUJOCO_ROS_INTERFACE_H_
