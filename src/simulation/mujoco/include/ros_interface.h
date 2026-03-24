#ifndef MUJOCO_ROS_INTERFACE_H_
#define MUJOCO_ROS_INTERFACE_H_

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <optional>

#include "interface_protocol/msg/imu_info.hpp"
#include "interface_protocol/msg/joint_command.hpp"
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/motion_state.hpp"
#include "interface_protocol/msg/contact_force.hpp"
#include "rclcpp/rclcpp.hpp"
#include <mujoco/mujoco.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
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

  /** 物理线程结束前调用：停写队列、join 写入线程并 flush，避免进程异常退出时 bin 截断或栈错乱 */
  void DrainBinaryWritersAndFlush();

  // Get the ROS node
  std::shared_ptr<rclcpp::Node> GetNode() const { return node_; }

  // 连续采集模式：切换到新实验目录，关闭旧CSV、打开新CSV、触发reset
  void PrepareNewExperiment(const std::string& csv_dir);

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
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr mujoco_reset_pub_;
  
  // Subscribers
  rclcpp::Subscription<interface_protocol::msg::JointCommand>::SharedPtr joint_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr new_experiment_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr next_direction_sub_;

  // Config loader
  std::shared_ptr<ConfigLoader> config_loader_;

  // Number of joints
  int num_total_joints_ = 0;
  
  // Maximum number of perturbations
  int max_perturbations_ = 2;

  // Current joint command
  interface_protocol::msg::JointCommand joint_command_;

  // MuJoCo model and data
  mjModel* model_;
  mjData* data_;

  // 上一帧仿真时间，用于检测用户/GUI 触发的 reset（时间回退时发布 reset_complete）
  double last_sim_time_ = -1.0;

  // Timer for publishing motion state
  rclcpp::TimerBase::SharedPtr motion_state_timer_;
  
  // Motion state timer callback
  void MotionStateTimerCallback();

  // Save joint forces to CSV file
  void SaveJointForcesToCSV(const mjModel* m, mjData* d);

  // Contact force publishing parameters
  bool export_contact_ = false;
  std::string contact_topic_ = "/mujoco/contact_forces";

  // CSV logging parameters
  bool save_contact_csv_ = false;
  bool save_perturbation_csv_ = false;
  std::string csv_file_path_;
  std::ofstream csv_file_;
  std::mutex csv_mutex_;
  int csv_save_frequency_ = 1;  // 保存频率，1表示每帧都保存
  std::string csv_format_ = "csv";  // 格式：csv 或 binary
  /// >0 时：contact_data 仅写入接触力模长≥该值的行（单位 N）。0=不过滤。
  double csv_min_force_n_ = 0.0;
  /// 与 csv_min_force_n 配合：joint_forces_data 保留条件为 (未设力阈值或 ||F||≥) OR (未设力矩阈值或 ||M||≥)。单位 N·m，0=仅按力阈值过滤（若力阈值亦为 0 则不过滤）。
  double csv_joint_forces_min_torque_nm_ = 0.0;

  // Perturbation CSV logging parameters
  std::string perturbation_csv_file_path_;
  std::ofstream perturbation_csv_file_;
  std::mutex perturbation_csv_mutex_;

  // Joint forces CSV logging parameters
  bool save_joint_forces_csv_ = false;
  std::string joint_forces_csv_file_path_;
  std::ofstream joint_forces_csv_file_;
  std::mutex joint_forces_csv_mutex_;

  // Sensor vibration CSV logging parameters (持续记录，不依赖接触力)
  bool save_sensor_vibration_csv_ = false;
  std::string sensor_vibration_csv_file_path_;
  std::ofstream sensor_vibration_csv_file_;
  std::mutex sensor_vibration_csv_mutex_;

  // Joint state CSV logging parameters (持续记录关节位置、速度、力矩)
  bool save_joint_state_csv_ = false;
  std::string joint_state_csv_file_path_;
  std::ofstream joint_state_csv_file_;
  std::mutex joint_state_csv_mutex_;

  // Link kinetic energy CSV logging parameters (持续记录每个link的动能)
  bool save_link_kinetic_energy_csv_ = false;
  std::string link_kinetic_energy_csv_file_path_;
  std::ofstream link_kinetic_energy_csv_file_;
  std::mutex link_kinetic_energy_csv_mutex_;
  bool link_kinetic_energy_csv_header_written_ = false;  // 每次打开新文件需重置，保证写表头
  /** binary：首帧前写入 MJLKEN02 + 各 body 名，供 Python 与 CSV 列名一致 */
  bool link_ke_bin_schema_written_ = false;

  // Policy switch CSV (RL policy 切换事件：walking↔mimic↔damping)
  bool save_policy_switch_csv_ = false;
  std::string policy_switch_csv_file_path_;
  std::ofstream policy_switch_csv_file_;
  std::mutex policy_switch_csv_mutex_;
  // CSV 模式下：由于需要用到 m->nbody 生成展开列表头，这个表头会在第一次写入时延迟生成
  bool policy_switch_csv_header_written_ = false;
  struct PendingPolicySwitch {
    std::string from_mode;
    std::string to_mode;
    std::string mimic_direction;
    bool has_data = false;
  } pending_policy_switch_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr policy_switch_sub_;
  void PolicySwitchCallback(const std_msgs::msg::String::SharedPtr msg);

  // 异步写入数据结构
  struct ContactDataRow {
    double sim_time;
    int contact_id;
    std::string body1_name;
    std::string body2_name;
    double red_ball_pos[3];
    double green_ball_pos[3];
    double world_forces[3];
    double force_magnitude;
    double force_normal;
    double force_friction;   // 接触坐标系下切向力大小 sqrt(f_c[1]^2 + f_c[2]^2)
    double protection_scale = 1.0;  // 1.0=未衰减，<1.0=护具生效后的缩放因子
    double world_torques[3];
    double base_link_pos[3];
    double base_link_quat[4];
    double base_link_vel[3];
    double base_link_angvel[3];
    double collision_link_pos[3];
    double collision_link_quat[4];
  };

  struct PerturbationDataRow {
    double sim_time;
    int perturbation_id;
    std::string body_name;
    double start_time;
    double duration;
    double elapsed_time;
    double force[3];
    double force_magnitude;
    double torque[3];
    double torque_magnitude;
    double std_pose[3];
    double world_force[3];
    double world_force_magnitude;
  };

  // 异步写入队列和线程
  std::queue<ContactDataRow> contact_data_queue_;
  std::queue<PerturbationDataRow> perturbation_data_queue_;
  std::mutex contact_queue_mutex_;
  std::mutex perturbation_queue_mutex_;
  std::thread contact_writer_thread_;
  std::thread perturbation_writer_thread_;
  std::atomic<bool> writer_threads_running_{false};
  std::condition_variable contact_queue_cv_;
  std::condition_variable perturbation_queue_cv_;
  int flush_interval_ = 100;  // 接触/扰动异步写线程：每 N 条 flush
  /** 关节力/振动/joint_state/link_energy 等同步写：每 N 帧 flush（binary 时用更小值减少异常退出丢尾） */
  int recording_flush_interval_ = 100;
  int contact_flush_counter_ = 0;
  int perturbation_flush_counter_ = 0;
  static constexpr size_t MAX_QUEUE_SIZE = 10000;  // CSV 模式下队列满则丢最旧；binary 模式下阻塞直至有空间
  size_t contact_queue_dropped_ = 0;  // 丢弃的数据计数
  size_t perturbation_queue_dropped_ = 0;  // 丢弃的数据计数

  // 异步写入线程函数
  void ContactWriterThread();
  void PerturbationWriterThread();
  
  // 将数据添加到队列（非阻塞）
  void EnqueueContactData(const ContactDataRow& row);
  void EnqueuePerturbationData(const PerturbationDataRow& row);
  
  // 二进制格式写入函数
  void WriteContactDataBinary(const ContactDataRow& row);
  void WritePerturbationDataBinary(const PerturbationDataRow& row);
  
  // 刷新剩余数据
  void FlushRemainingData();
  /** flush 所有正在写入的日志流（物理线程结束前调用，避免仅 contact 落盘而其它 bin 仍在缓冲区） */
  void FlushAllRecordingStreams();
  
  // 传感器震动数据记录函数
  void SaveSensorVibrationToCSV(const mjModel* m, mjData* d);
  
  // 关节状态数据记录函数（持续记录位置、速度、力矩）
  void SaveJointStateToCSV(const mjModel* m, mjData* d);
  
  // Link动能数据记录函数（持续记录每个link的动能 0.5*m*v^2）
  void SaveLinkKineticEnergyToCSV(const mjModel* m, mjData* d);

  // 在指定目录打开所有CSV文件（供 PrepareNewExperiment 复用）
  void OpenCsvFilesAtDir(const std::string& csv_dir);
  // 关闭所有CSV文件并停止写入线程
  void CloseAllCsvFiles();
  void NewExperimentCallback(const std_msgs::msg::String::SharedPtr msg);
  void NextDirectionCallback(const std_msgs::msg::Float64::SharedPtr msg);

  std::optional<double> next_direction_angle_;  // 连续采集时，下次实验的推力方向（度）

  // Mutex for thread safety
  mutable std::mutex mtx_;
  
  // Global mutex for contact force publishing synchronization
  mutable std::mutex contact_force_mutex_;

  // Flag indicating if we have a floating base robot
  bool is_floating_base_;
};

}  // namespace mujoco

#endif  // MUJOCO_ROS_INTERFACE_H_
