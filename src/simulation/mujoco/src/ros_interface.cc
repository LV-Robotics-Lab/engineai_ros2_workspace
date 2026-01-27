/**
 * @file ros_interface.cc
 * @brief MuJoCo仿真器与ROS2的接口实现
 * @details 该文件实现了MuJoCo仿真器与ROS2系统的接口，包括：
 *          - 关节状态发布
 *          - IMU数据发布
 *          - 接触力数据发布和CSV保存
 *          - 关节命令订阅
 *          - 运动状态定时发布
 */

#include "ros_interface.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <rclcpp/logging.hpp>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <cstdlib>

#include "config_loader.h"
#include "sim_manager.h"
#include "joint_forces_eigen.hpp"
#include "rclcpp/rclcpp.hpp"
#include <mujoco/mujoco.h>
#include <std_msgs/msg/empty.hpp>


// 常量定义
const int kDofFloatingBase = 6;        // 浮动基座的自由度数量
const int kNumFloatingBaseJoints = 7;  // 浮动基座的关节数量（四元数 + xyz位置）
const int kDimQuaternion = 4;          // 四元数的维度

// 全局标志位：控制CSV记录时机
static bool contact_csv_enabled = false;      // 是否启用contact CSV记录
static bool perturbation_csv_enabled = false;  // 是否启用perturbation CSV记录

// 3D旋转变换函数
/**
 * @brief 矩阵转置乘以向量：o = R^T * v
 * @param M 3x3旋转矩阵（按行存储）
 * @param v 输入向量
 * @param o 输出向量
 */
inline void rotT3(const double* M, const double* v, double* o) {
    o[0] = M[0]*v[0] + M[3]*v[1] + M[6]*v[2];
    o[1] = M[1]*v[0] + M[4]*v[1] + M[7]*v[2];
    o[2] = M[2]*v[0] + M[5]*v[1] + M[8]*v[2];
}

/**
 * @brief 矩阵乘以向量：o = R * v
 * @param M 3x3旋转矩阵（按行存储）
 * @param v 输入向量
 * @param o 输出向量
 */
inline void rot3(const double* M, const double* v, double* o) {
    o[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    o[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    o[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

namespace mujoco {

/**
 * @brief 构造函数
 * @param node ROS2节点指针
 * @param config_loader 配置加载器指针
 */
RosInterface::RosInterface(const std::shared_ptr<rclcpp::Node>& node, std::shared_ptr<ConfigLoader> config_loader)
    : node_(node), config_loader_(config_loader), model_(nullptr), data_(nullptr), is_floating_base_(false) {}

/**
 * @brief 析构函数
 * @details 关闭CSV文件并输出保存路径信息
 */
RosInterface::~RosInterface() {
  // 停止后台写入线程
  if (writer_threads_running_.load()) {
    writer_threads_running_ = false;
    
    // 通知所有等待的线程
    contact_queue_cv_.notify_all();
    perturbation_queue_cv_.notify_all();
    
    // 等待线程结束
    if (contact_writer_thread_.joinable()) {
      contact_writer_thread_.join();
    }
    if (perturbation_writer_thread_.joinable()) {
      perturbation_writer_thread_.join();
    }
    
    // 确保所有队列中的数据都被写入
    FlushRemainingData();
  }
  
  // 关闭接触力文件
  if (csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Contact data saved to: %s", csv_file_path_.c_str());
  }
  
  // 关闭推力文件
  if (perturbation_csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(perturbation_csv_mutex_);
    perturbation_csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Perturbation data saved to: %s", perturbation_csv_file_path_.c_str());
  }
  
  // 关闭关节反力CSV文件
  if (joint_forces_csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(joint_forces_csv_mutex_);
    joint_forces_csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Joint forces data saved to: %s", joint_forces_csv_file_path_.c_str());
  }

  // 关闭传感器震动CSV文件
  if (sensor_vibration_csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(sensor_vibration_csv_mutex_);
    sensor_vibration_csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Sensor vibration data saved to: %s", sensor_vibration_csv_file_path_.c_str());
  }
  
  // 报告队列统计信息
  if (contact_queue_dropped_ > 0) {
    RCLCPP_WARN(node_->get_logger(), "Contact data queue dropped %zu records during execution", contact_queue_dropped_);
  }
  if (perturbation_queue_dropped_ > 0) {
    RCLCPP_WARN(node_->get_logger(), "Perturbation data queue dropped %zu records during execution", perturbation_queue_dropped_);
  }
}

/**
 * @brief 初始化ROS接口
 * @return 初始化是否成功
 * @details 初始化包括：
 *          - 读取接触力导出参数
 *          - 设置CSV保存参数
 *          - 创建发布者和订阅者
 *          - 初始化关节命令数组
 *          - 创建定时器
 */
bool RosInterface::Initialize() {
  // 读取接触力导出参数
  node_->declare_parameter("export_contact", false);
  node_->declare_parameter("contact_topic", "/mujoco/contact_forces");
  
  export_contact_ = node_->get_parameter("export_contact").as_bool();
  contact_topic_ = node_->get_parameter("contact_topic").as_string();

  // 读取CSV保存参数
  node_->declare_parameter("save_contact_csv", false);
  node_->declare_parameter("save_perturbation_csv", false);
  node_->declare_parameter("save_joint_forces_csv", false);
  node_->declare_parameter("save_sensor_vibration_csv", false);
  node_->declare_parameter("save_joint_state_csv", false);
  node_->declare_parameter("csv_file_path", "");
  node_->declare_parameter("csv_save_frequency", 1);  // 每帧都保存
  node_->declare_parameter("csv_format", "csv");  // 格式：csv 或 binary
  
  save_contact_csv_ = node_->get_parameter("save_contact_csv").as_bool();
  save_perturbation_csv_ = node_->get_parameter("save_perturbation_csv").as_bool();
  save_joint_forces_csv_ = node_->get_parameter("save_joint_forces_csv").as_bool();
  save_sensor_vibration_csv_ = node_->get_parameter("save_sensor_vibration_csv").as_bool();
  save_joint_state_csv_ = node_->get_parameter("save_joint_state_csv").as_bool();
  csv_file_path_ = node_->get_parameter("csv_file_path").as_string();
  csv_save_frequency_ = node_->get_parameter("csv_save_frequency").as_int();
  csv_format_ = node_->get_parameter("csv_format").as_string();
  
  // 如果启用CSV保存，自动启用接触力导出（CSV保存需要接触力数据）
  if (save_contact_csv_ && !export_contact_) {
    export_contact_ = true;
    RCLCPP_INFO(node_->get_logger(), "Auto-enabled export_contact because save_contact_csv is enabled");
  }
  
  // 验证格式参数
  if (csv_format_ != "csv" && csv_format_ != "binary") {
    RCLCPP_WARN(node_->get_logger(), "Invalid csv_format '%s', using 'csv'", csv_format_.c_str());
    csv_format_ = "csv";
  }

  // 获取关节数量（必须在初始化CSV文件之前获取，因为CSV头部需要用到这个值）
  num_total_joints_ = config_loader_->GetNumTotalJoints();
  
  // 确定CSV文件保存的目录
  std::string csv_dir;
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  
  if (!csv_file_path_.empty()) {
    // 如果指定了路径，使用该路径作为目录
    csv_dir = csv_file_path_;
    std::filesystem::create_directories(csv_dir);
  } else {
    // 如果没有指定路径，使用默认路径
    const char* home_dir = std::getenv("HOME");
    csv_dir = (home_dir ? std::string(home_dir) : "") + "/data/mujoco_logs";
    std::filesystem::create_directories(csv_dir);
  }
  
  // 生成接触力文件路径（根据格式选择扩展名）
  if (save_contact_csv_) {
    std::stringstream ss;
    std::string ext = (csv_format_ == "binary") ? ".bin" : ".csv";
    ss << csv_dir << "/contact_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ext;
    csv_file_path_ = ss.str();
  }
  
  // 生成推力数据文件路径（与接触力文件在同一目录）
  if (save_perturbation_csv_) {
    std::stringstream ss_pert;
    std::string ext = (csv_format_ == "binary") ? ".bin" : ".csv";
    ss_pert << csv_dir << "/perturbation_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ext;
    perturbation_csv_file_path_ = ss_pert.str();
  }
  
  // 生成关节反力数据CSV文件路径（与接触力CSV在同一目录）
  if (save_joint_forces_csv_) {
    std::stringstream ss_joint;
    ss_joint << csv_dir << "/joint_forces_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
    joint_forces_csv_file_path_ = ss_joint.str();
  }

  // 初始化传感器震动CSV文件（持续记录，不依赖接触力）
  if (save_sensor_vibration_csv_) {
    std::stringstream ss_vibration;
    std::string ext = (csv_format_ == "binary") ? ".bin" : ".csv";
    ss_vibration << csv_dir << "/sensor_vibration_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ext;
    sensor_vibration_csv_file_path_ = ss_vibration.str();
    
    std::lock_guard<std::mutex> lock(sensor_vibration_csv_mutex_);
    if (csv_format_ == "binary") {
      sensor_vibration_csv_file_.open(sensor_vibration_csv_file_path_, std::ios::out | std::ios::binary);
    } else {
      sensor_vibration_csv_file_.open(sensor_vibration_csv_file_path_, std::ios::out);
    }
    if (sensor_vibration_csv_file_.is_open()) {
      if (csv_format_ == "csv") {
        // 写入CSV头部
        sensor_vibration_csv_file_ << "timestamp,"
                                   << "base_link_lin_acc_x,base_link_lin_acc_y,base_link_lin_acc_z,"
                                   << "base_link_ang_acc_x,base_link_ang_acc_y,base_link_ang_acc_z,"
                                   << "head_lin_acc_x,head_lin_acc_y,head_lin_acc_z,"
                                   << "head_ang_acc_x,head_ang_acc_y,head_ang_acc_z\n";
        sensor_vibration_csv_file_.flush();
      }
      RCLCPP_INFO(node_->get_logger(), "Sensor vibration data will be saved to: %s (format: %s)", 
                  sensor_vibration_csv_file_path_.c_str(), csv_format_.c_str());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open sensor vibration CSV file: %s", sensor_vibration_csv_file_path_.c_str());
      save_sensor_vibration_csv_ = false;
    }
  }

  // 初始化关节状态CSV文件（持续记录位置、速度、力矩）
  if (save_joint_state_csv_) {
    std::stringstream ss_joint_state;
    std::string ext = (csv_format_ == "binary") ? ".bin" : ".csv";
    ss_joint_state << csv_dir << "/joint_state_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ext;
    joint_state_csv_file_path_ = ss_joint_state.str();
    
    std::lock_guard<std::mutex> lock(joint_state_csv_mutex_);
    if (csv_format_ == "binary") {
      joint_state_csv_file_.open(joint_state_csv_file_path_, std::ios::out | std::ios::binary);
    } else {
      joint_state_csv_file_.open(joint_state_csv_file_path_, std::ios::out);
    }
    if (joint_state_csv_file_.is_open()) {
      if (csv_format_ == "csv") {
        // 写入CSV头部
        joint_state_csv_file_ << "timestamp";
        for (int i = 0; i < num_total_joints_; i++) {
          joint_state_csv_file_ << ",joint_" << i << "_position";
        }
        for (int i = 0; i < num_total_joints_; i++) {
          joint_state_csv_file_ << ",joint_" << i << "_velocity";
        }
        for (int i = 0; i < num_total_joints_; i++) {
          joint_state_csv_file_ << ",actuator_" << i << "_force";
        }
        joint_state_csv_file_ << "\n";
        joint_state_csv_file_.flush();
      }
      RCLCPP_INFO(node_->get_logger(), "Joint state data will be saved to: %s (format: %s)", 
                  joint_state_csv_file_path_.c_str(), csv_format_.c_str());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open joint state CSV file: %s", joint_state_csv_file_path_.c_str());
      save_joint_state_csv_ = false;
    }
  }
  
  // 初始化接触力文件
  if (save_contact_csv_) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    if (csv_format_ == "binary") {
      csv_file_.open(csv_file_path_, std::ios::out | std::ios::binary);
    } else {
      csv_file_.open(csv_file_path_, std::ios::out);
    }
    
    if (csv_file_.is_open()) {
      if (csv_format_ == "csv") {
        // 写入CSV头部
        // 写入CSV头部（移除关节角度、关节加速度和电机扭矩，已移到独立的 joint_state_data.csv）
        csv_file_ << "timestamp,contact_id,body1_name,body2_name,pos_x,pos_y,pos_z,robot_frame_x,robot_frame_y,robot_frame_z,force_x,force_y,force_z,force_magnitude,force_normal,torque_x,torque_y,torque_z,base_link_x,base_link_y,base_link_z,base_link_qw,base_link_qx,base_link_qy,base_link_qz,base_link_vel_x,base_link_vel_y,base_link_vel_z,base_link_angvel_x,base_link_angvel_y,base_link_angvel_z,collision_link_x,collision_link_y,collision_link_z,collision_link_qw,collision_link_qx,collision_link_qy,collision_link_qz\n";
        csv_file_.flush();
      } else {
        // 二进制格式：写入文件头（包含关节数量等信息）
        int32_t num_joints = num_total_joints_;
        csv_file_.write(reinterpret_cast<const char*>(&num_joints), sizeof(num_joints));
        csv_file_.flush();
      }
      
      RCLCPP_INFO(node_->get_logger(), "Contact data will be saved to: %s (format: %s)", 
                  csv_file_path_.c_str(), csv_format_.c_str());
      
      // 启动异步写入线程
      writer_threads_running_ = true;
      contact_writer_thread_ = std::thread(&RosInterface::ContactWriterThread, this);
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open contact file: %s", csv_file_path_.c_str());
      save_contact_csv_ = false;
    }
  }

  // 初始化推力文件
  if (save_perturbation_csv_) {
    std::lock_guard<std::mutex> lock(perturbation_csv_mutex_);
    if (csv_format_ == "binary") {
      perturbation_csv_file_.open(perturbation_csv_file_path_, std::ios::out | std::ios::binary);
    } else {
      perturbation_csv_file_.open(perturbation_csv_file_path_, std::ios::out);
    }
    
    if (perturbation_csv_file_.is_open()) {
      if (csv_format_ == "csv") {
        // 写入CSV头部
        perturbation_csv_file_ << "timestamp,perturbation_id,body_name,start_time,duration,elapsed_time,force_x,force_y,force_z,force_magnitude,torque_x,torque_y,torque_z,torque_magnitude,std_pose_x,std_pose_y,std_pose_z,world_force_x,world_force_y,world_force_z,world_force_magnitude\n";
        perturbation_csv_file_.flush();
      }
      // 二进制格式不需要头部
      
      RCLCPP_INFO(node_->get_logger(), "Perturbation data will be saved to: %s (format: %s)", 
                  perturbation_csv_file_path_.c_str(), csv_format_.c_str());
      
      // 启动异步写入线程
      if (!writer_threads_running_.load()) {
        writer_threads_running_ = true;
      }
      perturbation_writer_thread_ = std::thread(&RosInterface::PerturbationWriterThread, this);
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open perturbation file: %s", perturbation_csv_file_path_.c_str());
    }
  }

  // 初始化关节反力CSV文件
  if (save_joint_forces_csv_) {
    std::stringstream ss_joint_forces;
    std::string ext = (csv_format_ == "binary") ? ".bin" : ".csv";
    ss_joint_forces << csv_dir << "/joint_forces_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ext;
    joint_forces_csv_file_path_ = ss_joint_forces.str();
    
    std::lock_guard<std::mutex> lock(joint_forces_csv_mutex_);
    if (csv_format_ == "binary") {
      joint_forces_csv_file_.open(joint_forces_csv_file_path_, std::ios::out | std::ios::binary);
    } else {
      joint_forces_csv_file_.open(joint_forces_csv_file_path_, std::ios::out);
    }
    if (joint_forces_csv_file_.is_open()) {
      if (csv_format_ == "csv") {
        // 写入关节反力CSV头部
        joint_forces_csv_file_ << "timestamp,joint_id,joint_name,body_id,body_name,"
                               << "child_Mx,child_My,child_Mz,child_Fx,child_Fy,child_Fz,"
                               << "parent_Mx,parent_My,parent_Mz,parent_Fx,parent_Fy,parent_Fz,"
                               << "axis_x,axis_y,axis_z,"
                               << "F_axial_mag,F_shear_mag,M_torsion_mag,M_bend_mag,M_eq,"
                               << "F_axial_x,F_axial_y,F_axial_z,"
                               << "F_shear_x,F_shear_y,F_shear_z,"
                               << "M_torsion_x,M_torsion_y,M_torsion_z,"
                               << "M_bend_x,M_bend_y,M_bend_z\n";
        joint_forces_csv_file_.flush();
      }
      // 二进制格式不需要头部，直接写入数据
      RCLCPP_INFO(node_->get_logger(), "Joint forces data will be saved to: %s (format: %s)", 
                  joint_forces_csv_file_path_.c_str(), csv_format_.c_str());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open joint forces CSV file: %s", joint_forces_csv_file_path_.c_str());
      save_joint_forces_csv_ = false;
    }
  }

  // 创建发布者
  joint_state_pub_ =
      node_->create_publisher<interface_protocol::msg::JointState>(config_loader_->GetJointStateTopic(), 10);

  imu_pub_ = node_->create_publisher<interface_protocol::msg::ImuInfo>(config_loader_->GetImuTopic(), 10);
  
  // 创建运动状态发布者
  motion_state_pub_ = node_->create_publisher<interface_protocol::msg::MotionState>("/motion/motion_state", 10);

  // publish contact force
  contact_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>("/mujoco/contact_forces", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort());

  // contact visulization
  contact_marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/mujoco/contact_markers", 10);

  // MuJoCo重置完成发布者
  mujoco_reset_pub_ = node_->create_publisher<std_msgs::msg::Empty>("/mujoco/reset_complete", 10);

  // 如果启用接触力导出，创建接触力发布者
  if (export_contact_) {
    contact_force_pub_ = node_->create_publisher<interface_protocol::msg::ContactForce>(contact_topic_, 10);
    RCLCPP_INFO(node_->get_logger(), "Contact force publishing enabled on topic: %s", contact_topic_.c_str());
  }

  // 创建订阅者，使用更兼容的QoS设置
  using std::placeholders::_1;

  // 创建更兼容的QoS设置
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

  joint_cmd_sub_ = node_->create_subscription<interface_protocol::msg::JointCommand>(
      config_loader_->GetJointCommandTopic(), qos, std::bind(&RosInterface::JointCommandCallback, this, _1));

  // 注意：关节数量和最大干扰力数量已在CSV初始化时读取

  // 用零值初始化命令数组
  joint_command_.position.resize(num_total_joints_, 0.0);
  joint_command_.velocity.resize(num_total_joints_, 0.0);
  joint_command_.torque.resize(num_total_joints_, 0.0);
  joint_command_.feed_forward_torque.resize(num_total_joints_, 0.0);
  joint_command_.stiffness.resize(num_total_joints_, 0.0);
  joint_command_.damping.resize(num_total_joints_, 0.0);

  // 创建定时器，每秒发布一次运动状态
  motion_state_timer_ = node_->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&RosInterface::MotionStateTimerCallback, this));

  RCLCPP_INFO(node_->get_logger(), "MuJoCo ROS interface initialized successfully");
  return true;
}

/**
 * @brief 获取安全的关节命令
 * @return 当前的关节命令
 * @details 使用互斥锁保护，确保线程安全
 */
interface_protocol::msg::JointCommand RosInterface::GetCommandedSafe() {
  std::lock_guard<std::mutex> lock(mtx_);
  return joint_command_;
}

/**
 * @brief 关节命令回调函数
 * @param msg 接收到的关节命令消息
 * @details 更新关节命令并确保所有向量大小正确
 */
void RosInterface::JointCommandCallback(const interface_protocol::msg::JointCommand::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mtx_);

  // 更新命令值
  joint_command_ = *msg;

  // 确保所有向量大小正确
  if (joint_command_.position.size() > num_total_joints_) {
    joint_command_.position.resize(num_total_joints_);
  }
  if (joint_command_.velocity.size() > num_total_joints_) {
    joint_command_.velocity.resize(num_total_joints_);
  }
  if (joint_command_.torque.size() > num_total_joints_) {
    joint_command_.torque.resize(num_total_joints_);
  }
  if (joint_command_.feed_forward_torque.size() > num_total_joints_) {
    joint_command_.feed_forward_torque.resize(num_total_joints_);
  }
  if (joint_command_.stiffness.size() > num_total_joints_) {
    joint_command_.stiffness.resize(num_total_joints_);
  }
  if (joint_command_.damping.size() > num_total_joints_) {
    joint_command_.damping.resize(num_total_joints_);
  }
}

/**
 * @brief 更新仿真状态并发布数据
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 发布关节状态、IMU数据和接触力数据
 */
void RosInterface::UpdateSimState(const mjModel* m, mjData* d) {
  // 添加空指针检查
  if (!m || !d) {
    return;
  }
  
  is_floating_base_ = (m->nv != m->nu);

  // 创建消息
  auto joint_state_msg = std::make_unique<interface_protocol::msg::JointState>();
  auto imu_msg = std::make_unique<interface_protocol::msg::ImuInfo>();

  // 设置时间戳
  joint_state_msg->header.stamp = node_->now();
  imu_msg->header.stamp = node_->now();

  // 设置关节状态
  joint_state_msg->name.resize(num_total_joints_);
  joint_state_msg->position.resize(num_total_joints_);
  joint_state_msg->velocity.resize(num_total_joints_);
  joint_state_msg->torque.resize(num_total_joints_);

  if (is_floating_base_) {
    // 跳过浮动基座关节
    for (int i = 0; i < num_total_joints_; ++i) {
      // 从MuJoCo模型获取关节名称
      joint_state_msg->name[i] = m->names + m->name_jntadr[i + kNumFloatingBaseJoints];
      joint_state_msg->position[i] = d->qpos[i + kNumFloatingBaseJoints];
      joint_state_msg->velocity[i] = d->qvel[i + kDofFloatingBase];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  } else {
    for (int i = 0; i < num_total_joints_; ++i) {
      // 从MuJoCo模型获取关节名称
      joint_state_msg->name[i] = m->names + m->name_jntadr[i];
      joint_state_msg->position[i] = d->qpos[i];
      joint_state_msg->velocity[i] = d->qvel[i];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  }

  // IMU数据通常来自MuJoCo中的传感器
  int index = 0;

  // 设置IMU四元数
  imu_msg->quaternion.w = d->sensordata[index + 0];
  imu_msg->quaternion.x = d->sensordata[index + 1];
  imu_msg->quaternion.y = d->sensordata[index + 2];
  imu_msg->quaternion.z = d->sensordata[index + 3];
  index += kDimQuaternion;

  // 从传感器数据设置RPY值
  // 假设RPY值是四元数后的三个值
  imu_msg->rpy.x = d->sensordata[index + 0];  // 横滚角
  imu_msg->rpy.y = d->sensordata[index + 1];  // 俯仰角
  imu_msg->rpy.z = d->sensordata[index + 2];  // 偏航角
  index += 3;

  // 线性加速度
  imu_msg->linear_acceleration.x = d->sensordata[index + 0];
  imu_msg->linear_acceleration.y = d->sensordata[index + 1];
  imu_msg->linear_acceleration.z = d->sensordata[index + 2];
  index += 3;

  // 角速度
  imu_msg->angular_velocity.x = d->sensordata[index + 0];
  imu_msg->angular_velocity.y = d->sensordata[index + 1];
  imu_msg->angular_velocity.z = d->sensordata[index + 2];

  // 发布消息
  joint_state_pub_->publish(std::move(joint_state_msg));
  imu_pub_->publish(std::move(imu_msg));

  // 发布接触力（包含接触点发布和RViz可视化功能）
  if (export_contact_) {
    // 使用全局同步机制，确保所有情况下都有适当的同步
    std::lock_guard<std::mutex> global_lock(contact_force_mutex_);
    PublishContactForces(m, d);
  }

  // 保存关节反力到CSV
  if (save_joint_forces_csv_) {
    SaveJointForcesToCSV(m, d);
  }

  // 保存传感器震动数据到CSV（持续记录，不依赖接触力）
  if (save_sensor_vibration_csv_) {
    SaveSensorVibrationToCSV(m, d);
  }

  // 保存关节状态数据到CSV（持续记录位置、速度、力矩）
  if (save_joint_state_csv_) {
    SaveJointStateToCSV(m, d);
  }
}

/**
 * @brief 发布接触力数据
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 计算并发布所有接触点的力和力矩信息，同时保存到CSV文件
 */
/**
 * @brief 发布接触力信息到ROS话题
 * 
 * 该函数从MuJoCo仿真器中提取接触力数据，包括：
 * - 接触点位置
 * - 接触力（世界坐标系）
 * - 接触力矩（世界坐标系）
 * - 接触间隙
 * - 接触体信息
 * 
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 */
void RosInterface::PublishContactForces(const mjModel* m, mjData* d) {
  // 添加空指针检查，确保发布器和数据有效
  if (!contact_force_pub_ || !m || !d) {
    return;
  }

  // 获取当前仿真中的接触数量
  int ncon = d->ncon;
  
  // 减少调试信息频率 - 每5000帧打印一次
  static int frame_count = 0;
  frame_count++;
  if (frame_count % 5000 == 0) {
    RCLCPP_INFO(node_->get_logger(), "Current contacts: %d", ncon);
  }
  
  // 如果没有接触点，发布空消息并返回
  if (ncon == 0) {
    auto contact_msg = std::make_unique<interface_protocol::msg::ContactForce>();
    contact_msg->header.stamp = node_->now();
    contact_msg->header.frame_id = "world";
    contact_force_pub_->publish(std::move(contact_msg));
    return;
  }

  // ==================== 第一部分：声明所有需要的变量 ====================
  
  // 创建接触力消息并设置时间戳和坐标系
  auto contact_msg = std::make_unique<interface_protocol::msg::ContactForce>();
  contact_msg->header.stamp = node_->now();
  contact_msg->header.frame_id = "world";

  // 预分配消息中所有向量的空间，避免动态扩容
  contact_msg->contact_names.resize(ncon);           // 接触点名称
  contact_msg->contact_positions_x.resize(ncon);     // 接触点X坐标
  contact_msg->contact_positions_y.resize(ncon);     // 接触点Y坐标
  contact_msg->contact_positions_z.resize(ncon);     // 接触点Z坐标
  contact_msg->contact_forces_x.resize(ncon);        // 接触力X分量
  contact_msg->contact_forces_y.resize(ncon);        // 接触力Y分量
  contact_msg->contact_forces_z.resize(ncon);        // 接触力Z分量
  contact_msg->contact_torques_x.resize(ncon);       // 接触力矩X分量
  contact_msg->contact_torques_y.resize(ncon);       // 接触力矩Y分量
  contact_msg->contact_torques_z.resize(ncon);       // 接触力矩Z分量
  contact_msg->contact_gaps.resize(ncon);            // 接触间隙
  contact_msg->contact_bodies_1.resize(ncon);        // 第一个接触体ID
  contact_msg->contact_bodies_2.resize(ncon);        // 第二个接触体ID

  // CSV保存相关的变量声明 - 使用智能指针进行堆分配以避免栈溢出
  std::unique_ptr<std::vector<std::string>> csv_body1_names;
  std::unique_ptr<std::vector<std::string>> csv_body2_names;
  std::unique_ptr<std::vector<mjtNum>> csv_red_ball_pos_x;
  std::unique_ptr<std::vector<mjtNum>> csv_red_ball_pos_y;
  std::unique_ptr<std::vector<mjtNum>> csv_red_ball_pos_z;
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_x;
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_y;
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_z;
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_body2_x;  // body2坐标系下的绿球坐标（当两个geom都不是world时）
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_body2_y;
  std::unique_ptr<std::vector<mjtNum>> csv_green_ball_pos_body2_z;
  std::unique_ptr<std::vector<int>> csv_body1_ids;  // body1的ID（用于判断是否为world）
  std::unique_ptr<std::vector<int>> csv_body2_ids;  // body2的ID（用于判断是否为world）
  std::unique_ptr<std::vector<mjtNum>> csv_world_forces_x;
  std::unique_ptr<std::vector<mjtNum>> csv_world_forces_y;
  std::unique_ptr<std::vector<mjtNum>> csv_world_forces_z;
  std::unique_ptr<std::vector<mjtNum>> csv_contact_force_magnitudes;  // f_mag: 世界坐标系下的总力大小
  std::unique_ptr<std::vector<mjtNum>> csv_contact_force_normals;     // f_norm: 接触坐标系下的法向力分量
  std::unique_ptr<std::vector<mjtNum>> csv_world_torques_x;
  std::unique_ptr<std::vector<mjtNum>> csv_world_torques_y;
  std::unique_ptr<std::vector<mjtNum>> csv_world_torques_z;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_pos_x;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_pos_y;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_pos_z;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_quat_w;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_quat_x;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_quat_y;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_quat_z;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_vel_x;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_vel_y;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_vel_z;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_angvel_x;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_angvel_y;
  std::unique_ptr<std::vector<mjtNum>> csv_base_link_angvel_z;
  std::unique_ptr<std::vector<mjtNum>> csv_robot_frame_pos_x;
  std::unique_ptr<std::vector<mjtNum>> csv_robot_frame_pos_y;
  std::unique_ptr<std::vector<mjtNum>> csv_robot_frame_pos_z;
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_pos_x;    // 碰撞link的世界坐标位置
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_pos_y;
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_pos_z;
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_quat_w;  // 碰撞link的世界坐标姿态四元数
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_quat_x;
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_quat_y;
  std::unique_ptr<std::vector<mjtNum>> csv_collision_link_quat_z;
  
  // 只在需要CSV保存时才分配内存
  if (save_contact_csv_) {
    csv_body1_names = std::make_unique<std::vector<std::string>>(ncon);
    csv_body2_names = std::make_unique<std::vector<std::string>>(ncon);
    csv_red_ball_pos_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_red_ball_pos_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_red_ball_pos_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_body2_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_body2_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_green_ball_pos_body2_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_body1_ids = std::make_unique<std::vector<int>>(ncon);
    csv_body2_ids = std::make_unique<std::vector<int>>(ncon);
    csv_world_forces_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_world_forces_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_world_forces_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_contact_force_magnitudes = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_contact_force_normals = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_world_torques_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_world_torques_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_world_torques_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_pos_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_pos_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_pos_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_quat_w = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_quat_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_quat_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_quat_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_vel_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_vel_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_vel_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_angvel_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_angvel_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_base_link_angvel_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_robot_frame_pos_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_robot_frame_pos_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_robot_frame_pos_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_pos_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_pos_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_pos_z = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_quat_w = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_quat_x = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_quat_y = std::make_unique<std::vector<mjtNum>>(ncon);
    csv_collision_link_quat_z = std::make_unique<std::vector<mjtNum>>(ncon);
  }

  // 标准姿态缓存相关变量
  static std::vector<mjtNum> std_xpos_cache;    // 缓存标准位置
  static std::vector<mjtNum> std_xmat_cache;    // 缓存标准旋转矩阵
  static const mjModel* last_model_ptr = nullptr;  // 记录上次使用的模型

  // ==================== 第二部分：处理每个接触点，计算所有需要的数据 ====================
  
  // 检查是否需要重新计算标准姿态（当模型改变时，比如按reload）
  if (last_model_ptr != m) {
    // 模型改变，重新计算标准姿态
    std_xpos_cache.clear();
    std_xmat_cache.clear();
    
    // 创建临时mjData对象用于计算标准姿态
    mjData* dstd = mj_makeData(m);
    if (!dstd) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create temporary mjData for standard pose");
      // 如果创建失败，记录模型指针并返回，避免后续访问错误
      last_model_ptr = m;
      return;
    } else {
      // 设置标准姿态：优先使用关键帧，否则使用零位
      // 缓存所有刚体的位置 (xpos: 3*nbody)
      // dstd->xpos 是 MuJoCo 数据结构中存储所有body当前位置的一维数组
      // 初始状态：
      // dstd->qpos = [关节角度...]     ← 从keyframe读取
      // dstd->xpos = [body位置...]     ← 初始值（可能是0或随机值）
      // dstd->xmat = [body旋转...]     ← 初始值（可能是单位矩阵）

      // mj_resetDataKeyframe后：
      // dstd->qpos = [keyframe关节角度...]  ← 被keyframe值覆盖
      // dstd->xpos = [body位置...]          ← 仍然保持原值，未改变！
      // dstd->xmat = [body旋转...]          ← 仍然保持原值，未改变！

      // mj_forward后：
      // dstd->qpos = [keyframe关节角度...]  ← 保持keyframe值
      // dstd->xpos = [新计算的body位置...]   ← 被重新计算覆盖！
      // dstd->xmat = [新计算的body旋转...]   ← 被重新计算覆盖！
      int key_id = mj_name2id(m, mjOBJ_KEY, "floating_base_homing");
      if (key_id >= 0) {
        // 使用预定义的关键帧作为标准姿态
        mj_resetDataKeyframe(m, dstd, key_id);
      } else {
        // 兜底方案：设置为零位姿态
        mju_zero(dstd->qpos, m->nq);
        if (m->jnt_type[0] == mjJNT_FREE) {
          // 对于自由关节，设置四元数为单位四元数，位置为原点
          dstd->qpos[0] = 1; dstd->qpos[1] = dstd->qpos[2] = dstd->qpos[3] = 0; // quat = [1,0,0,0]
          dstd->qpos[4] = dstd->qpos[5] = dstd->qpos[6] = 0; // base at origin (0,0,0) 
        }
      }
      
      // 执行前向运动学计算，更新所有刚体的位置和姿态
      mj_forward(m, dstd);
      
      // 缓存标准姿态数据 - 添加边界检查确保安全
      if (m->nbody > 0) {
        // 缓存所有刚体的位置 (xpos: 3*nbody)
        std_xpos_cache.assign(dstd->xpos, dstd->xpos + 3*m->nbody);
        // 缓存所有刚体的旋转矩阵 (xmat: 9*nbody)
        std_xmat_cache.assign(dstd->xmat, dstd->xmat + 9*m->nbody);
      }
      
      // 正确释放临时mjData对象，避免内存泄漏
      mj_deleteData(dstd);
    }
    
    // 更新模型指针，标记已处理
    last_model_ptr = m;
  }

  // 坐标转换变量说明
  // p_W: 当前姿态下接触点在世界坐标系下的位置
  // p_W_std: 标准姿态下接触点在世界坐标系下的位置
  // p_L: 当前姿态下接触点在link局部坐标系下的位置
  // p_L_std: 标准姿态下接触点在link局部坐标系下的位置
  // R_LW: 当前姿态下body的旋转矩阵
  // R_LW_std: 标准姿态下body的旋转矩阵
  // f_L: 当前姿态下接触力在link局部坐标系下的力
  // f_L_std: 标准姿态下接触力在link局部坐标系下的力
  // t_L: 当前姿态下接触力在link局部坐标系下的力矩
  // t_L_std: 标准姿态下接触力在link局部坐标系下的力矩
  // f_norm: 接触力的法向分量（接触系x方向，正值，即正压力）
  // f_mag: 接触力的总大小（世界坐标系下）
  // f_norm_std: 标准姿态下接触力的法向分量（正压力）
  // f_mag_std: 标准姿态下接触力的总大小
  // dia: 接触点的大小
  // dia_std: 标准姿态下接触点的大小
  // color: 接触点的颜色
  // lifetime: 接触点的生存时间
  // ns: 接触点的名称空间
  // id: 接触点的ID
  // type: 接触点的类型
  // action: 接触点的动作
  // pose: 接触点的位置和姿态
  // scale: 接触点的大小
  // color: 接触点的颜色
  // lifetime: 接触点的生存时间
  // 计算公式
  // p_W = R_LW * p_L + x_LW
  // p_W_std = R_LW_std * p_L_std + x_LW_std
  // pW_minus_x = p_W - x_LW = R_LW * p_L
  // f_L = R_LW * f_w
  // f_L_std = R_LW_std * f_w
  // t_L = R_LW * t_w
  // t_L_std = R_LW_std * t_w
  // f_norm = f_c[0] (正值)               // 接触坐标系下的法向力分量
  // f_mag = norm(f_w)                     // 世界坐标系下的总力大小
  // f_norm_std = f_c[0] (标准姿态)       // 标准姿态下接触坐标系法向力分量


  // 遍历所有接触点，计算并存储所有需要的数据
  for (int i = 0; i < ncon; ++i) {
    const mjContact& contact = d->contact[i];
    
    // 获取body名称
    int body1_id = m->geom_bodyid[contact.geom[0]];
    int body2_id = m->geom_bodyid[contact.geom[1]];
    
    const char* body1_name = mj_id2name(m, mjOBJ_BODY, body1_id);
    const char* body2_name = mj_id2name(m, mjOBJ_BODY, body2_id);
    
    // 如果body没有名称，使用ID作为名称
    std::string name1 = body1_name ? body1_name : "body" + std::to_string(body1_id);
    std::string name2 = body2_name ? body2_name : "body" + std::to_string(body2_id);
    
    // 存储到消息和CSV变量中
    contact_msg->contact_names[i] = name1 + "_" + name2;
    if (save_contact_csv_ && csv_body1_names && csv_body2_names) {
      (*csv_body1_names)[i] = name1;
      (*csv_body2_names)[i] = name2;
    }
    // 存储body1_id和body2_id（用于判断是否为world）
    if (save_contact_csv_ && csv_body1_ids && csv_body2_ids) {
      (*csv_body1_ids)[i] = body1_id;
      (*csv_body2_ids)[i] = body2_id;
    }

    // 计算红球坐标：世界坐标系接触点（应用偏移量）
    mjtNum normal[3] = {contact.frame[0], contact.frame[1], contact.frame[2]};
    double position_offset = config_loader_->GetContactPositionOffset();
    
    mjtNum red_ball_pos[3];
    red_ball_pos[0] = contact.pos[0] + normal[0] * position_offset;
    red_ball_pos[1] = contact.pos[1] + normal[1] * position_offset;
    red_ball_pos[2] = contact.pos[2] + normal[2] * position_offset;
    
    // 存储红球坐标
    contact_msg->contact_positions_x[i] = red_ball_pos[0];
    contact_msg->contact_positions_y[i] = red_ball_pos[1];
    contact_msg->contact_positions_z[i] = red_ball_pos[2];
    if (save_contact_csv_ && csv_red_ball_pos_x && csv_red_ball_pos_y && csv_red_ball_pos_z) {
      (*csv_red_ball_pos_x)[i] = red_ball_pos[0];
      (*csv_red_ball_pos_y)[i] = red_ball_pos[1];
      (*csv_red_ball_pos_z)[i] = red_ball_pos[2];
    }

    // 计算绿球坐标：标准姿态下的接触点
    // 如果两个geom都不是world，需要分别计算两次（一次用body1，一次用body2）
    mjtNum green_ball_pos[3] = {0, 0, 0};
    mjtNum green_ball_pos_body2[3] = {0, 0, 0};
    
    // 判断两个geom是否都不是world
    bool both_not_world = (body1_id != 0 && body2_id != 0);
    
    // 计算body1坐标系下的绿球坐标（或如果body1是world，则用body2）
    int link1 = body1_id != 0 ? body1_id : body2_id;
    if (link1 > 0 && link1 < m->nbody && !std_xpos_cache.empty() && !std_xmat_cache.empty()) {
      // 当前姿态下接触点 → link 局部坐标系
      const mjtNum* x_LW = d->xpos + 3*link1;
      const mjtNum* R_LW = d->xmat + 9*link1;
      
      double pW_minus_x[3] = {contact.pos[0]-x_LW[0],
                              contact.pos[1]-x_LW[1],
                              contact.pos[2]-x_LW[2]};
      double p_L[3];
      rotT3(R_LW, pW_minus_x, p_L);
      
      // 标准姿态：link 局部 → 世界
      const mjtNum* x_LW_std = std_xpos_cache.data() + 3*link1;
      const mjtNum* R_LW_std = std_xmat_cache.data() + 9*link1;
      rot3(R_LW_std, p_L, green_ball_pos);
      green_ball_pos[0] += x_LW_std[0];
      green_ball_pos[1] += x_LW_std[1];
      green_ball_pos[2] += x_LW_std[2];
    }
    
    // 如果两个geom都不是world，还需要计算body2坐标系下的绿球坐标
    if (both_not_world && body2_id > 0 && body2_id < m->nbody && !std_xpos_cache.empty() && !std_xmat_cache.empty()) {
      const mjtNum* x_LW_body2 = d->xpos + 3*body2_id;
      const mjtNum* R_LW_body2 = d->xmat + 9*body2_id;
      
      double pW_minus_x_body2[3] = {contact.pos[0]-x_LW_body2[0],
                                    contact.pos[1]-x_LW_body2[1],
                                    contact.pos[2]-x_LW_body2[2]};
      double p_L_body2[3];
      rotT3(R_LW_body2, pW_minus_x_body2, p_L_body2);
      
      // 标准姿态：link 局部 → 世界
      const mjtNum* x_LW_std_body2 = std_xpos_cache.data() + 3*body2_id;
      const mjtNum* R_LW_std_body2 = std_xmat_cache.data() + 9*body2_id;
      rot3(R_LW_std_body2, p_L_body2, green_ball_pos_body2);
      green_ball_pos_body2[0] += x_LW_std_body2[0];
      green_ball_pos_body2[1] += x_LW_std_body2[1];
      green_ball_pos_body2[2] += x_LW_std_body2[2];
    }
    
    // 存储绿球坐标到CSV变量
    if (save_contact_csv_ && csv_green_ball_pos_x && csv_green_ball_pos_y && csv_green_ball_pos_z) {
      (*csv_green_ball_pos_x)[i] = green_ball_pos[0];
      (*csv_green_ball_pos_y)[i] = green_ball_pos[1];
      (*csv_green_ball_pos_z)[i] = green_ball_pos[2];
    }
    // 如果两个geom都不是world，存储body2坐标系下的绿球坐标
    if (save_contact_csv_ && both_not_world && csv_green_ball_pos_body2_x && csv_green_ball_pos_body2_y && csv_green_ball_pos_body2_z) {
      (*csv_green_ball_pos_body2_x)[i] = green_ball_pos_body2[0];
      (*csv_green_ball_pos_body2_y)[i] = green_ball_pos_body2[1];
      (*csv_green_ball_pos_body2_z)[i] = green_ball_pos_body2[2];
    }

    // 获取接触坐标系下的6D接触力 [fx, fy, fz, tx, ty, tz]
    mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mj_contactForce(m, d, i, contact_force);
    
    // 将接触坐标系下的力和力矩转换为世界坐标系
    mjtNum world_force[3] = {0.0, 0.0, 0.0};
    mjtNum world_torque[3] = {0.0, 0.0, 0.0};
    
    // 坐标系转换：从接触坐标系到世界坐标系
    // 使用全局定义的rot3函数进行3D旋转变换
    double f_c[3] = {contact_force[0], contact_force[1], contact_force[2]};
    double t_c[3] = {contact_force[3], contact_force[4], contact_force[5]};
    
    // 直接使用 contact.frame 进行旋转变换：world_force = frame * contact_force
    rot3(contact.frame, f_c, world_force);
    rot3(contact.frame, t_c, world_torque);
    
    // 计算单个接触点的合力大小
    double contact_force_magnitude = sqrt(world_force[0]*world_force[0] + 
                                         world_force[1]*world_force[1] + 
                                         world_force[2]*world_force[2]);
    
    // 计算接触坐标系下的法向力分量（接触系x方向，正值，即正压力）
    double contact_force_normal = std::max(0.0, f_c[0]);  // f_norm: 接触坐标系下的法向力分量
    
    // 将世界坐标系下的力和力矩存储到消息中
    contact_msg->contact_forces_x[i] = world_force[0];
    contact_msg->contact_forces_y[i] = world_force[1];
    contact_msg->contact_forces_z[i] = world_force[2];
    
    contact_msg->contact_torques_x[i] = world_torque[0];
    contact_msg->contact_torques_y[i] = world_torque[1];
    contact_msg->contact_torques_z[i] = world_torque[2];

    // 存储到CSV变量
    if (save_contact_csv_ && csv_world_forces_x && csv_world_forces_y && csv_world_forces_z &&
        csv_contact_force_magnitudes && csv_contact_force_normals &&
        csv_world_torques_x && csv_world_torques_y && csv_world_torques_z) {
      (*csv_world_forces_x)[i] = world_force[0];
      (*csv_world_forces_y)[i] = world_force[1];
      (*csv_world_forces_z)[i] = world_force[2];
      (*csv_contact_force_magnitudes)[i] = contact_force_magnitude;  // f_mag: 世界坐标系下的总力大小
      (*csv_contact_force_normals)[i] = contact_force_normal;        // f_norm: 接触坐标系下的法向力分量
      (*csv_world_torques_x)[i] = world_torque[0];
      (*csv_world_torques_y)[i] = world_torque[1];
      (*csv_world_torques_z)[i] = world_torque[2];
    }

    // 存储接触间隙距离（正值表示分离，负值表示穿透）
    contact_msg->contact_gaps[i] = contact.dist;

    // 存储接触的两个刚体ID
    contact_msg->contact_bodies_1[i] = body1_id;
    contact_msg->contact_bodies_2[i] = body2_id;

    // 获取base_link对应的body ID（LINK_BASE）
    int base_link_id = -1;
    for (int j = 0; j < m->nbody; j++) {
      if (std::string(m->names + m->name_bodyadr[j]) == "LINK_BASE") {
        base_link_id = j;
        break;
      }
    }
    
    // 获取base_link的pose（机器人基座的世界坐标位置）
    mjtNum base_link_pos[3] = {0, 0, 0};
    mjtNum base_link_quat[4] = {1, 0, 0, 0}; // 默认单位四元数
    mjtNum base_link_vel[3] = {0, 0, 0};    // 线速度
    mjtNum base_link_angvel[3] = {0, 0, 0};  // 角速度
    if (base_link_id >= 0) {
      base_link_pos[0] = d->xpos[base_link_id*3];
      base_link_pos[1] = d->xpos[base_link_id*3+1];
      base_link_pos[2] = d->xpos[base_link_id*3+2];
      base_link_quat[0] = d->xquat[base_link_id*4];
      base_link_quat[1] = d->xquat[base_link_id*4+1];
      base_link_quat[2] = d->xquat[base_link_id*4+2];
      base_link_quat[3] = d->xquat[base_link_id*4+3];
      
      // 获取base_link的速度（线速度和角速度）
      // 对于浮动基座，速度存储在d->cvel中
      // cvel格式：[angvel_x, angvel_y, angvel_z, vel_x, vel_y, vel_z]
      base_link_angvel[0] = d->cvel[base_link_id*6 + 0];  // 角速度x
      base_link_angvel[1] = d->cvel[base_link_id*6 + 1];  // 角速度y
      base_link_angvel[2] = d->cvel[base_link_id*6 + 2];  // 角速度z
      base_link_vel[0] = d->cvel[base_link_id*6 + 3];      // 线速度x
      base_link_vel[1] = d->cvel[base_link_id*6 + 4];      // 线速度y
      base_link_vel[2] = d->cvel[base_link_id*6 + 5];     // 线速度z
    }
    
    // 存储base_link信息到CSV变量
    if (save_contact_csv_ && csv_base_link_pos_x && csv_base_link_pos_y && csv_base_link_pos_z &&
        csv_base_link_quat_w && csv_base_link_quat_x && csv_base_link_quat_y && csv_base_link_quat_z &&
        csv_base_link_vel_x && csv_base_link_vel_y && csv_base_link_vel_z &&
        csv_base_link_angvel_x && csv_base_link_angvel_y && csv_base_link_angvel_z) {
      (*csv_base_link_pos_x)[i] = base_link_pos[0];
      (*csv_base_link_pos_y)[i] = base_link_pos[1];
      (*csv_base_link_pos_z)[i] = base_link_pos[2];
      (*csv_base_link_quat_w)[i] = base_link_quat[0];
      (*csv_base_link_quat_x)[i] = base_link_quat[1];
      (*csv_base_link_quat_y)[i] = base_link_quat[2];
      (*csv_base_link_quat_z)[i] = base_link_quat[3];
      (*csv_base_link_vel_x)[i] = base_link_vel[0];
      (*csv_base_link_vel_y)[i] = base_link_vel[1];
      (*csv_base_link_vel_z)[i] = base_link_vel[2];
      (*csv_base_link_angvel_x)[i] = base_link_angvel[0];
      (*csv_base_link_angvel_y)[i] = base_link_angvel[1];
      (*csv_base_link_angvel_z)[i] = base_link_angvel[2];
    }

    // 找到机器人body（非world的几何体）
    auto is_robot_geom = [&](int geom_id)->bool {
      int b = m->geom_bodyid[geom_id];
      return b != 0;   // 简单过滤：0通常是world
    };
    int robot_geom = is_robot_geom(contact.geom[0]) ? contact.geom[0] :
                    (is_robot_geom(contact.geom[1]) ? contact.geom[1] : -1);

    int body = -1;
    if (robot_geom >= 0) {
      body = m->geom_bodyid[robot_geom];
    }

    // 如果找到了机器人body，计算局部坐标
    mjtNum robot_frame_pos[3] = {0, 0, 0};
    if (body >= 0) {
      const mjtNum* x_LW = d->xpos + 3*body;   // body原点世界坐标
      const mjtNum* R_LW = d->xmat + 9*body;   // body旋转矩阵 (row-major)

      // 世界点 -> 局部点
      double pW_minus_x[3] = {red_ball_pos[0]-x_LW[0],
                              red_ball_pos[1]-x_LW[1],
                              red_ball_pos[2]-x_LW[2]};
      rotT3(R_LW, pW_minus_x, robot_frame_pos);
    }
    
    // 存储机器人坐标系下的位置到CSV变量
    if (save_contact_csv_ && csv_robot_frame_pos_x && csv_robot_frame_pos_y && csv_robot_frame_pos_z) {
      (*csv_robot_frame_pos_x)[i] = robot_frame_pos[0];
      (*csv_robot_frame_pos_y)[i] = robot_frame_pos[1];
      (*csv_robot_frame_pos_z)[i] = robot_frame_pos[2];
    }
    
    // 获取碰撞link的位姿信息
    mjtNum collision_link_pos[3] = {0, 0, 0};
    mjtNum collision_link_quat[4] = {1, 0, 0, 0}; // 默认单位四元数
    if (body >= 0) {
      // 获取碰撞link的世界坐标位置
      collision_link_pos[0] = d->xpos[body*3];
      collision_link_pos[1] = d->xpos[body*3+1];
      collision_link_pos[2] = d->xpos[body*3+2];
      
      // 获取碰撞link的世界坐标姿态四元数
      collision_link_quat[0] = d->xquat[body*4];
      collision_link_quat[1] = d->xquat[body*4+1];
      collision_link_quat[2] = d->xquat[body*4+2];
      collision_link_quat[3] = d->xquat[body*4+3];
    }
    
    // 存储碰撞link位姿到CSV变量
    if (save_contact_csv_ && csv_collision_link_pos_x && csv_collision_link_pos_y && csv_collision_link_pos_z &&
        csv_collision_link_quat_w && csv_collision_link_quat_x && csv_collision_link_quat_y && csv_collision_link_quat_z) {
      (*csv_collision_link_pos_x)[i] = collision_link_pos[0];
      (*csv_collision_link_pos_y)[i] = collision_link_pos[1];
      (*csv_collision_link_pos_z)[i] = collision_link_pos[2];
      (*csv_collision_link_quat_w)[i] = collision_link_quat[0];
      (*csv_collision_link_quat_x)[i] = collision_link_quat[1];
      (*csv_collision_link_quat_y)[i] = collision_link_quat[2];
      (*csv_collision_link_quat_z)[i] = collision_link_quat[3];
    }
  }

  // ==================== 第三部分：发布接触力消息到ROS话题 ====================
  contact_force_pub_->publish(std::move(contact_msg));

  // ==================== 第四部分：RViz可视化功能 ====================
  if (contact_marker_pub_) {
    visualization_msgs::msg::MarkerArray arr;

    // 先发送"清空"指令，清除之前的可视化标记，避免历史残留
    {
      visualization_msgs::msg::Marker clear;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(clear);
    }

    // 可视化参数：根据力大小动态缩放球的尺寸
    const double base_diameter = 0.03;   // 3 cm 基础直径
    const double max_diameter  = 0.12;   // 12 cm 最大直径
    const double scale_gain    = 1.0/200.0; // 200N对应1倍尺寸缩放

    // 遍历所有接触点，创建可视化标记
    for (int i = 0; i < ncon; ++i) {
      // -------------------
      // 红球：世界系接触点
      // -------------------
      visualization_msgs::msg::Marker mkr_red;
      mkr_red.header.frame_id = "world";
      mkr_red.header.stamp    = node_->now();
      mkr_red.ns   = "contact_world";
      mkr_red.id   = i;
      mkr_red.type = visualization_msgs::msg::Marker::SPHERE;
      mkr_red.action = visualization_msgs::msg::Marker::ADD;

      if (csv_red_ball_pos_x && csv_red_ball_pos_y && csv_red_ball_pos_z) {
        mkr_red.pose.position.x = (*csv_red_ball_pos_x)[i];
        mkr_red.pose.position.y = (*csv_red_ball_pos_y)[i];
        mkr_red.pose.position.z = (*csv_red_ball_pos_z)[i];
      }
      mkr_red.pose.orientation.w = 1.0;

      // 根据接触力大小动态计算球体直径
      double dia = base_diameter;
      if (csv_contact_force_magnitudes) {
        dia = base_diameter * (1.0 + std::clamp((*csv_contact_force_magnitudes)[i]*scale_gain, 0.0, 3.0));
      }
      dia = std::min(dia, max_diameter);
      mkr_red.scale.x = mkr_red.scale.y = mkr_red.scale.z = dia;

      mkr_red.color.r = 1.0; mkr_red.color.g = 0.0; mkr_red.color.b = 0.0; mkr_red.color.a = 0.9;
      mkr_red.lifetime = rclcpp::Duration::from_seconds(0.1);

      arr.markers.push_back(mkr_red);

      // -------------------
      // 绿球：link局部系下的接触点
      // -------------------
      visualization_msgs::msg::Marker mkr_green;
      mkr_green.header.frame_id = "world";      // RViz 的 fixed frame（和你系统一致）
      mkr_green.header.stamp    = node_->now();
      mkr_green.ns   = "contact_link_stdpose";
      mkr_green.id   = 10000 + i;               // 避免和红球 id 冲突
      mkr_green.type = visualization_msgs::msg::Marker::SPHERE;
      mkr_green.action = visualization_msgs::msg::Marker::ADD;

      if (csv_green_ball_pos_x && csv_green_ball_pos_y && csv_green_ball_pos_z) {
        mkr_green.pose.position.x = (*csv_green_ball_pos_x)[i];
        mkr_green.pose.position.y = (*csv_green_ball_pos_y)[i];
        mkr_green.pose.position.z = (*csv_green_ball_pos_z)[i];
      }
      mkr_green.pose.orientation.w = 1.0;

      // 尺寸：复用你算好的 dia（或独立给绿球一个固定尺寸）
      mkr_green.scale.x = mkr_green.scale.y = mkr_green.scale.z = dia;

      // 颜色：绿色
      mkr_green.color.r = 0.0; mkr_green.color.g = 1.0; mkr_green.color.b = 0.0; mkr_green.color.a = 0.9;

      mkr_green.lifetime = rclcpp::Duration::from_seconds(0.1);

      arr.markers.push_back(mkr_green);
    }

    contact_marker_pub_->publish(arr);
  }

  // ==================== 推力可视化功能 ====================
  if (contact_marker_pub_) {
    // 获取活跃的干扰力数据
    auto& sim_manager = SimManager::GetInstance();
    auto active_perturbations = sim_manager.GetActivePerturbations();
    
    if (!active_perturbations.empty()) {
      visualization_msgs::msg::MarkerArray perturbation_arr;
      
      // 推力可视化参数
      const double base_diameter = 0.05;   // 5 cm 基础直径
      const double max_diameter  = 0.15;   // 15 cm 最大直径
      const double scale_gain    = 1.0/100.0; // 100N对应1倍尺寸缩放
      
      // 遍历所有活跃的干扰力
      for (size_t i = 0; i < active_perturbations.size(); ++i) {
        const auto& pert = active_perturbations[i];
        
        // 计算推力大小
        double force_magnitude = pert.force.norm();
        if (force_magnitude < 0.1) continue; // 忽略太小的力
        
        // 获取推力作用的目标body ID
        int perturbation_body_id = mj_name2id(m, mjOBJ_BODY, pert.body_name.c_str());
        if (perturbation_body_id < 0) continue;
        
        // 计算推力在世界坐标系下的位置（body的当前位置）
        mjtNum world_pos[3] = {0, 0, 0};
        if (perturbation_body_id < m->nbody) {
          world_pos[0] = d->xpos[3 * perturbation_body_id];
          world_pos[1] = d->xpos[3 * perturbation_body_id + 1];
          world_pos[2] = d->xpos[3 * perturbation_body_id + 2];
        }
        
        // 计算推力在标准姿态下的位置
        mjtNum std_pose_pos[3] = {0, 0, 0};
        if (perturbation_body_id < m->nbody && !std_xpos_cache.empty()) {
          std_pose_pos[0] = std_xpos_cache[3 * perturbation_body_id];
          std_pose_pos[1] = std_xpos_cache[3 * perturbation_body_id + 1];
          std_pose_pos[2] = std_xpos_cache[3 * perturbation_body_id + 2];
        }
        
        // 直接使用MuJoCo已经计算好的世界坐标系推力方向
        // 这个方向已经在SimManager::TorqueController中计算过了
        mjtNum* body_quat = d->xquat + 4 * perturbation_body_id;
        mjtNum perturb_force_local[3] = {pert.force.x(), pert.force.y(), pert.force.z()};
        mjtNum perturb_force_world[3];
        mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);
        
        // 调试打印：显示RViz中使用的推力值
        static int debug_counter = 0;
        if (++debug_counter % 100 == 0) {
          // std::cout << "RViz推力调试 - ID " << pert.id << ": 局部力=[" 
          //           << perturb_force_local[0] << ", " << perturb_force_local[1] << ", " << perturb_force_local[2] 
          //           << "], 世界力=[" 
          //           << perturb_force_world[0] << ", " << perturb_force_world[1] << ", " << perturb_force_world[2] 
          //           << "], 大小=" << sqrt(perturb_force_world[0]*perturb_force_world[0] + 
          //                                perturb_force_world[1]*perturb_force_world[1] + 
          //                                perturb_force_world[2]*perturb_force_world[2]) << "N" << std::endl;
        }
        
        // -------------------
        // 蓝球：世界坐标系下的推力位置
        // -------------------
        visualization_msgs::msg::Marker mkr_blue;
        mkr_blue.header.frame_id = "world";
        mkr_blue.header.stamp = node_->now();
        mkr_blue.ns = "perturbation_world";
        mkr_blue.id = 20000 + i;  // 避免和其他标记冲突
        mkr_blue.type = visualization_msgs::msg::Marker::SPHERE;
        mkr_blue.action = visualization_msgs::msg::Marker::ADD;
        
        mkr_blue.pose.position.x = world_pos[0];
        mkr_blue.pose.position.y = world_pos[1];
        mkr_blue.pose.position.z = world_pos[2];
        mkr_blue.pose.orientation.w = 1.0;
        
        // 根据推力大小动态计算球体直径
        double dia = base_diameter * (1.0 + std::clamp(force_magnitude * scale_gain, 0.0, 3.0));
        dia = std::min(dia, max_diameter);
        mkr_blue.scale.x = mkr_blue.scale.y = mkr_blue.scale.z = dia;
        
        // 颜色：蓝色
        mkr_blue.color.r = 0.0; mkr_blue.color.g = 0.0; mkr_blue.color.b = 1.0; mkr_blue.color.a = 0.9;
        mkr_blue.lifetime = rclcpp::Duration::from_seconds(0.1);
        
        perturbation_arr.markers.push_back(mkr_blue);
        
        // -------------------
        // 黄球：标准姿态坐标系下的推力位置
        // -------------------
        visualization_msgs::msg::Marker mkr_yellow;
        mkr_yellow.header.frame_id = "world";
        mkr_yellow.header.stamp = node_->now();
        mkr_yellow.ns = "perturbation_stdpose";
        mkr_yellow.id = 30000 + i;  // 避免和其他标记冲突
        mkr_yellow.type = visualization_msgs::msg::Marker::SPHERE;
        mkr_yellow.action = visualization_msgs::msg::Marker::ADD;
        
        mkr_yellow.pose.position.x = std_pose_pos[0];
        mkr_yellow.pose.position.y = std_pose_pos[1];
        mkr_yellow.pose.position.z = std_pose_pos[2];
        mkr_yellow.pose.orientation.w = 1.0;
        
        // 使用相同的尺寸
        mkr_yellow.scale.x = mkr_yellow.scale.y = mkr_yellow.scale.z = dia;
        
        // 颜色：黄色
        mkr_yellow.color.r = 1.0; mkr_yellow.color.g = 1.0; mkr_yellow.color.b = 0.0; mkr_yellow.color.a = 0.9;
        mkr_yellow.lifetime = rclcpp::Duration::from_seconds(0.1);
        
        perturbation_arr.markers.push_back(mkr_yellow);
        
        // -------------------
        // 推力箭头：显示推力方向和大小（世界坐标系）
        // -------------------
        visualization_msgs::msg::Marker mkr_arrow;
        mkr_arrow.header.frame_id = "world";
        mkr_arrow.header.stamp = node_->now();
        mkr_arrow.ns = "perturbation_force";
        mkr_arrow.id = 40000 + i;  // 避免和其他标记冲突
        mkr_arrow.type = visualization_msgs::msg::Marker::ARROW;
        mkr_arrow.action = visualization_msgs::msg::Marker::ADD;
        
        // 箭头起点：推力作用位置
        mkr_arrow.pose.position.x = world_pos[0];
        mkr_arrow.pose.position.y = world_pos[1];
        mkr_arrow.pose.position.z = world_pos[2];
        
        // 箭头方向：推力方向
        double arrow_length = std::min(force_magnitude * 0.01, 0.5); // 缩放箭头长度
        mkr_arrow.scale.x = arrow_length;  // 箭头长度
        mkr_arrow.scale.y = 0.02;          // 箭头宽度
        mkr_arrow.scale.z = 0.02;          // 箭头高度
        
        // 使用更简单的方法：直接设置箭头的起点和终点
        // 起点：推力作用位置
        // 终点：起点 + 推力方向 * 箭头长度
        double norm = sqrt(perturb_force_world[0]*perturb_force_world[0] + 
                          perturb_force_world[1]*perturb_force_world[1] + 
                          perturb_force_world[2]*perturb_force_world[2]);
        if (norm > 0.001) {
          // 归一化推力向量
          double fx = perturb_force_world[0] / norm;
          double fy = perturb_force_world[1] / norm;
          double fz = perturb_force_world[2] / norm;
          
          // 计算箭头终点
          double end_x = world_pos[0] + fx * arrow_length;
          double end_y = world_pos[1] + fy * arrow_length;
          double end_z = world_pos[2] + fz * arrow_length;
          
          // 使用RViz的ARROW标记，设置起点和终点
          // 注意：RViz的ARROW标记的scale.x是长度，但我们需要通过pose来设置方向
          mkr_arrow.scale.x = arrow_length;
          mkr_arrow.scale.y = 0.02;
          mkr_arrow.scale.z = 0.02;
          
          // 设置箭头位置为起点
          mkr_arrow.pose.position.x = world_pos[0];
          mkr_arrow.pose.position.y = world_pos[1];
          mkr_arrow.pose.position.z = world_pos[2];
          
          // 计算从X轴(1,0,0)到推力方向的旋转（RViz ARROW默认指向X轴正方向）
          double axis_x = 0.0;  // 叉积 X × force
          double axis_y = -fz;  // 0*fz - 1*fy = -fz
          double axis_z = fy;   // 1*fx - 0*fz = fy
          
          double cos_angle = fx;  // 点积 X · force
          double angle = acos(std::clamp(cos_angle, -1.0, 1.0));
          
          if (angle > 0.001) {
            double axis_norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
            if (axis_norm > 0.001) {
              axis_x /= axis_norm;
              axis_y /= axis_norm;
              axis_z /= axis_norm;
            }
          } else {
            axis_x = 0; axis_y = 0; axis_z = 1;
          }
          
          // 转换为四元数
          double s = sin(angle / 2.0);
          mkr_arrow.pose.orientation.w = cos(angle / 2.0);
          mkr_arrow.pose.orientation.x = axis_x * s;
          mkr_arrow.pose.orientation.y = axis_y * s;
          mkr_arrow.pose.orientation.z = axis_z * s;
        } else {
          mkr_arrow.pose.orientation.w = 1.0;
          mkr_arrow.pose.orientation.x = 0.0;
          mkr_arrow.pose.orientation.y = 0.0;
          mkr_arrow.pose.orientation.z = 0.0;
        }
        
        // 颜色：红色（表示推力）
        mkr_arrow.color.r = 1.0; mkr_arrow.color.g = 0.0; mkr_arrow.color.b = 0.0; mkr_arrow.color.a = 0.8;
        mkr_arrow.lifetime = rclcpp::Duration::from_seconds(0.1);
        
        perturbation_arr.markers.push_back(mkr_arrow);
        
        // -------------------
        // 黄色箭头：显示推力在标准姿态坐标系下的方向
        // -------------------
        visualization_msgs::msg::Marker mkr_yellow_arrow;
        mkr_yellow_arrow.header.frame_id = "world";
        mkr_yellow_arrow.header.stamp = node_->now();
        mkr_yellow_arrow.ns = "perturbation_stdpose_force";
        mkr_yellow_arrow.id = 50000 + i;  // 避免和其他标记冲突
        mkr_yellow_arrow.type = visualization_msgs::msg::Marker::ARROW;
        mkr_yellow_arrow.action = visualization_msgs::msg::Marker::ADD;
        
        // 箭头起点：标准姿态坐标系下的推力位置
        mkr_yellow_arrow.pose.position.x = std_pose_pos[0];
        mkr_yellow_arrow.pose.position.y = std_pose_pos[1];
        mkr_yellow_arrow.pose.position.z = std_pose_pos[2];
        
        // 箭头方向：推力在标准姿态坐标系下的方向
        // 需要将推力从局部坐标系转换到标准姿态坐标系
        mkr_yellow_arrow.scale.x = arrow_length;  // 箭头长度
        mkr_yellow_arrow.scale.y = 0.02;          // 箭头宽度
        mkr_yellow_arrow.scale.z = 0.02;          // 箭头高度
        
        // 计算推力在标准姿态坐标系下的方向
        // 首先获取标准姿态下的body旋转矩阵
        mjtNum* std_rotmat = nullptr;
        if (perturbation_body_id < m->nbody && !std_xmat_cache.empty()) {
          std_rotmat = &std_xmat_cache[9 * perturbation_body_id];
        }
        
        // 将推力从局部坐标系转换到标准姿态坐标系
        mjtNum perturb_force_stdpose[3];
        if (std_rotmat) {
          mjtNum perturb_force_local[3] = {pert.force.x(), pert.force.y(), pert.force.z()};
          mju_rotVecMat(perturb_force_stdpose, perturb_force_local, std_rotmat);
        } else {
          // 如果没有标准姿态数据，使用原始局部方向
          perturb_force_stdpose[0] = pert.force.x();
          perturb_force_stdpose[1] = pert.force.y();
          perturb_force_stdpose[2] = pert.force.z();
        }
        
        // 使用与红色箭头相同的方向计算方法
        double stdpose_norm = sqrt(perturb_force_stdpose[0]*perturb_force_stdpose[0] + 
                                  perturb_force_stdpose[1]*perturb_force_stdpose[1] + 
                                  perturb_force_stdpose[2]*perturb_force_stdpose[2]);
        if (stdpose_norm > 0.001) {
          // 归一化标准姿态推力向量
          double fx = perturb_force_stdpose[0] / stdpose_norm;
          double fy = perturb_force_stdpose[1] / stdpose_norm;
          double fz = perturb_force_stdpose[2] / stdpose_norm;
          
          // 计算从X轴(1,0,0)到推力方向的旋转（RViz ARROW默认指向X轴正方向）
          double axis_x = 0.0;  // 叉积 X × force
          double axis_y = -fz;  // 0*fz - 1*fy = -fz
          double axis_z = fy;   // 1*fx - 0*fz = fy
          
          double cos_angle = fx;  // 点积 X · force
          double angle = acos(std::clamp(cos_angle, -1.0, 1.0));
          
          if (angle > 0.001) {
            double axis_norm = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
            if (axis_norm > 0.001) {
              axis_x /= axis_norm;
              axis_y /= axis_norm;
              axis_z /= axis_norm;
            }
          } else {
            axis_x = 0; axis_y = 0; axis_z = 1;
          }
          
          // 转换为四元数
          double s = sin(angle / 2.0);
          mkr_yellow_arrow.pose.orientation.w = cos(angle / 2.0);
          mkr_yellow_arrow.pose.orientation.x = axis_x * s;
          mkr_yellow_arrow.pose.orientation.y = axis_y * s;
          mkr_yellow_arrow.pose.orientation.z = axis_z * s;
        } else {
          mkr_yellow_arrow.pose.orientation.w = 1.0;
          mkr_yellow_arrow.pose.orientation.x = 0.0;
          mkr_yellow_arrow.pose.orientation.y = 0.0;
          mkr_yellow_arrow.pose.orientation.z = 0.0;
        }
        
        // 颜色：黄色（表示标准姿态坐标系下的推力）
        mkr_yellow_arrow.color.r = 1.0; mkr_yellow_arrow.color.g = 1.0; mkr_yellow_arrow.color.b = 0.0; mkr_yellow_arrow.color.a = 0.8;
        mkr_yellow_arrow.lifetime = rclcpp::Duration::from_seconds(0.1);
        
        perturbation_arr.markers.push_back(mkr_yellow_arrow);
      }
      
      // 发布推力可视化标记
      if (!perturbation_arr.markers.empty()) {
        contact_marker_pub_->publish(perturbation_arr);
      }
    }
  }

  // ==================== 第五部分：保存到CSV文件 ====================
  if (save_contact_csv_ && csv_file_.is_open()) {
    // 使用全局标志位控制CSV记录
    static double csv_start_time = -1.0;
    static bool mujoco_reset_done = false;
    
    // 如果还没有开始记录，检查是否已经过了3秒
    if (!contact_csv_enabled) {
      if (csv_start_time < 0) {
        csv_start_time = d->time;  // 记录开始时间
        RCLCPP_INFO(node_->get_logger(), "CSV记录延迟开始，等待3秒后重置MuJoCo...");
      }
      
      // 检查是否已经过了3秒
      if (d->time - csv_start_time >= 3.0) {
        if (!mujoco_reset_done) {
          // 执行MuJoCo重置
          RCLCPP_INFO(node_->get_logger(), "3秒延迟结束，开始重置MuJoCo...");
          
          // 重置到初始状态
          if (m) {
            // 查找并应用keyframe
            int keyframe_id = mj_name2id(m, mjOBJ_KEY, "floating_base_homing");
            if (keyframe_id >= 0) {
              mj_resetDataKeyframe(m, d, keyframe_id);
              RCLCPP_INFO(node_->get_logger(), "已应用keyframe 'floating_base_homing' 重置机器人到初始状态");
            } else {
              // 如果没有keyframe，重置到零位
              mju_zero(d->qpos, m->nq);
              if (m->jnt_type[0] == mjJNT_FREE) {
                // 对于自由关节，设置四元数为单位四元数，位置为原点
                d->qpos[0] = 1; d->qpos[1] = d->qpos[2] = d->qpos[3] = 0; // quat = [1,0,0,0]
                d->qpos[4] = d->qpos[5] = d->qpos[6] = 0; // base at origin (0,0,0)
              }
              RCLCPP_INFO(node_->get_logger(), "已重置机器人到零位状态");
            }
            
            // 执行前向运动学计算
            mj_forward(m, d);
            RCLCPP_INFO(node_->get_logger(), "MuJoCo重置完成");
            
            // 重置auto_sampling状态，让推力重新施加
            auto& sim_manager = SimManager::GetInstance();
            sim_manager.ResetAutoSampling();
            RCLCPP_INFO(node_->get_logger(), "已重置auto_sampling状态，推力将在auto_delay时间后重新施加");
            
            // 发布重置完成消息
            std_msgs::msg::Empty reset_msg;
            mujoco_reset_pub_->publish(reset_msg);
            RCLCPP_INFO(node_->get_logger(), "已发布MuJoCo重置完成消息");
          }
          
          mujoco_reset_done = true;
          // 重置后启用CSV记录
          contact_csv_enabled = true;
          perturbation_csv_enabled = true;
        }
      } else {
        // 还在延迟期间，不记录CSV
        return;
      }
    }
    
    // 使用帧计数器控制保存频率
    static int frame_counter = 0;
    frame_counter++;
    
    // 根据配置的频率决定是否保存，并且检查标志位
    if (contact_csv_enabled && frame_counter % csv_save_frequency_ == 0) {
      // 使用MuJoCo仿真时间而不是ROS时间
      double sim_time = d->time;
      
      // ==================== 获取多干扰力数据 ====================
      auto& sim_manager = SimManager::GetInstance();
      auto active_perturbations = sim_manager.GetActivePerturbations();
      int num_perturbations = active_perturbations.size();
      
      // 为每个接触点准备数据并加入队列（如果两个geom都不是world，写入两行；否则写入一行）
      for (int i = 0; i < ncon; ++i) {
        // 获取body1_id和body2_id
        int body1_id = 0;
        int body2_id = 0;
        if (csv_body1_ids && csv_body2_ids) {
          body1_id = (*csv_body1_ids)[i];
          body2_id = (*csv_body2_ids)[i];
        }
        
        // 判断是否需要写入两行（只有当两个geom都不是world时）
        bool write_two_rows = (body1_id != 0 && body2_id != 0);
        
        // 确定要写入的行数
        int num_rows = write_two_rows ? 2 : 1;
        
        for (int row = 0; row < num_rows; ++row) {
          // 确定当前行使用的绿球坐标和body名称
          mjtNum green_ball_x, green_ball_y, green_ball_z;
          std::string body1_name_for_row, body2_name_for_row;
          
          if (write_two_rows) {
            // 写入两行：第一行用body1的坐标系，第二行用body2的坐标系
            if (row == 0) {
              // 第一行：使用body1的坐标系，不交换名字
              green_ball_x = (*csv_green_ball_pos_x)[i];
              green_ball_y = (*csv_green_ball_pos_y)[i];
              green_ball_z = (*csv_green_ball_pos_z)[i];
              body1_name_for_row = (*csv_body1_names)[i];
              body2_name_for_row = (*csv_body2_names)[i];
            } else {
              // 第二行：使用body2的坐标系，交换body1和body2的名字
              green_ball_x = (*csv_green_ball_pos_body2_x)[i];
              green_ball_y = (*csv_green_ball_pos_body2_y)[i];
              green_ball_z = (*csv_green_ball_pos_body2_z)[i];
              body1_name_for_row = (*csv_body2_names)[i];  // 交换：body2变成body1
              body2_name_for_row = (*csv_body1_names)[i];  // 交换：body1变成body2
            }
          } else {
            // 只写入一行：使用非world的那个body的坐标系，不交换名字
            green_ball_x = (*csv_green_ball_pos_x)[i];
            green_ball_y = (*csv_green_ball_pos_y)[i];
            green_ball_z = (*csv_green_ball_pos_z)[i];
            body1_name_for_row = (*csv_body1_names)[i];
            body2_name_for_row = (*csv_body2_names)[i];
          }
          
          // 准备数据行并加入队列 - 使用智能指针访问
          if (csv_body1_names && csv_body2_names && csv_red_ball_pos_x && csv_red_ball_pos_y && csv_red_ball_pos_z &&
              csv_green_ball_pos_x && csv_green_ball_pos_y && csv_green_ball_pos_z &&
              csv_world_forces_x && csv_world_forces_y && csv_world_forces_z &&
              csv_contact_force_magnitudes && csv_contact_force_normals &&
              csv_world_torques_x && csv_world_torques_y && csv_world_torques_z &&
              csv_base_link_pos_x && csv_base_link_pos_y && csv_base_link_pos_z &&
              csv_base_link_quat_w && csv_base_link_quat_x && csv_base_link_quat_y && csv_base_link_quat_z &&
              csv_base_link_vel_x && csv_base_link_vel_y && csv_base_link_vel_z &&
              csv_base_link_angvel_x && csv_base_link_angvel_y && csv_base_link_angvel_z &&
              csv_collision_link_pos_x && csv_collision_link_pos_y && csv_collision_link_pos_z &&
              csv_collision_link_quat_w && csv_collision_link_quat_x && csv_collision_link_quat_y && csv_collision_link_quat_z) {
            ContactDataRow row;
            row.sim_time = sim_time;
            row.contact_id = i;
            row.body1_name = body1_name_for_row;
            row.body2_name = body2_name_for_row;
            row.red_ball_pos[0] = (*csv_red_ball_pos_x)[i];
            row.red_ball_pos[1] = (*csv_red_ball_pos_y)[i];
            row.red_ball_pos[2] = (*csv_red_ball_pos_z)[i];
            row.green_ball_pos[0] = green_ball_x;
            row.green_ball_pos[1] = green_ball_y;
            row.green_ball_pos[2] = green_ball_z;
            row.world_forces[0] = (*csv_world_forces_x)[i];
            row.world_forces[1] = (*csv_world_forces_y)[i];
            row.world_forces[2] = (*csv_world_forces_z)[i];
            row.force_magnitude = (*csv_contact_force_magnitudes)[i];
            row.force_normal = (*csv_contact_force_normals)[i];
            row.world_torques[0] = (*csv_world_torques_x)[i];
            row.world_torques[1] = (*csv_world_torques_y)[i];
            row.world_torques[2] = (*csv_world_torques_z)[i];
            row.base_link_pos[0] = (*csv_base_link_pos_x)[i];
            row.base_link_pos[1] = (*csv_base_link_pos_y)[i];
            row.base_link_pos[2] = (*csv_base_link_pos_z)[i];
            row.base_link_quat[0] = (*csv_base_link_quat_w)[i];
            row.base_link_quat[1] = (*csv_base_link_quat_x)[i];
            row.base_link_quat[2] = (*csv_base_link_quat_y)[i];
            row.base_link_quat[3] = (*csv_base_link_quat_z)[i];
            row.base_link_vel[0] = (*csv_base_link_vel_x)[i];
            row.base_link_vel[1] = (*csv_base_link_vel_y)[i];
            row.base_link_vel[2] = (*csv_base_link_vel_z)[i];
            row.base_link_angvel[0] = (*csv_base_link_angvel_x)[i];
            row.base_link_angvel[1] = (*csv_base_link_angvel_y)[i];
            row.base_link_angvel[2] = (*csv_base_link_angvel_z)[i];
            row.collision_link_pos[0] = (*csv_collision_link_pos_x)[i];
            row.collision_link_pos[1] = (*csv_collision_link_pos_y)[i];
            row.collision_link_pos[2] = (*csv_collision_link_pos_z)[i];
            row.collision_link_quat[0] = (*csv_collision_link_quat_w)[i];
            row.collision_link_quat[1] = (*csv_collision_link_quat_x)[i];
            row.collision_link_quat[2] = (*csv_collision_link_quat_y)[i];
            row.collision_link_quat[3] = (*csv_collision_link_quat_z)[i];
            
            // 将数据加入队列（非阻塞，异步写入）
            // 注意：关节角度、关节加速度和电机扭矩已移到独立的 joint_state_data.csv 文件中
            EnqueueContactData(row);
          }
        }
      }
    }
  }

  // ==================== 推力数据单独保存 ====================
  if (save_perturbation_csv_ && perturbation_csv_file_.is_open()) {
    // 使用全局标志位控制perturbation CSV记录
    
    // 检查是否启用perturbation CSV记录
    if (!perturbation_csv_enabled) {
      return;
    }
    
    // 使用帧计数器控制保存频率
    static int perturbation_frame_counter = 0;
    perturbation_frame_counter++;
    
    // 根据配置的频率决定是否保存，并且检查标志位
    if (perturbation_csv_enabled && perturbation_frame_counter % csv_save_frequency_ == 0) {
      // 使用MuJoCo仿真时间
      double sim_time = d->time;
      
      // 获取活跃的干扰力数据
      auto& sim_manager = SimManager::GetInstance();
      auto active_perturbations = sim_manager.GetActivePerturbations();
      
      // 为每个活跃的推力准备数据并加入队列
      for (const auto& pert : active_perturbations) {
        // 只记录真正活跃的推力
        if (!pert.is_active) {
          continue;
        }
        // 计算推力在标准姿态下的位置
        mjtNum perturbation_std_pose[3] = {0, 0, 0};
        int perturbation_body_id = mj_name2id(m, mjOBJ_BODY, pert.body_name.c_str());
        if (perturbation_body_id >= 0) {
          // 复用contact force计算中的标准姿态缓存
          if (perturbation_body_id < m->nbody && !std_xpos_cache.empty()) {
            perturbation_std_pose[0] = std_xpos_cache[3 * perturbation_body_id];
            perturbation_std_pose[1] = std_xpos_cache[3 * perturbation_body_id + 1];
            perturbation_std_pose[2] = std_xpos_cache[3 * perturbation_body_id + 2];
          }
        }
        
        // 计算推力在世界坐标系下的方向
        mjtNum* body_quat = d->xquat + 4 * perturbation_body_id;
        mjtNum perturb_force_local[3] = {pert.force.x(), pert.force.y(), pert.force.z()};
        mjtNum perturb_force_world[3];
        mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);
        
        // 计算推力大小
        double force_magnitude = pert.force.norm();
        double torque_magnitude = pert.torque.norm();
        double world_force_magnitude = sqrt(perturb_force_world[0]*perturb_force_world[0] + 
                                           perturb_force_world[1]*perturb_force_world[1] + 
                                           perturb_force_world[2]*perturb_force_world[2]);
        
        // 准备推力数据并加入队列
        PerturbationDataRow row;
        row.sim_time = sim_time;
        row.perturbation_id = pert.id;
        row.body_name = pert.body_name;
        row.start_time = pert.start_time;
        row.duration = pert.duration;
        row.elapsed_time = d->time - pert.start_time;
        row.force[0] = pert.force.x();
        row.force[1] = pert.force.y();
        row.force[2] = pert.force.z();
        row.force_magnitude = force_magnitude;
        row.torque[0] = pert.torque.x();
        row.torque[1] = pert.torque.y();
        row.torque[2] = pert.torque.z();
        row.torque_magnitude = torque_magnitude;
        row.std_pose[0] = perturbation_std_pose[0];
        row.std_pose[1] = perturbation_std_pose[1];
        row.std_pose[2] = perturbation_std_pose[2];
        row.world_force[0] = perturb_force_world[0];
        row.world_force[1] = perturb_force_world[1];
        row.world_force[2] = perturb_force_world[2];
        row.world_force_magnitude = world_force_magnitude;
        
        // 将数据加入队列（非阻塞，异步写入）
        EnqueuePerturbationData(row);
      }
    }
  }
}

/**
 * @brief 设置MuJoCo模型和数据指针
 * @param model MuJoCo模型指针
 * @param data MuJoCo数据指针
 */


void RosInterface::SetModelAndData(mjModel* model, mjData* data) {
  try {
    model_ = model;
    data_ = data;
    
    if (model && data) {
      RCLCPP_INFO(node_->get_logger(), "Model and data pointers set successfully");
    } else {
      RCLCPP_INFO(node_->get_logger(), "Model and data pointers cleared");
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Exception in SetModelAndData: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(node_->get_logger(), "Unknown exception in SetModelAndData");
  }
}

/**
 * @brief 运动状态定时器回调函数
 * @details 每秒发布一次运动状态消息，当前设置为"joint_bridge"模式
 */
void RosInterface::MotionStateTimerCallback() {
  // 创建运动状态消息
  auto motion_state_msg = std::make_unique<interface_protocol::msg::MotionState>();
  
  // 设置当前运动任务字段为"joint_bridge"
  motion_state_msg->current_motion_task = "joint_bridge";
  
  // 发布消息
  motion_state_pub_->publish(std::move(motion_state_msg));
}

/**
 * @brief 保存关节反力数据到CSV文件
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 从SimManager获取关节反力数据并保存到CSV文件
 */
void RosInterface::SaveJointForcesToCSV(const mjModel* m, mjData* d) {
  if (!m || !d) return;
  
  static int frame_counter = 0;
  frame_counter++;
  
  // 根据配置的频率决定是否保存
  if (frame_counter % csv_save_frequency_ != 0) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(joint_forces_csv_mutex_);
  
  if (!joint_forces_csv_file_.is_open()) {
    return;
  }
  
  // 使用MuJoCo仿真时间
  double sim_time = d->time;
  
  // 从SimManager获取关节反力数据
  auto& sim_manager = SimManager::GetInstance();
  const auto& joint_wrenches_child = sim_manager.GetJointWrenchesChild();
  const auto& joint_wrenches_parent = sim_manager.GetJointWrenchesParent();
  
  // 遍历所有关节
  for (int j = 0; j < m->njnt; ++j) {
    int jtype = m->jnt_type[j];
    if (jtype == mjJNT_FREE) {
      continue;  // 跳过自由关节
    }
    
    // 获取关节名称
    std::string joint_name = m->names + m->name_jntadr[j];
    
    // 获取关节所属的body
    int body_id = m->jnt_bodyid[j];
    std::string body_name = m->names + m->name_bodyadr[body_id];
    
    // 获取关节轴向量
    Eigen::Vector3d axis(
      m->jnt_axis[3 * j + 0],
      m->jnt_axis[3 * j + 1],
      m->jnt_axis[3 * j + 2]
    );
    
    // 获取子body坐标系下的反力
    const auto& child_wrench = joint_wrenches_child[j];
    const auto& parent_wrench = joint_wrenches_parent[j];
    
    // 计算载荷分解
    DecomposedWrenchEigen decomposed;
    try {
      decomposed = decomposeWrenchBodyFrameEigen(child_wrench, axis);
    } catch (const std::exception& e) {
      // 如果分解失败，使用零值
      decomposed.F_axial_mag = 0.0;
      decomposed.F_axial = Eigen::Vector3d::Zero();
      decomposed.F_shear_mag = 0.0;
      decomposed.F_shear = Eigen::Vector3d::Zero();
      decomposed.M_torsion_mag = 0.0;
      decomposed.M_torsion = Eigen::Vector3d::Zero();
      decomposed.M_bend_mag = 0.0;
      decomposed.M_bend = Eigen::Vector3d::Zero();
      decomposed.M_eq = 0.0;
    }
    
    // 根据格式选择写入方式
    if (csv_format_ == "binary") {
      // 二进制格式写入
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&sim_time), sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&j), sizeof(int));
      
      // 写入字符串
      int32_t joint_name_len = joint_name.length();
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&joint_name_len), sizeof(int32_t));
      joint_forces_csv_file_.write(joint_name.c_str(), joint_name_len);
      
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&body_id), sizeof(int));
      
      int32_t body_name_len = body_name.length();
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&body_name_len), sizeof(int32_t));
      joint_forces_csv_file_.write(body_name.c_str(), body_name_len);
      
      // 写入子body坐标系下的反力
      double child_M[3] = {child_wrench.M.x(), child_wrench.M.y(), child_wrench.M.z()};
      double child_F[3] = {child_wrench.F.x(), child_wrench.F.y(), child_wrench.F.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(child_M), 3 * sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(child_F), 3 * sizeof(double));
      
      // 写入父body坐标系下的反力
      double parent_M[3] = {parent_wrench.M.x(), parent_wrench.M.y(), parent_wrench.M.z()};
      double parent_F[3] = {parent_wrench.F.x(), parent_wrench.F.y(), parent_wrench.F.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(parent_M), 3 * sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(parent_F), 3 * sizeof(double));
      
      // 写入关节轴向量
      double axis_vec[3] = {axis.x(), axis.y(), axis.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(axis_vec), 3 * sizeof(double));
      
      // 写入载荷分解结果
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&decomposed.F_axial_mag), sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&decomposed.F_shear_mag), sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&decomposed.M_torsion_mag), sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&decomposed.M_bend_mag), sizeof(double));
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(&decomposed.M_eq), sizeof(double));
      
      // 写入轴向力向量
      double F_axial[3] = {decomposed.F_axial.x(), decomposed.F_axial.y(), decomposed.F_axial.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(F_axial), 3 * sizeof(double));
      
      // 写入剪切力向量
      double F_shear[3] = {decomposed.F_shear.x(), decomposed.F_shear.y(), decomposed.F_shear.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(F_shear), 3 * sizeof(double));
      
      // 写入扭转力矩向量
      double M_torsion[3] = {decomposed.M_torsion.x(), decomposed.M_torsion.y(), decomposed.M_torsion.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(M_torsion), 3 * sizeof(double));
      
      // 写入弯曲力矩向量
      double M_bend[3] = {decomposed.M_bend.x(), decomposed.M_bend.y(), decomposed.M_bend.z()};
      joint_forces_csv_file_.write(reinterpret_cast<const char*>(M_bend), 3 * sizeof(double));
      
      // 每N条记录flush一次
      static int flush_counter = 0;
      flush_counter++;
      if (flush_counter >= 100) {
        joint_forces_csv_file_.flush();
        flush_counter = 0;
      }
    } else {
      // CSV格式写入
      joint_forces_csv_file_ << std::fixed << std::setprecision(6)
                             << sim_time << ","
                             << j << ","
                             << "\"" << joint_name << "\","
                             << body_id << ","
                             << "\"" << body_name << "\","
                             // 子body坐标系下的反力
                             << child_wrench.M.x() << ","
                             << child_wrench.M.y() << ","
                             << child_wrench.M.z() << ","
                             << child_wrench.F.x() << ","
                             << child_wrench.F.y() << ","
                             << child_wrench.F.z() << ","
                             // 父body坐标系下的反力
                             << parent_wrench.M.x() << ","
                             << parent_wrench.M.y() << ","
                             << parent_wrench.M.z() << ","
                             << parent_wrench.F.x() << ","
                             << parent_wrench.F.y() << ","
                             << parent_wrench.F.z() << ","
                             // 关节轴向量
                             << axis.x() << ","
                             << axis.y() << ","
                             << axis.z() << ","
                             // 载荷分解的标量值
                             << decomposed.F_axial_mag << ","
                             << decomposed.F_shear_mag << ","
                             << decomposed.M_torsion_mag << ","
                             << decomposed.M_bend_mag << ","
                             << decomposed.M_eq << ","  // 综合破坏载荷
                             // 轴向力向量
                             << decomposed.F_axial.x() << ","
                             << decomposed.F_axial.y() << ","
                             << decomposed.F_axial.z() << ","
                             // 剪切力向量
                             << decomposed.F_shear.x() << ","
                             << decomposed.F_shear.y() << ","
                             << decomposed.F_shear.z() << ","
                             // 扭矩向量
                             << decomposed.M_torsion.x() << ","
                             << decomposed.M_torsion.y() << ","
                             << decomposed.M_torsion.z() << ","
                             // 弯矩向量
                             << decomposed.M_bend.x() << ","
                             << decomposed.M_bend.y() << ","
                             << decomposed.M_bend.z() << "\n";
    }
  }
  
  // 刷新CSV文件缓冲区
  joint_forces_csv_file_.flush();
}


// ==================== 异步写入函数实现 ====================

void RosInterface::EnqueueContactData(const ContactDataRow& row) {
  std::lock_guard<std::mutex> lock(contact_queue_mutex_);
  
  // 如果队列太大，丢弃最旧的数据（FIFO）
  if (contact_data_queue_.size() >= MAX_QUEUE_SIZE) {
    contact_data_queue_.pop();  // 移除最旧的数据
    contact_queue_dropped_++;
    
    // 每丢弃1000条记录警告一次
    if (contact_queue_dropped_ % 1000 == 0) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                          "Contact data queue full (%zu), dropped %zu records. "
                          "CSV writing may be too slow, consider reducing csv_save_frequency.",
                          contact_data_queue_.size(), contact_queue_dropped_);
    }
  }
  
  contact_data_queue_.push(row);
  contact_queue_cv_.notify_one();
}

void RosInterface::EnqueuePerturbationData(const PerturbationDataRow& row) {
  std::lock_guard<std::mutex> lock(perturbation_queue_mutex_);
  
  // 如果队列太大，丢弃最旧的数据（FIFO）
  if (perturbation_data_queue_.size() >= MAX_QUEUE_SIZE) {
    perturbation_data_queue_.pop();  // 移除最旧的数据
    perturbation_queue_dropped_++;
    
    // 每丢弃1000条记录警告一次
    if (perturbation_queue_dropped_ % 1000 == 0) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                          "Perturbation data queue full (%zu), dropped %zu records. "
                          "CSV writing may be too slow, consider reducing csv_save_frequency.",
                          perturbation_data_queue_.size(), perturbation_queue_dropped_);
    }
  }
  
  perturbation_data_queue_.push(row);
  perturbation_queue_cv_.notify_one();
}

void RosInterface::ContactWriterThread() {
  while (writer_threads_running_.load() || !contact_data_queue_.empty()) {
    std::unique_lock<std::mutex> lock(contact_queue_mutex_);
    
    // 等待队列中有数据或线程需要停止
    contact_queue_cv_.wait(lock, [this] {
      return !contact_data_queue_.empty() || !writer_threads_running_.load();
    });
    
    // 处理队列中的所有数据
    while (!contact_data_queue_.empty()) {
      ContactDataRow row = contact_data_queue_.front();
      contact_data_queue_.pop();
      lock.unlock();
      
      // 写入数据（根据格式选择）
      std::lock_guard<std::mutex> file_lock(csv_mutex_);
      if (csv_file_.is_open()) {
        if (csv_format_ == "binary") {
          WriteContactDataBinary(row);
        } else {
          // CSV格式写入
          csv_file_ << std::fixed << std::setprecision(6)
                    << row.sim_time << ","
                    << row.contact_id << ","
                    << "\"" << row.body1_name << "\","
                    << "\"" << row.body2_name << "\","
                    << row.red_ball_pos[0] << ","
                    << row.red_ball_pos[1] << ","
                    << row.red_ball_pos[2] << ","
                    << row.green_ball_pos[0] << ","
                    << row.green_ball_pos[1] << ","
                    << row.green_ball_pos[2] << ","
                    << row.world_forces[0] << ","
                    << row.world_forces[1] << ","
                    << row.world_forces[2] << ","
                    << row.force_magnitude << ","
                    << row.force_normal << ","
                    << row.world_torques[0] << ","
                    << row.world_torques[1] << ","
                    << row.world_torques[2] << ","
                    << row.base_link_pos[0] << ","
                    << row.base_link_pos[1] << ","
                    << row.base_link_pos[2] << ","
                    << row.base_link_quat[0] << ","
                    << row.base_link_quat[1] << ","
                    << row.base_link_quat[2] << ","
                    << row.base_link_quat[3] << ","
                    << row.base_link_vel[0] << ","
                    << row.base_link_vel[1] << ","
                    << row.base_link_vel[2] << ","
                    << row.base_link_angvel[0] << ","
                    << row.base_link_angvel[1] << ","
                    << row.base_link_angvel[2] << ","
                    << row.collision_link_pos[0] << ","
                    << row.collision_link_pos[1] << ","
                    << row.collision_link_pos[2] << ","
                    << row.collision_link_quat[0] << ","
                    << row.collision_link_quat[1] << ","
                    << row.collision_link_quat[2] << ","
                    << row.collision_link_quat[3] << "\n";
          // 注意：关节角度、关节加速度和电机扭矩已移到独立的 joint_state_data.csv 文件中
        }
        
        // 每N条记录flush一次，而不是每条都flush
        contact_flush_counter_++;
        if (contact_flush_counter_ >= flush_interval_) {
          csv_file_.flush();
          contact_flush_counter_ = 0;
        }
      }
      
      lock.lock();
    }
  }
  
  // 最后flush一次
  std::lock_guard<std::mutex> file_lock(csv_mutex_);
  if (csv_file_.is_open()) {
    csv_file_.flush();
  }
}

void RosInterface::PerturbationWriterThread() {
  while (writer_threads_running_.load() || !perturbation_data_queue_.empty()) {
    std::unique_lock<std::mutex> lock(perturbation_queue_mutex_);
    
    // 等待队列中有数据或线程需要停止
    perturbation_queue_cv_.wait(lock, [this] {
      return !perturbation_data_queue_.empty() || !writer_threads_running_.load();
    });
    
    // 处理队列中的所有数据
    while (!perturbation_data_queue_.empty()) {
      PerturbationDataRow row = perturbation_data_queue_.front();
      perturbation_data_queue_.pop();
      lock.unlock();
      
      // 写入数据（根据格式选择）
      std::lock_guard<std::mutex> file_lock(perturbation_csv_mutex_);
      if (perturbation_csv_file_.is_open()) {
        if (csv_format_ == "binary") {
          WritePerturbationDataBinary(row);
        } else {
          // CSV格式写入
          perturbation_csv_file_ << std::fixed << std::setprecision(6)
                                 << row.sim_time << ","
                                 << row.perturbation_id << ","
                                 << "\"" << row.body_name << "\","
                                 << row.start_time << ","
                                 << row.duration << ","
                                 << row.elapsed_time << ","
                                 << row.force[0] << ","
                                 << row.force[1] << ","
                                 << row.force[2] << ","
                                 << row.force_magnitude << ","
                                 << row.torque[0] << ","
                                 << row.torque[1] << ","
                                 << row.torque[2] << ","
                                 << row.torque_magnitude << ","
                                 << row.std_pose[0] << ","
                                 << row.std_pose[1] << ","
                                 << row.std_pose[2] << ","
                                 << row.world_force[0] << ","
                                 << row.world_force[1] << ","
                                 << row.world_force[2] << ","
                                 << row.world_force_magnitude << "\n";
        }
        
        // 每N条记录flush一次
        perturbation_flush_counter_++;
        if (perturbation_flush_counter_ >= flush_interval_) {
          perturbation_csv_file_.flush();
          perturbation_flush_counter_ = 0;
        }
      }
      
      lock.lock();
    }
  }
  
  // 最后flush一次
  std::lock_guard<std::mutex> file_lock(perturbation_csv_mutex_);
  if (perturbation_csv_file_.is_open()) {
    perturbation_csv_file_.flush();
  }
}

void RosInterface::WriteContactDataBinary(const ContactDataRow& row) {
  // 写入固定大小的数据
  csv_file_.write(reinterpret_cast<const char*>(&row.sim_time), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.contact_id), sizeof(int));
  
  // 写入字符串长度和内容
  int32_t body1_len = row.body1_name.length();
  csv_file_.write(reinterpret_cast<const char*>(&body1_len), sizeof(int32_t));
  csv_file_.write(row.body1_name.c_str(), body1_len);
  
  int32_t body2_len = row.body2_name.length();
  csv_file_.write(reinterpret_cast<const char*>(&body2_len), sizeof(int32_t));
  csv_file_.write(row.body2_name.c_str(), body2_len);
  
  // 写入数组数据
  csv_file_.write(reinterpret_cast<const char*>(row.red_ball_pos), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.green_ball_pos), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.world_forces), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.force_magnitude), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.force_normal), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.world_torques), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.base_link_pos), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.base_link_quat), 4 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.base_link_vel), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.base_link_angvel), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.collision_link_pos), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.collision_link_quat), 4 * sizeof(double));
  // 注意：关节角度、关节加速度和电机扭矩已移到独立的 joint_state_data.csv 文件中
}

void RosInterface::WritePerturbationDataBinary(const PerturbationDataRow& row) {
  csv_file_.write(reinterpret_cast<const char*>(&row.sim_time), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.perturbation_id), sizeof(int));
  
  // 写入字符串
  int32_t body_name_len = row.body_name.length();
  csv_file_.write(reinterpret_cast<const char*>(&body_name_len), sizeof(int32_t));
  csv_file_.write(row.body_name.c_str(), body_name_len);
  
  // 写入其他数据
  csv_file_.write(reinterpret_cast<const char*>(&row.start_time), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.duration), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.elapsed_time), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.force), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.force_magnitude), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.torque), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.torque_magnitude), sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.std_pose), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(row.world_force), 3 * sizeof(double));
  csv_file_.write(reinterpret_cast<const char*>(&row.world_force_magnitude), sizeof(double));
}

void RosInterface::FlushRemainingData() {
  // 处理剩余的接触力数据
  {
    std::lock_guard<std::mutex> lock(contact_queue_mutex_);
    while (!contact_data_queue_.empty()) {
      ContactDataRow row = contact_data_queue_.front();
      contact_data_queue_.pop();
      
      std::lock_guard<std::mutex> file_lock(csv_mutex_);
      if (csv_file_.is_open()) {
        if (csv_format_ == "binary") {
          WriteContactDataBinary(row);
        } else {
          // CSV格式（简化版，与ContactWriterThread中的相同）
          csv_file_ << std::fixed << std::setprecision(6)
                    << row.sim_time << ","
                    << row.contact_id << ","
                    << "\"" << row.body1_name << "\","
                    << "\"" << row.body2_name << "\","
                    << row.red_ball_pos[0] << ","
                    << row.red_ball_pos[1] << ","
                    << row.red_ball_pos[2] << ","
                    << row.green_ball_pos[0] << ","
                    << row.green_ball_pos[1] << ","
                    << row.green_ball_pos[2] << ","
                    << row.world_forces[0] << ","
                    << row.world_forces[1] << ","
                    << row.world_forces[2] << ","
                    << row.force_magnitude << ","
                    << row.force_normal << ","
                    << row.world_torques[0] << ","
                    << row.world_torques[1] << ","
                    << row.world_torques[2] << ","
                    << row.base_link_pos[0] << ","
                    << row.base_link_pos[1] << ","
                    << row.base_link_pos[2] << ","
                    << row.base_link_quat[0] << ","
                    << row.base_link_quat[1] << ","
                    << row.base_link_quat[2] << ","
                    << row.base_link_quat[3] << ","
                    << row.base_link_vel[0] << ","
                    << row.base_link_vel[1] << ","
                    << row.base_link_vel[2] << ","
                    << row.base_link_angvel[0] << ","
                    << row.base_link_angvel[1] << ","
                    << row.base_link_angvel[2] << ","
                    << row.collision_link_pos[0] << ","
                    << row.collision_link_pos[1] << ","
                    << row.collision_link_pos[2] << ","
                    << row.collision_link_quat[0] << ","
                    << row.collision_link_quat[1] << ","
                    << row.collision_link_quat[2] << ","
                    << row.collision_link_quat[3] << "\n";
          // 注意：关节角度、关节加速度和电机扭矩已移到独立的 joint_state_data.csv 文件中
        }
      }
    }
    if (csv_file_.is_open()) {
      csv_file_.flush();
    }
  }
  
  // 处理剩余的推力数据
  {
    std::lock_guard<std::mutex> lock(perturbation_queue_mutex_);
    while (!perturbation_data_queue_.empty()) {
      PerturbationDataRow row = perturbation_data_queue_.front();
      perturbation_data_queue_.pop();
      
      std::lock_guard<std::mutex> file_lock(perturbation_csv_mutex_);
      if (perturbation_csv_file_.is_open()) {
        if (csv_format_ == "binary") {
          WritePerturbationDataBinary(row);
        } else {
          // CSV格式
          perturbation_csv_file_ << std::fixed << std::setprecision(6)
                                  << row.sim_time << ","
                                  << row.perturbation_id << ","
                                  << "\"" << row.body_name << "\","
                                  << row.start_time << ","
                                  << row.duration << ","
                                  << row.elapsed_time << ","
                                  << row.force[0] << ","
                                  << row.force[1] << ","
                                  << row.force[2] << ","
                                  << row.force_magnitude << ","
                                  << row.torque[0] << ","
                                  << row.torque[1] << ","
                                  << row.torque[2] << ","
                                  << row.torque_magnitude << ","
                                  << row.std_pose[0] << ","
                                  << row.std_pose[1] << ","
                                  << row.std_pose[2] << ","
                                  << row.world_force[0] << ","
                                  << row.world_force[1] << ","
                                  << row.world_force[2] << ","
                                  << row.world_force_magnitude << "\n";
        }
      }
    }
    if (perturbation_csv_file_.is_open()) {
      perturbation_csv_file_.flush();
    }
  }
}

/**
 * @brief 保存传感器震动数据到CSV文件（持续记录，不依赖接触力）
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 记录 base_link 和 head 的加速度，用于传感器震动分析
 */
void RosInterface::SaveSensorVibrationToCSV(const mjModel* m, mjData* d) {
  if (!m || !d) return;
  
  static int frame_counter = 0;
  frame_counter++;
  
  // 根据配置的频率决定是否保存
  if (frame_counter % csv_save_frequency_ != 0) {
    return;
  }
  
  // 获取 base_link 的加速度（IMU震动分析）
  int base_link_id = -1;
  for (int j = 0; j < m->nbody; j++) {
    if (std::string(m->names + m->name_bodyadr[j]) == "LINK_BASE") {
      base_link_id = j;
      break;
    }
  }
  
  double base_link_lin_acc[3] = {0, 0, 0};
  double base_link_ang_acc[3] = {0, 0, 0};
  if (base_link_id >= 0) {
    // cacc格式：[angacc_x, angacc_y, angacc_z, linacc_x, linacc_y, linacc_z]
    base_link_ang_acc[0] = d->cacc[base_link_id*6 + 0];  // 角加速度x
    base_link_ang_acc[1] = d->cacc[base_link_id*6 + 1];  // 角加速度y
    base_link_ang_acc[2] = d->cacc[base_link_id*6 + 2];  // 角加速度z
    base_link_lin_acc[0] = d->cacc[base_link_id*6 + 3];   // 线加速度x
    base_link_lin_acc[1] = d->cacc[base_link_id*6 + 4];   // 线加速度y
    base_link_lin_acc[2] = d->cacc[base_link_id*6 + 5];   // 线加速度z
  }
  
  // 获取 head 的加速度（头部传感器震动分析）
  int head_id = -1;
  for (int j = 0; j < m->nbody; j++) {
    if (std::string(m->names + m->name_bodyadr[j]) == "LINK_HEAD_YAW") {
      head_id = j;
      break;
    }
  }
  
  double head_lin_acc[3] = {0, 0, 0};
  double head_ang_acc[3] = {0, 0, 0};
  if (head_id >= 0) {
    // cacc格式：[angacc_x, angacc_y, angacc_z, linacc_x, linacc_y, linacc_z]
    head_ang_acc[0] = d->cacc[head_id*6 + 0];  // 角加速度x
    head_ang_acc[1] = d->cacc[head_id*6 + 1];  // 角加速度y
    head_ang_acc[2] = d->cacc[head_id*6 + 2];  // 角加速度z
    head_lin_acc[0] = d->cacc[head_id*6 + 3];   // 线加速度x
    head_lin_acc[1] = d->cacc[head_id*6 + 4];   // 线加速度y
    head_lin_acc[2] = d->cacc[head_id*6 + 5];   // 线加速度z
  }
  
  std::lock_guard<std::mutex> lock(sensor_vibration_csv_mutex_);
  
  if (!sensor_vibration_csv_file_.is_open()) {
    return;
  }
  
  if (csv_format_ == "binary") {
    // 二进制格式写入
    double sim_time = d->time;
    sensor_vibration_csv_file_.write(reinterpret_cast<const char*>(&sim_time), sizeof(double));
    sensor_vibration_csv_file_.write(reinterpret_cast<const char*>(base_link_lin_acc), 3 * sizeof(double));
    sensor_vibration_csv_file_.write(reinterpret_cast<const char*>(base_link_ang_acc), 3 * sizeof(double));
    sensor_vibration_csv_file_.write(reinterpret_cast<const char*>(head_lin_acc), 3 * sizeof(double));
    sensor_vibration_csv_file_.write(reinterpret_cast<const char*>(head_ang_acc), 3 * sizeof(double));
    
    // 每N条记录flush一次
    static int flush_counter = 0;
    flush_counter++;
    if (flush_counter >= 100) {
      sensor_vibration_csv_file_.flush();
      flush_counter = 0;
    }
  } else {
    // CSV格式写入
    sensor_vibration_csv_file_ << std::fixed << std::setprecision(6)
                               << d->time << ","
                               << base_link_lin_acc[0] << "," << base_link_lin_acc[1] << "," << base_link_lin_acc[2] << ","
                               << base_link_ang_acc[0] << "," << base_link_ang_acc[1] << "," << base_link_ang_acc[2] << ","
                               << head_lin_acc[0] << "," << head_lin_acc[1] << "," << head_lin_acc[2] << ","
                               << head_ang_acc[0] << "," << head_ang_acc[1] << "," << head_ang_acc[2] << "\n";
    
    // 每N条记录flush一次
    static int flush_counter = 0;
    flush_counter++;
    if (flush_counter >= 100) {
      sensor_vibration_csv_file_.flush();
      flush_counter = 0;
    }
  }
}

/**
 * @brief 保存关节状态数据到CSV文件（持续记录位置、速度、力矩）
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 记录所有关节的位置、速度和电机输出力矩，通过时间戳可以与其他CSV文件关联
 */
void RosInterface::SaveJointStateToCSV(const mjModel* m, mjData* d) {
  if (!m || !d) return;
  
  static int frame_counter = 0;
  frame_counter++;
  
  // 根据配置的频率决定是否保存
  if (frame_counter % csv_save_frequency_ != 0) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(joint_state_csv_mutex_);
  
  if (!joint_state_csv_file_.is_open()) {
    return;
  }
  
  if (csv_format_ == "binary") {
    // 二进制格式写入
    double sim_time = d->time;
    joint_state_csv_file_.write(reinterpret_cast<const char*>(&sim_time), sizeof(double));
    
    // 写入关节位置
    int32_t num_joints = num_total_joints_;
    joint_state_csv_file_.write(reinterpret_cast<const char*>(&num_joints), sizeof(int32_t));
    if (is_floating_base_) {
      // 浮动基座：跳过前7个元素（基座位置和四元数）
      for (int j = 0; j < num_total_joints_; j++) {
        double pos = d->qpos[j + kNumFloatingBaseJoints];
        joint_state_csv_file_.write(reinterpret_cast<const char*>(&pos), sizeof(double));
      }
    } else {
      // 非浮动基座：直接从qpos读取
      for (int j = 0; j < num_total_joints_; j++) {
        double pos = d->qpos[j];
        joint_state_csv_file_.write(reinterpret_cast<const char*>(&pos), sizeof(double));
      }
    }
    
    // 写入关节速度
    joint_state_csv_file_.write(reinterpret_cast<const char*>(&num_joints), sizeof(int32_t));
    if (is_floating_base_) {
      // 浮动基座：跳过前6个元素（基座线速度和角速度）
      for (int j = 0; j < num_total_joints_; j++) {
        double vel = d->qvel[j + kDofFloatingBase];
        joint_state_csv_file_.write(reinterpret_cast<const char*>(&vel), sizeof(double));
      }
    } else {
      // 非浮动基座：直接从qvel读取
      for (int j = 0; j < num_total_joints_; j++) {
        double vel = d->qvel[j];
        joint_state_csv_file_.write(reinterpret_cast<const char*>(&vel), sizeof(double));
      }
    }
    
    // 写入电机输出力矩
    joint_state_csv_file_.write(reinterpret_cast<const char*>(&num_joints), sizeof(int32_t));
    for (int j = 0; j < num_total_joints_; j++) {
      double force = d->actuator_force[j];
      joint_state_csv_file_.write(reinterpret_cast<const char*>(&force), sizeof(double));
    }
    
    // 每N条记录flush一次
    static int flush_counter = 0;
    flush_counter++;
    if (flush_counter >= 100) {
      joint_state_csv_file_.flush();
      flush_counter = 0;
    }
  } else {
    // CSV格式写入
    joint_state_csv_file_ << std::fixed << std::setprecision(6) << d->time;
    
    // 写入关节位置
    if (is_floating_base_) {
      // 浮动基座：跳过前7个元素（基座位置和四元数）
      for (int j = 0; j < num_total_joints_; j++) {
        joint_state_csv_file_ << "," << d->qpos[j + kNumFloatingBaseJoints];
      }
    } else {
      // 非浮动基座：直接从qpos读取
      for (int j = 0; j < num_total_joints_; j++) {
        joint_state_csv_file_ << "," << d->qpos[j];
      }
    }
    
    // 写入关节速度
    if (is_floating_base_) {
      // 浮动基座：跳过前6个元素（基座线速度和角速度）
      for (int j = 0; j < num_total_joints_; j++) {
        joint_state_csv_file_ << "," << d->qvel[j + kDofFloatingBase];
      }
    } else {
      // 非浮动基座：直接从qvel读取
      for (int j = 0; j < num_total_joints_; j++) {
        joint_state_csv_file_ << "," << d->qvel[j];
      }
    }
    
    // 写入电机输出力矩
    for (int j = 0; j < num_total_joints_; j++) {
      joint_state_csv_file_ << "," << d->actuator_force[j];
    }
    
    joint_state_csv_file_ << "\n";
    
    // 每N条记录flush一次
    static int flush_counter = 0;
    flush_counter++;
    if (flush_counter >= 100) {
      joint_state_csv_file_.flush();
      flush_counter = 0;
    }
  }
}

}  // namespace mujoco
