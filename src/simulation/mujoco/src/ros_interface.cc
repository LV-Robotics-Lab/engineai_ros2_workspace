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

#include "config_loader.h"
#include "rclcpp/rclcpp.hpp"

// Constants
const int kDofFloatingBase = 6;        // Number of DoF for floating base
const int kNumFloatingBaseJoints = 7;  // Number of joints for floating base (quaternion + xyz)
const int kDimQuaternion = 4;          // Dimension of a quaternion

namespace mujoco {

RosInterface::RosInterface(const std::shared_ptr<rclcpp::Node>& node, std::shared_ptr<ConfigLoader> config_loader)
    : node_(node), config_loader_(config_loader), model_(nullptr), data_(nullptr), is_floating_base_(false) {}

RosInterface::~RosInterface() {
  // 关闭CSV文件
  if (csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Contact data saved to: %s", csv_file_path_.c_str());
  }
}

bool RosInterface::Initialize() {
  // 读取接触力导出参数
  node_->declare_parameter("export_contact", false);
  node_->declare_parameter("contact_topic", "/mujoco/contact_forces");
  
  export_contact_ = node_->get_parameter("export_contact").as_bool();
  contact_topic_ = node_->get_parameter("contact_topic").as_string();

  // 读取CSV保存参数
  node_->declare_parameter("save_contact_csv", false);
  node_->declare_parameter("csv_file_path", "");
  node_->declare_parameter("csv_save_frequency", 1);  // 每帧都保存
  
  save_contact_csv_ = node_->get_parameter("save_contact_csv").as_bool();
  csv_file_path_ = node_->get_parameter("csv_file_path").as_string();
  csv_save_frequency_ = node_->get_parameter("csv_save_frequency").as_int();

  // 如果启用CSV保存但没有指定路径，使用默认路径
  if (save_contact_csv_ && csv_file_path_.empty()) {
    // 创建logs目录（如果不存在）
    std::filesystem::create_directories("logs");
    
    // 生成带时间戳的文件名
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "logs/contact_data_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
    csv_file_path_ = ss.str();
  }

  // 初始化CSV文件
  if (save_contact_csv_) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    csv_file_.open(csv_file_path_, std::ios::out);
    if (csv_file_.is_open()) {
      // 写入CSV头部
      csv_file_ << "timestamp,contact_id,geom1_name,geom2_name,pos_x,pos_y,pos_z,force_x,force_y,force_z,torque_x,torque_y,torque_z,gap,body1_id,body2_id\n";
      csv_file_.flush();
      RCLCPP_INFO(node_->get_logger(), "Contact data will be saved to: %s", csv_file_path_.c_str());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open CSV file: %s", csv_file_path_.c_str());
      save_contact_csv_ = false;
    }
  }

  // Create publishers
  joint_state_pub_ =
      node_->create_publisher<interface_protocol::msg::JointState>(config_loader_->GetJointStateTopic(), 10);

  imu_pub_ = node_->create_publisher<interface_protocol::msg::ImuInfo>(config_loader_->GetImuTopic(), 10);
  
  // Create publisher for motion state
  motion_state_pub_ = node_->create_publisher<interface_protocol::msg::MotionState>("/motion/motion_state", 10);

  // Create publisher for contact forces if enabled
  if (export_contact_) {
    contact_force_pub_ = node_->create_publisher<interface_protocol::msg::ContactForce>(contact_topic_, 10);
    RCLCPP_INFO(node_->get_logger(), "Contact force publishing enabled on topic: %s", contact_topic_.c_str());
  }

  // Create subscriber with more compatible QoS settings
  using std::placeholders::_1;

  // 创建更兼容的QoS设置
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

  joint_cmd_sub_ = node_->create_subscription<interface_protocol::msg::JointCommand>(
      config_loader_->GetJointCommandTopic(), qos, std::bind(&RosInterface::JointCommandCallback, this, _1));

  // Get number of joints from config loader
  num_total_joints_ = config_loader_->GetNumTotalJoints();

  // Initialize commanded values with zeros
  joint_command_.position.resize(num_total_joints_, 0.0);
  joint_command_.velocity.resize(num_total_joints_, 0.0);
  joint_command_.torque.resize(num_total_joints_, 0.0);
  joint_command_.feed_forward_torque.resize(num_total_joints_, 0.0);
  joint_command_.stiffness.resize(num_total_joints_, 0.0);
  joint_command_.damping.resize(num_total_joints_, 0.0);

  // Create timer for publishing motion state every 1 second
  motion_state_timer_ = node_->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&RosInterface::MotionStateTimerCallback, this));

  RCLCPP_INFO(node_->get_logger(), "MuJoCo ROS interface initialized successfully");
  return true;
}

interface_protocol::msg::JointCommand RosInterface::GetCommandedSafe() {
  std::lock_guard<std::mutex> lock(mtx_);
  return joint_command_;
}

void RosInterface::JointCommandCallback(const interface_protocol::msg::JointCommand::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Update commanded values
  joint_command_ = *msg;

  // Ensure all vectors are properly sized
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

void RosInterface::UpdateSimState(const mjModel* m, mjData* d) {
  is_floating_base_ = (m->nv != m->nu);

  // Create messages
  auto joint_state_msg = std::make_unique<interface_protocol::msg::JointState>();
  auto imu_msg = std::make_unique<interface_protocol::msg::ImuInfo>();

  // Set timestamp
  joint_state_msg->header.stamp = node_->now();
  imu_msg->header.stamp = node_->now();

  // Set joint states
  joint_state_msg->name.resize(num_total_joints_);
  joint_state_msg->position.resize(num_total_joints_);
  joint_state_msg->velocity.resize(num_total_joints_);
  joint_state_msg->torque.resize(num_total_joints_);

  if (is_floating_base_) {
    // Skip the floating base joints
    for (int i = 0; i < num_total_joints_; ++i) {
      // Get joint name from MuJoCo model
      joint_state_msg->name[i] = m->names + m->name_jntadr[i + kNumFloatingBaseJoints];
      joint_state_msg->position[i] = d->qpos[i + kNumFloatingBaseJoints];
      joint_state_msg->velocity[i] = d->qvel[i + kDofFloatingBase];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  } else {
    for (int i = 0; i < num_total_joints_; ++i) {
      // Get joint name from MuJoCo model
      joint_state_msg->name[i] = m->names + m->name_jntadr[i];
      joint_state_msg->position[i] = d->qpos[i];
      joint_state_msg->velocity[i] = d->qvel[i];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  }

  // IMU data typically comes from sensors in MuJoCo
  int index = 0;

  // Set IMU quaternion
  imu_msg->quaternion.w = d->sensordata[index + 0];
  imu_msg->quaternion.x = d->sensordata[index + 1];
  imu_msg->quaternion.y = d->sensordata[index + 2];
  imu_msg->quaternion.z = d->sensordata[index + 3];
  index += kDimQuaternion;

  // Set RPY values from the sensor data
  // Assuming the RPY values are the next three values after the quaternion
  imu_msg->rpy.x = d->sensordata[index + 0];  // Roll
  imu_msg->rpy.y = d->sensordata[index + 1];  // Pitch
  imu_msg->rpy.z = d->sensordata[index + 2];  // Yaw
  index += 3;

  // Linear acceleration
  imu_msg->linear_acceleration.x = d->sensordata[index + 0];
  imu_msg->linear_acceleration.y = d->sensordata[index + 1];
  imu_msg->linear_acceleration.z = d->sensordata[index + 2];
  index += 3;

  // Angular velocity
  imu_msg->angular_velocity.x = d->sensordata[index + 0];
  imu_msg->angular_velocity.y = d->sensordata[index + 1];
  imu_msg->angular_velocity.z = d->sensordata[index + 2];

  // Publish messages
  joint_state_pub_->publish(std::move(joint_state_msg));
  imu_pub_->publish(std::move(imu_msg));

  // Publish contact forces if enabled
  if (export_contact_) {
    PublishContactForces(m, d);
  }
}

void RosInterface::PublishContactForces(const mjModel* m, mjData* d) {
  if (!contact_force_pub_) {
    return;
  }

  auto contact_msg = std::make_unique<interface_protocol::msg::ContactForce>();
  contact_msg->header.stamp = node_->now();
  contact_msg->header.frame_id = "world";

  // Get number of contacts
  int ncon = d->ncon;
  
  // 添加调试信息，每100帧打印一次
  static int frame_count = 0;
  frame_count++;
  if (frame_count % 100 == 0) {
    RCLCPP_INFO(node_->get_logger(), "Current contacts: %d", ncon);
  }
  
  if (ncon == 0) {
    // No contacts, publish empty message
    contact_force_pub_->publish(std::move(contact_msg));
    return;
  }

  // Resize vectors to hold contact data
  contact_msg->contact_names.resize(ncon);
  contact_msg->contact_positions_x.resize(ncon);
  contact_msg->contact_positions_y.resize(ncon);
  contact_msg->contact_positions_z.resize(ncon);
  contact_msg->contact_forces_x.resize(ncon);
  contact_msg->contact_forces_y.resize(ncon);
  contact_msg->contact_forces_z.resize(ncon);
  contact_msg->contact_torques_x.resize(ncon);
  contact_msg->contact_torques_y.resize(ncon);
  contact_msg->contact_torques_z.resize(ncon);
  contact_msg->contact_gaps.resize(ncon);
  contact_msg->contact_bodies_1.resize(ncon);
  contact_msg->contact_bodies_2.resize(ncon);

  // Fill contact data
  for (int i = 0; i < ncon; ++i) {
    const mjContact& contact = d->contact[i];
    
    // Contact name (using geom names)
    std::string geom1_name = m->names + m->name_geomadr[contact.geom[0]];
    std::string geom2_name = m->names + m->name_geomadr[contact.geom[1]];
    contact_msg->contact_names[i] = geom1_name + "_" + geom2_name;

    // 改进的接触点位置计算
    mjtNum pos[3];
    
    // 获取接触的两个几何体
    int geom1 = contact.geom[0];
    int geom2 = contact.geom[1];
    
    // 获取接触点的法向量和距离
    mjtNum normal[3] = {contact.frame[0], contact.frame[1], contact.frame[2]};
    mjtNum dist = contact.dist;
    
    // 使用MuJoCo内部计算的接触点位置（通常最准确）
    // contact.pos 是MuJoCo内部计算的接触点，考虑了几何体的实际形状
    pos[0] = contact.pos[0];
    pos[1] = contact.pos[1];
    pos[2] = contact.pos[2];
    
    // 使用配置文件中的位置偏移量
    double position_offset = config_loader_->GetContactPositionOffset();
    if (position_offset != 0.0) {
      pos[0] += normal[0] * position_offset;
      pos[1] += normal[1] * position_offset;
      pos[2] += normal[2] * position_offset;
    }
    
    contact_msg->contact_positions_x[i] = pos[0];
    contact_msg->contact_positions_y[i] = pos[1];
    contact_msg->contact_positions_z[i] = pos[2];

    // Contact force - 从efc_force获取（如果接触被包含在约束中）
    if (contact.efc_address >= 0 && contact.efc_address < d->nefc) {
      // 法向力在efc_force中
      contact_msg->contact_forces_x[i] = contact.frame[0] * d->efc_force[contact.efc_address];
      contact_msg->contact_forces_y[i] = contact.frame[1] * d->efc_force[contact.efc_address];
      contact_msg->contact_forces_z[i] = contact.frame[2] * d->efc_force[contact.efc_address];
    } else {
      // 接触未被包含在约束中，设置为0
      contact_msg->contact_forces_x[i] = 0.0;
      contact_msg->contact_forces_y[i] = 0.0;
      contact_msg->contact_forces_z[i] = 0.0;
    }

    // Contact torque - 暂时设置为0，因为需要更复杂的计算
    contact_msg->contact_torques_x[i] = 0.0;
    contact_msg->contact_torques_y[i] = 0.0;
    contact_msg->contact_torques_z[i] = 0.0;

    // Contact gap (distance)
    contact_msg->contact_gaps[i] = contact.dist;

    // Contact body indices
    contact_msg->contact_bodies_1[i] = m->geom_bodyid[contact.geom[0]];
    contact_msg->contact_bodies_2[i] = m->geom_bodyid[contact.geom[1]];
  }

  // Publish contact force message
  contact_force_pub_->publish(std::move(contact_msg));

  // 保存到CSV文件
  if (save_contact_csv_ && csv_file_.is_open()) {
    // 使用帧计数器控制保存频率
    static int frame_counter = 0;
    frame_counter++;
    
    // 根据配置的频率决定是否保存
    if (frame_counter % csv_save_frequency_ == 0) {
      std::lock_guard<std::mutex> lock(csv_mutex_);
      
      // 使用MuJoCo仿真时间而不是ROS时间
      double sim_time = d->time;
      
      // 为每个接触点写入一行数据
      for (int i = 0; i < ncon; ++i) {
        const mjContact& contact = d->contact[i];
        
        // 获取几何体名称
        std::string geom1_name = m->names + m->name_geomadr[contact.geom[0]];
        std::string geom2_name = m->names + m->name_geomadr[contact.geom[1]];
        
        // 计算接触点位置（与发布消息中相同的逻辑）
        mjtNum pos[3];
        mjtNum normal[3] = {contact.frame[0], contact.frame[1], contact.frame[2]};
        double position_offset = config_loader_->GetContactPositionOffset();
        
        pos[0] = contact.pos[0] + normal[0] * position_offset;
        pos[1] = contact.pos[1] + normal[1] * position_offset;
        pos[2] = contact.pos[2] + normal[2] * position_offset;
        
        // 计算接触力
        double force_x = 0.0, force_y = 0.0, force_z = 0.0;
        if (contact.efc_address >= 0 && contact.efc_address < d->nefc) {
          force_x = contact.frame[0] * d->efc_force[contact.efc_address];
          force_y = contact.frame[1] * d->efc_force[contact.efc_address];
          force_z = contact.frame[2] * d->efc_force[contact.efc_address];
        }
        
        // 写入CSV行
        csv_file_ << std::fixed << std::setprecision(6)
                  << sim_time << ","
                  << i << ","
                  << "\"" << geom1_name << "\","
                  << "\"" << geom2_name << "\","
                  << pos[0] << ","
                  << pos[1] << ","
                  << pos[2] << ","
                  << force_x << ","
                  << force_y << ","
                  << force_z << ","
                  << "0.0,0.0,0.0,"  // 扭矩暂时设为0
                  << contact.dist << ","
                  << m->geom_bodyid[contact.geom[0]] << ","
                  << m->geom_bodyid[contact.geom[1]] << "\n";
      }
      
      // 每帧都刷新文件缓冲区（10kHz频率）
      csv_file_.flush();
    }
  }
}

void RosInterface::SetModelAndData(mjModel* model, mjData* data) {
  model_ = model;
  data_ = data;
}

void RosInterface::MotionStateTimerCallback() {
  // Create a motion state message
  auto motion_state_msg = std::make_unique<interface_protocol::msg::MotionState>();
  
  // Set the current_motion_task field to "joint_bridge"
  motion_state_msg->current_motion_task = "joint_bridge";
  
  // Publish the message
  motion_state_pub_->publish(std::move(motion_state_msg));
}

}  // namespace mujoco
