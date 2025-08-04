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
      // 获取关节数量
      num_total_joints_ = config_loader_->GetNumTotalJoints();
      
      // 写入CSV头部
      csv_file_ << "timestamp,contact_id,geom1_name,geom2_name,pos_x,pos_y,pos_z,robot_frame_x,robot_frame_y,robot_frame_z,force_x,force_y,force_z,force_magnitude,torque_x,torque_y,torque_z,base_link_x,base_link_y,base_link_z,base_link_qw,base_link_qx,base_link_qy,base_link_qz";
      
      // 添加关节角度列
      for (int i = 0; i < num_total_joints_; i++) {
        csv_file_ << ",joint_" << i << "_angle";
      }
      csv_file_ << "\n";
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
  // num_total_joints_ = config_loader_->GetNumTotalJoints(); // This line is now moved to Initialize()

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
  
  // 添加调试信息，每1000帧打印一次（在10kHz频率下约每0.1秒一次）
  static int frame_count = 0;
  frame_count++;
  if (frame_count % 1000 == 0) {
    RCLCPP_INFO(node_->get_logger(), "Current contacts: %d", ncon);
    
    // 统计所有接触的力大小
    double total_force_magnitude = 0.0;
    double max_force_magnitude = 0.0;
    int contacts_with_force = 0;
    
    // 计算所有接触的合力
    mjtNum total_world_force[3] = {0.0, 0.0, 0.0};
    mjtNum total_world_torque[3] = {0.0, 0.0, 0.0};
    
    for (int i = 0; i < ncon; ++i) {
      const mjContact& contact = d->contact[i];
      mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      mj_contactForce(m, d, i, contact_force);
      
      // 转换到世界坐标系
      mjtNum world_force[3] = {0.0, 0.0, 0.0};
      mjtNum world_torque[3] = {0.0, 0.0, 0.0};
      
      for (int j = 0; j < 3; j++) {
        world_force[j] = contact.frame[j*3 + 0] * contact_force[0] + 
                         contact.frame[j*3 + 1] * contact_force[1] + 
                         contact.frame[j*3 + 2] * contact_force[2];
        
        world_torque[j] = contact.frame[j*3 + 0] * contact_force[3] + 
                          contact.frame[j*3 + 1] * contact_force[4] + 
                          contact.frame[j*3 + 2] * contact_force[5];
      }
      
      // 累加到合力
      for (int j = 0; j < 3; j++) {
        total_world_force[j] += world_force[j];
        total_world_torque[j] += world_torque[j];
      }
      
      double force_magnitude = sqrt(contact_force[0]*contact_force[0] + 
                                   contact_force[1]*contact_force[1] + 
                                   contact_force[2]*contact_force[2]);
      
      if (force_magnitude > 0.1) {  // 只统计有意义的力
        contacts_with_force++;
        total_force_magnitude += force_magnitude;
        if (force_magnitude > max_force_magnitude) {
          max_force_magnitude = force_magnitude;
        }
      }
    }
    
    double total_force_magnitude_world = sqrt(total_world_force[0]*total_world_force[0] + 
                                             total_world_force[1]*total_world_force[1] + 
                                             total_world_force[2]*total_world_force[2]);
    
    RCLCPP_INFO(node_->get_logger(), 
                "Force stats: total=%.3f, max=%.3f, contacts_with_force=%d/%d, total_world_force=(%.3f,%.3f,%.3f), magnitude=%.3f", 
                total_force_magnitude, max_force_magnitude, contacts_with_force, ncon,
                total_world_force[0], total_world_force[1], total_world_force[2], total_force_magnitude_world);
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

  // 计算所有接触的合力
  mjtNum total_world_force[3] = {0.0, 0.0, 0.0};
  mjtNum total_world_torque[3] = {0.0, 0.0, 0.0};

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

    // 使用mj_contactForce获取完整的6D接触力
    mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mj_contactForce(m, d, i, contact_force);
    
    // 将接触坐标系下的力转换为世界坐标系
    // contact_force[0:2] 是力，contact_force[3:5] 是力矩
    mjtNum world_force[3] = {0.0, 0.0, 0.0};
    mjtNum world_torque[3] = {0.0, 0.0, 0.0};
    
    // 正确的坐标系转换
    // contact.frame 在MuJoCo中存储为转置形式，轴在行中
    // frame[0-2]: X轴 (法向量)
    // frame[3-5]: Y轴 (第一个切向量)  
    // frame[6-8]: Z轴 (第二个切向量)
    for (int j = 0; j < 3; j++) {
      // 转换力：world_force = frame^T * contact_force
      world_force[j] = contact.frame[j*3 + 0] * contact_force[0] + 
                       contact.frame[j*3 + 1] * contact_force[1] + 
                       contact.frame[j*3 + 2] * contact_force[2];
      
      // 转换力矩：world_torque = frame^T * contact_torque
      world_torque[j] = contact.frame[j*3 + 0] * contact_force[3] + 
                        contact.frame[j*3 + 1] * contact_force[4] + 
                        contact.frame[j*3 + 2] * contact_force[5];
    }
    
    // 累加到合力
    for (int j = 0; j < 3; j++) {
      total_world_force[j] += world_force[j];
      total_world_torque[j] += world_torque[j];
    }
    
    contact_msg->contact_forces_x[i] = world_force[0];
    contact_msg->contact_forces_y[i] = world_force[1];
    contact_msg->contact_forces_z[i] = world_force[2];
    
    contact_msg->contact_torques_x[i] = world_torque[0];
    contact_msg->contact_torques_y[i] = world_torque[1];
    contact_msg->contact_torques_z[i] = world_torque[2];

    // Contact gap (distance)
    contact_msg->contact_gaps[i] = contact.dist;

    // Contact body indices
    contact_msg->contact_bodies_1[i] = m->geom_bodyid[contact.geom[0]];
    contact_msg->contact_bodies_2[i] = m->geom_bodyid[contact.geom[1]];
  }

  // 添加合力信息到消息中（如果有接触的话）
  if (ncon > 0) {
    // 可以在这里添加合力字段，如果消息类型支持的话
    // 或者通过其他方式发布合力信息
    RCLCPP_INFO(node_->get_logger(), 
                "Total contact force: (%.3f, %.3f, %.3f), magnitude: %.3f", 
                total_world_force[0], total_world_force[1], total_world_force[2],
                sqrt(total_world_force[0]*total_world_force[0] + 
                     total_world_force[1]*total_world_force[1] + 
                     total_world_force[2]*total_world_force[2]));
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
        
        // 使用mj_contactForce获取完整的6D接触力
        mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        mj_contactForce(m, d, i, contact_force);
        
        // 将接触坐标系下的力转换为世界坐标系
        mjtNum world_force[3] = {0.0, 0.0, 0.0};
        mjtNum world_torque[3] = {0.0, 0.0, 0.0};
        
        // 正确的坐标系转换
        // contact.frame 在MuJoCo中存储为转置形式，轴在行中
        // frame[0-2]: X轴 (法向量)
        // frame[3-5]: Y轴 (第一个切向量)  
        // frame[6-8]: Z轴 (第二个切向量)
        for (int j = 0; j < 3; j++) {
          // 转换力：world_force = frame^T * contact_force
          world_force[j] = contact.frame[j*3 + 0] * contact_force[0] + 
                           contact.frame[j*3 + 1] * contact_force[1] + 
                           contact.frame[j*3 + 2] * contact_force[2];
          
          // 转换力矩：world_torque = frame^T * contact_torque
          world_torque[j] = contact.frame[j*3 + 0] * contact_force[3] + 
                            contact.frame[j*3 + 1] * contact_force[4] + 
                            contact.frame[j*3 + 2] * contact_force[5];
        }
        
        // 计算单个接触点的合力大小
        double contact_force_magnitude = sqrt(world_force[0]*world_force[0] + 
                                             world_force[1]*world_force[1] + 
                                             world_force[2]*world_force[2]);
        
        // 获取body的pose信息并计算URDF坐标系中的位置
        int body1_id = m->geom_bodyid[contact.geom[0]];
        int body2_id = m->geom_bodyid[contact.geom[1]];
        
        // 找到base_link对应的body ID（LINK_BASE）
        int base_link_id = -1;
        for (int i = 0; i < m->nbody; i++) {
          if (std::string(m->names + m->name_bodyadr[i]) == "LINK_BASE") {
            base_link_id = i;
            break;
          }
        }
        
        // 获取base_link的pose（机器人基座的世界坐标位置）
        mjtNum base_link_pos[3] = {0, 0, 0};
        mjtNum base_link_quat[4] = {1, 0, 0, 0}; // 默认单位四元数
        if (base_link_id >= 0) {
          base_link_pos[0] = d->xpos[base_link_id*3];
          base_link_pos[1] = d->xpos[base_link_id*3+1];
          base_link_pos[2] = d->xpos[base_link_id*3+2];
          base_link_quat[0] = d->xquat[base_link_id*4];
          base_link_quat[1] = d->xquat[base_link_id*4+1];
          base_link_quat[2] = d->xquat[base_link_id*4+2];
          base_link_quat[3] = d->xquat[base_link_id*4+3];
        }
        
        // 计算碰撞点相对于base_link的位置（机器人坐标系）
        mjtNum robot_frame_pos[3];
        // 计算从base_link到接触点的相对位置（世界坐标系）
        mjtNum relative_to_base[3] = {pos[0] - base_link_pos[0], pos[1] - base_link_pos[1], pos[2] - base_link_pos[2]};
        // 将相对位置从世界坐标系转换到base_link的局部坐标系（机器人坐标系）
        mjtNum base_link_quat_conj[4];
        mju_negQuat(base_link_quat_conj, base_link_quat);
        mju_rotVecQuat(robot_frame_pos, relative_to_base, base_link_quat_conj);
        
        // 写入CSV行
        csv_file_ << std::fixed << std::setprecision(6)
                  << sim_time << ","
                  << i << ","
                  << "\"" << geom1_name << "\","
                  << "\"" << geom2_name << "\","
                  << pos[0] << ","
                  << pos[1] << ","
                  << pos[2] << ","
                  << robot_frame_pos[0] << ","  // 机器人坐标系中的位置
                  << robot_frame_pos[1] << ","
                  << robot_frame_pos[2] << ","
                  << world_force[0] << ","
                  << world_force[1] << ","
                  << world_force[2] << ","
                  << contact_force_magnitude << ","  // 单个接触点合力大小
                  << world_torque[0] << ","
                  << world_torque[1] << ","
                  << world_torque[2] << ","
                  << base_link_pos[0] << ","
                  << base_link_pos[1] << ","
                  << base_link_pos[2] << ","
                  << base_link_quat[0] << ","
                  << base_link_quat[1] << ","
                  << base_link_quat[2] << ","
                  << base_link_quat[3];

        // 添加关节角度列
        for (int i = 0; i < num_total_joints_; i++) {
          int joint_idx = i;
          if (is_floating_base_) {
            // 如果有浮动基座，跳过前6个自由度（3个位置+3个姿态）
            joint_idx = i + 6;
          }
          csv_file_ << "," << d->qpos[joint_idx];
        }
        csv_file_ << "\n";
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
