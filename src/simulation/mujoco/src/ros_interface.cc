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

#include "config_loader.h"
#include "rclcpp/rclcpp.hpp"
// #include <mujoco/mujoco.h>


// 常量定义
const int kDofFloatingBase = 6;        // 浮动基座的自由度数量
const int kNumFloatingBaseJoints = 7;  // 浮动基座的关节数量（四元数 + xyz位置）
const int kDimQuaternion = 4;          // 四元数的维度

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
  // 关闭CSV文件
  if (csv_file_.is_open()) {
    std::lock_guard<std::mutex> lock(csv_mutex_);
    csv_file_.close();
    RCLCPP_INFO(node_->get_logger(), "Contact data saved to: %s", csv_file_path_.c_str());
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

  // 从配置加载器获取关节数量
  // num_total_joints_ = config_loader_->GetNumTotalJoints(); // 这行现在移到Initialize()中

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
  PublishContacts(m, d);

  // 如果启用，发布接触力
  if (export_contact_) {
    PublishContactForces(m, d);
  }
}

/**
 * @brief 发布接触力数据
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 计算并发布所有接触点的力和力矩信息，同时保存到CSV文件
 */
void RosInterface::PublishContactForces(const mjModel* m, mjData* d) {
  if (!contact_force_pub_) {
    return;
  }

  auto contact_msg = std::make_unique<interface_protocol::msg::ContactForce>();
  contact_msg->header.stamp = node_->now();
  contact_msg->header.frame_id = "world";

  // 获取接触数量
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
    // 没有接触，发布空消息
    contact_force_pub_->publish(std::move(contact_msg));
    return;
  }

  // 调整向量大小以容纳接触数据
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

  // 填充接触数据
  for (int i = 0; i < ncon; ++i) {
    const mjContact& contact = d->contact[i];
    
    // 接触名称（使用几何体名称）
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

    // 接触间隙（距离）
    contact_msg->contact_gaps[i] = contact.dist;

    // 接触体索引
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

  // 发布接触力消息
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

/**
 * @brief 设置MuJoCo模型和数据指针
 * @param model MuJoCo模型指针
 * @param data MuJoCo数据指针
 */


void RosInterface::SetModelAndData(mjModel* model, mjData* data) {
  model_ = model;
  data_ = data;
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

void RosInterface::PublishContacts(const mjModel* m, const mjData* d) {
  if (!contact_pub_) return;

  std_msgs::msg::Float32MultiArray msg;
  msg.layout.dim.resize(2);
  msg.layout.dim[0].label = "contacts";
  msg.layout.dim[0].size = d->ncon;
  msg.layout.dim[0].stride = 21; //14
  msg.layout.dim[1].label = "fields";
  msg.layout.dim[1].size = 21; //14
  msg.layout.dim[1].stride = 1;
  msg.data.reserve(static_cast<size_t>(d->ncon) * 21); //14

  for (int i = 0; i < d->ncon; ++i) {
    const mjContact& con = d->contact[i];

    double w[6] = {0};
    mj_contactForce(m, d, i, w);  // [fx,fy,fz, tx,ty,tz] in contact frame

    // contact frame to world frame
    double R[9];
    if (con.dim > 0) std::memcpy(R, con.frame, sizeof(R));
    else { R[0]=1;R[1]=0;R[2]=0; R[3]=0;R[4]=1;R[5]=0; R[6]=0;R[7]=0;R[8]=1; }

    auto rot3 = [](const double* M, const double* v, double* o){
      o[0]=M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
      o[1]=M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
      o[2]=M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
    };

    double f_c[3] = {w[0], w[1], w[2]};
    double t_c[3] = {w[3], w[4], w[5]};
    double f_w[3], t_w[3];
    rot3(R, f_c, f_w);
    rot3(R, t_c, t_w);

    const double f_mag  = std::sqrt(f_w[0]*f_w[0] + f_w[1]*f_w[1] + f_w[2]*f_w[2]);
    const double f_norm = std::max(0.0, f_c[0]);  // 接触系 x 为法向

    /// trans to link frame
    auto is_robot_geom = [&](int geom_id)->bool {
      int b = m->geom_bodyid[geom_id];
      return b != 0;   // 简单过滤：0通常是world，你可改成白名单
    };
    int robot_geom = is_robot_geom(con.geom1) ? con.geom1 :
                     (is_robot_geom(con.geom2) ? con.geom2 : -1);

    int body = -1;
    if (robot_geom >= 0) {
      body = m->geom_bodyid[robot_geom];
    }

    // 如果找到了机器人body，算局部坐标
    double p_B[3] = {0}, f_B[3] = {0};
    if (body >= 0) {
      const mjtNum* x_BW = d->xpos + 3*body;   // body原点世界坐标
      const mjtNum* R_BW = d->xmat + 9*body;   // body旋转矩阵 (row-major)

      auto rotT3 = [](const double* M, const double* v, double* o){
        // R^T v
        o[0]=M[0]*v[0]+M[3]*v[1]+M[6]*v[2];
        o[1]=M[1]*v[0]+M[4]*v[1]+M[7]*v[2];
        o[2]=M[2]*v[0]+M[5]*v[1]+M[8]*v[2];
      };

      // 世界点 -> 局部点
      double pW_minus_x[3] = {con.pos[0]-x_BW[0],
                              con.pos[1]-x_BW[1],
                              con.pos[2]-x_BW[2]};
      rotT3(R_BW, pW_minus_x, p_B);

      // 世界力 -> 局部力
      rotT3(R_BW, f_w, f_B);
    }


    msg.data.push_back(static_cast<float>(con.geom1));
    msg.data.push_back(static_cast<float>(con.geom2));
    msg.data.push_back(static_cast<float>(con.pos[0]));
    msg.data.push_back(static_cast<float>(con.pos[1]));
    msg.data.push_back(static_cast<float>(con.pos[2]));
    msg.data.push_back(static_cast<float>(f_w[0]));
    msg.data.push_back(static_cast<float>(f_w[1]));
    msg.data.push_back(static_cast<float>(f_w[2]));
    msg.data.push_back(static_cast<float>(t_w[0]));
    msg.data.push_back(static_cast<float>(t_w[1]));
    msg.data.push_back(static_cast<float>(t_w[2]));
    msg.data.push_back(static_cast<float>(f_norm));
    msg.data.push_back(static_cast<float>(f_mag));

    // 新增7个: body_id, p_B(3), f_B(3)
    msg.data.push_back(static_cast<float>(body));
    msg.data.push_back(static_cast<float>(p_B[0]));
    msg.data.push_back(static_cast<float>(p_B[1]));
    msg.data.push_back(static_cast<float>(p_B[2]));
    msg.data.push_back(static_cast<float>(f_B[0]));
    msg.data.push_back(static_cast<float>(f_B[1]));
    msg.data.push_back(static_cast<float>(f_B[2]));
  }

  contact_pub_->publish(msg);

  //visuliazation
  if (!contact_marker_pub_) return;

  visualization_msgs::msg::MarkerArray arr;

  // 先发一个"清空"指令，避免历史残留
  {
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
  }

  // 可选：根据力大小缩放球的尺寸
  const double base_diameter = 0.03;   // 3 cm 基础直径
  const double max_diameter  = 0.12;   // 12 cm 上限
  const double scale_gain    = 1.0/200.0; // 200N -> +1 倍尺寸（按需调）

  // 修复：将dstd的创建和释放移到函数外部，避免内存泄漏
  // 使用静态变量缓存标准姿态数据，避免重复计算
  static std::vector<mjtNum> std_xpos_cache;
  static std::vector<mjtNum> std_xmat_cache;
  static int last_model_id = -1;
  
  // 检查是否需要重新计算标准姿态
  // 使用模型地址作为唯一标识符，因为mjModel没有id字段
  static const mjModel* last_model_ptr = nullptr;
  if (last_model_ptr != m) {
    // 重新计算标准姿态
    std_xpos_cache.clear();
    std_xmat_cache.clear();
    
    // 创建临时mjData对象
    mjData* dstd = mj_makeData(m);
    if (!dstd) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create temporary mjData for standard pose");
      return;
    }
    
    // 设置标准姿态
    int key_id = mj_name2id(m, mjOBJ_KEY, "floating_base_homing");
    if (key_id >= 0) {
      mj_resetDataKeyframe(m, dstd, key_id);
    } else {
      // 兜底：零位
      mju_zero(dstd->qpos, m->nq);
      if (m->jnt_type[0] == mjJNT_FREE) {
        dstd->qpos[0] = 1; dstd->qpos[1] = dstd->qpos[2] = dstd->qpos[3] = 0;
        dstd->qpos[4] = dstd->qpos[5] = dstd->qpos[6] = 0;
      }
    }
    
    mj_forward(m, dstd);
    
    // 缓存数据
    std_xpos_cache.assign(dstd->xpos, dstd->xpos + 3*m->nbody);
    std_xmat_cache.assign(dstd->xmat, dstd->xmat + 9*m->nbody);
    
    // 正确释放mjData对象
    mj_deleteData(dstd);
    
    last_model_ptr = m;
  }
  
  const mjtNum* std_xpos = std_xpos_cache.data();
  const mjtNum* std_xmat = std_xmat_cache.data();

  for (int i = 0; i < d->ncon; ++i) {
  const mjContact& con = d->contact[i];

  // 取 6D 接触力
  double w[6] = {0};
  mj_contactForce(m, d, i, w);

  // contact -> world
  double R[9];
  if (con.dim > 0) std::memcpy(R, con.frame, sizeof(R));
  else { R[0]=1;R[1]=0;R[2]=0; R[3]=0;R[4]=1;R[5]=0; R[6]=0;R[7]=0;R[8]=1; }

  auto rot3 = [](const double* M, const double* v, double* o){
    o[0]=M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
    o[1]=M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
    o[2]=M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
  };
  double f_c[3] = {w[0], w[1], w[2]}, f_w[3];
  rot3(R, f_c, f_w);
  const double f_mag = std::sqrt(f_w[0]*f_w[0] + f_w[1]*f_w[1] + f_w[2]*f_w[2]);

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

  mkr_red.pose.position.x = con.pos[0];
  mkr_red.pose.position.y = con.pos[1];
  mkr_red.pose.position.z = con.pos[2];
  mkr_red.pose.orientation.w = 1.0;

  double dia = base_diameter * (1.0 + std::clamp(f_mag*scale_gain, 0.0, 3.0));
  dia = std::min(dia, max_diameter);
  mkr_red.scale.x = mkr_red.scale.y = mkr_red.scale.z = dia;

  mkr_red.color.r = 1.0; mkr_red.color.g = 0.0; mkr_red.color.b = 0.0; mkr_red.color.a = 0.9;
  mkr_red.lifetime = rclcpp::Duration::from_seconds(0.1);

  arr.markers.push_back(mkr_red);

  // -------------------
  // 绿球：link局部系下的接触点
  // -------------------
  // 先找机器人 body
  int body = m->geom_bodyid[con.geom1];
  if (body == 0) body = m->geom_bodyid[con.geom2]; // 如果第一个是 world

  if (body > 0 && body < m->nbody) {  // 添加边界检查
    // 当前姿态：世界 -> body 局部
    const mjtNum* x_BW = d->xpos + 3*body;   // 当前 body 原点（世界）
    const mjtNum* R_BW = d->xmat + 9*body;   // 当前 body 旋转（行存 3x3）

    auto rotT3 = [](const double* M, const double* v, double* o){
      // o = R^T v
      o[0]=M[0]*v[0]+M[3]*v[1]+M[6]*v[2];
      o[1]=M[1]*v[0]+M[4]*v[1]+M[7]*v[2];
      o[2]=M[2]*v[0]+M[5]*v[1]+M[8]*v[2];
    };
    auto rot3 = [](const double* M, const double* v, double* o){
      // o = R v
      o[0]=M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
      o[1]=M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
      o[2]=M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
    };

    // 当前姿态下接触点 → body 局部 p_B
    double pW_minus_x[3] = {con.pos[0]-x_BW[0],
                            con.pos[1]-x_BW[1],
                            con.pos[2]-x_BW[2]};
    double p_B[3];
    rotT3(R_BW, pW_minus_x, p_B);

    // 【标准姿态】：body 局部 → 世界 p_W_std
    const mjtNum* x_BW_std = std_xpos + 3*body;   // 标准姿态 body 原点（世界）
    const mjtNum* R_BW_std = std_xmat + 9*body;   // 标准姿态 body 旋转
    double p_W_std[3];
    rot3(R_BW_std, p_B, p_W_std);
    p_W_std[0] += x_BW_std[0];
    p_W_std[1] += x_BW_std[1];
    p_W_std[2] += x_BW_std[2];

    visualization_msgs::msg::Marker mkr_green;
    mkr_green.header.frame_id = "world";      // RViz 的 fixed frame（和你系统一致）
    mkr_green.header.stamp    = node_->now();
    mkr_green.ns   = "contact_link_stdpose";
    mkr_green.id   = 10000 + i;               // 避免和红球 id 冲突
    mkr_green.type = visualization_msgs::msg::Marker::SPHERE;
    mkr_green.action = visualization_msgs::msg::Marker::ADD;

    mkr_green.pose.position.x = p_W_std[0];
    mkr_green.pose.position.y = p_W_std[1];
    mkr_green.pose.position.z = p_W_std[2];
    mkr_green.pose.orientation.w = 1.0;

    // 尺寸：复用你算好的 dia（或独立给绿球一个固定尺寸）
    mkr_green.scale.x = mkr_green.scale.y = mkr_green.scale.z = dia;

    // 颜色：绿色
    mkr_green.color.r = 0.0; mkr_green.color.g = 1.0; mkr_green.color.b = 0.0; mkr_green.color.a = 0.9;

    mkr_green.lifetime = rclcpp::Duration::from_seconds(0.1);

    arr.markers.push_back(mkr_green);
  }
}

  // for (int i = 0; i < d->ncon; ++i) {
  //   const mjContact& con = d->contact[i];

  //   // 取 6D 接触力（接触系）
  //   double w[6] = {0};
  //   mj_contactForce(m, d, i, w); // [fx, fy, fz, tx, ty, tz]

  //   // 旋到世界系
  //   double R[9];
  //   if (con.dim > 0) std::memcpy(R, con.frame, sizeof(R));
  //   else { R[0]=1;R[1]=0;R[2]=0; R[3]=0;R[4]=1;R[5]=0; R[6]=0;R[7]=0;R[8]=1; }
  //   auto rot3 = [](const double* M, const double* v, double* o){
  //     o[0]=M[0]*v[0]+M[1]*v[1]+M[2]*v[2];
  //     o[1]=M[3]*v[0]+M[4]*v[1]+M[5]*v[2];
  //     o[2]=M[6]*v[0]+M[7]*v[1]+M[8]*v[2];
  //   };
  //   double f_c[3] = {w[0], w[1], w[2]}, f_w[3];
  //   rot3(R, f_c, f_w);
  //   const double f_mag = std::sqrt(f_w[0]*f_w[0] + f_w[1]*f_w[1] + f_w[2]*f_w[2]);

  //   visualization_msgs::msg::Marker mkr;
  //   mkr.header.frame_id = "world";        // 你的世界系 TF 名称，如有不同请改
  //   mkr.header.stamp    = node_->now();
  //   mkr.ns   = "contact_points";
  //   mkr.id   = i;                          // 每个接触一个 id
  //   mkr.type = visualization_msgs::msg::Marker::SPHERE;
  //   mkr.action = visualization_msgs::msg::Marker::ADD;

  //   // 位置 = 接触点世界坐标
  //   mkr.pose.position.x = con.pos[0];
  //   mkr.pose.position.y = con.pos[1];
  //   mkr.pose.position.z = con.pos[2];
  //   mkr.pose.orientation.w = 1.0; // 球体无旋转

  //   // 尺寸（直径）：基础 + 按力大小放大，夹到上限
  //   double dia = base_diameter * (1.0 + std::clamp(f_mag*scale_gain, 0.0, 3.0));
  //   dia = std::min(dia, max_diameter);
  //   mkr.scale.x = mkr.scale.y = mkr.scale.z = dia;

  //   // 颜色：红色，半透明
  //   mkr.color.r = 1.0; mkr.color.g = 0.0; mkr.color.b = 0.0; mkr.color.a = 0.9;

  //   // 让 marker 有个短寿命，帧对帧自动更新
  //   mkr.lifetime = rclcpp::Duration::from_seconds(0.1);

  //   arr.markers.push_back(mkr);
  // }

  contact_marker_pub_->publish(arr);

}

}  // namespace mujoco
