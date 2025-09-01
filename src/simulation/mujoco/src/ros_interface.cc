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

#include "config_loader.h"
#include "rclcpp/rclcpp.hpp"
#include <mujoco/mujoco.h>


// 常量定义
const int kDofFloatingBase = 6;        // 浮动基座的自由度数量
const int kNumFloatingBaseJoints = 7;  // 浮动基座的关节数量（四元数 + xyz位置）
const int kDimQuaternion = 4;          // 四元数的维度

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
      csv_file_ << "timestamp,contact_id,body1_name,body2_name,pos_x,pos_y,pos_z,robot_frame_x,robot_frame_y,robot_frame_z,force_x,force_y,force_z,force_magnitude,torque_x,torque_y,torque_z,base_link_x,base_link_y,base_link_z,base_link_qw,base_link_qx,base_link_qy,base_link_qz";
      
      // 添加完整的31个参数列名
      if (is_floating_base_) {
        // 浮动基座位置 (3个参数)
        csv_file_ << ",floating_base_x,floating_base_y,floating_base_z";
        // 浮动基座姿态四元数 (4个参数)
        csv_file_ << ",floating_base_qw,floating_base_qx,floating_base_qy,floating_base_qz";
        // 24个关节角度
        for (int i = 0; i < num_total_joints_; i++) {
          csv_file_ << ",joint_" << i << "_angle";
        }
      } else {
        // 非浮动基座机器人，只添加关节角度
        for (int i = 0; i < num_total_joints_; i++) {
          csv_file_ << ",joint_" << i << "_angle";
        }
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
    PublishContactForces(m, d);
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

  // 创建接触力消息并设置时间戳和坐标系
  auto contact_msg = std::make_unique<interface_protocol::msg::ContactForce>();
  contact_msg->header.stamp = node_->now();
  contact_msg->header.frame_id = "world";

  // 获取当前仿真中的接触数量
  int ncon = d->ncon;
  
  // 添加调试信息，每1000帧打印一次（在10kHz频率下约每0.1秒一次）
  static int frame_count = 0;
  frame_count++;
  if (frame_count % 1000 == 0) {
    RCLCPP_INFO(node_->get_logger(), "Current contacts: %d", ncon);
  }
  
  // 如果没有接触点，发布空消息并返回
  if (ncon == 0) {
    contact_force_pub_->publish(std::move(contact_msg));
    return;
  }

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



  // 遍历所有接触点，填充消息数据
  for (int i = 0; i < ncon; ++i) {
    const mjContact& contact = d->contact[i];
    
    // 生成接触点名称：使用两个接触几何体对应的body名称组合
    int body1_id = m->geom_bodyid[contact.geom[0]];
    int body2_id = m->geom_bodyid[contact.geom[1]];
    
    const char* body1_name = mj_id2name(m, mjOBJ_BODY, body1_id);
    const char* body2_name = mj_id2name(m, mjOBJ_BODY, body2_id);
    
    // 如果body没有名称，使用ID作为名称
    std::string name1 = body1_name ? body1_name : "body" + std::to_string(body1_id);
    std::string name2 = body2_name ? body2_name : "body" + std::to_string(body2_id);
    contact_msg->contact_names[i] = name1 + "_" + name2;

    // 计算接触点在世界坐标系中的位置
    mjtNum pos[3];
    
    // 获取接触的两个几何体ID
    int geom1 = contact.geom[0];
    int geom2 = contact.geom[1];
    
    // 获取接触点的法向量（接触坐标系的第一轴）
    mjtNum normal[3] = {contact.frame[0], contact.frame[1], contact.frame[2]};
    mjtNum dist = contact.dist;  // 接触间隙距离
    
    // 使用MuJoCo内部计算的接触点位置（最准确）
    // contact.pos 是MuJoCo内部计算的接触点，考虑了几何体的实际形状和接触算法
    pos[0] = contact.pos[0];
    pos[1] = contact.pos[1];
    pos[2] = contact.pos[2];
    
    // 应用配置文件中的位置偏移量（用于可视化调整）
    double position_offset = config_loader_->GetContactPositionOffset();
    if (position_offset != 0.0) {
      // 沿法向量方向偏移接触点位置
      pos[0] += normal[0] * position_offset;
      pos[1] += normal[1] * position_offset;
      pos[2] += normal[2] * position_offset;
    }
    
    // 将计算得到的接触点位置存储到消息中
    contact_msg->contact_positions_x[i] = pos[0];
    contact_msg->contact_positions_y[i] = pos[1];
    contact_msg->contact_positions_z[i] = pos[2];

    // 获取接触坐标系下的6D接触力 [fx, fy, fz, tx, ty, tz]
    mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mj_contactForce(m, d, i, contact_force);
    
    // 将接触坐标系下的力和力矩转换为世界坐标系
    // contact_force[0:2] 是力向量，contact_force[3:5] 是力矩向量
    mjtNum world_force[3] = {0.0, 0.0, 0.0};
    mjtNum world_torque[3] = {0.0, 0.0, 0.0};
    
    // 坐标系转换：从接触坐标系到世界坐标系
    // contact.frame 在MuJoCo中存储为3x3旋转矩阵，表示接触坐标系到世界坐标系的变换
    // frame[0-2]: X轴 (法向量，指向接触点)
    // frame[3-5]: Y轴 (第一个切向量)  
    // frame[6-8]: Z轴 (第二个切向量)
    for (int j = 0; j < 3; j++) {
      // 转换力向量：world_force = frame^T * contact_force[0:2]
      world_force[j] = contact.frame[j*3 + 0] * contact_force[0] + 
                       contact.frame[j*3 + 1] * contact_force[1] + 
                       contact.frame[j*3 + 2] * contact_force[2];
      
      // 转换力矩向量：world_torque = frame^T * contact_force[3:5]
      world_torque[j] = contact.frame[j*3 + 0] * contact_force[3] + 
                        contact.frame[j*3 + 1] * contact_force[4] + 
                        contact.frame[j*3 + 2] * contact_force[5];
    }
    

    
    // 将世界坐标系下的力和力矩存储到消息中
    contact_msg->contact_forces_x[i] = world_force[0];
    contact_msg->contact_forces_y[i] = world_force[1];
    contact_msg->contact_forces_z[i] = world_force[2];
    
    contact_msg->contact_torques_x[i] = world_torque[0];
    contact_msg->contact_torques_y[i] = world_torque[1];
    contact_msg->contact_torques_z[i] = world_torque[2];

    // 存储接触间隙距离（正值表示分离，负值表示穿透）
    contact_msg->contact_gaps[i] = contact.dist;

    // 存储接触的两个刚体ID
    contact_msg->contact_bodies_1[i] = m->geom_bodyid[contact.geom[0]];
    contact_msg->contact_bodies_2[i] = m->geom_bodyid[contact.geom[1]];
  }



  // 发布接触力消息到ROS话题
  contact_force_pub_->publish(std::move(contact_msg));

  // RViz可视化功能：在RViz中显示接触点
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

    // 使用静态变量缓存标准姿态数据，避免重复计算
    // 标准姿态用于计算接触点相对于机器人的位置
    static std::vector<mjtNum> std_xpos_cache;    // 缓存标准位置
    static std::vector<mjtNum> std_xmat_cache;    // 缓存标准旋转矩阵
    static const mjModel* last_model_ptr = nullptr;  // 记录上次使用的模型
    
    // 检查是否需要重新计算标准姿态（当模型改变时）
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
            // 如需固定到特定高度（如0.82），可按模型顺序设置
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
    
    // 检查缓存是否有效，如果无效则跳过可视化
    if (std_xpos_cache.empty() || std_xmat_cache.empty()) {
      RCLCPP_WARN(node_->get_logger(), "Standard pose cache is empty, skipping visualization");
      return;
    }
    
    // 获取缓存数据的指针，用于后续计算
    const mjtNum* std_xpos = std_xpos_cache.data();  // 标准姿态下的位置数据
    const mjtNum* std_xmat = std_xmat_cache.data();  // 标准姿态下的旋转矩阵数据

    // 遍历所有接触点，创建可视化标记
    for (int i = 0; i < ncon; ++i) {
      const mjContact& contact = d->contact[i];

      // 获取6D接触力 [fx, fy, fz, tx, ty, tz]
      double w[6] = {0};
      mj_contactForce(m, d, i, w);

      // 获取接触坐标系到世界坐标系的旋转矩阵
      double R[9];
      if (contact.dim > 0) {
        // 复制接触坐标系变换矩阵
        std::memcpy(R, contact.frame, sizeof(R));
      } else {
        // 如果没有接触维度，使用单位矩阵
        R[0]=1;R[1]=0;R[2]=0; R[3]=0;R[4]=1;R[5]=0; R[6]=0;R[7]=0;R[8]=1;
      }

      // 使用全局定义的rot3函数进行3D旋转变换
      
      // 将接触坐标系下的力转换到世界坐标系
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

      mkr_red.pose.position.x = contact.pos[0];
      mkr_red.pose.position.y = contact.pos[1];
      mkr_red.pose.position.z = contact.pos[2];
      mkr_red.pose.orientation.w = 1.0;

      // 根据接触力大小动态计算球体直径
      // base_diameter: 基础直径（力为0时的默认大小）
      // f_mag: 当前接触点的力大小（标量）
      // scale_gain: 力到尺寸的缩放系数
      // std::clamp(f_mag*scale_gain, 0.0, 3.0): 限制缩放倍数在[0.0, 3.0]范围内
      // 最终直径 = 基础直径 × (1.0 + 缩放倍数)
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
      int body = m->geom_bodyid[contact.geom[0]];
      if (body == 0) body = m->geom_bodyid[contact.geom[1]]; // 如果第一个是 world

      if (body > 0 && body < m->nbody) {  // 添加边界检查
        // 进一步检查缓存边界
        if (body * 3 + 2 >= std_xpos_cache.size() || 
            body * 9 + 8 >= std_xmat_cache.size()) {
          RCLCPP_WARN(node_->get_logger(), "Body index %d out of cache bounds, skipping visualization", body);
          continue;
        }
        
        // 当前姿态：世界 -> body 局部
        // d->xpos + 3*body：
        // d->xpos 是 MuJoCo 数据结构中存储所有body当前位置的一维数组
        // 格式：[body0_x, body0_y, body0_z, body1_x, body1_y, body1_z, ...]
        // + 3*body 跳转到指定body的位置数据起始地址
        // x_BW 指向该body当前的 (x,y,z) 坐标
        // --------------------------------
        // d->xmat + 9*body：
        // d->xmat 是 MuJoCo 数据结构中存储所有body当前旋转矩阵的一维数组
        // 格式：[body0_R00,R01,R02,R10,R11,R12,R20,R21,R22, body1_R00,R01,...]
        // + 9*body 跳转到指定body的3x3旋转矩阵起始地址
        // R_BW 指向该body当前的旋转矩阵（按行存储）
        const mjtNum* x_BW = d->xpos + 3*body;   // 当前 body 原点（世界）
        const mjtNum* R_BW = d->xmat + 9*body;   // 当前 body 旋转（行存 3x3）

        // 使用全局定义的rotT3和rot3函数进行3D旋转变换

        // 当前姿态下接触点 → body 局部 p_B
        double pW_minus_x[3] = {contact.pos[0]-x_BW[0],
                                contact.pos[1]-x_BW[1],
                                contact.pos[2]-x_BW[2]};
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

    contact_marker_pub_->publish(arr);
  }

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
        
        // 获取body名称
        int body1_id = m->geom_bodyid[contact.geom[0]];
        int body2_id = m->geom_bodyid[contact.geom[1]];
        
        const char* body1_name = mj_id2name(m, mjOBJ_BODY, body1_id);
        const char* body2_name = mj_id2name(m, mjOBJ_BODY, body2_id);
        
        std::string name1 = body1_name ? body1_name : "body" + std::to_string(body1_id);
        std::string name2 = body2_name ? body2_name : "body" + std::to_string(body2_id);
        
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
        // body1_id 和 body2_id 已经在前面声明过了
        
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
        
        // 使用PublishContacts中正确的坐标系转换方法
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
        mjtNum robot_frame_force[3] = {0, 0, 0};
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
          double pW_minus_x[3] = {pos[0]-x_BW[0],
                                  pos[1]-x_BW[1],
                                  pos[2]-x_BW[2]};
          rotT3(R_BW, pW_minus_x, robot_frame_pos);

          // 世界力 -> 局部力
          rotT3(R_BW, world_force, robot_frame_force);
        }
        
        // 写入CSV行
        csv_file_ << std::fixed << std::setprecision(6)
                  << sim_time << ","
                  << i << ","
                  << "\"" << name1 << "\","
                  << "\"" << name2 << "\","
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

        // 添加完整的31个参数（浮动基座位置+姿态+关节角度）
        if (is_floating_base_) {
          // 保存浮动基座位置 (前3个参数)
          csv_file_ << "," << d->qpos[0] << "," << d->qpos[1] << "," << d->qpos[2];
          // 保存浮动基座姿态四元数 (接下来4个参数)
          csv_file_ << "," << d->qpos[3] << "," << d->qpos[4] << "," << d->qpos[5] << "," << d->qpos[6];
          // 保存24个关节角度 (接下来24个参数)
          for (int i = 0; i < num_total_joints_; i++) {
            csv_file_ << "," << d->qpos[i + 7];  // 从索引7开始，跳过前7个浮动基座参数
          }
        } else {
          // 非浮动基座机器人，只保存关节角度
          for (int i = 0; i < num_total_joints_; i++) {
            csv_file_ << "," << d->qpos[i];
          }
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

// 注意：此函数已被 PublishContactForces 替代，功能已整合到 PublishContactForces 中
// 保留此函数仅用于向后兼容，建议使用 PublishContactForces
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
