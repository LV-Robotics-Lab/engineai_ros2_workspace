#include "ros_interface.h"
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <rclcpp/logging.hpp>
#include <string>

#include "config_loader.h"
#include "rclcpp/rclcpp.hpp"
// #include <mujoco/mujoco.h>


// Constants
const int kDofFloatingBase = 6;        // Number of DoF for floating base
const int kNumFloatingBaseJoints = 7;  // Number of joints for floating base (quaternion + xyz)
const int kDimQuaternion = 4;          // Dimension of a quaternion

namespace mujoco {

RosInterface::RosInterface(const rclcpp::Node::SharedPtr& node, std::shared_ptr<ConfigLoader> config_loader)
    : node_(node), config_loader_(config_loader), model_(nullptr), data_(nullptr), is_floating_base_(false) {}

RosInterface::~RosInterface() {}

bool RosInterface::Initialize() {
  // Create publishers
  joint_state_pub_ =
      node_->create_publisher<interface_protocol::msg::JointState>(config_loader_->GetJointStateTopic(), 10);

  imu_pub_ = node_->create_publisher<interface_protocol::msg::ImuInfo>(config_loader_->GetImuTopic(), 10);
  
  // Create publisher for motion state
  motion_state_pub_ = node_->create_publisher<interface_protocol::msg::MotionState>("/motion/motion_state", 10);

  // publish contact force
  contact_pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>("/mujoco/contact_forces", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort());

  // contact visulization
  contact_marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/mujoco/contact_markers", 10);


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
  joint_state_msg->position.resize(num_total_joints_);
  joint_state_msg->velocity.resize(num_total_joints_);
  joint_state_msg->torque.resize(num_total_joints_);

  if (is_floating_base_) {
    // Skip the floating base joints
    for (int i = 0; i < num_total_joints_; ++i) {
      joint_state_msg->position[i] = d->qpos[i + kNumFloatingBaseJoints];
      joint_state_msg->velocity[i] = d->qvel[i + kDofFloatingBase];
      joint_state_msg->torque[i] = d->actuator_force[i];
    }
  } else {
    for (int i = 0; i < num_total_joints_; ++i) {
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
  PublishContacts(m, d);
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

  // 先发一个“清空”指令，避免历史残留
  {
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
  }

  // 可选：根据力大小缩放球的尺寸
  const double base_diameter = 0.03;   // 3 cm 基础直径
  const double max_diameter  = 0.12;   // 12 cm 上限
  const double scale_gain    = 1.0/200.0; // 200N -> +1 倍尺寸（按需调）

  //加“标准姿态”准备（一次性）
  // === 用 keyframe 'floating_base_homing' 得到【标准姿态】每个 body 的世界位姿 ===
  int key_id = mj_name2id(m, mjOBJ_KEY, "floating_base_homing");
  mjData* dstd = mj_makeData(m);
  if (key_id >= 0) {
    // 直接把 qpos/qvel/act 设为 keyframe
    mj_resetDataKeyframe(m, dstd, key_id);
  } else {
    // 兜底：零位（如需把 free-base z 设成 0.82 可在此设置）
    mju_zero(dstd->qpos, m->nq);
    if (m->jnt_type[0] == mjJNT_FREE) {
      dstd->qpos[0] = 1; dstd->qpos[1] = dstd->qpos[2] = dstd->qpos[3] = 0; // quat = [1,0,0,0]
      dstd->qpos[4] = dstd->qpos[5] = dstd->qpos[6] = 0;                     // base at origin
      // dstd->qpos[6] = 0.82; // 如需固定到 0.82，可按你的模型顺序解开
    }
  }
  mj_forward(m, dstd);
  const mjtNum* std_xpos = dstd->xpos;  // 3 * nbody
  const mjtNum* std_xmat = dstd->xmat;  // 9 * nbody

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

  if (body > 0) {
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
