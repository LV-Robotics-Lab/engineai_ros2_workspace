/**
 * @file sim_manager.cc
 * @brief MuJoCo仿真管理器实现
 * @details 该文件实现了MuJoCo仿真的核心管理功能，包括：
 *          - 仿真初始化和配置
 *          - 物理仿真循环
 *          - 模型加载和管理
 *          - 接触力可视化设置
 *          - 关节控制器实现
 *          - ROS接口集成
 */

#include "sim_manager.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include "simulate/array_safety.h"
#include "simulate/glfw_adapter.h"

namespace mj = mujoco;
namespace mju = mujoco::sample_util;

using namespace std::chrono_literals;

// 常量定义
const int kDofFloatingBase = 6;        // 浮动基座的自由度数量
const int kNumFloatingBaseJoints = 7;  // 浮动基座的关节数量（四元数 + xyz位置）
constexpr double kSyncMisalign = 0.1;  // 重新同步前的最大偏差
constexpr double kSimRefreshFraction = 0.7;  // 可用于仿真的刷新率分数
const std::chrono::milliseconds kBusyWaitTime(1);  // 忙等待时间

/**
 * @brief MuJoCo控制回调的静态包装函数
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 将MuJoCo控制回调转发到SimManager实例
 */
static void TorqueControllerWrapper(const mjModel* m, mjData* d) { 
  SimManager::GetInstance().TorqueController(m, d); 
}

/**
 * @brief 自定义GLFW适配器 - 实现推倒采样的键盘控制
 * 
 * 这个类重写了MuJoCo的键盘事件处理，实现了推倒采样的交互式控制。
 * 用户可以通过键盘快捷键实时控制干扰力的施加，用于测试机器人的平衡能力。
 */
class CustomGlfwAdapter : public mj::GlfwAdapter {
 protected:
  bool shift_pressed_ = false;  // Shift键状态
  
  /**
   * @brief 键盘事件处理函数
   * @param key 按键代码
   * @param scancode 扫描码
   * @param act 按键动作（按下/释放）
   * 
   * 实现推倒采样的键盘控制：
   * - Shift + F/B: 前后向干扰力
   * - Shift + L/R: 左右向干扰力  
   * - Shift + U/D: 上下向干扰力
   * - Shift + G/J: X轴干扰力矩
   * - Shift + Y/H: Y轴干扰力矩
   * - Shift + [/]: Z轴干扰力矩
   * - Shift + 0: 立即停止干扰力
   * - Shift + +/-: 调整干扰力大小
   * - Shift + ,/.: 调整干扰力持续时间
   */
  void OnKey(int key, int scancode, int act) override {
    // 首先调用父类的OnKey方法，保持原有功能
    mj::GlfwAdapter::OnKey(key, scancode, act);

    // 更新Shift键状态
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
      if (act == GLFW_PRESS) {
        shift_pressed_ = true;
      } else if (act == GLFW_RELEASE) {
        shift_pressed_ = false;
      }
      return;
    }

    // 处理按下和释放事件
    SimManager& sim_manager = SimManager::GetInstance();
    
    if (act == GLFW_PRESS && shift_pressed_) {
      // 处理按键按下事件（启动干扰力）
      switch (key) {
        case GLFW_KEY_F:  // Shift + F: 前向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("forward", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发前向干扰力 (推力可视化已启用)" << std::endl;
          }
          break;

        case GLFW_KEY_B:  // Shift + B: 后向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("backward", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发后向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_L:  // Shift + L: 左向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("left", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发左向干扰力: 力大小=" << sim_manager.GetPerturbationForceMagnitude() 
                      << "N, 目标物体=" << sim_manager.GetPerturbationBodyName() << " (推力可视化已启用)" << std::endl;
          }
          break;

        case GLFW_KEY_R:  // Shift + R: 右向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("right", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发右向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_U:  // Shift + U: 上向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("up", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发上向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_D:  // Shift + D: 下向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("down", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发下向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_G:  // Shift + G: X轴正方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("x_positive", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发X轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_J:  // Shift + J: X轴负方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("x_negative", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发X轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_Y:  // Shift + Y: Y轴正方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("y_positive", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发Y轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_H:  // Shift + H: Y轴负方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("y_negative", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发Y轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_LEFT_BRACKET:  // Shift + [: Z轴正方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("z_positive", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发Z轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_RIGHT_BRACKET:  // Shift + ]: Z轴负方向干扰力矩
          {
            double current_torque = sim_manager.GetPerturbationTorqueMagnitude();
            const auto& torque_vec = sim_manager.GetPerturbationTorque("z_negative", current_torque);
            sim_manager.SetPerturbationTorque(Eigen::Vector3d(torque_vec[0], torque_vec[1], torque_vec[2]));
            sim_manager.SetPerturbationForce(Eigen::Vector3d::Zero());  // 清除推力
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发Z轴干扰力矩" << std::endl;
          }
          break;

        case GLFW_KEY_0:  // Shift + 0: 立即停止所有干扰力/力矩
          sim_manager.StopPerturbation();
          std::cout << "立即停止所有干扰力/力矩" << std::endl;
          break;

        case GLFW_KEY_EQUAL:  // Shift + +: 增加干扰力/力矩大小
          {
            double force_step = sim_manager.GetForceStep();
            double torque_step = sim_manager.GetTorqueStep();
            sim_manager.SetPerturbationForceMagnitude(sim_manager.GetPerturbationForceMagnitude() + force_step);
            sim_manager.SetPerturbationTorqueMagnitude(sim_manager.GetPerturbationTorqueMagnitude() + torque_step);
            std::cout << "干扰力/力矩大小增加到: " << sim_manager.GetPerturbationForceMagnitude() 
                      << " N, " << sim_manager.GetPerturbationTorqueMagnitude() << " N.m" << std::endl;
          }
          break;

        case GLFW_KEY_MINUS:  // Shift + -: 减小干扰力/力矩大小
          {
            double force_step = sim_manager.GetForceStep();
            double torque_step = sim_manager.GetTorqueStep();
            sim_manager.SetPerturbationForceMagnitude(std::max(force_step, sim_manager.GetPerturbationForceMagnitude() - force_step));
            sim_manager.SetPerturbationTorqueMagnitude(std::max(torque_step, sim_manager.GetPerturbationTorqueMagnitude() - torque_step));
            std::cout << "干扰力/力矩大小减小到: " << sim_manager.GetPerturbationForceMagnitude() 
                      << " N, " << sim_manager.GetPerturbationTorqueMagnitude() << " N.m" << std::endl;
          }
          break;

        case GLFW_KEY_PERIOD:  // Shift + .: 增加干扰力持续时间
          sim_manager.SetPerturbationDuration(sim_manager.GetPerturbationDuration() + 0.1);
          std::cout << "干扰力/力矩持续时间增加到: " << sim_manager.GetPerturbationDuration() << "秒" << std::endl;
          break;

        case GLFW_KEY_COMMA:  // Shift + ,: 减小干扰力持续时间
          sim_manager.SetPerturbationDuration(std::max(0.1, sim_manager.GetPerturbationDuration() - 0.1));
          std::cout << "干扰力/力矩持续时间减小到: " << sim_manager.GetPerturbationDuration() << "秒" << std::endl;
          break;
      }
    } else if (act == GLFW_RELEASE) {
      // 处理按键释放事件（停止干扰力）
      switch (key) {
        case GLFW_KEY_F:  // 释放F键：停止前向干扰力
        case GLFW_KEY_B:  // 释放B键：停止后向干扰力
        case GLFW_KEY_L:  // 释放L键：停止左向干扰力
        case GLFW_KEY_R:  // 释放R键：停止右向干扰力
        case GLFW_KEY_U:  // 释放U键：停止上向干扰力
        case GLFW_KEY_D:  // 释放D键：停止下向干扰力
          sim_manager.ApplyPerturbation(false);
          std::cout << "停止推力干扰力" << std::endl;
          break;
          
        case GLFW_KEY_G:  // 释放G键：停止X轴干扰力矩
        case GLFW_KEY_J:  // 释放J键：停止X轴干扰力矩
        case GLFW_KEY_Y:  // 释放Y键：停止Y轴干扰力矩
        case GLFW_KEY_H:  // 释放H键：停止Y轴干扰力矩
        case GLFW_KEY_LEFT_BRACKET:   // 释放[键：停止Z轴干扰力矩
        case GLFW_KEY_RIGHT_BRACKET:  // 释放]键：停止Z轴干扰力矩
          sim_manager.ApplyPerturbation(false);
          std::cout << "停止力矩干扰力" << std::endl;
          break;
      }
    }
  }
};

/**
 * @brief 获取SimManager单例实例
 * @return SimManager的引用
 * @details 使用单例模式确保全局只有一个仿真管理器实例
 */
SimManager& SimManager::GetInstance() {
  static SimManager instance;
  return instance;
}

/**
 * @brief 构造函数
 * @details 初始化ROS节点和基本成员变量
 */
SimManager::SimManager() { 
  node_ = std::make_shared<rclcpp::Node>("mujoco_simulator"); 
}

/**
 * @brief 析构函数
 * @details 清理资源，包括物理线程、模型数据和ROS接口
 */
SimManager::~SimManager() {
  try {
    // 首先停止仿真
    if (sim_) {
      sim_->exitrequest.store(true);
    }
    
    // 等待物理线程结束
    if (physics_thread_.joinable()) {
      physics_thread_.join();
    }
    
    // 清理MuJoCo资源（添加空指针检查）
    if (d_) {
      mj_deleteData(d_);
      d_ = nullptr;
    }
    
    if (m_) {
      mj_deleteModel(m_);
      m_ = nullptr;
    }
    
    // 清理ROS接口
    ros_interface_.reset();
    
  } catch (const std::exception& e) {
    std::cerr << "Exception during SimManager destruction: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception during SimManager destruction" << std::endl;
  }
}

/**
 * @brief 关节力矩控制器
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 实现PD控制器和推倒采样的干扰力系统
 */
void SimManager::TorqueController(const mjModel* m, mjData* d) {
  // ==================== PD控制器部分 ====================
  if (ros_interface_) {
    // 获取线程安全的命令值副本
    auto cmd = ros_interface_->GetCommandedSafe();

    // 检查是否为浮动基座机器人
    bool is_floating_base = (m->nv != m->nu);

    // 应用命令控制
    for (int i = 0; i < m->nu; ++i) {
      if (i >= cmd.position.size() || i >= cmd.velocity.size() || i >= cmd.torque.size() ||
          i >= cmd.feed_forward_torque.size() || i >= cmd.stiffness.size() || i >= cmd.damping.size()) {
        continue;
      }

      // 获取位置和速度，考虑浮动基座
      double position;
      double velocity;

      if (is_floating_base) {
        position = d->qpos[i + kNumFloatingBaseJoints];
        velocity = d->qvel[i + kDofFloatingBase];
      } else {
        position = d->qpos[i];
        velocity = d->qvel[i];
      }

      // PD控制加前馈力矩
      double position_error = cmd.position[i] - position;
      double velocity_error = cmd.velocity[i] - velocity;

      d->ctrl[i] = cmd.feed_forward_torque[i] + cmd.stiffness[i] * position_error + cmd.damping[i] * velocity_error;
    }
  } else {
    // 如果没有ROS接口，使用零力矩控制（机器人自由运动）
    for (int i = 0; i < m->nu; ++i) {
      d->ctrl[i] = 0.0;
    }
  }

  // ==================== 多干扰力系统 ====================
  {
    std::lock_guard<std::mutex> lock(perturbation_mutex_);
    
    // 检查并清理过期的干扰力
    auto it = active_perturbations_.begin();
    while (it != active_perturbations_.end()) {
      if (d->time - it->start_time > it->duration) {
        std::cout << "干扰力ID " << it->id << " 自动停止" << std::endl;
        it = active_perturbations_.erase(it);
      } else {
        ++it;
      }
    }
    
    // 处理当前激活的干扰力（向后兼容）
    if (apply_perturb_) {
      // 检查是否已经存在相同的干扰力
      bool found_existing = false;
      for (auto& pert : active_perturbations_) {
        if (pert.body_name == perturb_body_name_ && 
            pert.force.isApprox(perturb_force_) && 
            pert.torque.isApprox(perturb_torque_)) {
          found_existing = true;
          break;
        }
      }
      
      // 如果没有找到相同的干扰力，添加新的
      if (!found_existing) {
        PerturbationData new_pert;
        new_pert.id = GetNextPerturbationId();
        new_pert.body_name = perturb_body_name_;
        new_pert.force = perturb_force_;
        new_pert.torque = perturb_torque_;
        new_pert.start_time = d->time;
        new_pert.duration = perturb_duration_;
        new_pert.is_active = true;
        active_perturbations_.push_back(new_pert);
        std::cout << "添加新干扰力ID " << new_pert.id << ", 持续时间: " << perturb_duration_ << "秒" << std::endl;
      }
    } else {
      // 如果apply_perturb_为false，清除所有干扰力（向后兼容）
      if (!active_perturbations_.empty()) {
        std::cout << "停止所有干扰力（apply_perturb_=false）" << std::endl;
        active_perturbations_.clear();
      }
    }
    
    // 施加所有活跃的干扰力
    for (const auto& pert : active_perturbations_) {
      int body_id = mj_name2id(m, mjOBJ_BODY, pert.body_name.c_str());
      if (body_id >= 0) {
        // 将干扰力从物体坐标系转换到世界坐标系
        mjtNum* body_quat = d->xquat + 4 * body_id;
        mjtNum perturb_force_local[3] = {pert.force.x(), pert.force.y(), pert.force.z()};
        mjtNum perturb_force_world[3];
        mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);

        // 施加干扰力到物体
        d->xfrc_applied[6 * body_id + 0] += perturb_force_world[0];
        d->xfrc_applied[6 * body_id + 1] += perturb_force_world[1];
        d->xfrc_applied[6 * body_id + 2] += perturb_force_world[2];

        // 施加干扰力矩到物体
        d->xfrc_applied[6 * body_id + 3] += pert.torque.x();
        d->xfrc_applied[6 * body_id + 4] += pert.torque.y();
        d->xfrc_applied[6 * body_id + 5] += pert.torque.z();
        
        // 调试信息：每100步打印一次
        static int debug_counter = 0;
        if (++debug_counter % 100 == 0) {
          double force_magnitude = sqrt(perturb_force_world[0]*perturb_force_world[0] + 
                                       perturb_force_world[1]*perturb_force_world[1] + 
                                       perturb_force_world[2]*perturb_force_world[2]);
          double torque_magnitude = sqrt(pert.torque.x()*pert.torque.x() + 
                                        pert.torque.y()*pert.torque.y() + 
                                        pert.torque.z()*pert.torque.z());
          std::cout << "施加干扰力ID " << pert.id << ": body_id=" << body_id << ", 世界力=[" 
                    << perturb_force_world[0] << ", " << perturb_force_world[1] << ", " << perturb_force_world[2] 
                    << "] (大小: " << force_magnitude << "N), 世界力矩=[" 
                    << pert.torque.x() << ", " << pert.torque.y() << ", " << pert.torque.z() 
                    << "] (大小: " << torque_magnitude << "N.m)" << std::endl;
        }
      } else {
        std::cout << "警告: 找不到目标物体 '" << pert.body_name << "'" << std::endl;
      }
    }
  }
}

/**
 * @brief 初始化仿真管理器
 * @return 初始化是否成功
 * @details 初始化包括：
 *          - 设置日志级别
 *          - 验证环境变量
 *          - 加载配置文件
 *          - 初始化ROS接口
 *          - 设置控制回调
 *          - 配置可视化选项
 */
bool SimManager::Initialize() {
  auto logger = node_->get_logger();
  if (rcutils_logging_set_logger_level(logger.get_name(), RCUTILS_LOG_SEVERITY_INFO) != RCUTILS_RET_OK) {
    RCLCPP_ERROR(logger, "Failed to set logger level");
    return false;
  }

  RCLCPP_INFO(logger, "MuJoCo Simulator node initialized");

  // 验证环境变量
  if (!std::getenv("PRODUCT") || !std::getenv("MUJOCO_ASSETS_PATH")) {
    RCLCPP_ERROR(logger, "Required environment variables not set! Please run from launch file.");
    return false;
  }

  // 从环境变量获取产品名称和资源路径
  std::string product_name = std::string(std::getenv("PRODUCT"));
  std::string assets_path = std::string(std::getenv("MUJOCO_ASSETS_PATH"));

  // 构建配置文件路径
  std::string config_file = assets_path + "/config/" + product_name + ".yaml";
  RCLCPP_INFO(logger, "Loading config from %s", config_file.c_str());

  // 初始化配置加载器
  config_loader_ = std::make_shared<ConfigLoader>(config_file);
  config_loader_->SetAssetsPath(assets_path);
  if (!config_loader_->LoadConfig()) {
    RCLCPP_ERROR(logger, "Failed to load config file: %s", config_file.c_str());
    return false;
  }

  // 创建MuJoCo ROS接口
  ros_interface_ = std::make_unique<mujoco::RosInterface>(node_, config_loader_);
  if (!ros_interface_->Initialize()) {
    RCLCPP_ERROR(logger, "Failed to initialize MuJoCo ROS interface");
    return false;
  }

  // 安装控制回调
  mjcb_control = TorqueControllerWrapper;

  // 记录版本信息
  RCLCPP_INFO(logger, "MuJoCo version %s", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    RCLCPP_ERROR(logger, "Headers and library have different versions");
    return false;
  }

  // 设置相机、选项、扰动
  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);
  
  // 启用接触点可视化 - 这是关键设置
  opt.flags[mjVIS_CONTACTPOINT] = 1;  // 显示接触点
  opt.flags[mjVIS_CONTACTFORCE] = 1;  // 显示接触力
  opt.flags[mjVIS_CONTACTSPLIT] = 1;  // 显示接触分离
  
  // 启用外力可视化 - 显示TorqueController施加的推力
  opt.flags[mjVIS_PERTFORCE] = 1;  // 显示施加的外力
  
  // 添加调试信息
  RCLCPP_INFO(logger, "Visualization enabled: CONTACTPOINT=%d, CONTACTFORCE=%d, CONTACTSPLIT=%d, PERTFORCE=%d", 
              opt.flags[mjVIS_CONTACTPOINT], opt.flags[mjVIS_CONTACTFORCE], opt.flags[mjVIS_CONTACTSPLIT], opt.flags[mjVIS_PERTFORCE]);
  RCLCPP_INFO(logger, "推力可视化已启用 - TorqueController施加的推力将在MuJoCo界面中显示为箭头");

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // 创建仿真对象，使用自定义的GLFW适配器以支持推倒采样的键盘交互
  sim_ = std::make_unique<mj::Simulate>(std::make_unique<CustomGlfwAdapter>(), &cam, &opt, &pert, false);

  return true;
}

// ==================== 推外力控制函数实现 ====================

/**
 * @brief 设置干扰力向量
 * @param force 干扰力向量（物体坐标系）
 */
void SimManager::SetPerturbationForce(const Eigen::Vector3d& force) {
  perturb_force_ = force;
}

/**
 * @brief 获取默认推力大小
 * @return 默认推力大小（牛顿）
 */
double SimManager::GetDefaultForceMagnitude() const {
  if (config_loader_) {
    return config_loader_->GetDefaultForceMagnitude();
  }
  return 20.0;  // 默认值
}

/**
 * @brief 获取默认扭矩大小
 * @return 默认扭矩大小（牛·米）
 */
double SimManager::GetDefaultTorqueMagnitude() const {
  if (config_loader_) {
    return config_loader_->GetDefaultTorqueMagnitude();
  }
  return 5.0;  // 默认值
}

/**
 * @brief 获取推力调整步长
 * @return 推力调整步长（牛顿）
 */
double SimManager::GetForceStep() const {
  if (config_loader_) {
    return config_loader_->GetForceStep();
  }
  return 20.0;  // 默认值
}

/**
 * @brief 获取扭矩调整步长
 * @return 扭矩调整步长（牛·米）
 */
double SimManager::GetTorqueStep() const {
  if (config_loader_) {
    return config_loader_->GetTorqueStep();
  }
  return 5.0;  // 默认值
}

/**
 * @brief 获取指定方向的推力配置
 * @param direction 推力方向（forward, backward, left, right, up, down）
 * @return 推力向量（3D向量）
 */
const std::vector<double>& SimManager::GetPerturbationForce(const std::string& direction) const {
  if (config_loader_) {
    return config_loader_->GetPerturbationForce(direction);
  }
  
  // 如果配置加载器不可用，返回默认前向推力
  static const std::vector<double> default_force = {20.0, 0.0, 0.0};
  return default_force;
}

/**
 * @brief 获取指定方向的推力配置（使用指定大小）
 * @param direction 推力方向（forward, backward, left, right, up, down）
 * @param magnitude 推力大小
 * @return 推力向量（3D向量）
 */
const std::vector<double>& SimManager::GetPerturbationForce(const std::string& direction, double magnitude) const {
  if (config_loader_) {
    return config_loader_->GetPerturbationForce(direction, magnitude);
  }
  
  // 如果配置加载器不可用，返回指定大小的前向推力
  static std::vector<double> force = {magnitude, 0.0, 0.0};
  return force;
}

/**
 * @brief 获取指定方向的扭矩配置
 * @param direction 扭矩方向（x_positive, x_negative, y_positive, y_negative, z_positive, z_negative）
 * @return 扭矩向量（3D向量）
 */
const std::vector<double>& SimManager::GetPerturbationTorque(const std::string& direction) const {
  if (config_loader_) {
    return config_loader_->GetPerturbationTorque(direction);
  }
  
  // 如果配置加载器不可用，返回默认X轴正方向扭矩
  static const std::vector<double> default_torque = {5.0, 0.0, 0.0};
  return default_torque;
}

/**
 * @brief 获取指定方向的扭矩配置（使用指定大小）
 * @param direction 扭矩方向（x_positive, x_negative, y_positive, y_negative, z_positive, z_negative）
 * @param magnitude 扭矩大小
 * @return 扭矩向量（3D向量）
 */
const std::vector<double>& SimManager::GetPerturbationTorque(const std::string& direction, double magnitude) const {
  if (config_loader_) {
    return config_loader_->GetPerturbationTorque(direction, magnitude);
  }
  
  // 如果配置加载器不可用，返回指定大小的X轴正方向扭矩
  static std::vector<double> torque = {magnitude, 0.0, 0.0};
  return torque;
}

/**
 * @brief 设置干扰力矩向量
 * @param torque 干扰力矩向量（世界坐标系）
 */
void SimManager::SetPerturbationTorque(const Eigen::Vector3d& torque) {
  perturb_torque_ = torque;
}

/**
 * @brief 设置施加干扰力的物体名称
 * @param body_name 物体名称
 */
void SimManager::SetPerturbationBody(const std::string& body_name) {
  perturb_body_name_ = body_name;
}


/**
 * @brief 应用或停止干扰力
 * @param apply 是否应用干扰力
 */
void SimManager::ApplyPerturbation(bool apply) {
  apply_perturb_ = apply;
  if (apply) {
    perturb_start_time_ = -1.0;  // 重置开始时间
  }
}

/**
 * @brief 立即停止所有干扰力
 */
void SimManager::StopPerturbation() {
  std::lock_guard<std::mutex> lock(perturbation_mutex_);
  apply_perturb_ = false;
  perturb_start_time_ = -1.0;
  perturb_force_ = Eigen::Vector3d::Zero();
  perturb_torque_ = Eigen::Vector3d::Zero();
  active_perturbations_.clear();
  std::cout << "停止所有干扰力" << std::endl;
}

/**
 * @brief 获取所有活跃的干扰力数据
 * @return 活跃干扰力数据向量
 */
std::vector<SimManager::PerturbationData> SimManager::GetActivePerturbations() const {
  std::lock_guard<std::mutex> lock(perturbation_mutex_);
  return active_perturbations_;
}

/**
 * @brief 运行仿真
 * @details 启动物理线程和UI渲染循环
 */
void SimManager::Run() {
  auto logger = node_->get_logger();
  std::string model_file = config_loader_->GetModelFilePath();
  RCLCPP_INFO(logger, "Model file path: %s", model_file.c_str());

  // 在启动物理线程前设置VFS目录
  const std::string resource_dir = config_loader_->GetResourceDir();
  setenv("MJCF_PATH", resource_dir.c_str(), 1);
  RCLCPP_INFO(logger, "Setting MJCF_PATH environment variable: %s", resource_dir.c_str());

  // 启动物理线程
  RCLCPP_INFO(logger, "Starting physics thread");
  physics_thread_ = std::thread([this, model_file]() { PhysicsThread(model_file); });

  // 启动UI循环
  RCLCPP_INFO(logger, "Starting UI rendering loop");
  sim_->RenderLoop();
  RCLCPP_INFO(logger, "UI rendering loop completed");
}

/**
 * @brief 加载MuJoCo模型
 * @param file 模型文件路径
 * @return 加载的模型指针，失败时返回nullptr
 * @details 支持加载.mjb二进制文件和.xml文本文件
 */
mjModel* SimManager::LoadModel(std::string_view file) {
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file.data());
  if (!filename[0]) {
    return nullptr;
  }

  mjModel* mnew = nullptr;
  auto load_start = mj::Simulate::Clock::now();
  auto logger = node_->get_logger();

  // 检查是否为二进制模型文件(.mjb)
  if (mju::strlen_arr(filename) > 4 && !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                                                     mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      RCLCPP_ERROR(logger, "Could not load binary model");
    }
  } else {
    // 获取碰撞模型条件
    std::string collision_condition = "simplified";  // 默认值
    std::string xml_filename = "pm_v2.xml";  // 默认使用主配置文件
    
    if (config_loader_) {
      collision_condition = config_loader_->GetCollisionModelCondition();
      // 根据碰撞类型选择对应的XML文件
      xml_filename = config_loader_->GetXmlFilenameByCollisionType();
    }
    
    RCLCPP_INFO(logger, "Loading XML with collision condition: %s", collision_condition.c_str());
    RCLCPP_INFO(logger, "Selected XML file: %s", xml_filename.c_str());
    
    // 构建完整的XML文件路径
    std::string full_xml_path = config_loader_ ? 
        config_loader_->GetResourceDir() + "/" + xml_filename : 
        filename;
    
    RCLCPP_INFO(logger, "Full XML path: %s", full_xml_path.c_str());
    
    // 设置编译条件
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    
    // 创建编译选项
    mjOption opt;
    mj_defaultOption(&opt);
    
    // 加载选择的XML文件
    char xml_path[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(xml_path, full_xml_path.c_str());
    
    mnew = mj_loadXML(xml_path, &vfs, mj_load_error_.data(), mj_load_error_.size());
    if (mj_load_error_[0]) {
      size_t error_length = std::strlen(mj_load_error_.data());
      if (mj_load_error_[error_length - 1] == '\n') {
        mj_load_error_[error_length - 1] = '\0';
      }
      RCLCPP_WARN(logger, "Model compiled with warning: %s", mj_load_error_.data());
      sim_->run = 0;
    }
  }

  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = std::chrono::duration<double>(load_interval).count();

  if (!mnew) {
    RCLCPP_ERROR(logger, "Failed to load model: %s", mj_load_error_.data());
    return nullptr;
  }

  if (load_seconds > 0.25) {
    RCLCPP_INFO(logger, "Model loaded in %.2g seconds", load_seconds);
  }

  try {
    mju::strcpy_arr(sim_->load_error, mj_load_error_.data());
  } catch (...) {
    RCLCPP_ERROR(logger, "Could not copy load error: %s", mj_load_error_.data());
  }

  return mnew;
}

/**
 * @brief 检查仿真是否发散
 * @param disableflags 禁用标志
 * @param d MuJoCo数据指针
 * @return 发散信息字符串，未发散时返回nullptr
 * @details 检查位置、速度、加速度是否超出合理范围
 */
const char* SimManager::Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

/**
 * @brief 处理拖拽加载请求
 * @details 加载用户拖拽的模型文件
 */
void SimManager::HandleDropLoad() {
  sim_->LoadMessage(sim_->dropfilename);
  mjModel* mnew = LoadModel(sim_->dropfilename);
  sim_->droploadrequest.store(false);

  mjData* dnew = nullptr;
  if (mnew) dnew = mj_makeData(mnew);
  if (dnew) {
    sim_->Load(mnew, dnew, sim_->dropfilename);

    const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);

    mj_deleteData(d_);
    mj_deleteModel(m_);

    m_ = mnew;
    d_ = dnew;
    
    // 应用 keyframe 中的初始位置
    int keyframe_id = mj_name2id(m_, mjOBJ_KEY, "floating_base_homing");
    if (keyframe_id >= 0) {
      mj_resetDataKeyframe(m_, d_, keyframe_id);
    }
    
    mj_forward(m_, d_);
  } else {
    sim_->LoadMessageClear();
  }
}

/**
 * @brief 处理UI加载请求
 * @details 加载UI界面请求的模型文件
 */
void SimManager::HandleUILoad() {
  sim_->uiloadrequest.fetch_sub(1);
  sim_->LoadMessage(sim_->filename);
  mjModel* mnew = LoadModel(sim_->filename);
  mjData* dnew = nullptr;
  if (mnew) dnew = mj_makeData(mnew);
  if (dnew) {
    sim_->Load(mnew, dnew, sim_->filename);

    const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);

    mj_deleteData(d_);
    mj_deleteModel(m_);

    m_ = mnew;
    d_ = dnew;
    
    // 应用 keyframe 中的初始位置
    int keyframe_id = mj_name2id(m_, mjOBJ_KEY, "floating_base_homing");
    if (keyframe_id >= 0) {
      mj_resetDataKeyframe(m_, d_, keyframe_id);
    }
    
    mj_forward(m_, d_);
  } else {
    sim_->LoadMessageClear();
  }
}

/**
 * @brief 物理仿真循环
 * @details 主要的仿真循环，包括：
 *          - ROS消息处理
 *          - 模型加载处理
 *          - 仿真步进
 *          - 状态更新
 *          - 发散检测
 */
void SimManager::PhysicsLoop() {
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  while (!sim_->exitrequest.load()) {
    // 处理ROS消息
    if (ros_interface_) {
      rclcpp::spin_some(ros_interface_->GetNode());
    }

    // 处理拖拽加载请求
    if (sim_->droploadrequest.load()) {
      HandleDropLoad();
    }

    // 处理UI加载请求
    if (sim_->uiloadrequest.load()) {
      HandleUILoad();
    }

    // 忙等待或睡眠
    if (sim_->run && sim_->busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(kBusyWaitTime);
    }

    {
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);

      if (m_) {
        if (sim_->run) {
          bool stepped = false;
          const auto startCPU = mj::Simulate::Clock::now();
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d_->time - syncSim;
          double slowdown = 100 / sim_->percentRealTime[sim_->real_time_index];
          bool misaligned = std::abs((elapsedCPU / slowdown).count() - elapsedSim) > kSyncMisalign;

          // 检查是否需要重新同步
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim_->speed_changed) {
            syncCPU = startCPU;
            syncSim = d_->time;
            sim_->speed_changed = false;

            // 执行仿真步进
            mj_step(m_, d_);
            ros_interface_->UpdateSimState(m_, d_);
            const char* message = Diverged(m_->opt.disableflags, d_);
            if (message) {
              sim_->run = 0;
              mju::strcpy_arr(sim_->load_error, message);
            } else {
              stepped = true;
            }
          } else {
            // 执行多个仿真步进以保持实时性
            bool measured = false;
            mjtNum prevSim = d_->time;
            double refreshTime = kSimRefreshFraction / sim_->refresh_rate;

            while ((d_->time - syncSim) * slowdown < (mj::Simulate::Clock::now() - syncCPU).count() &&
                   mj::Simulate::Clock::now() - startCPU < refreshTime * 1s) {
              if (!measured && elapsedSim) {
                sim_->measured_slowdown = elapsedCPU.count() / elapsedSim;
                measured = true;
              }

              sim_->InjectNoise();
              mj_step(m_, d_);
              ros_interface_->UpdateSimState(m_, d_);
              const char* message = Diverged(m_->opt.disableflags, d_);
              if (message) {
                sim_->run = 0;
                mju::strcpy_arr(sim_->load_error, message);
              } else {
                stepped = true;
              }

              if (d_->time < prevSim) {
                break;
              }
            }
          }

          if (stepped) {
            sim_->AddToHistory();
          }
        } else {
          // 仿真暂停时只进行前向计算
          mj_forward(m_, d_);
          sim_->speed_changed = true;
        }
      }
    }
  }
}

/**
 * @brief 物理线程主函数
 * @param filename 模型文件路径
 * @details 在独立线程中运行物理仿真，包括：
 *          - 模型加载
 *          - 数据初始化
 *          - 可视化设置
 *          - 物理循环
 */
void SimManager::PhysicsThread(std::string_view filename) {
  if (!rclcpp::ok()) {
    std::cerr << "ROS context not initialized in physics thread!" << std::endl;
    return;
  }

  auto logger = rclcpp::get_logger("mujoco_physics");
  if (rcutils_logging_set_logger_level(logger.get_name(), RCUTILS_LOG_SEVERITY_INFO) != RCUTILS_RET_OK) {
    RCLCPP_ERROR(logger, "Failed to set logger level");
  }

  RCLCPP_INFO(logger, "PhysicsThread started, filename: %s", filename.data());

  // 验证控制回调设置
  if (mjcb_control != &TorqueControllerWrapper) {
    RCLCPP_WARN(logger, "Control callback not set correctly, setting mjcb_control now");
    mjcb_control = &TorqueControllerWrapper;
  } else {
    RCLCPP_INFO(logger, "MuJoCo control callback is correctly set");
  }

  // 加载模型文件
  if (!filename.empty()) {
    sim_->LoadMessage(filename.data());
    m_ = LoadModel(filename);
    if (m_) {
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);
      d_ = mj_makeData(m_);

      RCLCPP_INFO(logger, "Setting model and data to ROS interface...");
      if (ros_interface_) {
        ros_interface_->SetModelAndData(m_, d_);
        RCLCPP_INFO(logger, "ROS interface successfully set model and data");
      } else {
        RCLCPP_ERROR(logger, "Error: ros_interface is null!");
      }
    }
    if (d_) {
      sim_->Load(m_, d_, filename.data());
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);

      // 设置可视化选项 - 在模型加载后设置
      sim_->opt.flags[mjVIS_CONTACTPOINT] = 1;  // 显示接触点
      sim_->opt.flags[mjVIS_CONTACTFORCE] = 1;  // 显示接触力
      sim_->opt.flags[mjVIS_CONTACTSPLIT] = 1;  // 显示接触分离
      sim_->opt.flags[mjVIS_PERTFORCE] = 1;  // 显示施加的外力
      RCLCPP_INFO(logger, "Visualization flags set in sim_->opt: CONTACTPOINT=%d, CONTACTFORCE=%d, CONTACTSPLIT=%d, PERTFORCE=%d", 
                  sim_->opt.flags[mjVIS_CONTACTPOINT], sim_->opt.flags[mjVIS_CONTACTFORCE], 
                  sim_->opt.flags[mjVIS_CONTACTSPLIT], sim_->opt.flags[mjVIS_PERTFORCE]);

      // 应用 keyframe 中的初始位置
      int keyframe_id = mj_name2id(m_, mjOBJ_KEY, "floating_base_homing");
      if (keyframe_id >= 0) {
        RCLCPP_INFO(logger, "Applying keyframe 'floating_base_homing' (id: %d) as initial position", keyframe_id);
        mj_resetDataKeyframe(m_, d_, keyframe_id);
        RCLCPP_INFO(logger, "Keyframe applied successfully");
      } else {
        RCLCPP_WARN(logger, "Keyframe 'floating_base_homing' not found, using default initial position");
      }

      RCLCPP_INFO(logger, "Performing initial mj_forward calculation...");
      mj_forward(m_, d_);
      RCLCPP_INFO(logger, "mj_forward calculation completed");
    } else {
      sim_->LoadMessageClear();
    }
  }

  if (ros_interface_) {
    RCLCPP_INFO(logger, "Starting physics loop, control callback status: %s",
                (mjcb_control == &TorqueControllerWrapper ? "set" : "not set"));
  }

  RCLCPP_INFO(logger, "Starting physics loop");
  
  try {
    PhysicsLoop();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception in physics loop: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(logger, "Unknown exception in physics loop");
  }

  RCLCPP_INFO(logger, "Physics thread ending, cleaning up resources");
  
  // 清理线程本地资源
  try {
    // 重置ROS接口中的模型和数据指针，避免悬空指针
    if (ros_interface_) {
      ros_interface_->SetModelAndData(nullptr, nullptr);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception during physics thread cleanup: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(logger, "Unknown exception during physics thread cleanup");
  }
}




