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
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include <sys/stat.h>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#endif
#include "simulate/array_safety.h"
#include "simulate/glfw_adapter.h"
#include "joint_forces_eigen.hpp"

namespace mj = mujoco;
namespace mju = mujoco::sample_util;

using namespace std::chrono_literals;

// 常量定义
const int kDofFloatingBase = 6;        // 浮动基座的自由度数量

// 排除防护的接触对列表定义
// 排除防护的接触对：脚/踝与地面的接触不施加防护，避免走路时地面打滑
// 脚掌碰撞几何体挂在 LINK_ANKLE_ROLL_L/R 下，需排除；LINK_ANKLE_PITCH 为踝俯仰轴
const std::vector<std::pair<std::string, std::string>> SimManager::excluded_contact_pairs_ = {
  {"LINK_ANKLE_PITCH_L", "world"},
  {"LINK_ANKLE_PITCH_R", "world"},
  {"LINK_ANKLE_ROLL_L", "world"},   // 左脚掌接地接触
  {"LINK_ANKLE_ROLL_R", "world"}    // 右脚掌接地接触
};
const int kNumFloatingBaseJoints = 7;  // 浮动基座的关节数量（四元数 + xyz位置）
// 未收到关节命令时用于“保持当前姿态”的阻尼，避免启动瞬间零力矩自由下落
constexpr double kHoldDampingWhenNoCmd = 15.0;
constexpr double kSyncMisalign = 0.1;  // 重新同步前的最大偏差
constexpr double kSimRefreshFraction = 0.7;  // 可用于仿真的刷新率分数
const std::chrono::milliseconds kBusyWaitTime(1);  // 忙等待时间

namespace {

// 与 ros_interface.cc 相同：xmat 行主序，o = R^T v / o = R v
inline void RotT3(const double* M, const double* v, double* o) {
  o[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
  o[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
  o[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
}
inline void Rot3(const double* M, const double* v, double* o) {
  o[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
  o[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
  o[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

/** 绿球：当前接触点 → link 局部 → 标准姿态下世界坐标（与 ros_interface 一致） */
void GreenBallWorldPos(const mjModel* m, const mjData* d,
                       const std::vector<mjtNum>& std_xpos, const std::vector<mjtNum>& std_xmat,
                       int body1_id, int body2_id, const mjtNum* contact_pos_w, double out[3]) {
  const int link1 = body1_id != 0 ? body1_id : body2_id;
  if (link1 <= 0 || link1 >= m->nbody ||
      std_xpos.size() < static_cast<size_t>(3 * m->nbody) ||
      std_xmat.size() < static_cast<size_t>(9 * m->nbody)) {
    out[0] = contact_pos_w[0];
    out[1] = contact_pos_w[1];
    out[2] = contact_pos_w[2];
    return;
  }
  const mjtNum* x_LW = d->xpos + 3 * link1;
  const mjtNum* R_LW = d->xmat + 9 * link1;
  double pW_minus_x[3] = {contact_pos_w[0] - x_LW[0], contact_pos_w[1] - x_LW[1],
                          contact_pos_w[2] - x_LW[2]};
  double p_L[3];
  RotT3(R_LW, pW_minus_x, p_L);
  const mjtNum* x_LW_std = std_xpos.data() + 3 * link1;
  const mjtNum* R_LW_std = std_xmat.data() + 9 * link1;
  Rot3(R_LW_std, p_L, out);
  out[0] += x_LW_std[0];
  out[1] += x_LW_std[1];
  out[2] += x_LW_std[2];
}

/**
 * @brief 最大法向力对应接触点：keyframe（绿球）坐标 + 世界坐标 + map 厚度，供 NO-protect WARN
 */
void FillMaxContactWarnPose(const mjModel* m, mjData* d,
                            const std::vector<mjtNum>& std_xpos, const std::vector<mjtNum>& std_xmat,
                            int ncon, int idx_max_normal, const ProtectorMap* protector_map, bool use_protector_map,
                            double std_pose_out[3], double world_out[3], double* map_th_mm_out,
                            const char** name_a_out, const char** name_b_out) {
  std_pose_out[0] = std_pose_out[1] = std_pose_out[2] = 0.0;
  world_out[0] = world_out[1] = world_out[2] = 0.0;
  *map_th_mm_out = -1.0;
  *name_a_out = "?";
  *name_b_out = "?";
  if (idx_max_normal < 0 || idx_max_normal >= ncon || !m || !d) {
    return;
  }
  const mjContact& cmax = d->contact[idx_max_normal];
  int b1m = m->geom_bodyid[cmax.geom[0]];
  int b2m = m->geom_bodyid[cmax.geom[1]];
  *name_a_out = mj_id2name(m, mjOBJ_BODY, b1m);
  *name_b_out = mj_id2name(m, mjOBJ_BODY, b2m);
  world_out[0] = cmax.pos[0];
  world_out[1] = cmax.pos[1];
  world_out[2] = cmax.pos[2];
  GreenBallWorldPos(m, d, std_xpos, std_xmat, b1m, b2m, cmax.pos, std_pose_out);
  if (protector_map && protector_map->IsLoaded() && use_protector_map) {
    *map_th_mm_out = protector_map->LookupThickness(std_pose_out[0], std_pose_out[1], std_pose_out[2]);
  }
}

}  // namespace

#if defined(__linux__)
namespace {
constexpr size_t kPhysicsThreadStackBytes = 64U * 1024U * 1024U;
}
void* SimManager::PhysicsThreadTrampoline(void* arg) {
  auto* p = static_cast<std::pair<SimManager*, std::string>*>(arg);
  SimManager* self = p->first;
  std::string file = std::move(p->second);
  delete p;
  try {
    self->PhysicsThread(file);
  } catch (const std::exception& e) {
    std::cerr << "[mujoco_physics] PhysicsThread: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[mujoco_physics] PhysicsThread: unknown exception" << std::endl;
  }
  return nullptr;
}
#endif

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
 public:
  explicit CustomGlfwAdapter(bool visible = true) : mj::GlfwAdapter(visible) {}

 protected:
  /**
   * @brief 键盘事件处理函数
   * @param key 按键代码
   * @param scancode 扫描码
   * @param act 按键动作（按下/释放）
   * @param mods 修饰键状态（GLFW 传入），用于判断 Shift 是否按下，不依赖按键顺序
   * 
   * 实现推倒采样的键盘控制：
   * - Shift + W/S: 前后向干扰力
   * - Shift + A/D: 左右向干扰力
   * - Shift + Q/E/Z/C: 左前/右前/左后/右后（斜 45°）
   * - Shift + U/K: 上下向干扰力
   * - Shift + G/J: X轴干扰力矩
   * - Shift + Y/H: Y轴干扰力矩
   * - Shift + [/]: Z轴干扰力矩
   * - Shift + 0: 立即停止干扰力
   * - Shift + +/-: 调整干扰力大小
   * - Shift + ,/.: 调整干扰力持续时间
   */
  void OnKey(int key, int scancode, int act, int mods) override {
    // 首先调用父类的OnKey方法，保持原有功能
    mj::GlfwAdapter::OnKey(key, scancode, act, mods);

    // 使用 mods 判断 Shift：按下时 GLFW 会传入修饰键状态，不依赖先按 Shift 再按 W 的顺序
    const bool shift_held = (mods & GLFW_MOD_SHIFT) != 0;

    // 处理按下和释放事件
    SimManager& sim_manager = SimManager::GetInstance();
    
    if (act == GLFW_PRESS && shift_held) {
      // 处理按键按下事件（启动干扰力）
      switch (key) {
        case GLFW_KEY_W:  // Shift + W: 前向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("forward", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发前向干扰力 (推力可视化已启用)" << std::endl;
          }
          break;

        case GLFW_KEY_S:  // Shift + S: 后向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("backward", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发后向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_A:  // Shift + A: 左向干扰力
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

        case GLFW_KEY_D:  // Shift + D: 右向干扰力
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const auto& force_vec = sim_manager.GetPerturbationForce("right", current_force);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(force_vec[0], force_vec[1], force_vec[2]));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());  // 清除力矩
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发右向干扰力" << std::endl;
          }
          break;

        case GLFW_KEY_Q:  // Shift + Q: 左前（斜 45°）
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const double s = current_force / std::sqrt(2.0);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(s, s, 0));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发左前干扰力 (斜 45°)" << std::endl;
          }
          break;

        case GLFW_KEY_E:  // Shift + E: 右前（斜 45°）
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const double s = current_force / std::sqrt(2.0);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(s, -s, 0));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发右前干扰力 (斜 45°)" << std::endl;
          }
          break;

        case GLFW_KEY_Z:  // Shift + Z: 左后（斜 45°）
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const double s = current_force / std::sqrt(2.0);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(-s, s, 0));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发左后干扰力 (斜 45°)" << std::endl;
          }
          break;

        case GLFW_KEY_C:  // Shift + C: 右后（斜 45°）
          {
            double current_force = sim_manager.GetPerturbationForceMagnitude();
            const double s = current_force / std::sqrt(2.0);
            sim_manager.SetPerturbationForce(Eigen::Vector3d(-s, -s, 0));
            sim_manager.SetPerturbationTorque(Eigen::Vector3d::Zero());
            sim_manager.ApplyPerturbation(true);
            std::cout << "触发右后干扰力 (斜 45°)" << std::endl;
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

        case GLFW_KEY_K:  // Shift + K: 下向干扰力
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
        case GLFW_KEY_W:  // 释放W键：停止前向干扰力
        case GLFW_KEY_S:  // 释放S键：停止后向干扰力
        case GLFW_KEY_A:  // 释放A键：停止左向干扰力
        case GLFW_KEY_D:  // 释放D键：停止右向干扰力
        case GLFW_KEY_Q:  // 释放Q键：停止左前干扰力
        case GLFW_KEY_E:  // 释放E键：停止右前干扰力
        case GLFW_KEY_Z:  // 释放Z键：停止左后干扰力
        case GLFW_KEY_C:  // 释放C键：停止右后干扰力
        case GLFW_KEY_U:  // 释放U键：停止上向干扰力
        case GLFW_KEY_K:  // 释放K键：停止下向干扰力
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
    
#if defined(__linux__)
    if (physics_pthread_started_) {
      pthread_join(physics_pthread_, nullptr);
      physics_pthread_started_ = false;
    }
#else
    if (physics_thread_.joinable()) {
      physics_thread_.join();
    }
#endif
    
    // 清理MuJoCo资源（添加空指针检查）
    if (d_) {
      mj_deleteData(d_);
      d_ = nullptr;
    }
    
    if (m_) {
      mj_deleteModel(m_);
      m_ = nullptr;
    }

    if (node_) {
      LogProtectionSessionSummary();
    }
    
    // 清理ROS接口
    ros_interface_.reset();
    
  } catch (const std::exception& e) {
    std::cerr << "Exception during SimManager destruction: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception during SimManager destruction" << std::endl;
  }
}

double SimManager::GetContactProtectionScale(int contact_idx) const {
  if (!protection_enabled_) {
    return 1.0;
  }
  if (contact_idx < 0 ||
      contact_idx >= static_cast<int>(last_contact_protection_scales_.size())) {
    return 1.0;
  }
  return last_contact_protection_scales_[static_cast<size_t>(contact_idx)];
}

void SimManager::LogProtectionSessionSummary() {
  if (!node_) {
    return;
  }
  RCLCPP_INFO(
      node_->get_logger(),
      "护具会话汇总: 有接触且走护具逻辑的帧数=%" PRIu64 ", 至少成功缩放1处接触的帧数=%" PRIu64
      ", 累计接触点缩放次数=%" PRIu64 ", 最大法向力>1kN的帧数=%" PRIu64 ", >10kN的帧数=%" PRIu64
      ", 高力(>1kN)但未产生任何缩放的帧数=%" PRIu64,
      prot_sess_frames_with_contact_, prot_sess_frames_with_protection_, prot_sess_total_contact_scalings_,
      prot_sess_frames_max_gt_1kn_, prot_sess_frames_max_gt_10kn_, prot_sess_warn_no_protect_frames_);
}

/**
 * @brief 关节力矩控制器
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * @details 实现PD控制器和推倒采样的干扰力系统
 */
void SimManager::TorqueController(const mjModel* m, mjData* d) {
  bool is_floating_base = (m->nv != m->nu);

  if (ros_interface_) {
    auto cmd = ros_interface_->GetCommandedSafe();

    // 命令未就绪（CHR 等尚未发布第一条 JointCommand）时不能留空 ctrl，否则零力矩会自由下落
    bool cmd_ready = (cmd.position.size() >= static_cast<size_t>(m->nu) &&
                      cmd.velocity.size() >= static_cast<size_t>(m->nu) &&
                      cmd.feed_forward_torque.size() >= static_cast<size_t>(m->nu) &&
                      cmd.stiffness.size() >= static_cast<size_t>(m->nu) &&
                      cmd.damping.size() >= static_cast<size_t>(m->nu));

    if (cmd_ready) {
      for (int i = 0; i < m->nu; ++i) {
        double position = is_floating_base ? d->qpos[i + kNumFloatingBaseJoints] : d->qpos[i];
        double velocity = is_floating_base ? d->qvel[i + kDofFloatingBase] : d->qvel[i];
        double position_error = cmd.position[i] - position;
        double velocity_error = cmd.velocity[i] - velocity;
        d->ctrl[i] = cmd.feed_forward_torque[i] + cmd.stiffness[i] * position_error + cmd.damping[i] * velocity_error;
      }
    } else {
      // 未收到命令前：用阻尼稳住当前姿态，避免启动瞬间摔倒
      for (int i = 0; i < m->nu; ++i) {
        double velocity = is_floating_base ? d->qvel[i + kDofFloatingBase] : d->qvel[i];
        d->ctrl[i] = -kHoldDampingWhenNoCmd * velocity;
      }
    }
  } else {
    // 如果没有ROS接口，使用零力矩控制（机器人自由运动）
    for (int i = 0; i < m->nu; ++i) {
      d->ctrl[i] = 0.0;
    }
  }

  // ==================== 多干扰力系统 ====================
  // 注意：推力施加已移动到ApplyPerturbationForces函数中
  // 在UI系统清空xfrc_applied之后调用，确保推力不被清空
}

/**
 * @brief 施加干扰力 - 在UI系统清空外力后调用
 * @details 这个函数在UI系统清空xfrc_applied之后调用，
 *          确保推力不会被UI系统清空，从而保证推力的正确效果
 */
void SimManager::ApplyPerturbationForces() {
  if (!m_ || !d_) return;
  
  std::lock_guard<std::mutex> lock(perturbation_mutex_);
  
  // 自动采样逻辑 - 只触发一次
  if (config_loader_->GetAutoSampling() && !apply_perturb_) {
    // 如果已经完成，不再执行
    if (auto_completed_) {
      return;
    }
    
    if (!auto_triggered_) {
      auto_start_time_ = d_->time;
      auto_triggered_ = true;
    }
    
    // 等待指定时间后自动触发推力
    if (d_->time - auto_start_time_ >= config_loader_->GetAutoDelay()) {
      // 根据角度设置推力：0°=前(X+), 90°=左(Y+), 180°=后(X-), 270°=右(Y-)
      const double angle_rad = config_loader_->GetAutoDirectionAngle() * M_PI / 180.0;
      perturb_force_ = Eigen::Vector3d(
        perturb_force_magnitude_ * std::cos(angle_rad),
        perturb_force_magnitude_ * std::sin(angle_rad),
        0.0
      );
      
      // 应用推力
      ApplyPerturbation(true);
      auto_completed_ = true;  // 标记已完成，不再重复触发
    }
  }
  
  // 检查并清理过期的干扰力
  auto it = active_perturbations_.begin();
  while (it != active_perturbations_.end()) {
    if (d_->time - it->start_time > it->duration) {
      std::cout << "干扰力ID " << it->id << " 自动停止" << std::endl;
      // 从列表中删除过期的推力，避免CSV记录"幽灵"推力
      it = active_perturbations_.erase(it);
    } else {
      ++it;
    }
  }
  
  // 处理定时推力逻辑
  if (apply_perturb_) {
    // 检查是否超过持续时间
    if (perturb_start_time_ > 0 && (d_->time - perturb_start_time_) >= perturb_duration_) {
      // 时间到了，自动停止推力
      double actual_duration = d_->time - perturb_start_time_;
      apply_perturb_ = false;
      perturb_start_time_ = -1.0;
      
      std::cout << "=== 自动停止推力 ===" << std::endl;
      std::cout << "时间: " << std::fixed << std::setprecision(3) << d_->time << "s" << std::endl;
      std::cout << "实际持续时间: " << actual_duration << "s" << std::endl;
      std::cout << "===================" << std::endl;
      
      // 清除推力
      int body_id = mj_name2id(m_, mjOBJ_BODY, perturb_body_name_.c_str());
      if (body_id >= 0) {
        d_->xfrc_applied[6 * body_id + 0] = 0.0;
        d_->xfrc_applied[6 * body_id + 1] = 0.0;
        d_->xfrc_applied[6 * body_id + 2] = 0.0;
        d_->xfrc_applied[6 * body_id + 3] = 0.0;
        d_->xfrc_applied[6 * body_id + 4] = 0.0;
        d_->xfrc_applied[6 * body_id + 5] = 0.0;
      }
    } else {
      // 还在持续时间内，继续施加推力
      int body_id = mj_name2id(m_, mjOBJ_BODY, perturb_body_name_.c_str());
      if (body_id >= 0) {
        // 将干扰力从物体坐标系转换到世界坐标系
        mjtNum* body_quat = d_->xquat + 4 * body_id;
        mjtNum perturb_force_local[3] = {perturb_force_.x(), perturb_force_.y(), perturb_force_.z()};
        mjtNum perturb_force_world[3];
        mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);

        // 直接设置xfrc_applied - 定时推力
        d_->xfrc_applied[6 * body_id + 0] = perturb_force_world[0];
        d_->xfrc_applied[6 * body_id + 1] = perturb_force_world[1];
        d_->xfrc_applied[6 * body_id + 2] = perturb_force_world[2];
        
        // 施加干扰力矩到物体
        d_->xfrc_applied[6 * body_id + 3] = perturb_torque_.x();
        d_->xfrc_applied[6 * body_id + 4] = perturb_torque_.y();
        d_->xfrc_applied[6 * body_id + 5] = perturb_torque_.z();

        // 确保推力可视化被启用
        if (sim_ && sim_->opt.flags[mjVIS_PERTFORCE] == 0) {
          sim_->opt.flags[mjVIS_PERTFORCE] = 1;
        }
      }
    }
  } else {
    // 没有推力，确保清除xfrc_applied
    int body_id = mj_name2id(m_, mjOBJ_BODY, perturb_body_name_.c_str());
    if (body_id >= 0) {
      d_->xfrc_applied[6 * body_id + 0] = 0.0;
      d_->xfrc_applied[6 * body_id + 1] = 0.0;
      d_->xfrc_applied[6 * body_id + 2] = 0.0;
      d_->xfrc_applied[6 * body_id + 3] = 0.0;
      d_->xfrc_applied[6 * body_id + 4] = 0.0;
      d_->xfrc_applied[6 * body_id + 5] = 0.0;
    }
  }
  
  // 注意：旧的干扰力系统已被持续推力系统替代
  // 持续推力逻辑在上面已经处理
}

/**
 * @brief 全局函数：从SimManager施加推力
 * @details 这个函数被simulate.cc调用，在UI系统清空外力后重新施加推力
 */
void ApplyPerturbationForcesFromSimManager() {
  SimManager::GetInstance().ApplyPerturbationForces();
}

/**
 * @brief 全局函数：获取接触力可视化设置
 * @return 是否启用接触力可视化
 * @details 这个函数被simulate.cc调用，用于获取配置文件中的接触力可视化设置
 */
bool IsContactVisualizationEnabled() {
  return SimManager::GetInstance().GetConfigLoader()->IsContactVisualizationEnabled();
}

/**
 * @brief 重置auto_sampling状态
 * @details 用于MuJoCo重置后重新触发推力施加
 */
void SimManager::ResetAutoSampling() {
  std::lock_guard<std::mutex> lock(perturbation_mutex_);
  
  // 重置auto_sampling状态变量
  auto_triggered_ = false;
  auto_start_time_ = -1.0;
  auto_completed_ = false;
  
  // 停止当前的推力
  apply_perturb_ = false;
  perturb_start_time_ = -1.0;
  
  // 清除所有外力
  if (m_ && d_) {
    mju_zero(d_->xfrc_applied, 6 * m_->nbody);
  }
  
  // 清除所有推力记录，避免CSV记录"幽灵"推力
  active_perturbations_.clear();
  
  std::cout << "Auto-sampling状态已重置，推力将在auto_delay时间后重新施加" << std::endl;
}

void SimManager::RequestReset() {
  if (sim_) {
    sim_->pending_.reset = true;
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
  
  // 从YAML配置加载推力参数
  perturb_force_magnitude_ = GetDefaultForceMagnitude();
  perturb_torque_magnitude_ = GetDefaultTorqueMagnitude();
  perturb_duration_ = config_loader_->GetForceDuration();
  RCLCPP_INFO(logger, "Loaded perturbation parameters from YAML: force=%.1fN, torque=%.1fN.m, duration=%.1fs", 
              perturb_force_magnitude_, perturb_torque_magnitude_, perturb_duration_);

  // 从YAML配置加载防护功能参数
  protection_enabled_ = config_loader_->IsProtectionEnabled();
  protection_thickness_ = config_loader_->GetProtectionThickness();
  use_protector_map_ = config_loader_->UseProtectorMap();
  force_method_ = config_loader_->GetForceMethod();
  protection_density_ = config_loader_->GetProtectionDensity();
  RCLCPP_INFO(logger, "Protection from YAML config: enabled=%s, thickness=%.1fmm, use_protector_map=%s, force_method=%s",
              protection_enabled_ ? "YES" : "NO", protection_thickness_, use_protector_map_ ? "YES" : "NO", force_method_.c_str());

  // 初始化力插值计算器（用于柔性护具防护力计算）
  try {
    // 尝试多个可能的路径来找到RT-FEM.tsv文件
    std::string tsv_path;
    std::vector<std::string> possible_paths = {
      // 源代码目录
      "/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/RT-FEM.tsv",
      // 相对于当前工作目录
      "scripts/ThicknessCalculate/RT-FEM.tsv",
      "../scripts/ThicknessCalculate/RT-FEM.tsv",
      "../../scripts/ThicknessCalculate/RT-FEM.tsv",
      // 安装目录（可能不存在）
      assets_path + "/scripts/ThicknessCalculate/RT-FEM.tsv",
    };
    
    // 检查文件是否存在
    auto file_exists = [](const std::string& path) -> bool {
      struct ::stat buffer;
      return (::stat(path.c_str(), &buffer) == 0);
    };
    
    bool found = false;
    for (const auto& path : possible_paths) {
      if (file_exists(path)) {
        tsv_path = path;
        found = true;
        break;
      }
    }
    
    // 如果找不到，让ForceInterpolation使用默认路径查找
    if (!found) {
      RCLCPP_WARN(logger, "RT-FEM.tsv not found in common paths, using default search");
      tsv_path = "";  // 空字符串让ForceInterpolation自己查找
    }
    
    if (force_method_ == "chr") {
      // CHR: 使用 fitted_parameters 公式 F_after=C*t^α*p^β*F_before^γ
      std::vector<std::string> params_candidates = {
        assets_path + "/../../../../scripts/ThicknessCalculate/fitted_parameters.json",   // source 布局
        assets_path + "/../../../../../scripts/ThicknessCalculate/fitted_parameters.json",  // install 布局
      };
      std::string params_path;
      for (const auto& cand : params_candidates) {
        struct ::stat st;
        if (::stat(cand.c_str(), &st) == 0) {
          params_path = cand;
          break;
        }
      }
      if (params_path.empty()) params_path = params_candidates[0];
      chr_zzq_force_ = std::make_unique<ChrZzqForce>(params_path);
      if (!chr_zzq_force_->IsLoaded()) {
        chr_zzq_force_ = std::make_unique<ChrZzqForce>("");  // 尝试默认路径
      }
      if (chr_zzq_force_->IsLoaded()) {
        RCLCPP_INFO(logger, "CHR force method initialized (fitted_parameters formula), density=%.2f", protection_density_);
      } else {
        RCLCPP_WARN(logger, "CHR init failed, falling back to ZZQ (RT-FEM table)");
        force_method_ = "zzq";
        chr_zzq_force_.reset();
      }
    }
    if (force_method_ == "zzq") {
      // ZZQ: 使用 RT-FEM 表查表
      force_interpolator_ = std::make_unique<ForceInterpolation>(tsv_path);
    }
    
    // 获取力插值范围信息（用于日志）
    std::pair<double, double> force_range{0, 0}, thickness_range{0, 0};
    if (force_interpolator_) {
      force_range = force_interpolator_->GetForceRange();
      thickness_range = force_interpolator_->GetThicknessRange();
    } else if (chr_zzq_force_) {
      force_range = chr_zzq_force_->GetForceRange();
      thickness_range = chr_zzq_force_->GetThicknessRange();
    }
    
    // 初始化护具地图（当 use_protector_map 时，路径写死，支持 source 与 install 两种布局）
    if (protection_enabled_ && use_protector_map_) {
      std::vector<std::string> map_dir_candidates = {
        assets_path + "/../../../../interface_example/share/interface_example/config/pm01/rl_basic/basic/protector_map",  // install 布局
        assets_path + "/../../../../../src/interface_example/config/pm01/rl_basic/basic/protector_map",  // source 布局
      };
      auto dir_exists = [](const std::string& p) {
        struct ::stat buffer;
        return (::stat(p.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
      };
      std::string map_dir;
      for (const auto& cand : map_dir_candidates) {
        if (dir_exists(cand)) {
          map_dir = cand;
          break;
        }
      }
      if (map_dir.empty()) {
        map_dir = map_dir_candidates[0];  // 默认尝试第一个
      }
      try {
        std::string front_tsv = config_loader_->GetProtectorMapFront();
        std::string back_tsv = config_loader_->GetProtectorMapBack();
        protector_map_ = std::make_unique<ProtectorMap>(map_dir, front_tsv, back_tsv);
        if (protector_map_->IsLoaded()) {
          RCLCPP_INFO(logger, "Protector map loaded from: %s (front=%s, back=%s)", map_dir.c_str(),
                      front_tsv.c_str(), back_tsv.c_str());
        } else {
          RCLCPP_WARN(logger, "Protector map failed to load, falling back to global thickness");
          protector_map_.reset();
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN(logger, "Protector map init failed: %s, falling back to global thickness", e.what());
        protector_map_.reset();
      }
    }

    RCLCPP_INFO(logger, "=== Protection Feature Initialized ===");
    RCLCPP_INFO(logger, "Protection enabled: %s", protection_enabled_ ? "YES" : "NO");
    RCLCPP_INFO(logger, "Protection thickness: %.1f mm (fallback when no map)", protection_thickness_);
    if (force_method_ == "zzq") {
      RCLCPP_INFO(logger, "RT-FEM data file: %s", tsv_path.c_str());
    }
    RCLCPP_INFO(logger, "Force range: [%.1f, %.1f] kN", force_range.first, force_range.second);
    RCLCPP_INFO(logger, "Thickness range: [%.1f, %.1f] mm", thickness_range.first, thickness_range.second);
    RCLCPP_INFO(logger, "Excluded contact pairs: %zu", excluded_contact_pairs_.size());
    for (const auto& pair : excluded_contact_pairs_) {
      RCLCPP_INFO(logger, "  - %s <-> %s", pair.first.c_str(), pair.second.c_str());
    }
    RCLCPP_INFO(logger, "=====================================");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Failed to initialize force method: %s", e.what());
    RCLCPP_ERROR(logger, "Protection feature will be DISABLED!");
    protection_enabled_ = false;
    force_interpolator_.reset();
    chr_zzq_force_.reset();
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
  opt.flags[mjVIS_CONTACTFORCE] = config_loader_->IsContactVisualizationEnabled() ? 1 : 0;  // 根据配置文件决定是否显示接触力
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
  if (headless_) {
    RCLCPP_INFO(logger, "Headless mode enabled: creating hidden GLFW window and skipping per-frame rendering");
  }
  sim_ = std::make_unique<mj::Simulate>(std::make_unique<CustomGlfwAdapter>(!headless_), &cam, &opt, &pert, false, headless_);

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
  if (apply) {
    // 按下键盘：开始定时推力
    perturb_start_time_ = d_->time;  // 记录开始时间
    apply_perturb_ = true;
    
    // 计算推力大小
    double force_magnitude = sqrt(perturb_force_.x()*perturb_force_.x() + 
                                 perturb_force_.y()*perturb_force_.y() + 
                                 perturb_force_.z()*perturb_force_.z());
    double torque_magnitude = sqrt(perturb_torque_.x()*perturb_torque_.x() + 
                                  perturb_torque_.y()*perturb_torque_.y() + 
                                  perturb_torque_.z()*perturb_torque_.z());
    
    // 将推力添加到active_perturbations_列表中，用于CSV记录
    PerturbationData new_perturbation;
    new_perturbation.id = GetNextPerturbationId();
    new_perturbation.body_name = perturb_body_name_;
    new_perturbation.force = perturb_force_;
    new_perturbation.torque = perturb_torque_;
    new_perturbation.start_time = d_->time;
    new_perturbation.duration = perturb_duration_;
    new_perturbation.is_active = true;
    
    active_perturbations_.push_back(new_perturbation);
    
    std::cout << "=== 开始定时推力 ===" << std::endl;
    std::cout << "当前active_perturbations_数量: " << active_perturbations_.size() << std::endl;
    std::cout << "时间: " << std::fixed << std::setprecision(3) << d_->time << "s" << std::endl;
    std::cout << "推力大小: " << force_magnitude << "N" << std::endl;
    std::cout << "推力方向分量: [" << perturb_force_.x() << ", " << perturb_force_.y() << ", " << perturb_force_.z() << "]" << std::endl;
    if (torque_magnitude > 0.001) {
      std::cout << "力矩大小: " << torque_magnitude << "N.m" << std::endl;
      std::cout << "力矩方向: [" << perturb_torque_.x() << ", " << perturb_torque_.y() << ", " << perturb_torque_.z() << "]" << std::endl;
    }
    std::cout << "目标物体: " << perturb_body_name_ << std::endl;
    std::cout << "持续时间: " << perturb_duration_ << "s" << std::endl;
    std::cout << "推力ID: " << new_perturbation.id << std::endl;
    std::cout << "===================" << std::endl;
  } else {
    // 松开键盘：立即停止推力
    if (apply_perturb_) {
      double actual_duration = (perturb_start_time_ > 0) ? (d_->time - perturb_start_time_) : 0.0;
      apply_perturb_ = false;
      perturb_start_time_ = -1.0;  // 重置开始时间
      
      // 标记所有活跃的推力为非活跃状态
      for (auto& pert : active_perturbations_) {
        if (pert.is_active) {
          pert.is_active = false;
        }
      }
      
      std::cout << "=== 手动停止推力 ===" << std::endl;
      std::cout << "时间: " << std::fixed << std::setprecision(3) << d_->time << "s" << std::endl;
      std::cout << "实际持续时间: " << actual_duration << "s" << std::endl;
      std::cout << "===================" << std::endl;
    }
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
void SimManager::RequestExitFromSignal() {
  if (sim_) {
    sim_->exitrequest.store(1);
  }
}

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
#if defined(__linux__)
  auto* pay = new std::pair<SimManager*, std::string>(this, std::string(model_file));
  pthread_attr_t pattr;
  pthread_attr_init(&pattr);
  if (pthread_attr_setstacksize(&pattr, kPhysicsThreadStackBytes) != 0) {
    RCLCPP_WARN(logger, "pthread_attr_setstacksize(%zu) failed, using default thread stack",
                kPhysicsThreadStackBytes);
  }
  const int perr = pthread_create(&physics_pthread_, &pattr, &PhysicsThreadTrampoline, pay);
  pthread_attr_destroy(&pattr);
  if (perr != 0) {
    delete pay;
    RCLCPP_ERROR(logger, "pthread_create(physics) failed: %d", perr);
    return;
  }
  physics_pthread_started_ = true;
  RCLCPP_INFO(logger, "Physics thread stack size: %zu MB (Linux pthread)",
              kPhysicsThreadStackBytes / (1024U * 1024U));
#else
  physics_thread_ = std::thread([this, model_file]() { PhysicsThread(model_file); });
#endif

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
    
    // 计算关节反力
    ComputeJointForces();

    RefreshStandardPoseCache();
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
    
    // 计算关节反力
    ComputeJointForces();

    RefreshStandardPoseCache();
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
          
          // 获取当前CPU时间戳，用于计算实际经过的时间
          const auto startCPU = mj::Simulate::Clock::now();
          
          // 计算自上次同步以来CPU实际经过的时间
          const auto elapsedCPU = startCPU - syncCPU;
          
          // 计算自上次同步以来仿真时间的变化
          double elapsedSim = d_->time - syncSim;
          
          // 计算时间缩放因子：100%实时 = 1.0，50%实时 = 0.5，200%实时 = 2.0
          // 例如：percentRealTime[real_time_index] = 50 时，slowdown = 100/50 = 2.0
          // 意味着仿真运行速度是实时的2倍（即仿真时间流逝比实际时间快2倍）
          double slowdown = 100 / sim_->percentRealTime[sim_->real_time_index];
          
          // 检查CPU时间和仿真时间是否同步
          // 如果两者差异超过阈值，则认为时间不同步，需要重新同步
          // 公式：|(CPU时间 / 缩放因子) - 仿真时间| > 同步容差
          bool misaligned = std::abs((elapsedCPU / slowdown).count() - elapsedSim) > kSyncMisalign;

          // 检查是否需要重新同步时间基准点
          // 重新同步的条件：
          // 1. elapsedSim < 0: 仿真时间异常（可能发生了重置）
          // 2. elapsedCPU.count() < 0: CPU时间异常
          // 3. syncCPU.time_since_epoch().count() == 0: 首次运行，同步点未初始化
          // 4. misaligned: CPU时间和仿真时间不同步
          // 5. sim_->speed_changed: 用户改变了仿真速度设置
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim_->speed_changed) {
            // 重新设置同步基准点
            syncCPU = startCPU;        // 重置CPU时间基准
            syncSim = d_->time;        // 重置仿真时间基准
            sim_->speed_changed = false;  // 清除速度变化标志

            // 根据防护功能是否启用，选择不同的步进方式
            if (protection_enabled_) {
              // 防护功能启用：使用mj_step1和mj_step2分离，以便在中间应用防护
              // mj_step1: 计算位置、速度，调用控制回调
              mj_step1(m_, d_);
              
              // 在mj_step1之后，mj_step2之前，施加推力（确保在UI系统清空外力之后）
              ApplyPerturbationForces();
              
              // mj_step2: 计算执行器力、加速度、约束力，并进行时间积分
              // 注意：我们需要手动调用mj_step2的内部步骤，以便在mj_fwdConstraint之后应用防护
              mj_fwdActuation(m_, d_);
              mj_fwdAcceleration(m_, d_);
              mj_fwdConstraint(m_, d_);
              
              // 在约束求解之后，应用防护到接触力
              ApplyProtectionToContactForces();
              
              // 继续mj_step2的剩余步骤
              mj_sensorAcc(m_, d_);
              mj_checkAcc(m_, d_);
              
              // 比较前向和逆向解（如果启用）
              if (m_->opt.enableflags & mjENBL_FWDINV) {
                mj_compareFwdInv(m_, d_);
              }
              
              // 时间积分（使用Euler积分，因为mj_step1/mj_step2只支持Euler）
              if (m_->opt.integrator == mjINT_IMPLICIT || m_->opt.integrator == mjINT_IMPLICITFAST) {
                mj_implicit(m_, d_);
              } else {
                mj_Euler(m_, d_);
              }
            } else {
              // 防护功能禁用：使用标准的mj_step，性能更好
              mj_step1(m_, d_);
              ApplyPerturbationForces();
              mj_step2(m_, d_);
            }
            
            // 计算关节反力
            ComputeJointForces();
            
            // 更新ROS接口状态，发送最新的仿真数据
            ros_interface_->UpdateSimState(m_, d_);
            
            // 检查仿真是否发散（数值不稳定）
            const char* message = Diverged(m_->opt.disableflags, d_);
            if (message) {
              // 如果发散，停止仿真并记录错误信息
              sim_->run = 0;
              mju::strcpy_arr(sim_->load_error, message);
            } else {
              // 仿真正常，标记为已步进
              stepped = true;
            }
          } else {
            // 时间同步正常，执行多步仿真以保持实时性
            // 当CPU时间领先于仿真时间时，需要执行多个仿真步进来追赶
            bool measured = false;           // 是否已测量实际运行速度
            mjtNum prevSim = d_->time;       // 记录当前仿真时间，用于检测时间倒退
            double refreshTime = kSimRefreshFraction / sim_->refresh_rate;  // 计算刷新时间间隔

            // 多步仿真循环条件：
            // 1. 仿真时间落后于CPU时间：需要追赶
            // 2. 单次循环时间不超过刷新间隔：避免阻塞UI
            while ((d_->time - syncSim) * slowdown < (mj::Simulate::Clock::now() - syncCPU).count() &&
                   mj::Simulate::Clock::now() - startCPU < refreshTime * 1s) {
              
              // 测量实际运行速度（仅测量一次）
              if (!measured && elapsedSim) {
                // 计算实际运行速度：CPU时间 / 仿真时间
                // 这个值用于显示实际性能，帮助用户了解仿真是否按预期速度运行
                sim_->measured_slowdown = elapsedCPU.count() / elapsedSim;
                measured = true;
              }

              // 注入噪声（如果启用）：模拟传感器噪声、环境扰动等
              sim_->InjectNoise();
              
              // 根据防护功能是否启用，选择不同的步进方式
              if (protection_enabled_) {
                // 防护功能启用：使用mj_step1和mj_step2分离，以便在中间应用防护
                // mj_step1: 计算位置、速度，调用控制回调
                mj_step1(m_, d_);
                
                // 在mj_step1之后，mj_step2之前，施加推力（确保在UI系统清空外力之后）
                ApplyPerturbationForces();
                
                // mj_step2: 计算执行器力、加速度、约束力，并进行时间积分
                // 注意：我们需要手动调用mj_step2的内部步骤，以便在mj_fwdConstraint之后应用防护
                mj_fwdActuation(m_, d_);
                mj_fwdAcceleration(m_, d_);
                mj_fwdConstraint(m_, d_);
                
                // 在约束求解之后，应用防护到接触力
                ApplyProtectionToContactForces();
                
                // 继续mj_step2的剩余步骤
                mj_sensorAcc(m_, d_);
                mj_checkAcc(m_, d_);
                
                // 比较前向和逆向解（如果启用）
                if (m_->opt.enableflags & mjENBL_FWDINV) {
                  mj_compareFwdInv(m_, d_);
                }
                
                // 时间积分（使用Euler积分，因为mj_step1/mj_step2只支持Euler）
                if (m_->opt.integrator == mjINT_IMPLICIT || m_->opt.integrator == mjINT_IMPLICITFAST) {
                  mj_implicit(m_, d_);
                } else {
                  mj_Euler(m_, d_);
                }
              } else {
                // 防护功能禁用：使用标准的mj_step，性能更好
                mj_step1(m_, d_);
                ApplyPerturbationForces();
                mj_step2(m_, d_);
              }
              
              // 计算关节反力
              ComputeJointForces();
              
              // 更新ROS接口状态，发送最新的仿真数据
              ros_interface_->UpdateSimState(m_, d_);
              
              // 检查仿真是否发散（数值不稳定）
              const char* message = Diverged(m_->opt.disableflags, d_);
              if (message) {
                // 如果发散，停止仿真并记录错误信息
                sim_->run = 0;
                mju::strcpy_arr(sim_->load_error, message);
              } else {
                // 仿真正常，标记为已步进
                stepped = true;
              }

              // 检测时间倒退：如果当前仿真时间小于之前记录的时间
              // 这通常表示发生了时间重置或异常情况，需要退出循环
              if (d_->time < prevSim) {
                break;
              }
            }
          }

          // 如果执行了仿真步进，将当前状态添加到历史记录中
          // 历史记录用于时间回放、状态恢复等功能
          if (stepped) {
            sim_->AddToHistory();
          }
        } else {
          // 仿真暂停状态：不推进时间，只更新当前状态的计算
          // 这确保用户交互（如拖拽、施加外力）和可视化能正常工作
          mj_forward(m_, d_);
          
          // 计算关节反力（即使暂停时也更新，以便可视化）
          ComputeJointForces();
          
          sim_->speed_changed = true;  // 标记速度已改变，下次运行时需要重新同步
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
      sim_->opt.flags[mjVIS_CONTACTFORCE] = config_loader_->IsContactVisualizationEnabled() ? 1 : 0;  // 根据配置文件决定是否显示接触力
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
      
      // 计算初始关节反力
      ComputeJointForces();

      // 护具 map 绿球坐标与 ros_interface 共用同一套标准姿态缓存（物理线程内，非 TF）
      RefreshStandardPoseCache();
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
    // 先排空接触/扰动异步写入并 flush，再清指针（避免退出时 bin 未刷盘或与写入线程竞态引发栈破坏）
    if (ros_interface_) {
      ros_interface_->DrainBinaryWritersAndFlush();
      ros_interface_->SetModelAndData(nullptr, nullptr);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception during physics thread cleanup: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(logger, "Unknown exception during physics thread cleanup");
  }
}

void SimManager::RefreshStandardPoseCache() {
  if (!m_) {
    return;
  }
  mjData* dstd = mj_makeData(m_);
  if (!dstd) {
    if (node_) {
      RCLCPP_WARN(node_->get_logger(), "RefreshStandardPoseCache: mj_makeData failed");
    }
    return;
  }
  const int key_id = mj_name2id(m_, mjOBJ_KEY, "floating_base_homing");
  if (key_id >= 0) {
    mj_resetDataKeyframe(m_, dstd, key_id);
  } else {
    mju_zero(dstd->qpos, m_->nq);
    if (m_->nq > 0 && m_->jnt_type[0] == mjJNT_FREE) {
      dstd->qpos[0] = 1;
      dstd->qpos[1] = dstd->qpos[2] = dstd->qpos[3] = 0;
      dstd->qpos[4] = dstd->qpos[5] = dstd->qpos[6] = 0;
    }
  }
  mj_forward(m_, dstd);
  if (m_->nbody > 0) {
    std_xpos_cache_.assign(dstd->xpos, dstd->xpos + 3 * m_->nbody);
    std_xmat_cache_.assign(dstd->xmat, dstd->xmat + 9 * m_->nbody);
  } else {
    std_xpos_cache_.clear();
    std_xmat_cache_.clear();
  }
  std_pose_model_ptr_ = m_;
  mj_deleteData(dstd);
}

/**
 * @brief 应用防护到接触力
 * @details 在mj_fwdConstraint之后调用，根据防护材料厚度减少接触力
 *          该函数会遍历所有接触点，计算每个接触点的力大小，
 *          使用ForceInterpolation计算防护后的力，然后按比例缩放efc_force
 */
void SimManager::ApplyProtectionToContactForces() {
  if (!m_ || !d_) {
    return;
  }
  
  // 检查防护功能是否启用
  if (!protection_enabled_) {
    static bool warned_once = false;
    if (!warned_once && node_) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000, 
                          "Protection feature is DISABLED");
      warned_once = true;
    }
    return;
  }
  
  if (!force_interpolator_ && !chr_zzq_force_) {
    static bool warned_once = false;
    if (!warned_once && node_) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000, 
                          "Force method (CHR/ZZQ) is not initialized - protection disabled");
      warned_once = true;
    }
    return;
  }
  
  // 获取当前接触数量
  int ncon = d_->ncon;
  // 对每个接触索引做一阶低通，减少 scale 帧间跳变导致的接触抖动/远滑
  static std::vector<double> filtered_scales_prev;
  if (ncon == 0) {
    last_contact_protection_scales_.clear();
    filtered_scales_prev.clear();
    return;
  }
  if (filtered_scales_prev.size() != static_cast<size_t>(ncon)) {
    filtered_scales_prev.assign(static_cast<size_t>(ncon), 1.0);
  }
  last_contact_protection_scales_.assign(static_cast<size_t>(ncon), 1.0);
  prot_sess_frames_with_contact_++;

  // 护具 map 与 ros_interface 绿球一致：每帧按需保证标准姿态缓存与当前 m_ 一致（物理线程内，与 TF 无关）
  if (use_protector_map_ && protector_map_ && protector_map_->IsLoaded()) {
    if (std_pose_model_ptr_ != m_ || std_xpos_cache_.size() != static_cast<size_t>(3 * m_->nbody)) {
      RefreshStandardPoseCache();
    }
  }
  
  // 每1000帧打印一次，确认防护功能被调用
  static int call_count = 0;
  call_count++;
  if (call_count % 1000 == 0 && node_) {
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                        "ApplyProtectionToContactForces called: ncon=%d", ncon);
  }
  
  // 对于高接触数量，立即记录（用于调试，但只记录前几次）
  // if (ncon > 5 && call_count <= 10) {
  //   RCLCPP_INFO(node_->get_logger(), 
  //               "ApplyProtectionToContactForces: ncon=%d, protection_enabled=%s, thickness=%.1fmm",
  //               ncon, protection_enabled_ ? "YES" : "NO", protection_thickness_);
  // }
  
  // 统计信息
  int protected_contacts = 0;
  int skipped_small_force = 0;
  int skipped_out_of_range = 0;
  int skipped_no_effect = 0;
  int skipped_thickness_below_min = 0;
  /** 已通过厚度与 CHR，但 efc_address 无效或 pyramid 写入越界，未改 efc */
  int skipped_efc_unavailable = 0;
  double total_force_reduction = 0.0;
  double max_force_before = 0.0;
  double max_force_after = 0.0;
  /** 非排除接触中、法向力最大的接触索引，供 WARN 打印 keyframe 坐标 */
  int idx_max_normal = -1;
  static int frame_count = 0;  // 用于调试输出频率控制

  // 遍历所有接触点
  for (int i = 0; i < ncon; ++i) {
    const mjContact& contact = d_->contact[i];
    
    // 获取接触体的名称，检查是否在排除列表中
    int body1_id = m_->geom_bodyid[contact.geom[0]];
    int body2_id = m_->geom_bodyid[contact.geom[1]];
    
    const char* body1_name = mj_id2name(m_, mjOBJ_BODY, body1_id);
    const char* body2_name = mj_id2name(m_, mjOBJ_BODY, body2_id);
    
    // 如果body没有名称，使用ID作为名称
    std::string name1 = body1_name ? body1_name : "body" + std::to_string(body1_id);
    std::string name2 = body2_name ? body2_name : "body" + std::to_string(body2_id);
    
    // 检查是否在排除列表中（顺序无关）
    bool is_excluded = false;
    for (const auto& excluded_pair : excluded_contact_pairs_) {
      if ((excluded_pair.first == name1 && excluded_pair.second == name2) ||
          (excluded_pair.first == name2 && excluded_pair.second == name1)) {
        is_excluded = true;
        break;
      }
    }
    
    // 如果在排除列表中，跳过防护处理
    if (is_excluded) {
      continue;
    }
    
    // 获取接触坐标系下的6D接触力 [fx, fy, fz, tx, ty, tz]
    mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mj_contactForce(m_, d_, i, contact_force);
    
    // 计算接触坐标系下的法向力大小（接触系x方向，正值，即正压力）
    double contact_force_normal = std::max(0.0, static_cast<double>(contact_force[0]));
    
    // 记录最大力（用于调试）及对应接触，便于与 protector map（keyframe 坐标）对照
    if (contact_force_normal > max_force_before) {
      max_force_before = contact_force_normal;
      idx_max_normal = i;
    }
    
    // 如果法向力太小，跳过（避免对微小接触力进行插值）
    // 阈值 500N：躺着时体重分布的力通常 < 500N，不处理；摔倒冲击力 > 500N 才做防护
    if (contact_force_normal < 500) {
      skipped_small_force++;
      continue;
    }
    
    // 根据碰撞点位置确定护具厚度：护具地图查表（绿球坐标 = 标准姿态下世界系，与 ros_interface 一致）或全局厚度
    double thickness_mm = protection_thickness_;
    double map_lookup_pos[3] = {contact.pos[0], contact.pos[1], contact.pos[2]};
    if (protector_map_ && protector_map_->IsLoaded() && use_protector_map_) {
      GreenBallWorldPos(m_, d_, std_xpos_cache_, std_xmat_cache_, body1_id, body2_id, contact.pos,
                        map_lookup_pos);
      thickness_mm = protector_map_->LookupThickness(map_lookup_pos[0], map_lookup_pos[1], map_lookup_pos[2]);
    }
    // 厚度 < 6mm 时，不进行衰减计算（chr 与 zzq 均要求厚度 >= 6）
    if (thickness_mm < 6.0) {
      skipped_thickness_below_min++;
      continue;
    }

    // 将力从N转换为kN
    double force_unprotected_kN = contact_force_normal / 1000.0;
    
    // 根据 force_method 计算防护后的力
    double force_protected_kN;
    try {
      bool input_valid = false;
      if (force_method_ == "chr" && chr_zzq_force_) {
        input_valid = chr_zzq_force_->IsInputValid(force_unprotected_kN, thickness_mm);
      } else if (force_interpolator_) {
        input_valid = force_interpolator_->IsInputValid(force_unprotected_kN, thickness_mm);
      }
      if (!input_valid) {
        // 如果超出范围，跳过该接触点
        skipped_out_of_range++;
        // 对于超出范围的力，立即记录调试信息（不依赖frame_count）
        // 对于超过1kN的力，总是打印（不限制次数）
        if (contact_force_normal > 1000) {
          double std_pose_log[3];
          GreenBallWorldPos(m_, d_, std_xpos_cache_, std_xmat_cache_, body1_id, body2_id, contact.pos, std_pose_log);
          auto force_range = force_interpolator_ ? force_interpolator_->GetForceRange()
                                                : chr_zzq_force_->GetForceRange();
          RCLCPP_WARN(node_->get_logger(),
                      "Force %.2f kN (%.2f N) out of range for thickness %.1f mm. "
                      "std_pose_xyz=%.4f,%.4f,%.4f (keyframe/green, m), world_pos=%.4f,%.4f,%.4f. "
                      "Valid range: [%.2f, %.2f] kN",
                      force_unprotected_kN, contact_force_normal, thickness_mm,
                      std_pose_log[0], std_pose_log[1], std_pose_log[2],
                      contact.pos[0], contact.pos[1], contact.pos[2],
                      force_range.first, force_range.second);
        } else if (contact_force_normal > 20000) {
          static int out_of_range_count = 0;
          out_of_range_count++;
          if (out_of_range_count <= 5) {
            auto force_range = force_interpolator_ ? force_interpolator_->GetForceRange()
                                                  : chr_zzq_force_->GetForceRange();
            RCLCPP_WARN(node_->get_logger(),
                        "Force %.2f kN (%.2f N) out of range for thickness %.1f mm. "
                        "Valid range: [%.2f, %.2f] kN",
                        force_unprotected_kN, contact_force_normal, thickness_mm,
                        force_range.first, force_range.second);
          }
        }
        continue;
      }
      if (force_method_ == "chr" && chr_zzq_force_) {
        double density = use_protector_map_ && protector_map_ && protector_map_->IsLoaded()
                            ? protector_map_->GetDensity()
                            : protection_density_;
        force_protected_kN = chr_zzq_force_->GetProtectedForce(force_unprotected_kN, thickness_mm, density);
      } else {
        force_protected_kN = force_interpolator_->GetProtectedForce(force_unprotected_kN, thickness_mm);
      }
    } catch (const std::exception& e) {
      // 如果插值失败，跳过该接触点
      skipped_out_of_range++;
      if (contact_force_normal > 20000) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                            "Interpolation failed for force %.2f kN: %s",
                            force_unprotected_kN, e.what());
      }
      continue;
    }
    
    // 计算缩放因子：防护后的力 / 原始力
    double scale_factor = force_protected_kN / force_unprotected_kN;
    
    // 确保缩放因子在合理范围内（0到1之间，因为防护应该减少力）
    scale_factor = std::max(0.0, std::min(1.0, scale_factor));
    
    // CHR 公式在中小冲击力下衰减过强，可能导致约束力不足、地面穿透。
    // 使用“线性过渡”的最小 scale，减少阈值附近突变：
    // - <=500N:       min_scale = 1.0
    // - 500N~2000N:   min_scale 线性从 1.0 -> 0.4
    // - 2000N~3500N:  min_scale 线性从 0.4 -> 0.1
    // - >3500N:       min_scale = 0.1
    if (force_method_ == "chr" && chr_zzq_force_) {
      constexpr double kChrMinScaleHighForce = 0.1;   // 高冲击下限
      constexpr double kChrMinScaleMidForce = 0.5;    // 中冲击下限
      constexpr double kChrMinScaleLowForce = 1.0;    // 低冲击下限（等效不缩放）
      constexpr double kN1 = 500.0;                   // 低->中过渡起点
      constexpr double kN2 = 2000.0;                  // 中->高过渡拐点
      constexpr double kN3 = 10000.0;                  // 中->高过渡终点

      double chr_min_scale = kChrMinScaleHighForce;
      if (contact_force_normal <= kN1) {
        chr_min_scale = kChrMinScaleLowForce;
      } else if (contact_force_normal <= kN2) {
        const double t = (contact_force_normal - kN1) / (kN2 - kN1);
        chr_min_scale = kChrMinScaleLowForce +
                        t * (kChrMinScaleMidForce - kChrMinScaleLowForce);
      } else if (contact_force_normal <= kN3) {
        const double t = (contact_force_normal - kN2) / (kN3 - kN2);
        chr_min_scale = kChrMinScaleMidForce +
                        t * (kChrMinScaleHighForce - kChrMinScaleMidForce);
      } else {
        chr_min_scale = kChrMinScaleHighForce;
      }
      if (scale_factor < chr_min_scale) {
        scale_factor = chr_min_scale;
      }
    }

    // 双速率低通：
    // - scale 下降（防护增强）时：快跟随，避免冲击峰值压不住
    // - scale 上升（防护减弱）时：慢回升，抑制阈值附近抖动/远滑
    // - 对 CHR 的高冲击段(>kN3=3500N)，无论升降都使用快跟随，降低高力段相位滞后
    constexpr double kScaleLpAlphaDown = 0.2;  // 快响应
    constexpr double kScaleLpAlphaUp = 0.8;    // 慢回升
    constexpr double kChrFastFollowForceN = 2000.0;
    const double prev_scale = filtered_scales_prev[static_cast<size_t>(i)];
    double alpha = (scale_factor < prev_scale) ? kScaleLpAlphaDown : kScaleLpAlphaUp;
    if (force_method_ == "chr" && chr_zzq_force_ && contact_force_normal > kChrFastFollowForceN) {
      alpha = kScaleLpAlphaDown;
    }
    scale_factor = alpha * prev_scale + (1.0 - alpha) * scale_factor;
    // 低通后再次夹紧，防止数值漂移
    scale_factor = std::max(0.0, std::min(1.0, scale_factor));
    filtered_scales_prev[static_cast<size_t>(i)] = scale_factor;

    // 实际施加到 efc_force 的是 scale_factor，CHR 原始 force_protected_kN 可能极小；日志与统计用有效力
    const double effective_force_kN = force_unprotected_kN * scale_factor;
    
    // 如果缩放因子接近1，说明防护效果不明显，跳过
    // 但是，即使缩放因子不是1.0，如果防护效果很小（比如只减少1%），也可能被跳过
    // 这里我们只跳过真正没有效果的情况（缩放因子 = 1.0）
    if (std::abs(scale_factor - 1.0) < 1e-6) {
      skipped_no_effect++;
      // 对于高力，立即记录为什么没有效果（不依赖frame_count）
      if (contact_force_normal > 20000) {
        static int no_effect_count = 0;
        no_effect_count++;
        if (no_effect_count <= 5) {
          double std_pose_log[3];
          GreenBallWorldPos(m_, d_, std_xpos_cache_, std_xmat_cache_, body1_id, body2_id, contact.pos, std_pose_log);
          RCLCPP_WARN(node_->get_logger(),
                      "Force %.2f kN: scale_factor=%.6f (too close to 1.0, no effect), "
                      "std_pose_xyz=%.4f,%.4f,%.4f (keyframe / green ball, m)",
                      force_unprotected_kN, scale_factor, std_pose_log[0], std_pose_log[1], std_pose_log[2]);
        }
      }
      continue;
    }
    
    // 对于高力，总是应用防护，即使效果不明显
    // 这样可以确保防护功能总是生效
    
    // 对于高力，立即记录防护应用信息（不依赖frame_count）
    // 对于超过1kN的力，总是打印（不限制次数）
    if (contact_force_normal > 1000) {
      double std_pose_log[3];
      GreenBallWorldPos(m_, d_, std_xpos_cache_, std_xmat_cache_, body1_id, body2_id, contact.pos, std_pose_log);
      RCLCPP_INFO(node_->get_logger(),
                  "Applying protection: force=%.2f kN (%.2f N) -> %.2f kN (%.2f N) effective (scale=%.4f, chr_raw=%.4f kN), "
                  "std_pose_xyz=%.4f,%.4f,%.4f (keyframe/green, m), world_pos=%.4f,%.4f,%.4f",
                  force_unprotected_kN, contact_force_normal, effective_force_kN,
                  effective_force_kN * 1000.0, scale_factor, force_protected_kN,
                  std_pose_log[0], std_pose_log[1], std_pose_log[2],
                  contact.pos[0], contact.pos[1], contact.pos[2]);
    } else if (contact_force_normal > 400) {
      static int protection_applied_count = 0;
      protection_applied_count++;
      if (protection_applied_count <= 10) {
        RCLCPP_INFO(node_->get_logger(),
                    "Applying protection: force=%.2f kN -> %.2f kN effective (scale=%.4f, chr_raw=%.4f kN)",
                    force_unprotected_kN, effective_force_kN, scale_factor, force_protected_kN);
      }
    }
    
    // 记录最大防护后的力（用于调试）：与 efc 缩放一致
    double force_after = effective_force_kN * 1000.0;  // N
    if (force_after > max_force_after) {
      max_force_after = force_after;
    }
    
    // 获取该接触点对应的约束索引
    // 每个接触点可能对应多个约束（法向约束和摩擦约束）
    // 我们需要找到该接触点对应的所有约束并缩放它们的efc_force
    
    // 查找该接触点对应的约束范围
    // 接触约束在efc_force中的索引可以通过contact.efc_address获取
    int efc_address = contact.efc_address;
    if (efc_address < 0 || efc_address >= d_->nefc) {
      skipped_efc_unavailable++;
      continue;
    }
    
    // 接触约束通常包括：
    // 1. 法向约束（1个）
    // 2. 摩擦约束（最多4个，取决于摩擦维数）
    // 法向与摩擦同比例缩放，保证仍在摩擦锥内（|friction| <= mu*normal），否则仿真会失稳
    
    // 获取接触的约束维数（dim = 1, 3, 4, 或 6）
    // dim = 1: 只有法向约束
    // dim = 3: 法向约束 + 2个切向摩擦约束
    // dim = 4: 法向约束 + 3个切向摩擦约束（椭圆摩擦）
    // dim = 6: 法向约束 + 5个切向摩擦约束（椭球摩擦）
    int dim = contact.dim;
    
    // 检查摩擦锥类型
    // 对于 pyramidal 摩擦锥，efc_force 存储的是 pyramid 编码，需要先解码、缩放、再编码
    // 对于 elliptic 摩擦锥，efc_force 直接存储力值，可以直接缩放
    bool is_pyramidal = mj_isPyramidal(m_);
    
    if (is_pyramidal && dim > 1) {
      // Pyramidal 摩擦锥：需要解码、缩放、再编码
      // pyramid 表示：对于 dim=3，有 2*(dim-1) = 4 个元素
      // 对于 dim=4，有 2*(dim-1) = 6 个元素
      // 对于 dim=6，有 2*(dim-1) = 10 个元素（最大）
      int pyramid_size = 2 * (dim - 1);
      
      // 安全检查：确保不会越界
      if (efc_address < 0 || efc_address + pyramid_size > d_->nefc) {
        skipped_efc_unavailable++;
        continue;
      }
      
      // 保存原始的 pyramid 表示（用于验证）
      // 使用足够大的数组（最大 dim=6，pyramid_size=10）
      mjtNum original_pyramid[10];
      for (int k = 0; k < pyramid_size; ++k) {
        original_pyramid[k] = d_->efc_force[efc_address + k];
      }
      
      // 解码 pyramid 表示得到实际的接触力
      mjtNum decoded_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      mju_decodePyramid(decoded_force, d_->efc_force + efc_address, contact.friction, dim);
      
      // 记录原始法向力（用于验证）
      double original_normal = decoded_force[0];
      
      decoded_force[0] *= scale_factor;  // 法向力
      int tang_dim = std::min(dim, 6);   // 防止 dim 异常导致越界
      for (int j = 1; j < tang_dim; ++j) {
        decoded_force[j] *= scale_factor;  // 切向同比例缩放
      }
      
      // 重新编码为 pyramid 表示
      mju_encodePyramid(d_->efc_force + efc_address, decoded_force, contact.friction, dim);
      
      // 验证：重新解码检查是否正确（仅对高力进行验证，避免性能影响）
      if (contact_force_normal > 20000 && frame_count % 10 == 0) {
        mjtNum verify_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        mju_decodePyramid(verify_force, d_->efc_force + efc_address, contact.friction, dim);
        double verify_normal = verify_force[0];
        double expected_normal = original_normal * scale_factor;
        double error = std::abs(verify_normal - expected_normal) / (expected_normal + 1e-6);  // 避免除零
        if (error > 0.01) {  // 1% 误差阈值
          RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                              "Pyramid encode/decode error: original=%.2f, expected=%.2f, got=%.2f, error=%.2f%%",
                              original_normal, expected_normal, verify_normal, error * 100.0);
        }
      }
    } else {
      // Elliptic 摩擦锥或 dim=1
      int efc_size = (dim == 1) ? 1 : dim;
      if (efc_address >= 0 && efc_address < d_->nefc && efc_size > 0) {
        int max_size = std::min(efc_size, d_->nefc - efc_address);
        for (int j = 0; j < max_size; ++j) {
          int efc_idx = efc_address + j;
          if (efc_idx >= 0 && efc_idx < d_->nefc) {
            d_->efc_force[efc_idx] *= scale_factor;
          }
        }
      }
    }

    last_contact_protection_scales_[static_cast<size_t>(i)] = scale_factor;
    
    // 记录统计信息
    protected_contacts++;
    total_force_reduction += (1.0 - scale_factor) * contact_force_normal;
  }
  
  // 重要：修改efc_force后，需要重新计算qfrc_constraint
  // 因为qfrc_constraint = J^T * efc_force，其中J是约束雅可比矩阵
  // 如果不重新计算，修改的efc_force不会影响后续的动力学计算
  if (protected_contacts > 0 && m_->nv > 0 && m_->nv <= 1024) {
    // 用增量方式更新 qacc：qacc_new = qacc_old + M^{-1} * (qfrc_constraint_new - qfrc_constraint_old)
    // 注意：mj_solveM 的最后一个参数 n 表示"解多少组向量"（每组长度 nv），不是向量长度！
    // 之前误传 nv 导致越界崩溃，正确值是 n=1。
    //
    // 阈值 800N 确保：躺着时小力不处理（仿真正常），只有摔倒冲击大力才重算 acc。
    
    // 1. 保存防护前的 qfrc_constraint
    std::vector<mjtNum> qfrc_old(d_->qfrc_constraint, d_->qfrc_constraint + m_->nv);
    
    // 2. 用新的 efc_force 计算新的 qfrc_constraint
    mj_mulJacTVec(m_, d_, d_->qfrc_constraint, d_->efc_force);
    
    // 3. 计算增量 delta_qfrc = qfrc_constraint_new - qfrc_constraint_old
    std::vector<mjtNum> delta_qfrc(static_cast<size_t>(m_->nv));
    mju_sub(delta_qfrc.data(), d_->qfrc_constraint, qfrc_old.data(), m_->nv);
    
    // 4. 计算 delta_qacc = M^{-1} * delta_qfrc（n=1 表示解 1 组向量）
    std::vector<mjtNum> delta_qacc(static_cast<size_t>(m_->nv));
    mj_solveM(m_, d_, delta_qacc.data(), delta_qfrc.data(), 1);
    
    // 5. qacc_new = qacc_old + delta_qacc
    mju_addTo(d_->qacc, delta_qacc.data(), m_->nv);
    
    // 6. 更新 body 笛卡尔加速度 cacc（CSV 的 base_link_lin_acc 等来自 cacc）
    mj_rnePostConstraint(m_, d_);
  } else if (protected_contacts > 0) {
    // nv 超范围时只更新 qfrc_constraint，不重算 qacc/cacc
    mj_mulJacTVec(m_, d_, d_->qfrc_constraint, d_->efc_force);
  }

  // 每1000帧打印一次统计信息（避免日志过多）
  frame_count++;
  if (frame_count % 1000 == 0 && node_) {
    auto logger = node_->get_logger();
    auto clock = node_->get_clock();
    if (clock) {
      double reduction_percent = (max_force_before > 0) ? 
        (1.0 - max_force_after / max_force_before) * 100.0 : 0.0;
      RCLCPP_INFO_THROTTLE(logger, *clock, 5000, 
                          "Protection stats: contacts=%d, protected=%d, skipped_small=%d, skipped_range=%d, skipped_no_effect=%d, skipped_thickness_lt_min=%d, skipped_efc=%d, max_force_before=%.2fN (%.2f kN), max_force_after=%.2fN (%.2f kN), reduction=%.1f%%, avg_reduction=%.2fN",
                          ncon, protected_contacts, skipped_small_force, skipped_out_of_range, skipped_no_effect, skipped_thickness_below_min, skipped_efc_unavailable,
                          max_force_before, max_force_before/1000.0, max_force_after, max_force_after/1000.0,
                          reduction_percent,
                          protected_contacts > 0 ? total_force_reduction / protected_contacts : 0.0);
    }
  }
  
  // 对于高力，即使没有达到1000帧，也输出统计信息
  // 对于超过1kN的力，总是打印（不限制次数）
  if (max_force_before > 1000 && node_) {
    double reduction_percent = (max_force_before > 0) ? 
      (1.0 - max_force_after / max_force_before) * 100.0 : 0.0;
    RCLCPP_INFO(node_->get_logger(),
                "Protection stats (high force): contacts=%d, protected=%d, skipped_small=%d, skipped_range=%d, skipped_no_effect=%d, skipped_thickness_lt_min=%d, skipped_efc=%d, max_force_before=%.2fN (%.2f kN), max_force_after=%.2fN (%.2f kN), reduction=%.1f%%",
                ncon, protected_contacts, skipped_small_force, skipped_out_of_range, skipped_no_effect, skipped_thickness_below_min, skipped_efc_unavailable,
                max_force_before, max_force_before/1000.0, max_force_after, max_force_after/1000.0,
                reduction_percent);
  } else if (max_force_before > 20000 && node_) {
    static int high_force_stats_count = 0;
    high_force_stats_count++;
    if (high_force_stats_count <= 5) {  // 只记录前5次
      double reduction_percent = (max_force_before > 0) ? 
        (1.0 - max_force_after / max_force_before) * 100.0 : 0.0;
      RCLCPP_INFO(node_->get_logger(),
                  "Protection stats (high force): contacts=%d, protected=%d, skipped_small=%d, skipped_range=%d, skipped_no_effect=%d, skipped_thickness_lt_min=%d, skipped_efc=%d, max_force_before=%.2fN (%.2f kN), max_force_after=%.2fN (%.2f kN), reduction=%.1f%%",
                  ncon, protected_contacts, skipped_small_force, skipped_out_of_range, skipped_no_effect, skipped_thickness_below_min, skipped_efc_unavailable,
                  max_force_before, max_force_before/1000.0, max_force_after, max_force_after/1000.0,
                  reduction_percent);
    }
  }
  
  // 对于非常大的力，立即记录调试信息
  // 对于超过1kN的力，总是打印（不限制次数）
  if (max_force_before > 1000 && protected_contacts == 0 && node_) {
    double std_pose_warn[3] = {0.0, 0.0, 0.0};
    double world_warn[3] = {0.0, 0.0, 0.0};
    double map_th_warn = -1.0;
    const char* max_pair_a = "?";
    const char* max_pair_b = "?";
    FillMaxContactWarnPose(m_, d_, std_xpos_cache_, std_xmat_cache_, ncon, idx_max_normal,
                           protector_map_.get(), use_protector_map_, std_pose_warn, world_warn, &map_th_warn,
                           &max_pair_a, &max_pair_b);
    RCLCPP_WARN(node_->get_logger(), 
                "High force detected (%.2f N, %.2f kN) but NO contacts were protected! "
                "skipped_small=%d, skipped_range=%d, skipped_no_effect=%d, skipped_thickness_lt_min=%d, skipped_efc=%d (efc_address/pyramid), protection_enabled=%s, use_map=%s, "
                "max_contact_pair=%s <-> %s, idx=%d, std_pose_xyz(keyframe/map)=%.4f,%.4f,%.4f (m), world_pos=%.4f,%.4f,%.4f (m), map_thickness_mm=%.3f",
                max_force_before, max_force_before/1000.0, skipped_small_force, skipped_out_of_range, skipped_no_effect,
                skipped_thickness_below_min, skipped_efc_unavailable,
                protection_enabled_ ? "YES" : "NO", use_protector_map_ ? "YES" : "NO",
                max_pair_a ? max_pair_a : "?", max_pair_b ? max_pair_b : "?",
                idx_max_normal,
                std_pose_warn[0], std_pose_warn[1], std_pose_warn[2],
                world_warn[0], world_warn[1], world_warn[2],
                map_th_warn);
  } else if (max_force_before > 20000 && protected_contacts == 0 && node_) {
    static int high_force_warn_count = 0;
    high_force_warn_count++;
    if (high_force_warn_count <= 10) {  // 只记录前10次
      double std_pose_warn[3] = {0.0, 0.0, 0.0};
      double world_warn[3] = {0.0, 0.0, 0.0};
      double map_th_warn = -1.0;
      const char* max_pair_a = "?";
      const char* max_pair_b = "?";
      FillMaxContactWarnPose(m_, d_, std_xpos_cache_, std_xmat_cache_, ncon, idx_max_normal,
                             protector_map_.get(), use_protector_map_, std_pose_warn, world_warn, &map_th_warn,
                             &max_pair_a, &max_pair_b);
      RCLCPP_WARN(node_->get_logger(), 
                  "High force detected (%.2f N) but NO contacts were protected! "
                  "skipped_small=%d, skipped_range=%d, skipped_no_effect=%d, skipped_thickness_lt_min=%d, skipped_efc=%d, protection_enabled=%s, use_map=%s, "
                  "max_contact_pair=%s <-> %s, idx=%d, std_pose_xyz(keyframe/map)=%.4f,%.4f,%.4f (m), world_pos=%.4f,%.4f,%.4f (m), map_thickness_mm=%.3f",
                  max_force_before, skipped_small_force, skipped_out_of_range, skipped_no_effect, skipped_thickness_below_min, skipped_efc_unavailable,
                  protection_enabled_ ? "YES" : "NO", use_protector_map_ ? "YES" : "NO",
                  max_pair_a ? max_pair_a : "?", max_pair_b ? max_pair_b : "?",
                  idx_max_normal,
                  std_pose_warn[0], std_pose_warn[1], std_pose_warn[2],
                  world_warn[0], world_warn[1], world_warn[2],
                  map_th_warn);
    }
  }

  if (protected_contacts > 0) {
    prot_sess_frames_with_protection_++;
    prot_sess_total_contact_scalings_ += static_cast<uint64_t>(protected_contacts);
  }
  if (max_force_before > 1000.0) {
    prot_sess_frames_max_gt_1kn_++;
  }
  if (max_force_before > 10000.0) {
    prot_sess_frames_max_gt_10kn_++;
  }
  if (max_force_before > 1000.0 && protected_contacts == 0) {
    prot_sess_warn_no_protect_frames_++;
  }
}

/**
 * @brief 计算并更新关节反力数据
 * @details 在mj_step或mj_forward之后调用，计算所有关节的反力
 * 
 * 使用示例：
 * @code
 * // 1. 获取子body坐标系下的关节反力
 * const auto& jointChild = sim_manager.GetJointWrenchesChild();
 * 
 * // 2. 获取父body坐标系下的关节反力
 * const auto& jointParent = sim_manager.GetJointWrenchesParent();
 * 
 * // 3. 获取每个link两端的关节受力
 * const auto& linkEnds = sim_manager.GetLinkEndWrenches();
 * 
 * // 4. 获取特定关节的载荷分解（轴向力、剪切力、扭矩、弯矩）
 * auto comp = sim_manager.GetJointDecomposedWrench("J03_KNEE_PITCH_L");
 * if (comp) {
 *   std::cout << "轴向力: " << comp->F_axial_mag << std::endl;
 *   std::cout << "剪切力: " << comp->F_shear_mag << std::endl;
 *   std::cout << "扭矩: " << comp->M_torsion_mag << std::endl;
 *   std::cout << "弯矩: " << comp->M_bend_mag << std::endl;
 * }
 * @endcode
 */
void SimManager::ComputeJointForces() {
  if (!m_ || !d_) return;
  
  std::lock_guard<std::mutex> lock(joint_forces_mutex_);
  
  // 计算子body坐标系下的关节反力
  joint_wrenches_child_ = computeJointWrenchesChildBodyEigen(m_, d_);
  
  // 计算父body坐标系下的关节反力
  joint_wrenches_parent_ = computeJointWrenchesParentBodyEigen(m_, d_, joint_wrenches_child_);
  
  // 计算每个link两端的关节受力
  link_end_wrenches_ = collectLinkEndWrenchesEigen(m_, joint_wrenches_child_, joint_wrenches_parent_);
}

/**
 * @brief 获取指定关节的载荷分解
 * @param joint_name 关节名称
 * @return 载荷分解结果，如果关节不存在则返回nullptr
 */
std::unique_ptr<DecomposedWrenchEigen> SimManager::GetJointDecomposedWrench(const std::string& joint_name) const {
  if (!m_ || !d_) return nullptr;
  
  std::lock_guard<std::mutex> lock(joint_forces_mutex_);
  
  // 查找关节ID
  int joint_id = mj_name2id(m_, mjOBJ_JOINT, joint_name.c_str());
  if (joint_id < 0 || joint_id >= static_cast<int>(joint_wrenches_child_.size())) {
    return nullptr;
  }
  
  // 获取关节轴向量
  Eigen::Vector3d axis(
    m_->jnt_axis[3 * joint_id + 0],
    m_->jnt_axis[3 * joint_id + 1],
    m_->jnt_axis[3 * joint_id + 2]
  );
  
  // 分解载荷
  try {
    DecomposedWrenchEigen result = decomposeWrenchBodyFrameEigen(joint_wrenches_child_[joint_id], axis);
    return std::make_unique<DecomposedWrenchEigen>(result);
  } catch (const std::exception& e) {
    std::cerr << "Error decomposing wrench for joint " << joint_name << ": " << e.what() << std::endl;
    return nullptr;
  }
}



