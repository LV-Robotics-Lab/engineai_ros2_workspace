#include <mujoco/mujoco.h>
#include "array_safety.h"
#include "glfw_adapter.h"
#include "simulate.h"
#include "config_loader.h"
#include "ros_interface.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <cstdlib>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#  include <windows.h>
#else
#  include <unistd.h>
#endif
}

using namespace std::chrono_literals;
namespace mju = mujoco::sample_util;

namespace {
// 键盘控制变量 - 用于推倒采样实验
bool apply_perturb = false;                        // 是否施加干扰力
double perturb_force_magnitude = 20.0;             // 干扰力大小 (N)
double perturb_torque_magnitude = 5.0;             // 干扰力矩大小 (N.m)
Eigen::Vector3d perturb_force = Eigen::Vector3d::Zero();   // 干扰力向量 (物体坐标系)
Eigen::Vector3d perturb_torque = Eigen::Vector3d::Zero();  // 干扰力矩向量 (世界坐标系)
std::string perturb_body_name = "LINK_TORSO_YAW";  // 施加干扰力的物体名称

// 干扰力持续时间控制
double perturb_duration = 0.2;     // 干扰力持续时间（秒）
double perturb_start_time = -1.0;  // 干扰力开始时间（-1表示未开始）

// 仿真参数常量
const constexpr auto kBusyWaitTime = 100us;        // 忙等待时间
const constexpr int kDofFloatingBase = 6;          // 浮动基座自由度
const constexpr int kNumFloatingBaseJoints = 7;    // 浮动基座关节数
const constexpr int kDimQuaternion = 4;            // 四元数维度
}  // namespace

/**
 * @brief 力矩控制器 - MuJoCo控制回调函数
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * 
 * 这个函数是MuJoCo的控制回调函数，在每个仿真步骤中被调用。
 * 它实现了PD控制器和推倒采样的干扰力系统。
 */
void TorqueController(const mjModel* m, mjData* d) {
  // 简化版本：使用零力矩控制（机器人自由运动）
  // 在实际应用中，可以添加PD控制器或其他控制策略
  Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(m->nu);
  Eigen::Map<Eigen::VectorXd>(d->ctrl, m->nu) = tau_cmd;

  // ==================== 推倒采样干扰力系统 ====================
  // 检查干扰力是否应该自动停止（基于持续时间）
  if (apply_perturb && perturb_start_time > 0) {
    if (d->time - perturb_start_time > perturb_duration) {
      apply_perturb = false;
      perturb_start_time = -1.0;
      std::cout << "干扰力自动停止" << std::endl;

      // 清除所有外力
      mju_zero(d->xfrc_applied, 6 * m->nbody);
    }
  }

  // 施加干扰力和力矩（仅当apply_perturb为true时）
  if (apply_perturb) {
    // 如果是新的干扰力，记录开始时间
    if (perturb_start_time < 0) {
      perturb_start_time = d->time;
      std::cout << "开始施加干扰力，持续时间: " << perturb_duration << "秒" << std::endl;
    }

    // 获取目标物体的ID
    int body_id = mj_name2id(m, mjOBJ_BODY, perturb_body_name.c_str());
    if (body_id >= 0) {
      // 将干扰力从物体坐标系转换到世界坐标系
      mjtNum* body_quat = d->xquat + 4 * body_id;  // 获取物体四元数
      mjtNum perturb_force_local[3] = {perturb_force.x(), perturb_force.y(), perturb_force.z()};
      mjtNum perturb_force_world[3];
      mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);

      // 施加干扰力到物体
      d->xfrc_applied[6 * body_id + 0] += perturb_force_world[0];
      d->xfrc_applied[6 * body_id + 1] += perturb_force_world[1];
      d->xfrc_applied[6 * body_id + 2] += perturb_force_world[2];

      // 施加干扰力矩到物体
      d->xfrc_applied[6 * body_id + 3] += perturb_torque.x();
      d->xfrc_applied[6 * body_id + 4] += perturb_torque.y();
      d->xfrc_applied[6 * body_id + 5] += perturb_torque.z();
    }
  }
}

// 加载MuJoCo模型
mjModel* LoadModel(mujoco::Simulate* sim, const std::string& filename, std::shared_ptr<ConfigLoader> config_loader) {
  char filepath[mujoco::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filepath, filename.c_str());
  
  if (!filepath[0]) {
    return nullptr;
  }

  mjModel* mnew = nullptr;
  auto load_start = mujoco::Simulate::Clock::now();
  
  // 检查是否为二进制模型文件(.mjb)
  if (mju::strlen_arr(filepath) > 4 && !std::strncmp(filepath + mju::strlen_arr(filepath) - 4, ".mjb",
                                                     mju::sizeof_arr(filepath) - mju::strlen_arr(filepath) + 4)) {
    mnew = mj_loadModel(filepath, nullptr);
    if (!mnew) {
      std::cerr << "Could not load binary model" << std::endl;
    }
  } else {
    // 获取碰撞模型条件
    std::string collision_condition = "simplified";  // 默认值
    std::string xml_filename = "pm_v2.xml";  // 默认使用主配置文件
    
    if (config_loader) {
      collision_condition = config_loader->GetCollisionModelCondition();
      // 根据碰撞类型选择对应的XML文件
      xml_filename = config_loader->GetXmlFilenameByCollisionType();
      std::cout << "ConfigLoader found! Collision condition: " << collision_condition << std::endl;
      std::cout << "XML filename from config: " << xml_filename << std::endl;
    } else {
      std::cout << "ConfigLoader is null! Using default values." << std::endl;
    }
    
    std::cout << "Loading XML with collision condition: " << collision_condition << std::endl;
    std::cout << "Selected XML file: " << xml_filename << std::endl;
    
    // 构建完整的XML文件路径
    std::string full_xml_path = config_loader ? 
        config_loader->GetResourceDir() + "/" + xml_filename : 
        filename;
    
    std::cout << "Full XML path: " << full_xml_path << std::endl;
    
    // 检查文件是否存在
    if (std::filesystem::exists(full_xml_path)) {
      std::cout << "XML file exists: " << full_xml_path << std::endl;
    } else {
      std::cout << "ERROR: XML file does not exist: " << full_xml_path << std::endl;
      std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
      
      // 列出资源目录中的文件
      if (config_loader) {
        std::string resource_dir = config_loader->GetResourceDir();
        std::cout << "Resource directory: " << resource_dir << std::endl;
        if (std::filesystem::exists(resource_dir)) {
          std::cout << "Files in resource directory:" << std::endl;
          for (const auto& entry : std::filesystem::directory_iterator(resource_dir)) {
            std::cout << "  " << entry.path().filename() << std::endl;
          }
        } else {
          std::cout << "Resource directory does not exist: " << resource_dir << std::endl;
        }
      }
    }
    
    // 设置编译条件
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    
    // 创建编译选项
    mjOption opt;
    mj_defaultOption(&opt);
    
    // 加载选择的XML文件
    char xml_path[mujoco::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(xml_path, full_xml_path.c_str());
    
    char error[1000] = "Could not load XML model";
    mnew = mj_loadXML(xml_path, &vfs, error, sizeof(error));
    if (error[0]) {
      size_t error_length = std::strlen(error);
      if (error_length > 0 && error[error_length - 1] == '\n') {
        error[error_length - 1] = '\0';
      }
      std::cout << "Model compiled with warning: " << error << std::endl;
      sim->run = 0;
    }
  }

  auto load_interval = mujoco::Simulate::Clock::now() - load_start;
  double load_seconds = std::chrono::duration<double>(load_interval).count();

  if (!mnew) {
    std::cerr << "Failed to load model" << std::endl;
    return nullptr;
  }

  if (load_seconds > 0.25) {
    std::cout << "Model loaded in " << load_seconds << " seconds" << std::endl;
  }

  return mnew;
}

// 仿真参数常量
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation

// 全局模型和数据指针
mjModel* m = nullptr;
mjData* d = nullptr;

// 仿真速度监控变量
std::chrono::time_point<std::chrono::steady_clock> last_speed_report;
int step_count = 0;
double last_sim_time = 0.0;

// 完整的物理循环函数
void PhysicsLoop(mujoco::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mujoco::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;
  
  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(&sim, sim.dropfilename, nullptr);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(&sim, sim.filename, nullptr);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(kBusyWaitTime);
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mujoco::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned = std::abs(std::chrono::duration<double>(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            stepped = true;
            
            // 监控仿真速度
            step_count++;
            if (step_count % 1000 == 0) {  // 每1000步报告一次
              auto now = std::chrono::steady_clock::now();
              auto elapsed = std::chrono::duration<double>(now - last_speed_report).count();
              if (elapsed >= 1.0) {  // 每秒报告一次
                double sim_elapsed = d->time - last_sim_time;
                double real_time_factor = sim_elapsed / elapsed;
                std::cout << "Simulation Speed: " << real_time_factor << "x real-time" 
                         << " (Sim: " << sim_elapsed << "s, Real: " << elapsed << "s)" << std::endl;
                last_speed_report = now;
                last_sim_time = d->time;
              }
            }
          }

          // in-sync: step until ahead of cpu
          else {
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (std::chrono::duration<double>((d->time - syncSim) * slowdown) < mujoco::Simulate::Clock::now() - syncCPU &&
                   mujoco::Simulate::Clock::now() - startCPU < std::chrono::duration<double>(refreshTime)) {

              // call mj_step
              mj_step(m, d);
              stepped = true;
              
              // 监控仿真速度
              step_count++;
              if (step_count % 1000 == 0) {  // 每1000步报告一次
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration<double>(now - last_speed_report).count();
                if (elapsed >= 1.0) {  // 每秒报告一次
                  double sim_elapsed = d->time - last_sim_time;
                  double real_time_factor = sim_elapsed / elapsed;
                  std::cout << "Simulation Speed: " << real_time_factor << "x real-time" 
                           << " (Sim: " << sim_elapsed << "s, Real: " << elapsed << "s)" << std::endl;
                  last_speed_report = now;
                  last_sim_time = d->time;
                }
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}

// 完整的物理线程函数
void PhysicsThread(mujoco::Simulate* sim, const char* filename, std::shared_ptr<ConfigLoader> config_loader) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(sim, filename, config_loader);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      sim->Load(m, d, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      // 应用 keyframe 中的初始位置
      int keyframe_id = mj_name2id(m, mjOBJ_KEY, "floating_base_homing");
      if (keyframe_id >= 0) {
        std::cout << "Applying keyframe 'floating_base_homing' (id: " << keyframe_id << ") as initial position" << std::endl;
        mj_resetDataKeyframe(m, d, keyframe_id);
        std::cout << "Keyframe applied successfully" << std::endl;
      } else {
        std::cout << "Keyframe 'floating_base_homing' not found, using default initial position" << std::endl;
      }

      std::cout << "Performing initial mj_forward calculation..." << std::endl;
      mj_forward(m, d);
      std::cout << "mj_forward calculation completed" << std::endl;
      
      // 显示仿真参数
      std::cout << "Simulation parameters:" << std::endl;
      std::cout << "  Timestep: " << m->opt.timestep << " seconds" << std::endl;
      std::cout << "  Tolerance: " << m->opt.tolerance << std::endl;
      std::cout << "  Number of bodies: " << m->nbody << std::endl;
      std::cout << "  Number of joints: " << m->njnt << std::endl;
      std::cout << "  Number of actuators: " << m->nu << std::endl;
      
      // 初始化速度监控
      last_speed_report = std::chrono::steady_clock::now();
      step_count = 0;
      last_sim_time = d->time;
      std::cout << "Speed monitoring initialized" << std::endl;

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}

// ==================== 键盘控制类 ====================
/**
 * @brief 自定义GLFW适配器 - 实现推倒采样的键盘控制
 * 
 * 这个类重写了MuJoCo的键盘事件处理，实现了推倒采样的交互式控制。
 * 用户可以通过键盘快捷键实时控制干扰力的施加，用于测试机器人的平衡能力。
 */
class CustomGlfwAdapter : public mujoco::GlfwAdapter {
 protected:
  bool shift_pressed = false;  // Shift键状态
  
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
    mujoco::GlfwAdapter::OnKey(key, scancode, act);

    // 更新Shift键状态
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
      if (act == GLFW_PRESS) {
        shift_pressed = true;
      } else if (act == GLFW_RELEASE) {
        shift_pressed = false;
      }
      return;
    }

    // 只处理按下事件
    if (act != GLFW_PRESS) return;

    // 只有在Shift键按下时才处理推倒采样控制
    if (shift_pressed) {
      // 重置干扰力和力矩
      perturb_force = Eigen::Vector3d::Zero();
      perturb_torque = Eigen::Vector3d::Zero();
      
      switch (key) {
        case GLFW_KEY_F:  // Shift + F: 前向干扰力
          apply_perturb = true;
          perturb_force.x() = perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发前向干扰力: " << perturb_force.x() << std::endl;
          break;

        case GLFW_KEY_B:  // Shift + B: 后向干扰力
          apply_perturb = true;
          perturb_force.x() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发后向干扰力: " << perturb_force.x() << std::endl;
          break;

        case GLFW_KEY_L:  // Shift + L: 左向干扰力
          apply_perturb = true;
          perturb_force.y() = perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发左向干扰力: " << perturb_force.y() << std::endl;
          break;

        case GLFW_KEY_R:  // Shift + R: 右向干扰力
          apply_perturb = true;
          perturb_force.y() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发右向干扰力: " << perturb_force.y() << std::endl;
          break;

        case GLFW_KEY_U:  // Shift + U: 上向干扰力
          apply_perturb = true;
          perturb_force.z() = perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发上向干扰力: " << perturb_force.z() << std::endl;
          break;

        case GLFW_KEY_D:  // Shift + D: 下向干扰力
          apply_perturb = true;
          perturb_force.z() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发下向干扰力: " << perturb_force.z() << std::endl;
          break;

        case GLFW_KEY_G:  // Shift + G: X轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.x() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发X轴干扰力矩: " << perturb_torque.x() << std::endl;
          break;

        case GLFW_KEY_J:  // Shift + J: X轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.x() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发X轴干扰力矩: " << perturb_torque.x() << std::endl;
          break;

        case GLFW_KEY_Y:  // Shift + Y: Y轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.y() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发Y轴干扰力矩: " << perturb_torque.y() << std::endl;
          break;

        case GLFW_KEY_H:  // Shift + H: Y轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.y() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发Y轴干扰力矩: " << perturb_torque.y() << std::endl;
          break;

        case GLFW_KEY_LEFT_BRACKET:  // Shift + [: Z轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.z() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发Z轴干扰力矩: " << perturb_torque.z() << std::endl;
          break;

        case GLFW_KEY_RIGHT_BRACKET:  // Shift + ]: Z轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.z() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          std::cout << "触发Z轴干扰力矩: " << perturb_torque.z() << std::endl;
          break;

        case GLFW_KEY_0:  // Shift + 0: 立即停止所有干扰力/力矩
          apply_perturb = false;
          perturb_start_time = -1.0;
          std::cout << "立即停止所有干扰力/力矩" << std::endl;
          break;

        case GLFW_KEY_EQUAL:  // Shift + +: 增加干扰力/力矩大小
          perturb_force_magnitude += 20.0;
          perturb_torque_magnitude += 5.0;
          std::cout << "干扰力/力矩大小增加到: " << perturb_force_magnitude << " N, " << perturb_torque_magnitude
                    << " N.m" << std::endl;
          break;

        case GLFW_KEY_MINUS:  // Shift + -: 减小干扰力/力矩大小
          perturb_force_magnitude = std::max(20.0, perturb_force_magnitude - 20.0);
          perturb_torque_magnitude = std::max(5.0, perturb_torque_magnitude - 5.0);
          std::cout << "干扰力/力矩大小减小到: " << perturb_force_magnitude << " N, " << perturb_torque_magnitude
                    << " N.m" << std::endl;
          break;

        case GLFW_KEY_PERIOD:  // Shift + .: 增加干扰力持续时间
          perturb_duration += 0.1;
          std::cout << "干扰力/力矩持续时间增加到: " << perturb_duration << "秒" << std::endl;
          break;

        case GLFW_KEY_COMMA:  // Shift + ,: 减小干扰力持续时间
          perturb_duration = std::max(0.1, perturb_duration - 0.1);
          std::cout << "干扰力/力矩持续时间减小到: " << perturb_duration << "秒" << std::endl;
          break;
      }
    }
  }
};

// 获取可执行文件目录
std::string getExecutableDir() {
  char path[1024];
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len != -1) {
    path[len] = '\0';
    std::string full_path(path);
    size_t last_slash = full_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return full_path.substr(0, last_slash);
    }
  }
  return ".";
}

// 完整的主函数
int main(int argc, char** argv) {
  // 初始化ROS2
  rclcpp::init(argc, argv);
  
  // 创建ROS2节点
  auto node = rclcpp::Node::make_shared("perturbation_simulator");
  
  // 检查MuJoCo版本
  std::cout << "MuJoCo version " << mj_versionString() << std::endl;
  if (mjVERSION_HEADER != mj_version()) {
    std::cerr << "Headers and library have different versions" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  // 设置环境变量（如果没有设置的话）
  if (!std::getenv("PRODUCT")) {
    setenv("PRODUCT", "pm_v2", 1);
    std::cout << "Set PRODUCT environment variable to: pm_v2" << std::endl;
  }
  
  if (!std::getenv("MUJOCO_ASSETS_PATH")) {
    // 尝试从可执行文件目录推断资源路径
    std::string exe_dir = getExecutableDir();
    std::cout << "Executable directory: " << exe_dir << std::endl;
    
    // 尝试多个可能的路径
    std::vector<std::string> possible_paths = {
      exe_dir + "/../share/mujoco_simulator/assets",
      exe_dir + "/../../share/mujoco_simulator/assets", 
      exe_dir + "/../../../share/mujoco_simulator/assets",
      "./install/mujoco_simulator/share/mujoco_simulator/assets",
      "./src/simulation/mujoco/assets"
    };
    
    std::string assets_path = ".";
    for (const auto& path : possible_paths) {
      std::cout << "Checking path: " << path << std::endl;
      if (std::filesystem::exists(path)) {
        assets_path = path;
        std::cout << "Found assets path: " << assets_path << std::endl;
        break;
      }
    }
    
    setenv("MUJOCO_ASSETS_PATH", assets_path.c_str(), 1);
    std::cout << "Set MUJOCO_ASSETS_PATH environment variable to: " << assets_path << std::endl;
  }

  // 获取产品名称和资源路径
  std::string product_name = std::string(std::getenv("PRODUCT"));
  std::string assets_path = std::string(std::getenv("MUJOCO_ASSETS_PATH"));

  // 构建配置文件路径
  std::string config_file = assets_path + "/config/" + product_name + ".yaml";
  std::cout << "Loading config from: " << config_file << std::endl;

  // 初始化配置加载器
  std::cout << "Creating ConfigLoader with config file: " << config_file << std::endl;
  auto config_loader = std::make_shared<ConfigLoader>(config_file);
  std::cout << "Setting assets path: " << assets_path << std::endl;
  config_loader->SetAssetsPath(assets_path);
  
  std::cout << "Loading config..." << std::endl;
  if (!config_loader->LoadConfig()) {
    std::cerr << "Failed to load config file: " << config_file << std::endl;
    std::cerr << "Using default configuration..." << std::endl;
  } else {
    std::cout << "Config loaded successfully!" << std::endl;
  }

  // 获取ROS2参数（使用get_parameter避免重复声明）
  bool export_contact = false;
  std::string contact_topic = "/mujoco/contact_forces";
  bool save_contact_csv = false;
  std::string csv_file_path = "";
  
  // 检查参数是否存在并获取
  if (node->has_parameter("export_contact")) {
    export_contact = node->get_parameter("export_contact").as_bool();
  }
  if (node->has_parameter("contact_topic")) {
    contact_topic = node->get_parameter("contact_topic").as_string();
  }
  if (node->has_parameter("save_contact_csv")) {
    save_contact_csv = node->get_parameter("save_contact_csv").as_bool();
  }
  if (node->has_parameter("csv_file_path")) {
    csv_file_path = node->get_parameter("csv_file_path").as_string();
  }
  
  // 也检查命令行参数
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--export_contact") {
      export_contact = true;
    } else if (std::string(argv[i]) == "--save_contact_csv") {
      save_contact_csv = true;
    } else if (std::string(argv[i]) == "--contact_topic" && i + 1 < argc) {
      contact_topic = argv[++i];
    } else if (std::string(argv[i]) == "--csv_file_path" && i + 1 < argc) {
      csv_file_path = argv[++i];
    }
  }
  
  std::cout << "ROS2 Parameters:" << std::endl;
  std::cout << "  export_contact: " << (export_contact ? "true" : "false") << std::endl;
  std::cout << "  contact_topic: " << contact_topic << std::endl;
  std::cout << "  save_contact_csv: " << (save_contact_csv ? "true" : "false") << std::endl;
  std::cout << "  csv_file_path: " << csv_file_path << std::endl;

  // 获取模型文件路径
  std::string model_file = config_loader->GetModelFilePath();
  std::cout << "Model file path: " << model_file << std::endl;

  // 设置VFS目录
  const std::string resource_dir = config_loader->GetResourceDir();
  setenv("MJCF_PATH", resource_dir.c_str(), 1);
  std::cout << "Setting MJCF_PATH environment variable: " << resource_dir << std::endl;

  // 初始化MuJoCo可视化组件
  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);
  
  // 启用接触点可视化
  opt.flags[mjVIS_CONTACTPOINT] = 1;  // 显示接触点
  opt.flags[mjVIS_CONTACTFORCE] = 1;  // 显示接触力
  opt.flags[mjVIS_CONTACTSPLIT] = 1;  // 显示接触分离
  opt.flags[mjVIS_PERTFORCE] = 1;  // 显示施加的外力

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // 安装控制回调函数 - 实现推倒采样干扰力
  mjcb_control = TorqueController;

  // 创建仿真对象，封装UI界面
  // 使用自定义的GLFW适配器以支持推倒采样的键盘交互
  auto sim = std::make_unique<mujoco::Simulate>(std::make_unique<CustomGlfwAdapter>(), &cam, &opt, &pert,
                                            /* is_passive = */ false);

  // 初始化ROS接口（如果需要）
  std::shared_ptr<mujoco::RosInterface> ros_interface = nullptr;
  if (export_contact || save_contact_csv) {
    std::cout << "Initializing ROS interface..." << std::endl;
    try {
      ros_interface = std::make_shared<mujoco::RosInterface>(node, config_loader);
      if (!ros_interface->Initialize()) {
        std::cerr << "Failed to initialize ROS interface" << std::endl;
        // 不退出，继续运行但不使用ROS接口
        ros_interface = nullptr;
      } else {
        std::cout << "ROS interface initialized successfully" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "ROS interface initialization error: " << e.what() << std::endl;
      ros_interface = nullptr;
    }
  } else {
    std::cout << "Skipping ROS interface initialization (not needed)" << std::endl;
  }

  // 启动物理仿真线程 - 在后台运行物理计算
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), model_file.c_str(), config_loader);

  // 启动ROS2 spin线程（非阻塞）
  std::thread ros_spin_thread([node]() {
    try {
      rclcpp::spin(node);
    } catch (const std::exception& e) {
      std::cout << "ROS2 spin thread error: " << e.what() << std::endl;
    }
  });

  std::cout << "Starting MuJoCo GUI..." << std::endl;
  
  // 启动仿真UI循环 - 阻塞调用，处理渲染和用户交互
  try {
    sim->RenderLoop();
  } catch (const std::exception& e) {
    std::cout << "MuJoCo GUI error: " << e.what() << std::endl;
  }
  
  std::cout << "MuJoCo GUI closed, shutting down..." << std::endl;
  
  // 关闭ROS2
  rclcpp::shutdown();
  
  // 等待ROS2 spin线程结束
  if (ros_spin_thread.joinable()) {
    ros_spin_thread.join();
  }
  
  // 等待物理线程结束
  if (physicsthreadhandle.joinable()) {
    physicsthreadhandle.join();
  }

  return 0;
}

