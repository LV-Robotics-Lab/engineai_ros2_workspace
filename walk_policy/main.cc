// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file perturbation_simulator.cc
 * @brief MuJoCo推倒采样仿真器主程序 - 用于机器人推倒采样和步态策略测试
 * 
 * 这个程序实现了一个完整的MuJoCo物理仿真环境，主要用于：
 * 1. 机器人步态策略的测试和验证
 * 2. 推倒采样实验 - 通过施加干扰力测试机器人的平衡能力
 * 3. 实时交互式仿真 - 支持键盘控制干扰力/力矩
 * 4. LCM通信 - 与外部控制器进行数据交换
 * 
 * 与ROS2版本的main.cc相比，这个版本：
 * - 使用LCM而不是ROS2进行通信
 * - 包含完整的UI界面和交互功能
 * - 实现了推倒采样的干扰力系统
 * - 支持实时键盘控制干扰力参数
 */

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "array_safety.h"
#include "glfw_adapter.h"
#include "simulate.h"

#include "lcm_interface/lcm_data_store.h"
#include "lcm_interface/lcm_publisher.h"
#include "lcm_interface/lcm_subscriber.h"
#include "lcm_param/lcm_param.h"
#include "model_param/model_param.h"
#include "monitor/google_logger.h"
#include "parameter/global_config_initializer.h"
#include "parameter/parameter_loader.h"
#include "tool/string_join.h"
#include "tool/timer.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#  include <windows.h>
#else
#  if defined(__APPLE__)
#    include <mach-o/dyld.h>
#  endif
#  include <sys/errno.h>
#  include <unistd.h>
#endif
}
using namespace std::chrono_literals;

namespace {
// LCM数据存储 - 用于与外部控制器通信
std::shared_ptr<LcmDataStore> lcm_data_store;

// 仿真参数常量
const constexpr auto kBusyWaitTime = 100us;        // 忙等待时间
const constexpr int kDofFloatingBase = 6;          // 浮动基座自由度
const constexpr int kNumFloatingBaseJoints = 7;    // 浮动基座关节数
const constexpr int kDimQuaternion = 4;            // 四元数维度

// 接触力维度映射 - 根据接触类型确定力的维度
const std::unordered_map<data::ContactType, int> single_contact_dimensions_map = {
    {data::ContactType::kPoint, 3},                // 点接触：3D力
    {data::ContactType::kRectangularSurface, 6},   // 面接触：3D力+3D力矩
};

// ==================== 推倒采样干扰力系统 ====================
// 这些变量用于实现推倒采样实验，通过施加外部干扰力测试机器人的平衡能力

bool apply_perturb = false;                        // 是否施加干扰力
double perturb_force_magnitude = 20.0;             // 干扰力大小 (N)
double perturb_torque_magnitude = 5.0;             // 干扰力矩大小 (N.m)
Eigen::Vector3d perturb_force = Eigen::Vector3d::Zero();   // 干扰力向量 (物体坐标系)
Eigen::Vector3d perturb_torque = Eigen::Vector3d::Zero();  // 干扰力矩向量 (世界坐标系)
std::string perturb_body_name = "LINK_TORSO_YAW";  // 施加干扰力的物体名称

// 干扰力持续时间控制
double perturb_duration = 0.2;     // 干扰力持续时间（秒）
double perturb_start_time = -1.0;  // 干扰力开始时间（-1表示未开始）
}  // namespace

/**
 * @brief 更新仿真状态 - 从MuJoCo数据中提取机器人状态信息
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * 
 * 这个函数负责从MuJoCo的仿真数据中提取机器人的各种状态信息，
 * 包括关节状态、基座状态、IMU状态和接触力信息，并存储到LCM数据存储中
 */
void UpdateSimState(const mjModel* m, mjData* d) {
  bool is_floating_base = (m->nv != m->nu);  // 判断是否为浮动基座机器人

  // 设置关节状态 - 提取关节位置、速度和力矩
  SimState sim_state;
  Eigen::VectorXd q(m->nu);   // 关节位置
  Eigen::VectorXd qd(m->nu);  // 关节速度
  if (is_floating_base) {
    // 浮动基座机器人：跳过基座的7个自由度（位置3+四元数4）
    sim_state.q = Eigen::Map<const Eigen::VectorXd>(d->qpos + kNumFloatingBaseJoints, m->nu);
    sim_state.qd = Eigen::Map<const Eigen::VectorXd>(d->qvel + kDofFloatingBase, m->nu);
  } else {
    // 固定基座机器人：直接使用所有关节
    sim_state.q = Eigen::Map<const Eigen::VectorXd>(d->qpos, m->nu);
    sim_state.qd = Eigen::Map<const Eigen::VectorXd>(d->qvel, m->nu);
  }
  sim_state.tau = Eigen::Map<const Eigen::VectorXd>(d->actuator_force, m->nu);  // 关节力矩

  int index = 0;
  // 设置基座链接状态 - 数据序列由sensors.xml文件确定
  // 基座四元数 (w, x, y, z)
  sim_state.base_link_quaternion.w() = d->sensordata[index + 0];
  sim_state.base_link_quaternion.x() = d->sensordata[index + 1];
  sim_state.base_link_quaternion.y() = d->sensordata[index + 2];
  sim_state.base_link_quaternion.z() = d->sensordata[index + 3];
  index += kDimQuaternion;

  // 基座位置 (x, y, z)
  sim_state.base_link_position =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.base_link_position.size());
  index += sim_state.base_link_position.size();

  // 基座线速度 (vx, vy, vz)
  sim_state.base_link_linear_velocity =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.base_link_linear_velocity.size());
  index += sim_state.base_link_linear_velocity.size();

  // 基座角速度 (wx, wy, wz)
  sim_state.base_link_angular_velocity =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.base_link_angular_velocity.size());
  index += sim_state.base_link_angular_velocity.size();

  // 设置IMU链接状态 - 数据序列由sensors.xml文件确定
  // IMU链接四元数 (w, x, y, z)
  sim_state.imu_link_quaternion.w() = d->sensordata[index + 0];
  sim_state.imu_link_quaternion.x() = d->sensordata[index + 1];
  sim_state.imu_link_quaternion.y() = d->sensordata[index + 2];
  sim_state.imu_link_quaternion.z() = d->sensordata[index + 3];
  index += kDimQuaternion;

  // IMU链接位置 (x, y, z)
  sim_state.imu_link_position =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.imu_link_position.size());
  index += sim_state.imu_link_position.size();

  // IMU链接线速度 (vx, vy, vz)
  sim_state.imu_link_linear_velocity =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.imu_link_linear_velocity.size());
  index += sim_state.imu_link_linear_velocity.size();

  // IMU链接角速度 (wx, wy, wz)
  sim_state.imu_link_angular_velocity =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.imu_link_angular_velocity.size());
  index += sim_state.imu_link_angular_velocity.size();

  // 设置IMU传感器状态 - 数据序列由sensors.xml文件确定
  // IMU传感器四元数 (w, x, y, z)
  sim_state.imu_sensor_quaternion.w() = d->sensordata[index + 0];
  sim_state.imu_sensor_quaternion.x() = d->sensordata[index + 1];
  sim_state.imu_sensor_quaternion.y() = d->sensordata[index + 2];
  sim_state.imu_sensor_quaternion.z() = d->sensordata[index + 3];
  index += kDimQuaternion;

  // IMU传感器线性加速度 (ax, ay, az)
  sim_state.imu_sensor_linear_acceleration =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.imu_sensor_linear_acceleration.size());
  index += sim_state.imu_sensor_linear_acceleration.size();

  // IMU传感器角速度 (wx, wy, wz)
  sim_state.imu_sensor_angular_velocity =
      Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, sim_state.imu_sensor_angular_velocity.size());
  index += sim_state.imu_sensor_angular_velocity.size();

  // 设置接触力和力矩状态 - 数据序列由sensors.xml文件确定
  // 初始化接触力向量数组
  sim_state.contact_force = std::vector<Eigen::VectorXd>(
      lcm_data_store->num_contacts, Eigen::VectorXd::Zero(lcm_data_store->num_single_contact_dimensions));
  
  // 提取每个接触点的力/力矩数据
  for (int i = 0; i < lcm_data_store->num_contacts; ++i) {
    sim_state.contact_force[i] =
        Eigen::Map<const Eigen::VectorXd>(d->sensordata + index, lcm_data_store->num_single_contact_dimensions);
    sim_state.contact_force[i] *= -1.0;  // 取反号，因为MuJoCo返回的是物体对环境的力
    index += lcm_data_store->num_single_contact_dimensions;
  }
  
  // 将更新后的仿真状态存储到LCM数据存储中
  lcm_data_store->sim_state.Set(sim_state);
}

/**
 * @brief 力矩控制器 - MuJoCo控制回调函数
 * @param m MuJoCo模型指针
 * @param d MuJoCo数据指针
 * 
 * 这个函数是MuJoCo的控制回调函数，在每个仿真步骤中被调用。
 * 它实现了PD控制器和推倒采样的干扰力系统。
 */
void TorqueController(const mjModel* m, mjData* d) {
  bool is_floating_base = (m->nv != m->nu);  // 判断是否为浮动基座机器人

  // 提取当前关节状态
  Eigen::VectorXd q(m->nu);   // 关节位置
  Eigen::VectorXd qd(m->nu);  // 关节速度
  if (is_floating_base) {
    // 浮动基座机器人：跳过基座的7个自由度
    q = Eigen::Map<const Eigen::VectorXd>(d->qpos + kNumFloatingBaseJoints, m->nu);
    qd = Eigen::Map<const Eigen::VectorXd>(d->qvel + kDofFloatingBase, m->nu);
  } else {
    // 固定基座机器人：直接使用所有关节
    q = Eigen::Map<const Eigen::VectorXd>(d->qpos, m->nu);
    qd = Eigen::Map<const Eigen::VectorXd>(d->qvel, m->nu);
  }

  // 从LCM数据存储中获取控制命令
  SimCommand sim_command = lcm_data_store->sim_command();
  
  // 计算PD控制力矩：tau = Kp*(q_desired - q) + Kd*(qd_desired - qd) + tau_feedforward
  Eigen::VectorXd tau_cmd(m->nu);
  tau_cmd = sim_command.kp.cwiseProduct(sim_command.q - q) + 
            sim_command.kd.cwiseProduct(sim_command.qd - qd) +
            sim_command.tau_ff;

  // 将计算出的控制力矩应用到仿真中
  Eigen::Map<Eigen::VectorXd>(d->ctrl, m->nu) = tau_cmd;

  // ==================== 推倒采样干扰力系统 ====================
  // 检查干扰力是否应该自动停止（基于持续时间）
  if (apply_perturb && perturb_start_time > 0) {
    if (d->time - perturb_start_time > perturb_duration) {
      apply_perturb = false;
      perturb_start_time = -1.0;
      LOG(INFO) << "干扰力自动停止";

      // 清除所有外力
      mju_zero(d->xfrc_applied, 6 * m->nbody);
    }
  }

  // 施加干扰力和力矩（仅当apply_perturb为true时）
  if (apply_perturb) {
    // 如果是新的干扰力，记录开始时间
    if (perturb_start_time < 0) {
      perturb_start_time = d->time;
      LOG(INFO) << "开始施加干扰力，持续时间: " << perturb_duration << "秒";
    }

    // 获取目标物体的ID
    int body_id = mj_name2id(m, mjOBJ_BODY, perturb_body_name.c_str());
    if (body_id >= 0) {
      // 将干扰力从物体坐标系转换到世界坐标系
      mjtNum* body_quat = d->xquat + 4 * body_id;  // 获取物体四元数
      mjtNum perturb_force_local[3] = {perturb_force.x(), perturb_force.y(), perturb_force.z()};
      mjtNum perturb_force_world[3] = {0., 0., 0.};
      mju_rotVecQuat(perturb_force_world, perturb_force_local, body_quat);

      // 施加力（物体坐标系，质心为原点）
      d->xfrc_applied[6 * body_id + 0] = perturb_force_world[0];  // Fx
      d->xfrc_applied[6 * body_id + 1] = perturb_force_world[1];  // Fy
      d->xfrc_applied[6 * body_id + 2] = perturb_force_world[2];  // Fz
      
      // 施加扭矩（世界坐标系）
      d->xfrc_applied[6 * body_id + 3] = perturb_torque.x();  // Tx
      d->xfrc_applied[6 * body_id + 4] = perturb_torque.y();  // Ty
      d->xfrc_applied[6 * body_id + 5] = perturb_torque.z();  // Tz
    }
  }
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

using Seconds = std::chrono::duration<double>;

//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#  if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#  else
  const char* path = "/proc/self/exe";
#  endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

//------------------------------------------- simulation -------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();
  if (mju::strlen_arr(filename) > 4 && !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                                                     mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }
  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  if (!loadError[0] && load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;
  common::Timer loop_timer;
  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
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
      mjModel* mnew = LoadModel(sim.filename, sim);
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

    loop_timer.StartTimer();
    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned = std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            UpdateSimState(m, d);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown = std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              // call mj_step
              mj_step(m, d);
              UpdateSimState(m, d);
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
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
    auto elapsed = loop_timer.GetElapsedSeconds();
    if (elapsed > m->opt.timestep) {
      LOG(WARNING) << "Physics loop took too long: " << elapsed << " s"
                   << ", Current timestep: " << m->opt.timestep << " s";
    }
  }
}
}  // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate* sim, const char* filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      sim->Load(m, d, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}

/**
 * @brief 自定义GLFW适配器 - 实现推倒采样的键盘交互控制
 * 
 * 这个类重写了MuJoCo的键盘事件处理，实现了推倒采样的交互式控制。
 * 用户可以通过键盘快捷键实时控制干扰力的施加，用于测试机器人的平衡能力。
 */
class CustomGlfwAdapter : public mj::GlfwAdapter {
 protected:
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
          LOG(INFO) << "触发前向干扰力: " << perturb_force.x();
          break;

        case GLFW_KEY_B:  // Shift + B: 后向干扰力
          apply_perturb = true;
          perturb_force.x() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发后向干扰力: " << perturb_force.x();
          break;

        case GLFW_KEY_L:  // Shift + L: 左向干扰力
          apply_perturb = true;
          perturb_force.y() = perturb_force_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发左向干扰力: " << perturb_force.y();
          break;

        case GLFW_KEY_R:  // Shift + R: 右向干扰力
          apply_perturb = true;
          perturb_force.y() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发右向干扰力: " << perturb_force.y();
          break;

        case GLFW_KEY_U:  // Shift + U: 上向干扰力
          apply_perturb = true;
          perturb_force.z() = perturb_force_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发上向干扰力: " << perturb_force.z();
          break;

        case GLFW_KEY_D:  // Shift + D: 下向干扰力
          apply_perturb = true;
          perturb_force.z() = -perturb_force_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发下向干扰力: " << perturb_force.z();
          break;

        case GLFW_KEY_G:  // Shift + G: X轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.x() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发X轴干扰力矩: " << perturb_torque.x();
          break;

        case GLFW_KEY_J:  // Shift + J: X轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.x() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发X轴干扰力矩: " << perturb_torque.x();
          break;

        case GLFW_KEY_Y:  // Shift + Y: Y轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.y() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发Y轴干扰力矩: " << perturb_torque.y();
          break;

        case GLFW_KEY_H:  // Shift + H: Y轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.y() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发Y轴干扰力矩: " << perturb_torque.y();
          break;

        case GLFW_KEY_LEFT_BRACKET:  // Shift + [: Z轴正方向干扰力矩
          apply_perturb = true;
          perturb_torque.z() = perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发Z轴干扰力矩: " << perturb_torque.z();
          break;

        case GLFW_KEY_RIGHT_BRACKET:  // Shift + ]: Z轴负方向干扰力矩
          apply_perturb = true;
          perturb_torque.z() = -perturb_torque_magnitude;
          perturb_start_time = -1.0;
          LOG(INFO) << "触发Z轴干扰力矩: " << perturb_torque.z();
          break;

        case GLFW_KEY_0:  // Shift + 0: 立即停止所有干扰力/力矩
          apply_perturb = false;
          perturb_start_time = -1.0;
          LOG(INFO) << "立即停止所有干扰力/力矩";
          break;

        case GLFW_KEY_EQUAL:  // Shift + +: 增加干扰力/力矩大小
          perturb_force_magnitude += 20.0;
          perturb_torque_magnitude += 5.0;
          LOG(INFO) << "干扰力/力矩大小增加到: " << perturb_force_magnitude << " N, " << perturb_torque_magnitude
                    << " N.m";
          break;

        case GLFW_KEY_MINUS:  // Shift + -: 减小干扰力/力矩大小
          perturb_force_magnitude = std::max(20.0, perturb_force_magnitude - 20.0);
          perturb_torque_magnitude = std::max(5.0, perturb_torque_magnitude - 5.0);
          LOG(INFO) << "干扰力/力矩大小减小到: " << perturb_force_magnitude << " N, " << perturb_torque_magnitude
                    << " N.m";
          break;

        case GLFW_KEY_PERIOD:  // Shift + .: 增加干扰力持续时间
          perturb_duration += 0.1;
          LOG(INFO) << "干扰力/力矩持续时间增加到: " << perturb_duration << "秒";
          break;

        case GLFW_KEY_COMMA:  // Shift + ,: 减小干扰力持续时间
          perturb_duration = std::max(0.1, perturb_duration - 0.1);
          LOG(INFO) << "干扰力/力矩持续时间减小到: " << perturb_duration << "秒";
          break;
      }
    }
  }

 private:
  bool shift_pressed{false};
};

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop

bool InitMain(int argc, char** argv) {
  std::string default_robot_name = argc > 1 ? argv[1] : "";
  if (!common::InitializeGlobalConfigPath(default_robot_name)) {
    std::cout << "[ERROR] Terminated since failed to initialize global config path." << std::endl;
    return false;
  }

  common::GoogleLogger google_logger(argv[0]);
  LOG(INFO) << "Loaded config files from: " << common::GlobalPathManager::GetInstance().GetConfigPath();

  return true;
}

/**
 * @brief 主函数 - 推倒采样MuJoCo仿真器入口点
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return 程序退出状态
 * 
 * 这个主函数实现了完整的推倒采样仿真环境：
 * 1. 初始化MuJoCo和配置系统
 * 2. 设置LCM通信（替代ROS2）
 * 3. 加载机器人模型和插件
 * 4. 启动物理仿真线程和UI渲染循环
 * 5. 实现推倒采样的交互式控制
 */
int main(int argc, char** argv) {
  // 打印MuJoCo版本并检查兼容性
  LOG(INFO) << "MuJoCo version " << mj_versionString();
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // 初始化全局配置路径
  InitMain(argc, argv);
  data::ModelParam model_param;
  std::string filename = common::PathJoin(common::GlobalPathManager::GetInstance().GetResourcePath(), model_param.xml);

  // 初始化LCM数据存储 - 用于与外部控制器通信
  int num_contacts = model_param.contact.size();
  int num_single_contact_dimensions = single_contact_dimensions_map.at(model_param.contact[0].type);
  lcm_data_store =
      std::make_shared<LcmDataStore>(model_param.num_total_joints, num_contacts, num_single_contact_dimensions);

  // 初始化LCM订阅者和发布者 - 替代ROS2通信
  data::LcmParam lcm_param;
  std::shared_ptr<LcmSubscriber> lcm_subscriber = std::make_shared<LcmSubscriber>(lcm_param, lcm_data_store);
  lcm_subscriber->TaskCreate();
  lcm_subscriber->TaskInit();
  lcm_subscriber->TaskStart();

  std::shared_ptr<LcmPublisher> lcm_publisher = std::make_shared<LcmPublisher>(lcm_param, lcm_data_store);
  lcm_publisher->TaskCreate();
  lcm_publisher->TaskInit();
  lcm_publisher->TaskStart();

  // 扫描插件目录以加载额外的MuJoCo插件
  scanPluginLibraries();

  // 初始化MuJoCo可视化组件
  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // 安装控制回调函数 - 实现PD控制和推倒采样干扰力
  mjcb_control = TorqueController;

  // 创建仿真对象，封装UI界面
  // 使用自定义的GLFW适配器以支持推倒采样的键盘交互
  auto sim = std::make_unique<mj::Simulate>(std::make_unique<CustomGlfwAdapter>(), &cam, &opt, &pert,
                                            /* is_passive = */ false);

  // 启动物理仿真线程 - 在后台运行物理计算
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());

  // 启动仿真UI循环 - 阻塞调用，处理渲染和用户交互
  sim->RenderLoop();
  
  // 等待物理线程结束
  physicsthreadhandle.join();

  return 0;
}
