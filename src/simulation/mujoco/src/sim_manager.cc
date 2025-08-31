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
 * @details 实现PD控制器，根据关节命令计算控制力矩
 */
void SimManager::TorqueController(const mjModel* m, mjData* d) {
  if (!ros_interface_) {
    return;
  }

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
  
  // 添加调试信息
  RCLCPP_INFO(logger, "Contact visualization enabled: CONTACTPOINT=%d, CONTACTFORCE=%d, CONTACTSPLIT=%d", 
              opt.flags[mjVIS_CONTACTPOINT], opt.flags[mjVIS_CONTACTFORCE], opt.flags[mjVIS_CONTACTSPLIT]);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // 创建仿真对象
  sim_ = std::make_unique<mj::Simulate>(std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, false);

  return true;
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
      RCLCPP_INFO(logger, "Contact visualization flags set in sim_->opt");

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




