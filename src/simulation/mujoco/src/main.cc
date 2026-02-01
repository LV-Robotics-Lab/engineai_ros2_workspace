#include <rclcpp/rclcpp.hpp>
#include <cstring>
#include <signal.h>
#include <iostream>
#include "sim_manager.h"

// 全局变量用于信号处理
static bool g_shutdown_requested = false;

// 信号处理函数
void signal_handler(int signal) {
  std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
  g_shutdown_requested = true;
  
  // 强制关闭ROS上下文
  try {
    rclcpp::shutdown();
  } catch (const std::exception& e) {
    std::cerr << "Exception during shutdown: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception during shutdown" << std::endl;
  }
}

int main(int argc, char** argv) {
  // 设置信号处理
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);
  
  // 打印推外力控制说明
  std::cout << "=== MuJoCo 仿真器启动 ===" << std::endl;
  std::cout << "推外力控制快捷键：" << std::endl;
  std::cout << "  Shift + W/S: 前后向干扰力" << std::endl;
  std::cout << "  Shift + A/D: 左右向干扰力" << std::endl;
  std::cout << "  Shift + Q/E/Z/C: 左前/右前/左后/右后（斜 45°）" << std::endl;
  std::cout << "  Shift + U/K: 上下向干扰力" << std::endl;
  std::cout << "  Shift + G/J: X轴干扰力矩" << std::endl;
  std::cout << "  Shift + Y/H: Y轴干扰力矩" << std::endl;
  std::cout << "  Shift + [/]: Z轴干扰力矩" << std::endl;
  std::cout << "  Shift + 0: 立即停止干扰力" << std::endl;
  std::cout << "  Shift + +/-: 调整干扰力大小" << std::endl;
  std::cout << "  Shift + ,/.: 调整干扰力持续时间" << std::endl;
  std::cout << "=========================" << std::endl;
  
  // 解析命令行参数
  bool export_contact = false;
  std::string contact_topic = "/mujoco/contact_forces";
  bool save_contact_csv = false;
  std::string csv_file_path = "";
  
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--export_contact") == 0) {
      export_contact = true;
    } else if (strcmp(argv[i], "--contact_topic") == 0 && i + 1 < argc) {
      contact_topic = argv[++i];
    } else if (strcmp(argv[i], "--save_contact_csv") == 0) {
      save_contact_csv = true;
    } else if (strcmp(argv[i], "--csv_file_path") == 0 && i + 1 < argc) {
      csv_file_path = argv[++i];
    }
  }

  rclcpp::init(argc, argv);
  auto& sim_manager = SimManager::GetInstance();

  if (!sim_manager.Initialize()) {
    return 1;
  }

  try {
    sim_manager.Run();
  } catch (const std::exception& e) {
    std::cerr << "Exception in simulation: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception in simulation" << std::endl;
  }
  
  std::cout << "Shutting down ROS..." << std::endl;
  
  // 强制清理资源
  try {
    rclcpp::shutdown();
  } catch (const std::exception& e) {
    std::cerr << "Exception during ROS shutdown: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception during ROS shutdown" << std::endl;
  }
  
  // 等待一段时间确保资源清理完成
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  return 0;
}
