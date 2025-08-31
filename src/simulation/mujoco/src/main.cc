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
  rclcpp::shutdown();
}

int main(int argc, char** argv) {
  // 设置信号处理
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);
  
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
  rclcpp::shutdown();
  return 0;
}
